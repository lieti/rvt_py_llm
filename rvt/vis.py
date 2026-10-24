"""
Relief Visualization Toolbox – visualization algorithms (``rvt.vis``).

This module holds the numerical algorithms of rvt. Every function takes a 2D elevation array in memory and
returns arrays; there is no file I/O here. Callers:

- ``rvt.default.DefaultValues.get_*`` (and through them ``save_*`` and ``rvt.tile``) call the public
  functions with the settings of a ``DefaultValues`` object;
- ``rvt.blend`` calls ``hillshade``, ``slope_aspect`` and ``byte_scale`` for blending layers and the
  e3MSTP / e4MSTP recipes;
- ``rvt.tile`` runs any of them tile by tile on large rasters (via ``rvt.default``).

Contents:

- terrain derivatives: ``slope_aspect``, ``roll_fill_nans``, ``hillshade``, ``multi_hillshade``;
- local relief: ``mean_filter``, ``slrm`` (simple local relief model), ``msrm`` (multi-scale relief model);
- horizon based: ``horizon_shift_vector``, ``sky_view_factor_compute``, ``sky_view_factor`` (SVF,
  anisotropic SVF, openness), ``local_dominance``;
- sky illumination: ``horizon_generate_coarse_dem``, ``horizon_generate_pyramids``, ``sky_illumination``,
  ``shadow_horizon``;
- topographic position: ``integral_image``, ``topographic_dev``, ``max_elevation_deviation``, ``mstp``;
- utilities: ``byte_scale`` (float to uint8), ``fill_where_nan`` (NaN interpolation).

Accelerated paths
-----------------
Five functions have a faster path that returns bitwise identical results to the original numpy code:

- ``mean_filter``, ``sky_view_factor_compute``, ``local_dominance``, ``sky_illumination`` and
  ``max_elevation_deviation`` (used by ``mstp``) compute only the output window, as slices of the padded
  array instead of ``np.roll`` copies. The slices equal the rolled arrays wherever the output is kept,
  because every shift is at most the padding width; where this is not guaranteed by construction the
  function checks it at run time and otherwise runs its original full-extent loop, which is kept as the
  reference. SVF, local dominance and sky illumination work in row blocks of about 32768 elements
  (``rvt._blocking.BLOCK_ELEMS``, 256 KiB of float64 per buffer) so that the per-block buffers stay in the CPU cache;
- the inner loops of four of them (SVF horizon scan, local dominance, sky illumination level search,
  MSTP deviation) also have a numba kernel, registered with ``rvt._accel.jit_kernel`` and run through
  ``rvt._accel.call``. When numba is missing, fails, or ``RVT_NO_ACCEL=1`` is set, the call returns False
  and the numpy path runs. Each kernel performs the same IEEE operations as its numpy path, in the same
  order and in the same dtypes (no fastmath, no FMA, no reassociation); its docstring states the exact
  equivalence contract;
- the kernels are serial, so results do not depend on the thread count (``RVT_NUM_THREADS``,
  ``NUMBA_NUM_THREADS``; see ``rvt._accel``).

Conventions (docs/CONVENTIONS.md)
---------------------------------
- Elevation in metres, 2D ``numpy.ndarray``, row 0 is north. Pixel size (``resolution*``) in m/px.
- Radii and distances in px unless the parameter says m (MSRM feature sizes are in m).
- ``no_data``: pixels equal to it are set to NaN. Several functions do this in the caller's array before
  converting the dtype; each docstring states whether the input is modified.
- Inputs are converted to float32; outputs are float32 or float64, stated per function.
- NaN in the input stays NaN in the output at the same pixels unless stated otherwise.
- Edges: each function pads with its own mode (edge, reflect, symmetric, constant), stated per function.
- ``ve_factor`` (vertical exaggeration) is checked to lie in [-10000, 10000] (``mstp``: [-1000, 10000]).

Behaviour that looks unintended is described in the Notes of the function concerned and listed in
docs/KNOWN_ISSUES.md; it is kept unchanged.

Credits:
    Žiga Kokalj (ziga.kokalj@zrc-sazu.si)
    Krištof Oštir (kristof.ostir@fgg.uni-lj.si)
    Klemen Zakšek
    Peter Pehani
    Klemen Čotar
    Maja Somrak
    Žiga Maroh
    Nejc Čož

Copyright:
    2010-2020 Research Centre of the Slovenian Academy of Sciences and Arts
    2016-2020 University of Ljubljana, Faculty of Civil and Geodetic Engineering
"""

# python libraries
import numpy as np
from rvt import _accel
from rvt._blocking import BLOCK_ELEMS as _BLOCK_ELEMS
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# Window shapes of mean_filter, slrm, msrm, max_elevation_deviation and mstp (keyword ``kernel_shape``). Windows of
# radius r >= DISK_MIN_RADIUS_PX px take the disk footprint when the shape is "disk" or "disk_equal_area"; smaller
# windows stay square, because a disk of a few px is dominated by its discretisation (docs/DISK_WINDOWS.md).
DISK_MIN_RADIUS_PX = 8
_KERNEL_SHAPES = ("square", "disk", "disk_equal_area")


def byte_scale(data,
               c_min=None,
               c_max=None,
               high=255,
               low=0,
               no_data=None
               ):
    """
    Scale an array linearly to uint8 (byte scaling, after IDL BYTSCL).

    Each band is scaled with ``(high + 0.9999) * (data - c_min) / (c_max - c_min)``, clipped to [0, high],
    NaN set to 255, truncated to uint8, and ``low`` is added. This replaces the removed
    ``scipy.misc.bytescale``; its rounding differs from it.

    Parameters
    ----------
    data : numpy.ndarray
        2D array (H, W), or 3D array (B, H, W) whose bands are scaled independently. Any numeric dtype;
        uint8 input takes a separate branch (see Notes).
    c_min : int or float or None
        Value mapped to 0. Default None: ``np.nanmin`` of each band.
    c_max : int or float or None
        Value mapped to ``high``. Default None: ``np.nanmax`` of each band.
    high : int
        Upper end of the output range before ``low`` is added. Default 255.
    low : int
        Offset added after scaling and clipping (uint8 addition, wraps above 255). Default 0.
    no_data : int or float or None
        Pixels equal to it are set to NaN and end up as 255 + ``low``. For a 3D input this is done in the
        caller's array (the bands are views of it); a 2D input is copied first and is not modified.
        Default None.

    Returns
    -------
    numpy.ndarray
        uint8, same shape as ``data`` (for uint8 input see Notes).

    Raises
    ------
    ValueError
        If ``high < low`` or ``c_max < c_min``.
    OverflowError
        For uint8 input with ``high = 255`` (numpy 2: ``high + 1 = 256`` does not fit uint8).

    Notes
    -----
    Known issues (see docs/KNOWN_ISSUES.md):

    - ``low`` is added after clipping, so the maximum is ``high + low``, and the NaN code 255 wraps
      (255 + 10 = 9 in uint8);
    - for a constant band ``c_scale`` is set to 1 but the division still uses ``c_max - c_min = 0``, so
      every pixel becomes 0 / 0 = NaN and then 255;
    - uint8 input uses ``(high + 1) * (data - c_min - 1) / (c_max - c_min)`` in uint8 arithmetic and
      returns from inside the band loop, so a 3D uint8 input returns only its first band.

    See Also
    --------
    rvt.default.DefaultValues.float_to_8bit
    """
    is_2d_arr = False
    data_bands = data
    if len(data.shape) == 2:
        is_2d_arr = True
        data_bands = np.array([data])

    c_min_orig = c_min
    c_max_orig = c_max

    byte_data_bands = []
    for i_band in data_bands:
        data = i_band
        c_min = c_min_orig
        c_max = c_max_orig
        if high < low:
            raise ValueError("`high` should be larger than `low`.")

        if no_data is not None:  # change no data to np.nan
            data[data == no_data] = np.nan

        if c_min is None:
            c_min = np.nanmin(data)
        if c_max is None:
            c_max = np.nanmax(data)

        c_scale = c_max - c_min
        if c_scale < 0:
            raise ValueError("`cmax` should be larger than `cmin`.")
        elif c_scale == 0:
            c_scale = 1
        # c_scale is not used below: the divisions use c_max - c_min, so a constant band gives 0 / 0 = NaN, i.e. 255
        # (known issue, see docs/KNOWN_ISSUES.md)

        if data.dtype == np.uint8:
            # TODO: the following line seems not good to  me - if cmin=0, then that pixel will get negative value
            byte_data = (high + 1) * (data - c_min - 1) / (c_max - c_min)  # copied from IDL BYTSCL
            byte_data[byte_data > high] = high
            byte_data[byte_data < 0] = 0
            byte_data[np.isnan(byte_data)] = 0  # change no_data to 0
            return np.asarray(byte_data, dtype=np.uint8) + np.asarray(low, dtype=np.uint8)

        byte_data = (high + 0.9999) * (data - c_min) / (c_max - c_min)  # copied from IDL BYTSCL
        byte_data[byte_data > high] = high
        byte_data[byte_data < 0] = 0
        byte_data[np.isnan(byte_data)] = 255  # change no_data to 255
        byte_data = np.asarray(byte_data, dtype=np.uint8) + np.asarray(low, dtype=np.uint8)
        byte_data_bands.append(byte_data)

    if is_2d_arr:  # if only one band
        return byte_data_bands[0]
    else:  # multiple bands
        return np.array(byte_data_bands)


def slope_aspect(dem,
                 resolution_x=1,
                 resolution_y=1,
                 output_units="radian",
                 ve_factor=1,
                 no_data=None
                 ):
    """
    Compute slope and aspect with central finite differences.

    The DEM is converted to float32, padded by 1 px in edge mode, multiplied by ``ve_factor``, and
    differentiated with central differences ``dz/dx = (z[i, j-1] - z[i, j+1]) / (2 * resolution_x)`` and
    ``dz/dy = (z[i+1, j] - z[i-1, j]) / (2 * resolution_y)``. A NaN neighbour is replaced by the centre
    pixel (``roll_fill_nans``), so next to NaN and at the raster edge the difference is half the one-sided
    difference. Slope is 0 for a horizontal plane and pi/2 for a vertical one. NaN input pixels are NaN
    in both outputs.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W). Converted to float32 (a copy); the caller's array is not
        modified.
    resolution_x : float
        Pixel size in x (m/px). Default 1. Negative values raise; 0 is not rejected.
    resolution_y : float
        Pixel size in y (m/px). Default 1.
    output_units : str
        ``"radian"`` (default), ``"degree"`` or ``"percent"``. Applies to the slope; the aspect is in
        degrees for ``"degree"`` and in radians otherwise.
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it (after the float32 conversion) are set to NaN. Default None.

    Returns
    -------
    dict
        ``{"slope": numpy.ndarray float32 (H, W), "aspect": numpy.ndarray float32 (H, W)}``.

    Raises
    ------
    Exception
        If ``dem`` is not 2D, ``ve_factor`` is outside [-10000, 10000], a resolution is negative, or
        ``output_units`` is not one of the three names.

    Notes
    -----
    Aspect is ``arctan2(dz/dx, dz/dy)``, the down-slope direction measured clockwise from north, in
    (-pi, pi]: north 0, east pi/2, south pi, west -pi/2. Where ``dz/dy == 0`` it is replaced by 1e-8 for
    numeric stability, so a flat pixel has aspect 0. Known issue (see docs/KNOWN_ISSUES.md): earlier
    documentation said "0 or 2pi for north", i.e. [0, 2pi); the code returns (-pi, pi].

    See Also
    --------
    hillshade, roll_fill_nans
    """
    if dem.ndim != 2:
        raise Exception("rvt.vis.slope_aspect: dem has to be 2D np.array!")
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.slope_aspect: ve_factor must be between -10000 and 10000!")
    if resolution_x < 0 or resolution_y < 0:
        raise Exception("rvt.vis.slope_aspect: resolution must be a positive number!")

    # Make sure array has the correct dtype!
    dem = dem.astype(np.float32)

    # Change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    # Save NaN mask
    nan_dem = np.isnan(dem)

    # Add 1 pixel edge padding
    dem = np.pad(array=dem, pad_width=1, mode="edge")

    # Vertical exaggeration
    dem = dem * ve_factor

    # Derivatives in X and Y direction
    dzdx = ((roll_fill_nans(dem, 1, axis=1) - roll_fill_nans(dem, -1, axis=1)) / 2) / resolution_x
    dzdy = ((roll_fill_nans(dem, -1, axis=0) - roll_fill_nans(dem, 1, axis=0)) / 2) / resolution_y
    tan_slope = np.sqrt(dzdx ** 2 + dzdy ** 2)

    # Compute slope
    if output_units == "percent":
        slope_out = tan_slope * 100
    elif output_units == "degree":
        slope_out = np.rad2deg(np.arctan(tan_slope))
    elif output_units == "radian":
        slope_out = np.arctan(tan_slope)
    else:
        raise Exception("rvt.vis.slope_aspect: Wrong function input 'output_units'!")

    # Compute Aspect
    # aspect identifies the down slope direction of the maximum rate of change in value from each cell to its neighbors:
    #     0
    # 270    90
    #    180
    dzdy[dzdy == 0] = 10e-9  # important for numeric stability - where dzdy is zero, make tangent to really high value
    aspect_out = np.arctan2(dzdx, dzdy)  # atan2 took care of the quadrants
    if output_units == "degree":
        aspect_out = np.rad2deg(aspect_out)

    # Remove padding
    aspect_out = aspect_out[1:-1, 1:-1]
    slope_out = slope_out[1:-1, 1:-1]

    # Apply NaN mask
    slope_out[nan_dem] = np.nan
    aspect_out[nan_dem] = np.nan

    return {"slope": slope_out, "aspect": aspect_out}


def roll_fill_nans(dem, shift, axis):
    """
    Return ``np.roll(dem, shift, axis)`` with NaN replaced by the pixel's own value.

    Where the rolled array and ``dem`` differ in NaN-ness, the rolled value is replaced by the value of
    ``dem`` at the same position: a NaN neighbour becomes the centre value, and a NaN centre stays NaN.
    For a finite difference this acts like edge padding at NaN borders.

    Parameters
    ----------
    dem : numpy.ndarray
        Array to roll (any shape, floating dtype). Not modified.
    shift : int
        Number of pixels to roll by, as in ``numpy.roll``.
    axis : int
        Axis to roll along.

    Returns
    -------
    numpy.ndarray
        New array, same shape and dtype as ``dem``.

    Notes
    -----
    ``np.roll`` wraps around, so the first/last row or column receives the opposite edge.
    ``slope_aspect`` pads by 1 px before rolling and crops afterwards, so wrapped values never reach its
    output.

    See Also
    --------
    slope_aspect, numpy.roll
    """
    out = np.roll(dem, shift, axis=axis)

    # Fill NaNs with values from dem
    out[np.isnan(out) != np.isnan(dem)] = dem[np.isnan(out) != np.isnan(dem)]

    return out


def hillshade(dem,
              resolution_x,
              resolution_y,
              sun_azimuth=315,
              sun_elevation=35,
              slope=None,
              aspect=None,
              ve_factor=1,
              no_data=None
              ):
    """
    Compute analytical hillshading (cosine of the solar incidence angle).

    ``cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(aspect - azimuth)``, with negative values
    set to 0. When ``slope`` or ``aspect`` is None, both are computed with ``slope_aspect`` from the DEM
    padded by 1 px in edge mode and multiplied by ``ve_factor``; the result is then cropped by 1 px on each
    side back to (H, W).

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W).
    resolution_x : float
        Pixel size in x (m/px). Must not be negative.
    resolution_y : float
        Pixel size in y (m/px). Must not be negative.
    sun_azimuth : int or float
        Sun azimuth in degrees clockwise from north, in [0, 360]. Default 315.
    sun_elevation : int or float
        Sun elevation in degrees above the horizon, in [0, 90]. Default 35.
    slope : numpy.ndarray or None
        Precomputed slope (radians). Used only together with ``aspect``; see Notes for the shape.
        Default None (computed).
    aspect : numpy.ndarray or None
        Precomputed aspect (radians, as returned by ``slope_aspect``). Default None (computed).
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1. Not used when ``slope`` and ``aspect`` are
        given.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**, before the float32 conversion (an
        integer array raises ValueError then). Default None.

    Returns
    -------
    numpy.ndarray
        float64, (H, W) when slope and aspect are computed; values in [0, 1], NaN where the input is NaN.

    Raises
    ------
    Exception
        If ``dem`` is not 2D, ``ve_factor`` is outside [-10000, 10000], the sun angles are outside their
        ranges, or a resolution is negative.

    Notes
    -----
    The output is always cropped by 1 px on each side. This fits the internally computed slope, which has
    the padded shape (H + 2, W + 2). A ``slope`` and ``aspect`` passed in must therefore have shape
    (H + 2, W + 2) (computed from the edge-padded DEM) to give an (H, W) result; an (H, W) slope gives an
    (H - 2, W - 2) result. Known issue, see docs/KNOWN_ISSUES.md; ``multi_hillshade`` is affected.

    See Also
    --------
    slope_aspect, multi_hillshade
    """
    if dem.ndim != 2:
        raise Exception("rvt.vis.hillshade: dem has to be 2D np.array!")
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.hillshade: ve_factor must be between -10000 and 10000!")
    if sun_azimuth > 360 or sun_elevation > 90 or sun_azimuth < 0 or sun_elevation < 0:
        raise Exception("rvt.vis.hillshade: sun_azimuth must be [0-360] and sun_elevation [0-90]!")
    if resolution_x < 0 or resolution_y < 0:
        raise Exception("rvt.vis.hillshade: resolution must be a positive number!")

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    # add 1 pixel edge padding
    dem = np.pad(array=dem, pad_width=1, mode="edge")
    dem = dem * ve_factor

    # Convert solar position (degrees) to radians
    sun_azimuth_rad = np.deg2rad(sun_azimuth)
    sun_elevation_rad = np.deg2rad(sun_elevation)

    # Convert to solar zenith angle
    sun_zenith_rad = np.pi / 2 - sun_elevation_rad

    # are slope and aspect already calculated and presented
    if slope is None or aspect is None:
        # calculates slope and aspect
        dict_slp_asp = slope_aspect(dem=dem, resolution_x=resolution_x, resolution_y=resolution_y,
                                    output_units="radian")
        slope = dict_slp_asp["slope"]
        aspect = dict_slp_asp["aspect"]

    # Compute solar incidence angle, hillshading
    hillshade_out = np.cos(sun_zenith_rad) * np.cos(slope) + np.sin(sun_zenith_rad) * np.sin(slope) * np.cos(
        aspect - sun_azimuth_rad)

    hillshade_out[hillshade_out < 0] = 0  # set all negative to 0

    # remove padding
    hillshade_out = hillshade_out[1:-1, 1:-1]

    return hillshade_out


def multi_hillshade(dem,
                    resolution_x,
                    resolution_y,
                    nr_directions=16,
                    sun_elevation=35,
                    slope=None,
                    aspect=None,
                    ve_factor=1,
                    no_data=None
                    ):
    """
    Compute hillshades for ``nr_directions`` sun azimuths spaced evenly from north.

    The azimuths are ``360 / nr_directions * k`` degrees for k = 0 .. nr_directions - 1 (0, 22.5, ... for
    16). Slope and aspect are computed once from the DEM multiplied by ``ve_factor`` (unless given) and
    passed to ``hillshade`` for every azimuth.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W).
    resolution_x : float
        Pixel size in x (m/px). Must not be negative.
    resolution_y : float
        Pixel size in y (m/px). Must not be negative.
    nr_directions : int
        Number of sun azimuths, at least 1. Default 16.
    sun_elevation : int or float
        Sun elevation in degrees above the horizon, in [0, 90]. Default 35.
    slope : numpy.ndarray or None
        Precomputed slope (radians), used together with ``aspect``. Default None (computed).
    aspect : numpy.ndarray or None
        Precomputed aspect (radians). Default None (computed).
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**, before the float32 conversion.
        Default None.

    Returns
    -------
    numpy.ndarray
        float64, shape (nr_directions, H - 2, W - 2) (see Notes); band k is lit from azimuth
        ``360 / nr_directions * k``.

    Raises
    ------
    Exception
        If ``dem`` is not 2D, ``sun_elevation`` is outside [0, 90], a resolution is negative,
        ``nr_directions < 1``, or ``ve_factor`` is outside [-10000, 10000].

    Notes
    -----
    Known issue (see docs/KNOWN_ISSUES.md): the slope and aspect computed here have the shape (H, W) of
    the unpadded DEM, and ``hillshade`` crops 1 px on each side of whatever it receives, so the result is
    (nr_directions, H - 2, W - 2), not (nr_directions, H, W). The one-pixel border is lost; the other
    pixels equal ``hillshade`` of the same DEM.

    See Also
    --------
    hillshade, slope_aspect
    """
    if dem.ndim != 2:
        raise Exception("rvt.vis.multi_hillshade: dem has to be 2D np.array!")
    if sun_elevation > 90 or sun_elevation < 0:
        raise Exception("rvt.vis.multi_hillshade: sun_elevation must be [0-90]!")
    if resolution_x < 0 or resolution_y < 0:
        raise Exception("rvt.vis.multi_hillshade: resolution must be a positive number!")
    if nr_directions < 1:
        raise Exception("rvt.vis.multi_hillshade: nr_directions must be a positive number!")
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.multi_hillshade: ve_factor must be between -10000 and 10000!")

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    dem = dem * ve_factor

    # calculates slope and aspect if they are not added
    if slope is None or aspect is None:  # slope and aspect are the same, so we have to calculate it once
        dict_slp_asp = slope_aspect(dem=dem, resolution_x=resolution_x, resolution_y=resolution_y,
                                    output_units="radian")
        slope = dict_slp_asp["slope"]
        aspect = dict_slp_asp["aspect"]

    hillshades_arr_list = []  # list of all hillshades in different directions
    for i_direction in range(nr_directions):
        sun_azimuth = (360 / nr_directions) * i_direction
        hillshading = hillshade(dem=dem, resolution_x=resolution_x, resolution_y=resolution_y,
                                sun_elevation=sun_elevation, sun_azimuth=sun_azimuth, slope=slope, aspect=aspect)
        hillshades_arr_list.append(hillshading)
    multi_hillshade_out = np.asarray(hillshades_arr_list)

    return multi_hillshade_out


def _box_sum(table, lo, hi, n_rows, n_cols):
    """
    Return the box sums of an integral image ``table`` on an output window, (n_rows, n_cols), in its dtype.

    ``lo`` and ``hi`` are the offsets of the top-left and bottom-right corner terms of the window in the
    table: for a box of radius r in a DEM padded by (p + 1, p), ``lo = p - r`` and ``hi = p + r + 1``. The
    result is ``((I[lo, lo] + I[hi, hi]) - I[hi, lo]) - I[lo, hi]``, the terms of the reference
    ``roll(r, r) + roll(-r-1, -r-1) - roll(-r-1, r) - roll(r, -r-1)`` in the same order and dtype, taken as
    slices: on the output window no roll wraps, so the values are bitwise identical.

    Used by ``mean_filter`` (p = r, so ``lo = 0``, ``hi = 2r + 1``) and by ``max_elevation_deviation``
    (p = ``maximum_radius``, one call per radius and table).
    """
    box = np.add(table[lo:lo + n_rows, lo:lo + n_cols], table[hi:hi + n_rows, hi:hi + n_cols])
    box -= table[hi:hi + n_rows, lo:lo + n_cols]
    box -= table[lo:lo + n_rows, hi:hi + n_cols]
    return box


def _check_kernel_shape(kernel_shape, where, error):
    """
    Raise ``error`` (the exception type the calling function uses for bad arguments), with a message that starts
    with ``where``, unless ``kernel_shape`` is "square", "disk" or "disk_equal_area".
    """
    if not (isinstance(kernel_shape, str) and kernel_shape in _KERNEL_SHAPES):
        raise error("{}: kernel_shape must be 'square', 'disk' or 'disk_equal_area'!".format(where))


def _disk_path(kernel_shape, radii, where, error):
    """
    Validate ``kernel_shape`` (``_check_kernel_shape``) and return True when the call takes the disk path.

    ``radii`` is a callable that returns the window radii of the call in px; it is evaluated only for "disk" and
    "disk_equal_area", so on the square path this function only checks the value. The disk path is taken when the
    shape is not "square" and the largest radius (``int``) is at least ``DISK_MIN_RADIUS_PX``; a call whose radii are
    all smaller runs the square code unchanged.
    """
    _check_kernel_shape(kernel_shape, where, error)
    if kernel_shape == "square":
        return False
    radii = [int(r) for r in radii()]
    return len(radii) > 0 and max(radii) >= DISK_MIN_RADIUS_PX


def _window_radius(r, kernel_shape):
    """
    Return the radius r_w (px, float) of the disk window that stands for the square window of radius r px.

    "disk": ``r_w = r`` (the disk inscribed in the square); "disk_equal_area": ``r_w = (2r + 1) / sqrt(pi)``, the disk
    with the area (2r + 1)^2 of the square window (e.g. r = 20 px gives r_w = 23.13 px = 5.78 m at 0.25 m/px).
    """
    if kernel_shape == "disk_equal_area":
        return float((2 * r + 1) / np.sqrt(np.pi))
    return float(r)


def _window_footprint(r_w, disk):
    """
    Return the boolean footprint of a window of radius ``r_w`` px, shape (2b + 1, 2b + 1) with b = floor(r_w).

    Disk (``disk`` True): offset (dy, dx) from the centre pixel belongs to the window iff ``dx^2 + dy^2 <= r_w^2``
    (the exact discrete disk; the convention of GRASS r.neighbors -c, SAGA and ArcGIS). Square: every offset with
    |dx|, |dy| <= b, the (2r + 1) x (2r + 1) window of the square path for an integer ``r_w = r``.
    """
    b = int(np.floor(r_w))
    if not disk:
        return np.ones((2 * b + 1, 2 * b + 1), dtype=bool)
    d = np.arange(-b, b + 1)
    return d[:, None] * d[:, None] + d[None, :] * d[None, :] <= r_w * r_w


def _disk_window_sums(dem_pad, out_shape, windows, squares):
    """
    Return the mean-removed window sums of a padded tile for several windows, by FFT convolution.

    Contract
    --------
    Inputs: ``dem_pad`` the 2D tile padded by p px on every side with the padding mode of the calling function
    (float32 or float64, NaN = no data), p >= floor(r_w) of every window; ``out_shape`` (H, W) the shape of the tile
    before padding; ``windows`` a sequence of (r_w, disk) pairs (radius in px, footprint of ``_window_footprint``);
    ``squares`` True to compute the sums of squares too.

    Returns ``(zbar, sums)``: ``zbar`` (float) the mean of the non-NaN pixels of ``dem_pad`` in float64 (0.0 when all
    are NaN); ``sums`` an iterator that yields, per window in the order given, float64 arrays (H, W) on the output
    window (padded pixel (p + i, p + j) for output pixel (i, j)):

    - ``s1``: the sum of ``z - zbar`` over the valid pixels of the window;
    - ``s2``: the sum of ``(z - zbar)^2`` over the valid pixels (None if ``squares`` is False);
    - ``count``: the number of valid pixels, rounded to the nearest integer (``np.rint``) from the convolution of
      the valid mask when the tile has NaN; without NaN the footprint size (a float).

    Algorithm: ``z - zbar`` (NaN set to 0) and, if needed, its square and the valid mask are transformed once per call
    with ``scipy.fft.rfft2`` (float64, ``workers=1``); per window the footprint, centred on index (0, 0) with
    wrap-around, is transformed once, and each product is transformed back with ``irfft2``. The FFT size is
    ``next_fast_len`` of the padded size (deterministic); circular convolution cannot wrap into the output window,
    because the padding p >= floor(r_w) separates it from the far edge by at least the window radius. The kernel
    spectrum is real (the footprint is symmetric under (dy, dx) -> (-dy, -dx)), so its real part is used.

    Why mean removal: sums of z and z^2 at real elevations (hundreds of metres) carry the FFT rounding of the large
    values into the small within-window variance (E[z^2] - E[z]^2 cancels); sums of ``z - zbar`` are exact
    translations of the same statistics, so the rounding error scales with the relief of the tile, not with its
    elevation.

    Cost: three forward transforms of the padded tile per call, and per window one forward and two (three with NaN)
    inverse transforms; memory about four float64 / complex128 arrays of the FFT size. No numba, no threads,
    nothing is kept between calls.
    """
    from scipy import fft as sp_fft  # imported here so that the square path never loads scipy.fft

    n_rows, n_cols = out_shape
    pad = (dem_pad.shape[0] - n_rows) // 2
    fft_shape = (sp_fft.next_fast_len(dem_pad.shape[0], real=True),
                 sp_fft.next_fast_len(dem_pad.shape[1], real=True))
    valid = ~np.isnan(dem_pad)
    has_nan = not bool(valid.all())
    zbar = float(np.nanmean(dem_pad, dtype=np.float64)) if valid.any() else 0.0

    # data spectra, once per call: z - zbar (NaN -> 0), its square, the valid mask (only with NaN)
    z = dem_pad.astype(np.float64)
    z -= zbar
    if has_nan:
        z[~valid] = 0.0
    spec_1 = sp_fft.rfft2(z, s=fft_shape, workers=1)
    spec_2 = None
    if squares:
        np.square(z, out=z)
        spec_2 = sp_fft.rfft2(z, s=fft_shape, workers=1)
    del z
    spec_n = sp_fft.rfft2(valid.astype(np.float64), s=fft_shape, workers=1) if has_nan else None
    del valid
    out = (slice(pad, pad + n_rows), slice(pad, pad + n_cols))

    def _sums():
        for r_w, disk in windows:
            footprint = _window_footprint(r_w, disk)
            b = footprint.shape[0] // 2
            kernel = np.zeros(fft_shape)
            offsets = np.arange(-b, b + 1)
            kernel[np.ix_(offsets % fft_shape[0], offsets % fft_shape[1])] = footprint
            spec_k = sp_fft.rfft2(kernel, workers=1).real
            del kernel
            s1 = sp_fft.irfft2(spec_1 * spec_k, s=fft_shape, workers=1)[out].copy()
            s2 = None
            if spec_2 is not None:
                s2 = sp_fft.irfft2(spec_2 * spec_k, s=fft_shape, workers=1)[out].copy()
            if spec_n is not None:
                count = np.rint(sp_fft.irfft2(spec_n * spec_k, s=fft_shape, workers=1)[out])
            else:
                count = float(np.count_nonzero(footprint))
            del spec_k
            yield s1, s2, count

    return zbar, _sums()


def _mean_filter_disk(dem, radii, kernel_shape):
    """
    Yield the disk mean filter of ``dem`` for each radius in ``radii`` (px), float32 (H, W), NaN where ``dem`` is NaN.

    Every radius gets the disk of ``_window_radius(r, kernel_shape)`` (the caller applies ``DISK_MIN_RADIUS_PX``).
    The DEM is padded in edge mode by floor(r_w) of the largest window, as the square path pads; the window sums of
    ``z - zbar`` come from ``_disk_window_sums`` (data spectra once for all radii) and the mean is
    ``s1 / count + zbar`` in float64, rounded to float32.
    """
    windows = [(_window_radius(r, kernel_shape), True) for r in radii]
    pad = int(np.floor(max(r_w for r_w, _ in windows)))
    idx_nan_dem = np.isnan(dem)
    dem_pad = np.pad(dem, pad, mode="edge")
    zbar, sums = _disk_window_sums(dem_pad, dem.shape, windows, False)
    del dem_pad
    for s1, _, count in sums:
        with np.errstate(divide="ignore", invalid="ignore"):  # windows without a valid pixel: output NaN anyway
            mean = s1 / count
        mean += zbar
        mean = mean.astype(np.float32)
        mean[idx_nan_dem] = np.nan
        yield mean


def mean_filter(dem, kernel_radius, kernel_shape="square"):
    """
    Apply a mean (low-pass) filter with a square or disk window, ignoring NaN.

    Each output pixel is the mean of the non-NaN pixels in the (2r + 1) x (2r + 1) window centred on it,
    with r = ``int(kernel_radius)``. The DEM is padded in edge mode, so windows at the border repeat the
    edge pixels. Window sums come from a float64 summed-area table (integral image).

    Parameters
    ----------
    dem : numpy.ndarray
        2D array (m), shape (H, W), floating dtype (NaN is detected with ``np.isnan``). Not modified.
    kernel_radius : int or float
        Window radius r in px (window 2r + 1 px wide), e.g. r = 10 px = 2.5 m at 0.25 m/px. Truncated with
        ``int``.
    kernel_shape : str
        "square" (default, the (2r + 1) x (2r + 1) window), "disk" (pixels with dx^2 + dy^2 <= r^2) or
        "disk_equal_area" (dx^2 + dy^2 <= r_w^2 with r_w = (2r + 1) / sqrt(pi), the disk with the square's area).
        The disk shapes apply for r >= ``DISK_MIN_RADIUS_PX`` (8 px = 2 m at 0.25 m/px); smaller windows stay
        square (bitwise the default result).

    Returns
    -------
    numpy.ndarray
        float32 (H, W); NaN where ``dem`` is NaN. For ``kernel_radius == 0`` the input array itself is
        returned (same object, not a copy).

    Raises
    ------
    ValueError
        If ``kernel_shape`` is not one of the three values.

    Notes
    -----
    The sums are taken in float64 because a float32 cumulative sum over a large raster loses the small
    differences between neighbouring windows; the mean is rounded to float32 at the end.

    Implementation: the padding is (r + 1) before and r after each axis, so the table has a leading zero
    row and column and the box sum of output pixel (i, j) is ``I[i, j] + I[i+2r+1, j+2r+1] - I[i+2r+1, j]
    - I[i, j+2r+1]`` with plain slices. The original code took these four terms with ``np.roll`` of the
    padded tables and cropped afterwards; on the kept window no roll wraps, so the slices give the same
    float64 values in the same order (bitwise identical). Without NaN the pixel count is the constant
    (2r + 1)^2 as int64; with NaN it is a box sum of an int64 table of the non-NaN mask.

    A ``kernel_radius`` in (0, 1) is not the shortcut for 0 but gives r = 0, a 1 x 1 window, i.e. a float32
    copy of ``dem``.

    Disk windows (``kernel_shape`` "disk" or "disk_equal_area", r >= 8 px): the DEM is padded in edge mode by
    floor(r_w) px and the window sums of ``z - zbar`` (``zbar`` the float64 mean of the non-NaN padded pixels) and
    of the valid mask are FFT convolutions with the disk footprint (``_disk_window_sums``, scipy.fft, float64,
    ``workers=1``); the mean is ``s1 / count + zbar``, rounded to float32. No numba kernel and no threads, so
    ``RVT_NO_ACCEL`` and the thread count do not change the result. See docs/DISK_WINDOWS.md.

    See Also
    --------
    slrm, msrm, integral_image
    """
    if _disk_path(kernel_shape, lambda: (kernel_radius,), "rvt.vis.mean_filter", ValueError):
        return next(_mean_filter_disk(dem, (int(kernel_radius),), kernel_shape))
    radius_cell = int(kernel_radius)

    if kernel_radius == 0:
        return dem

    # store nans
    idx_nan_dem = np.isnan(dem)
    has_nan = bool(idx_nan_dem.any())

    # mean filter
    dem_pad = np.pad(dem, (radius_cell + 1, radius_cell), mode="edge")
    if has_nan:
        # store nans
        idx_nan_dem_pad = np.isnan(dem_pad)
        # change nan to 0
        dem_pad[idx_nan_dem_pad] = 0

    # Box sums over the output window only. With padding (r + 1, r) the four np.roll shifts of the
    # integral image I never wrap inside the window kept after removing the padding, so they are
    # plain slices: roll(I, (r, r)) -> I[:h, :w], roll(I, (-r-1, -r-1)) -> I[2r+1:, 2r+1:],
    # roll(I, (-r-1, r)) -> I[2r+1:, :w], roll(I, (r, -r-1)) -> I[:h, 2r+1:].
    nr_rows, nr_cols = dem_pad.shape[0] - 2 * radius_cell - 1, dem_pad.shape[1] - 2 * radius_cell - 1
    off = 2 * radius_cell + 1

    # integral image (same values as integral_image(dem_pad), cumsum done in place)
    dem_i1 = dem_pad.astype(np.float64, copy=dem_pad.dtype != np.float64)
    np.cumsum(dem_i1, axis=0, out=dem_i1)
    np.cumsum(dem_i1, axis=1, out=dem_i1)
    mean_out = _box_sum(dem_i1, 0, off, nr_rows, nr_cols)
    del dem_i1

    if has_nan:
        # kernel nr pixel integral image
        dem_i_nr_pixels = integral_image(~idx_nan_dem_pad, np.int64)
        kernel_nr_pix_arr = _box_sum(dem_i_nr_pixels, 0, off, nr_rows, nr_cols)
        del dem_i_nr_pixels
    else:
        # no nan: every window holds exactly (2r + 1)^2 pixels (exact int64, as the integral sums)
        kernel_nr_pix_arr = np.int64(off * off)
    # Ignore warnings for dividing with nan
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(mean_out, kernel_nr_pix_arr, out=mean_out)
    mean_out = mean_out.astype(np.float32)
    # nan back to nan
    if has_nan:
        mean_out[idx_nan_dem] = np.nan

    return mean_out


def slrm(dem,
         radius_cell=20,
         ve_factor=1,
         no_data=None
         ):
    """
    Compute the simple local relief model (SLRM): the DEM minus its mean-filtered trend.

    ``slrm = z - mean_filter(z, radius_cell)`` with z the float32 DEM multiplied by ``ve_factor``
    (Hesse 2010; method description in docs/listofvis_slrm.rst).

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W).
    radius_cell : int
        Radius of the trend window in px, in [10, 50]; e.g. r = 20 px = 5 m at 0.25 m/px. Default 20.
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**, before the float32 conversion.
        Default None.

    Returns
    -------
    numpy.ndarray
        float32 (H, W), local relief in m (times ``ve_factor``); NaN where the input is NaN.

    Raises
    ------
    Exception
        If ``dem`` is not 2D, ``radius_cell`` is outside [10, 50], or ``ve_factor`` is outside
        [-10000, 10000].

    Notes
    -----
    Edges: ``mean_filter`` pads in edge mode. NaN pixels are ignored in the trend.

    See Also
    --------
    mean_filter, msrm
    """
    if dem.ndim != 2:
        raise Exception("rvt.vis.slrm: dem has to be 2D np.array!")
    if radius_cell < 10 or radius_cell > 50:
        raise Exception("rvt.vis.slrm: Radius for trend assessment needs to be in interval 10-50 pixels!")
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.slrm: ve_factor must be between -10000 and 10000!")

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    dem = dem * ve_factor

    # mean filter
    dem_mean_filter = mean_filter(dem=dem, kernel_radius=radius_cell)
    slrm_out = dem - dem_mean_filter

    return slrm_out


def horizon_shift_vector(num_directions=16,
                         radius_pixels=10,
                         min_radius=1
                         ):
    """
    Return the pixel shifts and distances of a horizon search along ``num_directions`` directions.

    For direction angle a_k = 360 / num_directions * k degrees, points are sampled on the ray at radii from
    ``min_radius`` to ``radius_pixels`` px in steps of 1/3 px, rounded to whole pixels, de-duplicated and
    sorted by their distance.

    Parameters
    ----------
    num_directions : int
        Number of search directions. Default 16.
    radius_pixels : int or float
        Maximum search radius in px (not m), e.g. r = 10 px = 2.5 m at 0.25 m/px. Default 10.
    min_radius : int or float
        Minimum search radius in px. Default 1. With 0 the shift (0, 0) at distance 0 is included.

    Returns
    -------
    dict
        ``{a_k: {"shift": [(dy, dx), ...], "distance": numpy.ndarray float64}}``, keys are a_k in degrees
        rounded to 1 decimal (``numpy.float64``), in increasing order. ``shift`` holds integer pairs for
        ``np.roll(height, (dy, dx), axis=(0, 1))``; ``distance`` holds ``sqrt(dy**2 + dx**2)`` in px, in
        ascending order, one per shift.

    Notes
    -----
    ``dy = round(cos(a_k) * r)`` and ``dx = round(sin(a_k) * r)``. Since ``np.roll(h, (dy, dx))[i, j] ==
    h[i - dy, j - dx]``, key 0 looks north (towards row 0), 90 west, 180 south and 270 east: the keys are
    measured counterclockwise from north, unlike ``sun_azimuth`` and ``shadow_az``, which are clockwise.
    ``sky_illumination`` converts with ``360 - shadow_az``. Equal distances are ordered by ``np.argsort``
    of the de-duplicated set; the order is deterministic for given arguments.

    See Also
    --------
    sky_view_factor_compute, horizon_generate_pyramids
    """

    # Initialize the output dict
    shift = {}

    # Generate angles and corresponding normal shifts in X (columns)
    # and Y (lines) direction
    angles = (2 * np.pi / num_directions) * np.arange(num_directions)
    x = np.cos(angles)
    y = np.sin(angles)
    angles = np.round(np.degrees(angles), decimals=1)

    # Generate a range of radius values in pixels.
    # Make it finer for the selected scaling.
    # By adding the last constant we make sure that we do not start with
    # point (0,0).
    scale = 3.
    radii = np.arange((radius_pixels - min_radius) * scale + 1) / scale + min_radius

    # For each direction compute all possible horizon point position
    # and round them to integers
    for i in range(num_directions):
        x_int = np.round(x[i] * radii, decimals=0)
        y_int = np.round(y[i] * radii, decimals=0)
        # consider only the minimal number of points
        # use the trick with set and complex number as the input
        coord_complex = set(x_int + 1j * y_int)
        # to sort proportional with increasing radius, 
        # set has to be converted to numpy array
        shift_pairs = np.array([(k.real, k.imag) for k in coord_complex]).astype(int)
        distance = np.sqrt(np.sum(shift_pairs ** 2, axis=1))
        sort_index = np.argsort(distance)
        # write for each direction shifts and corresponding distances
        shift[angles[i]] = {
            "shift": [(k[0], k[1]) for k in shift_pairs[sort_index]],
            "distance": distance[sort_index],
        }

    return shift


def _svf_window_ok(height, radius_max, move):
    """
    Return True if the output-window path of ``sky_view_factor_compute`` is exact for these arguments.

    ``height`` is the DEM padded by ``radius_max`` px. The path replaces ``np.roll(height, (dy, dx))`` by
    slices of ``height``; on the output window (the padded array without its ``radius_max`` border) this is
    the same array only if no roll wraps there, i.e. ``|dy|, |dx| <= radius_max`` for every shift. Also
    required: a 2D array larger than twice the padding, ``radius_max >= 1``, at least one direction, and
    for every direction a non-empty float64 ``distance`` array with one shift per distance (so that the
    division by the radius promotes exactly as in the reference loop). False sends the caller to the
    full-extent reference loop.
    """
    if height.ndim != 2 or radius_max < 1 or len(move) == 0:
        return False
    if height.shape[0] <= 2 * radius_max or height.shape[1] <= 2 * radius_max:
        return False
    for direction in move:
        distance = move[direction]["distance"]
        if not isinstance(distance, np.ndarray) or distance.dtype != np.float64:
            return False
        if len(distance) == 0 or len(move[direction]["shift"]) != len(distance):
            return False
        for dy, dx in move[direction]["shift"]:
            if abs(int(dy)) > radius_max or abs(int(dx)) > radius_max:
                return False
    return True


def _accumulate(acc, term):
    """
    Return ``acc + term``, written into ``acc`` when that gives the same result.

    In place only when ``acc`` already has the promoted dtype of ``acc + term``; then ``np.add(acc, term,
    out=acc)`` runs the same loop as ``acc + term`` and the values are identical. Otherwise (the first
    float32 + float64 step of an accumulator that starts as ``dem * 0``) a new array is returned, as in the
    reference expression ``acc = acc + term``. ``acc`` must be a private buffer of the caller, never a user
    array. Used by the SVF horizon scan (``_svf_compute_window``) and by ``sky_illumination``; both pass
    2D ndarrays for ``acc`` and ``term``, for which the dtype test below gives the same answer as
    ``np.result_type(acc, term)`` under every numpy version (value-based promotion only ever applied to
    0-d arrays and scalars).
    """
    if acc.dtype == np.result_type(acc.dtype, term.dtype):
        return np.add(acc, term, out=acc)
    return acc + term


@_accel.jit_kernel
def _svf_max_slope_kernel(height, row0, col0, dys, dxs, radii, out):
    """
    Numba kernel: running maximum slope of one direction of the SVF horizon scan, for one row block.

    Inputs: ``height`` the padded DEM (float32 or float64, C-contiguous); ``row0``, ``col0`` the padded
    coordinates of the block's first output pixel; ``dys``, ``dxs`` (int64) and ``radii`` (float64) the
    shifts and distances of the direction, in scan order. Output: ``out[i, j]`` (float64) is the maximum
    over k of ``(height[row0 + i - dys[k], col0 + j - dxs[k]] - height[row0 + i, col0 + j]) / radii[k]``,
    starting from ``float64(float32(-1000))``. Serial, so the result does not depend on the thread count.

    Equivalence with the numpy scan of ``_svf_compute_window`` (bitwise):

    - the difference is taken in the height dtype (numba keeps float32 - float32 in float32, like the
      ``d_buf`` buffer), widened exactly to float64 and divided by the same float64 radius (``q_buf``);
    - the start value is the float32 -1000 of the reference promoted to float64;
    - ``q if q > m else m`` equals ``np.fmax(m, q)``: for a NaN q the comparison is False and m is kept, as
      fmax ignores NaN; m starts finite and so never becomes NaN. The only case where the two could pick
      different operands is a +0 / -0 tie; the sign of a zero maximum cannot reach the outputs, because
      ``1 - sin(fmax(angle, 0))`` and ``pi / 2 - opns / n`` give the same value for +0 and -0;
    - radii are visited in the same order, so the maximum sees the same sequence of operands.
    """
    n_rows, n_cols = out.shape
    start = np.float64(np.float32(-1000.0))
    for i in range(n_rows):
        out_r = out[i]
        h_row = row0 + i
        h_r = height[h_row, col0:col0 + n_cols]
        for c in range(n_cols):
            out_r[c] = start
        for k in range(dys.shape[0]):
            s_col = col0 - dxs[k]
            s_r = height[h_row - dys[k], s_col:s_col + n_cols]
            rad = radii[k]
            for c in range(n_cols):
                q = np.float64(s_r[c] - h_r[c]) / rad
                m = out_r[c]
                out_r[c] = q if q > m else m


def _svf_kernel_ok(height, scan):
    """
    Return True if ``_svf_max_slope_kernel`` performs the same operations as the numpy scan for this input.

    Checks by probing numpy's own type promotion on a 1 x 1 slice: ``height`` is float32 or float64 and
    C-contiguous (the kernel's assumed layout); ``shifted - height`` stays in the height dtype; every radius
    is an ``np.float64``, so the quotient is float64; and ``np.fmax`` of the float32 -1000 start with that
    quotient is float64. If numpy promoted differently (another dtype, another numpy version), the kernel's
    fixed float64 arithmetic would not match and the numpy scan is used instead.
    """
    if height.dtype not in (np.float32, np.float64) or not height.flags.c_contiguous:
        return False
    probe = height[:1, :1]
    d = probe - probe
    if d.dtype != height.dtype:
        return False
    for steps in scan:
        for _, _, radius in steps:
            if not isinstance(radius, np.float64):
                return False
    q = d / scan[0][0][2]
    if q.dtype != np.float64:
        return False
    return np.fmax(np.zeros(probe.shape, dtype=np.float32) - 1000, q).dtype == np.float64


def _svf_compute_window(height, radius_max, move, num_directions, compute_svf, compute_opns, compute_asvf,
                        a_main_direction, a_poly_level, a_min_weight):
    """
    Horizon scan of sky_view_factor_compute evaluated only on the output window, in blocks of rows.

    Arguments are those of ``sky_view_factor_compute`` after padding: ``height`` is the DEM padded by
    ``radius_max`` px, ``move`` the result of ``horizon_shift_vector``; the caller has checked
    ``_svf_window_ok``. Returns the same dict as ``sky_view_factor_compute`` (float64 (H, W) arrays).

    Per pixel the arithmetic is that of the full-extent loop: (shifted - height) / radius with the np.float64
    radius, np.fmax with the float32 -1000 start, np.arctan, then per direction svf + (1 - sin(fmax(angle, 0))),
    asvf + (1 - sin(fmax(angle, 0))) * weight[i_dir], opns + angle, starting from height * 0. Pixels are
    independent, so the window is processed in row blocks of ``_BLOCK_ELEMS`` elements (``rvt/_blocking.py``),
    and every shifted array is the slice ``height[r + r0 - dy : r + r1 - dy, r - dx : r - dx + W]``, which equals the
    rolled array there because no shift exceeds the padding.

    Per block and direction the running maximum comes from ``_svf_max_slope_kernel`` when numba is usable
    (``_svf_kernel_ok``); if a kernel call fails, the numpy scan takes over from that direction on, with a
    fresh maximum, so a partially written kernel buffer is never used. The numpy scan reuses two buffers
    (difference and quotient) after the first radius, whose dtypes numpy chose by promotion, so later
    radii run the same ufunc loops in place.
    """
    r = radius_max
    n_rows = height.shape[0] - 2 * r
    n_cols = height.shape[1] - 2 * r

    if compute_asvf:
        w_m = a_min_weight
        w_a = np.deg2rad(a_main_direction)
        weight = np.arange(num_directions) * (2 * np.pi / num_directions)
        weight = (1 - w_m) * (np.cos((weight - w_a) / 2)) ** a_poly_level + w_m
    else:
        weight = None

    # Per direction: (dy, dx, radius) with the np.float64 radius element of the distance array
    scan = [[(int(dy), int(dx), radius) for (dy, dx), radius in zip(move[direction]["shift"],
                                                                    move[direction]["distance"])]
            for direction in move]

    # Accelerated horizon search (numba kernel, same operations); the numpy scan below is the fallback
    use_kernel = _accel.enabled() and _svf_kernel_ok(height, scan)
    if use_kernel:
        scan_arr = [(np.array([s[0] for s in steps], dtype=np.int64), np.array([s[1] for s in steps], dtype=np.int64),
                     np.array([s[2] for s in steps], dtype=np.float64)) for steps in scan]

    svf_win = asvf_win = opns_win = None
    block_rows = max(1, _BLOCK_ELEMS // max(1, n_cols))
    for r0 in range(0, n_rows, block_rows):
        r1 = min(n_rows, r0 + block_rows)
        h_blk = height[r + r0:r + r1, r:r + n_cols]

        svf_blk = h_blk * 0 if compute_svf else None
        asvf_blk = h_blk * 0 if compute_asvf else None
        opns_blk = h_blk * 0 if compute_opns else None

        d_buf = q_buf = None
        ms_buf = np.empty(h_blk.shape, dtype=np.float64) if use_kernel else None
        for i_dir, steps in enumerate(scan):
            if use_kernel:
                dys, dxs, rads = scan_arr[i_dir]
                if _accel.call(_svf_max_slope_kernel, height, r + r0, r, dys, dxs, rads, ms_buf):
                    steps = ()  # max_slope is ms_buf; np.arctan below writes a new array, so ms_buf is reused
                    max_slope = ms_buf
                else:
                    use_kernel = False  # fallback: numpy scan from this direction on
            if not use_kernel:
                max_slope = np.zeros(h_blk.shape, dtype=np.float32) - 1000
            for i_rad, (dy, dx, radius) in enumerate(steps):
                # np.roll(height, (dy, dx))[i, j] == height[i - dy, j - dx] (no wrap on the window)
                shifted = height[r + r0 - dy:r + r1 - dy, r - dx:r - dx + n_cols]
                if d_buf is None:
                    # First step of the block: let numpy promote (float32 difference, / np.float64 radius)
                    d_buf = shifted - h_blk
                    q_buf = d_buf / radius
                else:
                    # Same operand dtypes as the first step (all radii are np.float64), so the same loops run
                    np.subtract(shifted, h_blk, out=d_buf)
                    np.divide(d_buf, radius, out=q_buf)
                if i_rad == 0:
                    # float32 -1000 start promoted with the slope dtype, as in the full-extent loop
                    max_slope = np.fmax(max_slope, q_buf)
                else:
                    # max_slope already has the promoted dtype, so the in-place fmax is the same operation
                    np.fmax(max_slope, q_buf, out=max_slope)

            # Convert to angle in radians
            max_slope = np.arctan(max_slope)

            if compute_svf or compute_asvf:
                # 1 - sin(fmax(angle, 0)), computed once and used for svf and asvf
                term = 1 - np.sin(np.fmax(max_slope, 0))
                if compute_svf:
                    svf_blk = _accumulate(svf_blk, term)
                if compute_asvf:
                    asvf_blk = _accumulate(asvf_blk, term * weight[i_dir])
            if compute_opns:
                opns_blk = _accumulate(opns_blk, max_slope)

        if compute_svf:
            if svf_win is None:
                svf_win = np.empty((n_rows, n_cols), dtype=svf_blk.dtype)
            svf_win[r0:r1] = svf_blk
        if compute_asvf:
            if asvf_win is None:
                asvf_win = np.empty((n_rows, n_cols), dtype=asvf_blk.dtype)
            asvf_win[r0:r1] = asvf_blk
        if compute_opns:
            if opns_win is None:
                opns_win = np.empty((n_rows, n_cols), dtype=opns_blk.dtype)
            opns_win[r0:r1] = opns_blk

    # Average the directional output over all directions (same expressions as the full-extent path)
    svf_out = asvf_out = opns_out = None
    if compute_svf:
        svf_out = svf_win / num_directions
    if compute_asvf:
        asvf_out = asvf_win / np.sum(weight)
    if compute_opns:
        opns_out = np.rad2deg(0.5 * np.pi - (opns_win / num_directions))

    dict_svf_asvf_opns = {"svf": svf_out, "asvf": asvf_out, "opns": opns_out}
    dict_svf_asvf_opns = {k: v for k, v in dict_svf_asvf_opns.items() if v is not None}  # filter out none

    return dict_svf_asvf_opns


def sky_view_factor_compute(height_arr,
                            radius_max=10,
                            radius_min=1,
                            num_directions=16,
                            compute_svf=True,
                            compute_opns=False,
                            compute_asvf=False,
                            a_main_direction=315.,
                            a_poly_level=4,
                            a_min_weight=0.4
                            ):
    """
    Compute sky-view factor, anisotropic sky-view factor and openness from a horizon search.

    For each of ``num_directions`` directions the maximum elevation angle of the horizon is searched at the
    shifts of ``horizon_shift_vector`` (radii ``radius_min`` .. ``radius_max`` px). Then
    SVF = mean over directions of ``1 - sin(max(angle, 0))``; ASVF = the same terms weighted by
    ``w_k = (1 - a_min_weight) * cos((a_k - a_main_direction) / 2) ** a_poly_level + a_min_weight`` and divided
    by the sum of the weights; openness = ``90 - mean(angle)`` in degrees (Zakšek et al. 2011, Yokoyama et
    al. 2002; docs/listofvis_svf.rst, docs/listofvis_openness.rst).

    Parameters
    ----------
    height_arr : numpy.ndarray
        2D elevation array, shape (H, W), in the same unit as one pixel (``sky_view_factor`` divides the DEM
        by the resolution first). Use NaN for no-data. Not modified.
    radius_max : int
        Maximum search radius in px, e.g. r = 10 px = 2.5 m at 0.25 m/px; also the padding width. Default 10.
    radius_min : int or float
        Minimum search radius in px (noise reduction). Default 1.
    num_directions : int
        Number of search directions. Default 16.
    compute_svf : bool
        Return ``"svf"``. Default True.
    compute_opns : bool
        Return ``"opns"``. Default False.
    compute_asvf : bool
        Return ``"asvf"``. Default False.
    a_main_direction : int or float
        Main direction of anisotropy in degrees, in the convention of the ``horizon_shift_vector`` keys
        (counterclockwise from north, see Notes). Default 315.
    a_poly_level : int
        Exponent of the anisotropy weight (4 low, 8 high in ``sky_view_factor``). Default 4.
    a_min_weight : float
        Minimum weight, in [0, 1], given to the direction opposite the main one (0.4 low, 0.1 high
        anisotropy in ``sky_view_factor``). Default 0.4.

    Returns
    -------
    dict
        Keys ``"svf"``, ``"asvf"``, ``"opns"``, each present only if requested; values are
        numpy.ndarray float64 (H, W) for float32 or float64 input. SVF and ASVF are in [0, 1], openness in
        degrees. NaN where ``height_arr`` is NaN.

    Notes
    -----
    Edges: the array is padded by ``radius_max`` in reflect mode, so pixels near the edge see a mirrored
    terrain. NaN neighbours are skipped (``np.fmax``), i.e. treated as open sky; the search maximum starts
    at slope -1000 (angle about -89.94°).

    Known issue (see docs/KNOWN_ISSUES.md): ``a_main_direction`` is compared with the direction keys,
    which are counterclockwise from north, while ``sun_azimuth`` elsewhere is clockwise. With the default
    315 the weight is largest for the north-east sky, not the north-west.

    Implementation: when every shift is at most ``radius_max`` px (checked at run time by
    ``_svf_window_ok``; true for ``radius_max >= 1`` and ``radius_min <= radius_max``), ``_svf_compute_window``
    computes only the output window from slices of the padded array, in row blocks, with an optional numba
    kernel for the radius loop (``_svf_max_slope_kernel``; ``RVT_NO_ACCEL=1`` switches it off). The
    results are bitwise identical to the full-extent ``np.roll`` loop below, which is kept as the
    reference and the fallback.

    See Also
    --------
    sky_view_factor, horizon_shift_vector
    """
    # Structure:
    #   1. pad by radius_max (reflect) and build the shifts of every direction;
    #   2. fast path: output window only, in row blocks (_svf_compute_window), when no shift exceeds the padding;
    #   3. reference path: per direction, running np.fmax of (rolled - height) / distance over all radii on the
    #      padded extent, arctan, accumulate svf / asvf / opns terms;
    #   4. crop the padding and average over the directions (asvf: over the weights).

    # Pad the array for the radius_max on all 4 sides
    height = np.pad(height_arr, radius_max, mode='reflect')

    # Compute the vector of movement and corresponding distances
    move = horizon_shift_vector(num_directions=num_directions, radius_pixels=radius_max, min_radius=radius_min)

    # Fast path: only the output window (the padded array without its radius_max border) is computed. On that
    # window np.roll never wraps when every shift satisfies |dy|, |dx| <= radius_max, so the rolled array equals a
    # slice of the padded array there. Otherwise the original full-extent loop below runs.
    if _svf_window_ok(height, radius_max, move):
        return _svf_compute_window(height, radius_max, move, num_directions, compute_svf, compute_opns,
                                   compute_asvf, a_main_direction, a_poly_level, a_min_weight)

    # Reference path: the original full-extent np.roll loop. It runs when _svf_window_ok is False and defines the
    # values the fast path above reproduces bitwise; it is not dead code.
    # Initiate the output for SVF
    if compute_svf:
        svf_out = height * 0  # Multiply with 0 instead of using np.zeros to preserve nodata
    else:
        svf_out = None

    # Initiate the output for azimuth dependent SVF
    if compute_asvf:
        asvf_out = height * 0  # Multiply with 0 instead of using np.zeros to preserve nodata
        w_m = a_min_weight
        w_a = np.deg2rad(a_main_direction)
        weight = np.arange(num_directions) * (2 * np.pi / num_directions)
        weight = (1 - w_m) * (np.cos((weight - w_a) / 2)) ** a_poly_level + w_m
    else:
        asvf_out = None
        weight = None

    # Initiate the output for Openness
    if compute_opns:
        opns_out = height * 0  # Multiply with 0 instead of using np.zeros to preserve nodata
    else:
        opns_out = None

    # Search for horizon in each direction...
    for i_dir, direction in enumerate(move):
        # Reset maximum at each iteration (i.e. at the start of new direction),
        # the start is a slope (tangent) of -1000, i.e. an elevation angle of about -89.94 deg, below any real horizon
        max_slope = np.zeros(height.shape, dtype=np.float32) - 1000

        # ... and for each search radius
        for i_rad, radius in enumerate(move[direction]["distance"]):
            # Get shift index from move dictionary
            shift_indx = move[direction]["shift"][i_rad]
            # Estimate the slope
            _ = (np.roll(height, shift_indx, axis=(0, 1)) - height) / radius
            # Compare to the previous max slope and keep the largest values (element wise). Use np.fmax to prevent NaN
            # values contaminating the edge of the image (if one of the elements is NaN, pick non-NaN element)
            max_slope = np.fmax(max_slope, _)

        # Convert to angle in radians and compute directional output
        max_slope = np.arctan(max_slope)

        # Sum max angle for all directions
        if compute_svf:
            # For SVF minimum possible angle is 0 (hemisphere), use np.fmax() to change NaNs to 0
            svf_out = svf_out + (1 - np.sin(np.fmax(max_slope, 0)))
        if compute_asvf:
            # For SVF minimum possible angle is 0 (hemisphere), use np.fmax() to change NaNs to 0
            asvf_out = asvf_out + (1 - np.sin(np.fmax(max_slope, 0))) * weight[i_dir]
        if compute_opns:
            # For Openness taking the entire sphere
            opns_out = opns_out + max_slope

    # Cut to original extent and average the directional output over all directions
    if compute_svf:
        svf_out = svf_out[radius_max:-radius_max, radius_max:-radius_max] / num_directions
    if compute_asvf:
        asvf_out = asvf_out[radius_max:-radius_max, radius_max:-radius_max] / np.sum(weight)
    if compute_opns:
        opns_out = np.rad2deg(0.5 * np.pi - (opns_out[radius_max:-radius_max, radius_max:-radius_max] / num_directions))

    # Return results within dict
    dict_svf_asvf_opns = {"svf": svf_out, "asvf": asvf_out, "opns": opns_out}
    dict_svf_asvf_opns = {k: v for k, v in dict_svf_asvf_opns.items() if v is not None}  # filter out none

    return dict_svf_asvf_opns


def sky_view_factor(dem,
                    resolution,
                    compute_svf=True,
                    compute_opns=False,
                    compute_asvf=False,
                    svf_n_dir=16,
                    svf_r_max=10,
                    svf_noise=0,
                    asvf_dir=315,
                    asvf_level=1,
                    ve_factor=1,
                    no_data=None
                    ):
    """
    Compute sky-view factor, anisotropic sky-view factor and/or positive openness of a DEM.

    Converts the DEM to float32, sets ``no_data`` to NaN, multiplies by ``ve_factor``, divides by
    ``resolution`` (so that heights are in pixel units and the elevation angles are correct), derives the
    minimum radius and the anisotropy parameters from the noise and level settings, and calls
    ``sky_view_factor_compute``. Negative openness is obtained by calling this function with the negated
    DEM (``rvt.default.DefaultValues.get_neg_opns``).

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W). Converted to float32 (a copy); not modified.
    resolution : float
        Pixel size (m/px). Must not be negative.
    compute_svf : bool
        Return ``"svf"``. Default True.
    compute_opns : bool
        Return ``"opns"`` (openness, degrees). Default False.
    compute_asvf : bool
        Return ``"asvf"``. Default False.
    svf_n_dir : int
        Number of search directions. Default 16.
    svf_r_max : int
        Maximum search radius in px, e.g. r = 10 px = 2.5 m at 0.25 m/px. Default 10.
    svf_noise : int
        Noise removal level 0 (none), 1 (low), 2 (medium), 3 (high): the search starts at 0 %, 10 %, 20 % or
        40 % of ``svf_r_max``, rounded, at least 1 px. Default 0.
    asvf_dir : int or float
        Main direction of anisotropy in degrees (see the known issue in ``sky_view_factor_compute``).
        Default 315.
    asvf_level : int
        Anisotropy level 1 (low: exponent 4, minimum weight 0.4) or 2 (high: exponent 8, minimum weight 0.1).
        Default 1.
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it (after the float32 conversion) are set to NaN. Default None.

    Returns
    -------
    dict
        Keys ``"svf"``, ``"asvf"``, ``"opns"``, each present only if requested; values are numpy.ndarray
        float64 (H, W), NaN where the input is NaN or ``no_data``.

    Raises
    ------
    Exception
        If ``dem`` is not 2D, ``ve_factor`` is outside [-10000, 10000], ``svf_noise`` is not 0-3,
        ``asvf_level`` is not 1 or 2, all three ``compute_*`` are False, or ``resolution`` is negative.

    Notes
    -----
    Edges are padded by ``svf_r_max`` in reflect mode. For seamless tiles pass a DEM with a buffer of at
    least ``svf_r_max`` px. The accelerated path is described in
    ``sky_view_factor_compute``.

    See Also
    --------
    sky_view_factor_compute, horizon_shift_vector
    """

    # Checks for input parameters
    if dem.ndim != 2:
        raise Exception("rvt.vis.sky_view_factor: dem has to be 2D np.array!")
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.sky_view_factor: ve_factor must be between -10000 and 10000!")
    if svf_noise != 0 and svf_noise != 1 and svf_noise != 2 and svf_noise != 3:
        raise Exception("rvt.vis.sky_view_factor: svf_noise must be one of the following "
                        "values (0-don't remove, 1-low, 2-med, 3-high)!")
    if asvf_level != 1 and asvf_level != 2:
        raise Exception("rvt.vis.sky_view_factor: asvf_level must be one of the following "
                        "values (1-low, 2-high)!")
    if not compute_svf and not compute_asvf and not compute_opns:
        raise Exception("rvt.vis.sky_view_factor: All computes are false!")
    if resolution < 0:
        raise Exception("rvt.vis.sky_view_factor: resolution must be a positive number!")

    # Make sure array has the correct dtype!
    dem = dem.astype(np.float32)

    # CONSTANTS
    # Level of polynomial that determines the anisotropy, selected with asvf_level (1 - low, 2 - high)
    sc_asvf_pol = [4, 8]
    sc_asvf_min = [0.4, 0.1]
    # The portion (percent) of the maximal search radius to ignore in horizon estimation; for each noise level,
    # selected with svf_noise (0-3)
    sc_svf_r_min = [0., 10., 20., 40.]

    # Before doing anything to the array, make sure all NODATA values are set to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan
    # Save NaN mask (processing may change NaNs to arbitrary values)
    nan_mask = np.isnan(dem)

    # Vertical exaggeration
    dem = dem * ve_factor
    # Pixel size (adjust elevation to correctly calculate the vertical elevation angle, calculation thinks 1px == 1m)
    dem = dem / resolution

    # Minimal search radius depends on the noise level, it has to be an integer not smaller than 1
    svf_r_min = max(np.round(svf_r_max * sc_svf_r_min[svf_noise] * 0.01, decimals=0), 1)

    # Set anisotropy parameters
    poly_level = sc_asvf_pol[asvf_level - 1]
    min_weight = sc_asvf_min[asvf_level - 1]

    # Main routine for SVF processing
    dict_svf_asvf_opns = sky_view_factor_compute(
        height_arr=dem,
        radius_max=svf_r_max,
        radius_min=svf_r_min,
        num_directions=svf_n_dir,
        compute_svf=compute_svf,
        compute_opns=compute_opns,
        compute_asvf=compute_asvf,
        a_main_direction=asvf_dir,
        a_poly_level=poly_level,
        a_min_weight=min_weight
    )

    # Apply NaN mask to outputs
    for item in dict_svf_asvf_opns.values():
        item[nan_mask] = np.nan

    return dict_svf_asvf_opns


@_accel.jit_kernel
def _local_dominance_kernel(dem, dem_obs, out, pad_width, s_y, s_x, distances, dist_factor):
    """
    Numba kernel: local dominance sum over all shifts for the output window.

    Inputs: ``dem`` the DEM padded by ``pad_width`` px (float32 or float64); ``dem_obs`` the observer
    heights ``dem + observer_height`` on the output window (H, W); ``s_y``, ``s_x`` (int64) the shifts and
    ``distances``, ``dist_factor`` (float64) the per-shift distance d and factor 2d + rad_inc, all in the
    angle-major order of ``local_dominance``. Output: ``out`` (H, W), the float32 accumulator starting at
    ``dem_in * 0``, updated in place. Serial, so independent of the thread count.

    Equivalence with the sliced numpy path of ``local_dominance`` (bitwise), per shift and pixel:

    - the comparison ``obs > moved`` in the input dtypes (``np.greater`` into the bool mask);
    - the difference ``obs - moved`` rounded to float32 (the float32 ``buf_diff``), then widened to float64;
    - float64 division by ``distances[i]`` and multiplication by ``dist_factor[i]`` (``buf_val``);
    - float64 addition to the accumulator value, rounded to the accumulator dtype on store
      (``np.add(..., where=mask, casting="same_kind")``);
    - shifts in the same order. A NaN operand compares False and adds nothing, as with ``where=mask``.

    The guards in ``local_dominance`` (float dtypes, shifts within the padding) exclude out-of-range reads.
    """
    n_rows, n_cols = out.shape
    for r in range(n_rows):
        obs_r = dem_obs[r]
        out_r = out[r]
        for i_s in range(s_y.shape[0]):
            row = pad_width - s_y[i_s] + r
            c0 = pad_width - s_x[i_s]
            moved_r = dem[row, c0:c0 + n_cols]
            dist = distances[i_s]
            fac = dist_factor[i_s]
            for c in range(n_cols):
                x = obs_r[c]
                m = moved_r[c]
                o = out_r[c]
                v = np.float64(np.float32(x - m)) / dist * fac
                out_r[c] = o + v if x > m else o


def local_dominance(dem,
                    min_rad=10,
                    max_rad=20,
                    rad_inc=1,
                    angular_res=15,
                    observer_height=1.7,
                    ve_factor=1,
                    no_data=None
                    ):
    """
    Compute local dominance: how much an observer above each pixel looks down on its surroundings.

    Adapted from the Lidar Visualisation Toolbox (LiVT) by Ralf Hesse (docs/listofvis_localdom.rst). For
    every shift (angle a, distance d) of a ring of radii ``min_rad`` .. ``max_rad`` px, where the observer
    at ``z + observer_height`` is above the shifted pixel ``z_s``, the term
    ``(z + observer_height - z_s) / d * (2 d + rad_inc)`` is added. The sum is divided by
    ``norma = n_angles * sum_d(observer_height / d * (2 d + rad_inc))``, the value for flat terrain.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W).
    min_rad : int
        Minimum distance in px, e.g. r = 10 px = 2.5 m at 0.25 m/px. Default 10.
    max_rad : int
        Maximum distance in px; also the edge padding width. Default 20.
    rad_inc : int
        Distance step in px. Default 1.
    angular_res : int
        Angle step in degrees; the angles are 0, ``angular_res``, ... below 360 (24 angles for 15).
        Default 15.
    observer_height : int or float
        Observer height above the terrain (m). Default 1.7. Not multiplied by ``ve_factor``.
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**, before the float32 conversion.
        Default None.

    Returns
    -------
    numpy.ndarray
        float64 (H, W); 1 for flat terrain, larger where the pixel dominates its surroundings; NaN where the
        input is NaN.

    Raises
    ------
    Exception
        If ``dem`` is not 2D or ``ve_factor`` is outside [-10000, 10000].

    Notes
    -----
    Edges: the DEM is padded by ``max_rad`` in edge mode. NaN neighbours compare False and add nothing.
    There is no resolution parameter: heights in m are divided by distances in px, in the sum and in
    ``norma`` alike.

    Implementation: the shifts never exceed the padding (``|round(d * sin a)|, |round(d * cos a)| <=
    max_rad``), so the shifted DEM over the output window is a slice of the padded DEM. The output window
    is computed in row blocks of about ``_BLOCK_ELEMS`` = 32768 elements (numpy), or by the numba kernel
    ``_local_dominance_kernel`` (``RVT_NO_ACCEL=1`` switches it off), with the same float32 difference,
    float64 division and multiplication, float32 accumulator and shift order as the original full-extent
    ``np.roll`` loop at the end, which is kept as the reference (bitwise identical).

    Known issue (see docs/KNOWN_ISSUES.md): with ``max_rad = 0`` the padding is 0, the reference loop
    runs and its crop ``[0:-0]`` returns an empty (0, 0) array.

    See Also
    --------
    sky_view_factor
    """
    # Structure:
    #   1. checks, no_data -> NaN (in place), float32, edge padding by max_rad, ve_factor;
    #   2. distances, angles, normalisation norma, the list of (dy, dx) shifts (angle-major order);
    #   3. fast path (shifts within the padding): numba kernel, else numpy row blocks; divide by norma;
    #   4. reference path: np.roll per shift on the padded extent, masked accumulation, divide, crop.
    if dem.ndim != 2:
        raise Exception("rvt.vis.local_dominance: dem has to be 2D np.array!")
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.local_dominance: ve_factor must be between -10000 and 10000!")

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    # add max_rad pixel edge padding
    pad_width = max_rad
    dem = np.pad(array=dem, pad_width=pad_width, mode="edge")
    dem = dem * ve_factor

    # create a vector with possible distances
    n_dist = int((max_rad - min_rad) / rad_inc + 1)
    distances = np.arange(n_dist * rad_inc, step=rad_inc) + min_rad
    # create vector with possible angles
    n_ang = int(359 / angular_res + 1)
    angles = np.arange(n_ang * angular_res, step=angular_res)
    # determine total area within radius range
    norma = np.sum((observer_height / distances) * (2 * distances + rad_inc)) * n_ang

    # image shifts
    n_shifts = distances.size * angles.size
    x_t = (np.outer(np.cos(np.deg2rad(angles)), distances)).reshape(n_shifts)
    y_t = (np.outer(np.sin(np.deg2rad(angles)), distances)).reshape(n_shifts)
    distances = (np.outer(np.ones(n_ang), distances)).reshape(n_shifts)
    dist_factor = 2 * distances + rad_inc

    shifts = [(int(round(y_t[i_s])), int(round(x_t[i_s]))) for i_s in range(n_shifts)]
    if pad_width > 0 and n_shifts > 0 and all(abs(s_y) <= pad_width and abs(s_x) <= pad_width for s_y, s_x in shifts):
        # Shifts never exceed the padding, so np.roll never wraps for output pixels: the moved DEM over the output
        # window is a slice of the padded DEM. Pixels are independent, so only the output window is computed,
        # in row blocks, with the same per-shift float32/float64 arithmetic and order as the loop below.
        n_rows = dem.shape[0] - 2 * pad_width
        n_cols = dem.shape[1] - 2 * pad_width
        dem_in = dem[pad_width:pad_width + n_rows, pad_width:pad_width + n_cols]
        dem_obs = dem_in + observer_height  # float32, as (dem + observer_height)
        local_dom_out = dem_in * 0
        if (dem.dtype in (np.float32, np.float64) and dem_obs.dtype in (np.float32, np.float64)
                and distances.dtype == np.float64 and dist_factor.dtype == np.float64):
            s_y_arr = np.array([s[0] for s in shifts], dtype=np.int64)
            s_x_arr = np.array([s[1] for s in shifts], dtype=np.int64)
            if _accel.call(_local_dominance_kernel, dem, dem_obs, local_dom_out, pad_width, s_y_arr, s_x_arr,
                           distances, dist_factor):
                return local_dom_out / norma
            local_dom_out = dem_in * 0  # fallback: start again from the numpy path
        blk_rows = max(1, _BLOCK_ELEMS // max(1, n_cols))
        buf_mask = np.empty((blk_rows, n_cols), dtype=bool)
        buf_diff = np.empty((blk_rows, n_cols), dtype=np.float32)
        buf_val = np.empty((blk_rows, n_cols), dtype=np.result_type(buf_diff, distances[0], dist_factor[0]))
        for r0 in range(0, n_rows, blk_rows):
            r1 = min(r0 + blk_rows, n_rows)
            nr = r1 - r0
            obs_b = dem_obs[r0:r1]
            out_b = local_dom_out[r0:r1]
            m_b = buf_mask[:nr]
            d_b = buf_diff[:nr]
            v_b = buf_val[:nr]
            for i_s in range(n_shifts):
                s_y, s_x = shifts[i_s]
                moved_b = dem[pad_width - s_y + r0:pad_width - s_y + r1, pad_width - s_x:pad_width - s_x + n_cols]
                np.greater(obs_b, moved_b, out=m_b)
                np.subtract(obs_b, moved_b, out=d_b)
                np.divide(d_b, distances[i_s], out=v_b)
                np.multiply(v_b, dist_factor[i_s], out=v_b)
                np.add(out_b, v_b, out=out_b, where=m_b, casting="same_kind")
        return local_dom_out / norma

    # Reference path: the original full-extent np.roll loop. It runs when the guard above fails (max_rad <= 0,
    # no shifts, or a shift beyond the padding) and defines the values the sliced and numba paths above reproduce bitwise; not dead code.
    local_dom_out = dem * 0
    for i_s in range(n_shifts):
        dem_moved = np.roll(dem, int(round(y_t[i_s])), axis=0)
        dem_moved = np.roll(dem_moved, int(round(x_t[i_s])), axis=1)
        idx_lower = np.where((dem + observer_height) > dem_moved)
        if idx_lower[0].size > 0:
            local_dom_out[idx_lower[0], idx_lower[1]] = local_dom_out[idx_lower[0], idx_lower[1]] + \
                                                        (dem[idx_lower[0], idx_lower[1]] + observer_height -
                                                         dem_moved[idx_lower[0], idx_lower[1]]) / \
                                                        distances[i_s] * dist_factor[i_s]
    local_dom_out = local_dom_out / norma

    # Remove padding
    local_dom_out = local_dom_out[pad_width:-pad_width, pad_width:-pad_width]

    return local_dom_out


def horizon_generate_coarse_dem(dem_fine,
                                pyramid_scale,
                                conv_from,
                                conv_to,
                                max_radius
                                ):
    """
    Build the next coarser pyramid level for the sky illumination horizon search.

    The ``max_radius`` border of ``dem_fine`` is removed, the rest is padded at the right and bottom so
    that the coarse grid contains the corner pixels, and each coarse pixel takes the maximum of a
    ``pyramid_scale`` x ``pyramid_scale`` window divided by ``pyramid_scale`` (heights in coarse-pixel
    units, so that height / distance stays an angle tangent). The result is padded by ``max_radius``
    with its own minimum.

    Parameters
    ----------
    dem_fine : numpy.ndarray
        2D level (heights in fine-pixel units) including a ``max_radius`` border, shape
        (n + 2 * max_radius, m + 2 * max_radius).
    pyramid_scale : int
        Resolution ratio between two levels (2 in ``sky_illumination``).
    conv_from : int
        First offset of the maximum window: ``1 - pyramid_scale // 2`` for an even scale,
        ``-(pyramid_scale // 2)`` for an odd one.
    conv_to : int
        ``pyramid_scale // 2``; padding added after the data for the window.
    max_radius : int
        Border width in px, removed first and added again at the end.

    Returns
    -------
    numpy.ndarray
        float64, shape (n // pyramid_scale + 1 + 2 * max_radius, m // pyramid_scale + 1 + 2 * max_radius).

    Notes
    -----
    Padding: edge mode at the right and bottom (``|1 - n % pyramid_scale|`` px, none when the remainder is
    1), then symmetric ``(-conv_from, conv_to)`` for the window, then constant (the coarse minimum) for the
    border. NaN propagates through ``np.maximum`` and ``min``.

    Known issues (see docs/KNOWN_ISSUES.md): the running maximum starts from ``np.zeros``, so a level
    entirely below 0 becomes 0; for an even scale the window reaches one row/column before the first one,
    where the padding supplies nothing and ``np.roll`` wraps to the last row/column.

    See Also
    --------
    horizon_generate_pyramids, sky_illumination
    """
    # first reduce the size for the edge required for horizon search
    dem_fine = dem_fine[max_radius:-max_radius, max_radius:-max_radius]

    # get and adjust the array sizes
    in_shape = dem_fine.shape
    n_col_fine = in_shape[1]
    n_lin_fine = in_shape[0]
    n_lin_coarse = int(np.floor(n_lin_fine / pyramid_scale)) + 1
    n_col_coarse = int(np.floor(n_col_fine / pyramid_scale)) + 1
    # The corner points must fit in the new grid.
    # This is always the case with the left most column or the upper line.
    # But you have to adjust ne number of columns to the right and number of lines below.
    # The final number of columns/lines has to fulfill:
    #     n_coarse = pyramid_scale * n_fine + 1
    # columns
    mod_col = n_col_fine % pyramid_scale
    pad_col = 0
    if mod_col != 1:
        pad_col = np.abs(1 - mod_col)
    # lines
    mod_lin = n_lin_fine % pyramid_scale
    pad_lin = 0
    if mod_lin != 1:
        pad_lin = np.abs(1 - mod_lin)
    # Here we extend it to right and below, so padding with edge is OK
    # Edge-mode otherwise creates artefacts on left and above.
    dem_fine = np.pad(dem_fine, ((0, pad_lin), (0, pad_col)), mode="edge")

    # Once you have data in the shape appropriate for resizing,
    # pad the data to support np.move.
    dem_fine = np.pad(dem_fine, ((-conv_from, conv_to), (-conv_from, conv_to)), mode="symmetric")

    # Convolution (keep maximum)
    dem_convolve = np.zeros(dem_fine.shape)
    for i in np.arange(pyramid_scale) + conv_from:
        for j in np.arange(pyramid_scale) + conv_from:
            dem_convolve = np.maximum(dem_convolve, np.roll(dem_fine, (i, j), axis=(0, 1)))
    # Divide by pyramid_scale to account for the change of resolution
    # (important for the angle computation later on)
    dem_convolve = dem_convolve / pyramid_scale

    # Consider only the selected convoluted points according to the scale change.
    # As we select slice's end point make sure to consider at least 1 point more 
    # to the right / below to really include it (Python way of considering end index).
    dem_coarse = dem_convolve[-conv_from:(n_lin_coarse * pyramid_scale + 1):pyramid_scale,
                              -conv_from:(n_col_coarse * pyramid_scale + 1):pyramid_scale]

    # Final padding to enable searching the horizon over the edge:
    # use constant-mode set to the minimal height, so it doesn't 
    # affect the horizon estimation.
    dem_coarse = np.pad(dem_coarse, ((max_radius, max_radius), (max_radius, max_radius)), mode="constant",
                        constant_values=dem_coarse.min())

    return dem_coarse


def horizon_generate_pyramids(dem,
                              num_directions=4,
                              max_fine_radius=100,
                              max_pyramid_radius=7,
                              pyramid_scale=3,
                              ):
    """
    Build the DEM pyramid and the search shifts of every level for the sky illumination horizon search.

    Level 0 is the DEM padded by ``max_pyramid_radius`` px with its minimum. Each further level is
    ``horizon_generate_coarse_dem`` of the previous one. Levels are added until
    ``max_fine_radius / pyramid_scale ** L <= max_pyramid_radius``. Level 0 searches radii 1 ..
    ``max_pyramid_radius``; higher levels start at ``max_pyramid_radius // pyramid_scale`` (in their own
    pixels), so that each level continues where the finer one stopped; the last level searches up to
    ``round(max_fine_radius / pyramid_scale ** L)``.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array, shape (H, W) (float32 in ``sky_illumination``). Not modified.
    num_directions : int
        Number of search directions. Default 4.
    max_fine_radius : int
        Total search distance in level-0 px, e.g. 100 px = 25 m at 0.25 m/px. Default 100.
    max_pyramid_radius : int
        Search radius within one level, in that level's px; also the border width. Default 7.
    pyramid_scale : int
        Resolution ratio between two levels. Default 3.

    Returns
    -------
    dict
        ``{level: {"num_directions", "radius_pixels", "min_radius", "shift", "dem", "i_lin", "i_col"}}``
        for level 0 .. L: ``shift`` from ``horizon_shift_vector``, ``dem`` the level array with its border
        (level 0 in the input dtype, higher levels float64), ``i_lin`` / ``i_col`` the row and column
        indices ``np.arange`` of that array.

    Notes
    -----
    With the ``sky_illumination`` settings (scale 2, radius 20 px, ``max_fine_radius`` 100 px) there are
    3 coarse levels, radii 20, 20, 20, 12 px. Any NaN in ``dem`` makes ``dem.min()`` NaN, so the padding
    and every coarse level are NaN (see ``sky_illumination``). The pyramid is anchored at the top-left
    corner of ``dem``, so it depends on the origin of a tile (see ``sky_illumination``).

    See Also
    --------
    horizon_generate_coarse_dem, horizon_shift_vector, sky_illumination
    """
    # In the levels higher than 1, determine the minimal search distance
    # and number of search distances.
    # If you have for instance
    #     pyramid_scale = 3
    #     max_pyramid_radius = 10
    #     num_directions = 8
    # then you have original distances in level 0:
    # 1, 2, 3, ... 9, 10
    # In level 1, your resolution is 3-times coarser.
    # The first pixel that takes that this new resolution,
    # has in original distance value 12 (in coarse resolution 4):
    # 12->4, 15->5, 18->6 ... 27->9, 30->10
    # So you start in the level 1 with min_pyramid_radius=4
    # and you search from 4 to 10 distances (n_pyramid_radius=7)
    min_pyramid_radius = int(np.floor(max_pyramid_radius / pyramid_scale)) + 1
    n_pyramid_radius = max_pyramid_radius - min_pyramid_radius + 1

    # get the convolution window indices
    conv_to = int(np.floor(pyramid_scale / 2.))
    if (pyramid_scale % 2) == 0:
        conv_from = 1 - conv_to
    else:
        conv_from = -conv_to

    # initializations
    pyramid_levels = 0
    work = True
    pyramid = {}

    # Determine the number of levels and
    # the last radius to be used in the highest level.
    while work:
        _ = max_fine_radius / pyramid_scale ** pyramid_levels
        if _ > max_pyramid_radius:
            pyramid_levels = pyramid_levels + 1
        else:
            work = False
            last_radius = np.round(max_fine_radius / pyramid_scale ** pyramid_levels, decimals=0)

    # fill out the pyramid dict with the metadata required for horizon searching.
    dem_coarse = None
    for level in np.arange(pyramid_levels + 1):
        # the level 0 contains the other min_radius as the rest of levels
        if level == 0:
            min_radius = 1
            dem_fine = np.copy(np.pad(dem, max_pyramid_radius, mode="constant", constant_values=dem.min()))
        else:
            min_radius = min_pyramid_radius - 1
            dem_fine = np.copy(dem_coarse)
        # the last level contains the other radius_pixels as the rest of levels
        if level == pyramid_levels:
            max_radius = last_radius
        else:
            max_radius = max_pyramid_radius
        # determine the dict of shifts
        shift = horizon_shift_vector(num_directions, max_radius, min_radius)
        dem_coarse = horizon_generate_coarse_dem(dem_fine, pyramid_scale, conv_from, conv_to, max_pyramid_radius)
        i_lin = np.arange(dem_fine.shape[0])
        i_col = np.arange(dem_fine.shape[1])

        pyramid[level] = {
            "num_directions": num_directions,
            "radius_pixels": max_radius,
            "min_radius": min_radius,
            "shift": shift,
            "dem": dem_fine,
            "i_lin": i_lin,
            "i_col": i_col,
        }

    return pyramid


def _sky_illum_level_search(height, move, direction, max_slope, inner):
    """
    Horizon search of one pyramid level and direction on whole arrays (path used when not blocked).

    For every shift (dy, dx) and distance of ``move[direction]``, in order, computes the reference
    expression of the original code::

        max_slope = np.maximum(max_slope, np.maximum((np.roll(height, (dy, dx), axis=(0, 1)) - height) / radius, 0.))

    with reused buffers. ``inner=None``: on the full array with ``np.roll`` semantics (wrapped values
    included), the roll being written as four slice subtractions. ``inner=p``: only on
    ``height[p:-p, p:-p]`` (``max_slope`` has that shape); requires ``|dy|, |dx| <= p``, so the shifted
    window stays inside ``height`` and no wrapped value is used. ``max_slope`` is promoted (copied) when
    the quotient has a wider dtype, otherwise updated in place; the returned array is the new maximum.
    The values are bitwise those of the reference expression: same ufuncs, operand order and dtypes.
    """
    n0, n1 = height.shape
    if inner is None:
        out_shape = (n0, n1)
    else:
        out_shape = (n0 - 2 * inner, n1 - 2 * inner)
    diff = np.empty(out_shape, dtype=height.dtype)
    quot = None
    for i_rad, radius in enumerate(move[direction]["distance"]):
        dy, dx = move[direction]["shift"][i_rad]
        if inner is None:
            # reference: diff = np.roll(height, (dy, dx), axis=(0, 1)) - height, written as the four
            # wrapped quadrants of the roll so that no rolled copy is allocated
            dy = int(dy) % n0
            dx = int(dx) % n1
            for o0, s0 in ((slice(dy, None), slice(0, n0 - dy)), (slice(0, dy), slice(n0 - dy, None))):
                for o1, s1 in ((slice(dx, None), slice(0, n1 - dx)), (slice(0, dx), slice(n1 - dx, None))):
                    np.subtract(height[s0, s1], height[o0, o1], out=diff[o0, o1])
        else:
            p = inner
            np.subtract(height[p - dy:n0 - p - dy, p - dx:n1 - p - dx], height[p:n0 - p, p:n1 - p], out=diff)
        q_dtype = np.result_type(diff, radius)
        if q_dtype == diff.dtype:
            quot = diff
        elif quot is None or quot.dtype != q_dtype:
            quot = np.empty(out_shape, dtype=q_dtype)
        np.divide(diff, radius, out=quot)
        np.maximum(quot, 0., out=quot)
        if np.result_type(max_slope, quot) != max_slope.dtype:
            max_slope = max_slope.astype(np.result_type(max_slope, quot))
        np.maximum(max_slope, quot, out=max_slope)
    return max_slope


def _sky_illum_block_rows(n_cols):
    """
    Return the number of rows per block of the blocked sky illumination path (at least 16).

    About ``_BLOCK_ELEMS`` = 32768 pixels per block (``rvt/_blocking.py`` gives the cache reasoning),
    rounded down to a multiple of 16 rows. With a multiple of 16 rows every block except the last holds a
    multiple of 16 elements, so each element sits at the same position relative to the SIMD vector width
    (at most 16 lanes) as in the whole-array operation. numpy's vectorized transcendental functions (sin,
    cos, arctan, power) may round differently in the vector body and in the scalar tail of a loop; keeping
    the positions keeps every element on the same code path and the results bitwise identical.
    """
    return max(16, (_BLOCK_ELEMS // max(1, n_cols)) // 16 * 16)


# Two serial numba kernels reproduce the numpy path of _sky_illum_search_blocked row by row, with the same operations
# per pixel and shift in the same order: the difference in the dtype of hp (float32 subtraction for the float32
# level 0, float64 for the coarse levels), widened to float64 and divided by the float64 radius, then np.maximum.
#
# np.maximum(a, b) is written as "a if (a > b or a != a) else b". This reproduces numpy's float64 np.maximum
# bitwise, including the two cases where IEEE leaves a choice:
# - NaN: numpy returns a NaN operand, the first one if both are NaN (so the NaN payload is that of a). Here a NaN a
#   is returned by "a != a"; for a NaN b only, "a > b" is False and b is returned;
# - a tie of +0 and -0 (a > b is False): numpy returns the second operand b. This was checked on numpy 2.x / x86-64
#   for arrays of 1 to 33 elements (SIMD bodies and scalar tails), with an array or a scalar as b, and is what the
#   expression returns.
# The zero sign matters because max_slope at level 0 becomes the horizon angle of the shadow direction, whose sign
# of zero is part of the bitwise result. Two kernels instead of one with a flag, so that a process compiles only the
# variant it runs (the literal one serves only the shadow direction).
@_accel.jit_kernel
def _sky_illum_search_kernel(hp, p, s_y, s_x, radii, ms):
    """
    Numba kernel of ``_sky_illum_search_blocked`` with ``floor_once=True``.

    Computes ``ms = np.maximum(ms, 0.)`` once, then ``ms = np.maximum(ms, q)`` for every shift k in order,
    with ``q = float64(hp[p + r - s_y[k], p + c - s_x[k]] - hp[p + r, p + c]) / radii[k]``, for every pixel
    (r, c) of ``ms`` (float64, updated in place). ``hp`` is the padded level (float32 or float64), ``p`` the
    padding, ``s_y``, ``s_x`` int64 shifts, ``radii`` float64 distances. Same operations, operand order and
    tie/NaN rule as the numpy loop (see the comment above), so the result is bitwise identical to it.
    """
    n0, n1 = ms.shape
    n_s = s_y.shape[0]
    for r in range(n0):
        ms_r = ms[r]
        base_r = hp[p + r, p:p + n1]
        for c in range(n1):
            a = ms_r[c]
            ms_r[c] = a if (a > 0. or a != a) else 0.
        for i_s in range(n_s):
            c0 = p - s_x[i_s]
            sh_r = hp[p + r - s_y[i_s], c0:c0 + n1]
            rad = radii[i_s]
            for c in range(n1):
                q = np.float64(sh_r[c] - base_r[c]) / rad
                a = ms_r[c]
                ms_r[c] = a if (a > q or a != a) else q


@_accel.jit_kernel
def _sky_illum_search_literal_kernel(hp, p, s_y, s_x, radii, ms):
    """
    Numba kernel of ``_sky_illum_search_blocked`` with ``floor_once=False`` (literal order).

    For every shift k in order: ``q = np.maximum(q, 0.)``, then ``ms = np.maximum(ms, q)``, with q as in
    ``_sky_illum_search_kernel``; ``ms`` (float64) is updated in place. This is the reference expression
    ``np.maximum(ms, np.maximum(q, 0.))`` operation by operation, so even the sign of a zero maximum is that
    of the reference; used for the shadow direction at level 0.
    """
    n0, n1 = ms.shape
    n_s = s_y.shape[0]
    for r in range(n0):
        ms_r = ms[r]
        base_r = hp[p + r, p:p + n1]
        for i_s in range(n_s):
            c0 = p - s_x[i_s]
            sh_r = hp[p + r - s_y[i_s], c0:c0 + n1]
            rad = radii[i_s]
            for c in range(n1):
                q = np.float64(sh_r[c] - base_r[c]) / rad
                q = q if (q > 0. or q != q) else 0.
                a = ms_r[c]
                ms_r[c] = a if (a > q or a != a) else q


def _sky_illum_search_accel(hp, p, shifts, max_slope, floor_once):
    """
    Run the numba level-search kernel for ``_sky_illum_search_blocked``, if possible.

    Arguments as in ``_sky_illum_search_blocked``; ``shifts`` is a list of ``(dy, dx, radius)``. Returns True
    when ``max_slope`` (float64) was updated in place by ``_sky_illum_search_kernel`` (``floor_once``) or
    ``_sky_illum_search_literal_kernel``; False when the numpy path has to run (accelerators off, numba
    missing, unsupported dtype or layout, compile failure).

    The guards exclude every case in which the kernel could raise after writing: a zero radius
    (ZeroDivisionError in numba's default error model), shifts larger than ``p`` or a too small ``hp``
    (indices outside ``hp``), a read-only or empty ``max_slope``. What remains can fail only at typing or
    compilation, before the kernel writes, so a False return leaves ``max_slope`` untouched and the numpy
    path starts from the same values.
    """
    if not _accel.enabled():
        return False
    n0, n1 = max_slope.shape
    if not (hp.ndim == 2 and hp.dtype in (np.float32, np.float64) and max_slope.dtype == np.float64
            and max_slope.flags.writeable and n0 > 0 and n1 > 0
            and hp.shape[0] >= n0 + 2 * p and hp.shape[1] >= n1 + 2 * p
            and all(type(s[2]) is np.float64 and s[2] != 0 and abs(s[0]) <= p and abs(s[1]) <= p
                    for s in shifts)):
        return False
    s_y = np.array([s[0] for s in shifts], dtype=np.int64)
    s_x = np.array([s[1] for s in shifts], dtype=np.int64)
    radii = np.array([s[2] for s in shifts], dtype=np.float64)
    kernel = _sky_illum_search_kernel if floor_once else _sky_illum_search_literal_kernel
    return _accel.call(kernel, hp, int(p), s_y, s_x, radii, max_slope)


def _sky_illum_search_blocked(hp, p, shifts, max_slope, floor_once):
    """
    Horizon search of one pyramid level and direction on a window, in row blocks (blocked path).

    ``hp`` is the padded level, ``p`` the padding, ``max_slope`` (n0, n1) the running maximum on the window
    ``hp[p:p + n0, p:p + n1]``, ``shifts`` a list of ``(dy, dx, radius)`` with ``|dy|, |dx| <= p``, so the
    shifted windows ``hp[p - dy:, p - dx:]`` are slices that never leave ``hp``. Level 0 is passed with its
    own padding; a coarse level as a wrap-padded copy, whose slices equal ``np.roll`` of the level.
    Returns the new maximum: ``max_slope`` itself, updated in place, when its dtype is already the promoted
    one, else a promoted copy.

    Per pixel the same ufuncs, operand order and dtypes as the reference expression of
    ``_sky_illum_level_search``::

        max_slope = np.maximum(max_slope, np.maximum((shifted - base) / radius, 0.))

    except that ``max_slope`` is promoted before the first radius instead of at it (exact for
    float32 -> float64). ``floor_once=True`` computes ``max(ms, max(q1, 0), ..., max(qn, 0))`` as
    ``max(max(ms, 0), q1, ..., qn)``: the same numbers, because np.maximum returns the NaN operand (the
    first one if both are NaN), but the sign of a zero maximum may differ. The caller uses it where the
    zero sign cannot reach the output (coarse levels, which only feed the spline, and level 0 of the
    directions other than the shadow direction) and the literal order (``floor_once=False``) elsewhere.
    The row blocks (``_sky_illum_block_rows``) keep the buffers in cache; the numba kernels of
    ``_sky_illum_search_accel`` replace the block loop when available.
    """
    if len(shifts) == 0:
        return max_slope
    n0, n1 = max_slope.shape
    d_dtype = hp.dtype
    q_dtype = np.result_type(np.empty(0, dtype=d_dtype), shifts[0][2])
    if np.result_type(max_slope, np.empty(0, dtype=q_dtype)) != max_slope.dtype:
        max_slope = max_slope.astype(np.result_type(max_slope, np.empty(0, dtype=q_dtype)))
    if q_dtype == np.float64 and _sky_illum_search_accel(hp, p, shifts, max_slope, floor_once):
        return max_slope
    rows = _sky_illum_block_rows(n1)
    shape_b = (min(rows, n0), n1)
    diff_b = np.empty(shape_b, dtype=d_dtype)
    quot_b = diff_b if q_dtype == d_dtype else np.empty(shape_b, dtype=q_dtype)
    for r0 in range(0, n0, rows):
        r1 = min(r0 + rows, n0)
        ms = max_slope[r0:r1]
        d_b = diff_b[:r1 - r0]
        q_b = quot_b[:r1 - r0]
        base = hp[p + r0:p + r1, p:p + n1]
        if floor_once:
            np.maximum(ms, 0., out=ms)
            for dy, dx, radius in shifts:
                np.subtract(hp[p + r0 - dy:p + r1 - dy, p - dx:p - dx + n1], base, out=d_b)
                np.divide(d_b, radius, out=q_b)
                np.maximum(ms, q_b, out=ms)
        else:
            for dy, dx, radius in shifts:
                np.subtract(hp[p + r0 - dy:p + r1 - dy, p - dx:p - dx + n1], base, out=d_b)
                np.divide(d_b, radius, out=q_b)
                np.maximum(q_b, 0., out=q_b)
                np.maximum(ms, q_b, out=ms)
    return max_slope


def _sky_illum_resample(x, y, z, x_fine, y_fine):
    """
    Return ``RectBivariateSpline(x, y, z, kx=1, ky=1)(x_fine, y_fine)`` (bilinear), skipping the fit when exact.

    ``x``, ``y`` are the coarse row and column coordinates (in level-0 px), ``z`` the coarse max_slope
    (len(x), len(y)), ``x_fine``, ``y_fine`` the coordinates of the next finer level. Returns float64
    (len(x_fine), len(y_fine)), bitwise equal to the scipy call.

    Why the fit can be skipped: for kx = ky = 1 and s = 0 the knots are (x[0], x, x[-1]) and the fitted
    coefficients are ``z`` itself whenever ``(1 / dx) * dx == 1`` for every knot step (the B-spline values
    at the data points are then exactly 1 and 0, every Givens rotation of the FITPACK least-squares solve
    has cos = 0, sin = 1, and the back substitution only subtracts exact zeros) and ``z`` is finite; NaN
    or inf would be spread by the fit, so then the fit is run. The evaluation is the same FITPACK
    ``bispev`` (``scipy.interpolate.bisplev``) with the same knots. Inputs that do not meet the conditions
    (other dtypes or shapes, non-increasing or inexact steps, non-finite ``z``) go to
    ``RectBivariateSpline``, the reference.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    direct = (z.ndim == 2 and z.dtype == np.float64 and x.ndim == 1 and y.ndim == 1
              and z.shape == (x.size, y.size) and x.size >= 2 and y.size >= 2
              and np.size(x_fine) > 0 and np.size(y_fine) > 0)
    if direct:
        tx = np.concatenate((x[:1], x, x[-1:])).astype(np.float64)
        ty = np.concatenate((y[:1], y, y[-1:])).astype(np.float64)
        dtx = np.diff(tx[1:-1])
        dty = np.diff(ty[1:-1])
        direct = (bool(np.all(dtx > 0)) and bool(np.all(dty > 0))
                  and bool(np.all((1. / dtx) * dtx == 1.)) and bool(np.all((1. / dty) * dty == 1.))
                  and bool(np.isfinite(z).all()))
    if not direct:
        return RectBivariateSpline(x, y, z, kx=1, ky=1)(x_fine, y_fine)
    from scipy.interpolate import bisplev
    x_fine = np.asarray(x_fine)
    y_fine = np.asarray(y_fine)
    out = bisplev(x_fine, y_fine, (tx, ty, np.ascontiguousarray(z).ravel(), 1, 1))
    return np.asarray(out).reshape((x_fine.size, y_fine.size))


def sky_illumination(dem,
                     resolution,
                     sky_model="overcast",
                     compute_shadow=False,
                     shadow_horizon_only=False,
                     max_fine_radius=100,
                     num_directions=32,
                     shadow_az=315,
                     shadow_el=35,
                     ve_factor=1,
                     no_data=None
                     ):
    """
    Compute sky illumination (diffuse light from a uniform or overcast sky), optionally with cast shadow.

    For each of ``num_directions`` directions the horizon angle h is found with a multi-resolution horizon
    search (``horizon_generate_pyramids``: scale 2, 20 px per level, up to ``max_fine_radius`` px), and the
    irradiance of the visible sky is integrated per direction with the pixel's slope and aspect (Kokalj and
    Hesse 2017; docs/listofvis_skyillum.rst). With da = pi / num_directions and
    ``g = -2 sin(da) cos(direction - aspect)``:

    - uniform = ``da cos(slope) sum cos(h)^2 + sin(slope) min(sum max(g (pi/4 - h/2 - sin(2h)/4), 0), pi)``,
      returned divided by pi (1 on flat open terrain);
    - overcast = ``0.33 uniform + 0.67 (2 da / 3 cos(slope) sum max(cos(h)^3, 0)
      + sin(slope) sum max(g (2/3 - cos h + cos(h)^3 / 3), 0))``, divided by its maximum over the array;
    - shadow (``compute_shadow``): ``horizon`` = h in degrees in the direction nearest to the sun, and
      ``shadow`` = 1 where ``horizon < shadow_el`` (lit), 0 where the pixel is in cast shadow. The shaded
      results are ``0.8 * model + 0.2 * shadow``.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W). Any NaN makes the whole result NaN (see Notes).
    resolution : float
        Pixel size (m/px), used for slope and aspect only (see Notes). Must not be negative.
    sky_model : str
        ``"overcast"`` (default) or ``"uniform"``, case-insensitive.
    compute_shadow : bool
        Compute the cast shadow and return the shaded model. Default False.
    shadow_horizon_only : bool
        With ``compute_shadow``: return ``{"shadow", "horizon"}`` right after the horizon search of the sun
        direction, skipping the remaining directions. Ignored without ``compute_shadow``. Default False.
    max_fine_radius : int
        Horizon search distance in px, e.g. 100 px = 25 m at 0.25 m/px. Default 100.
    num_directions : int
        Number of horizon search directions. Default 32.
    shadow_az : int or float
        Sun azimuth for the shadow, degrees clockwise from north, in [0, 360]. Replaced by the nearest
        search direction. Default 315.
    shadow_el : int or float
        Sun elevation for the shadow, degrees above the horizon, in [0, 90]. Default 35.
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**, before the float32 conversion.
        Default None.

    Returns
    -------
    numpy.ndarray or dict
        float64 (H, W): uniform / pi, ``0.8 * uniform + 0.2 * shadow`` (uniform not divided by pi, see
        Notes), overcast, or
        ``0.8 * overcast + 0.2 * shadow``, according to ``sky_model`` and ``compute_shadow``. With
        ``compute_shadow`` and ``shadow_horizon_only``: ``{"shadow": numpy.ndarray int64 (H, W) of 0/1,
        "horizon": numpy.ndarray float64 (H, W) in degrees}``.

    Raises
    ------
    Exception
        If ``ve_factor`` is outside [-10000, 10000], ``shadow_az`` is outside [0, 360], ``shadow_el`` is
        outside [0, 90], ``resolution`` is negative, or ``sky_model`` is unknown.

    Notes
    -----
    Edges: slope and aspect come from the DEM padded by 20 px in symmetric mode; the pyramid pads with the
    DEM minimum (constant), so the terrain beyond the edge is treated as low and open.

    Known issues (see docs/KNOWN_ISSUES.md):

    - the heights are not divided by ``resolution`` (``sky_view_factor`` does divide), so the horizon
      angles use m per px: ``horizon`` and ``shadow`` are the same for any resolution;
    - any NaN in the DEM makes ``dem.min()`` NaN, which spreads through the pyramid padding, the coarse
      levels and the spline to every pixel: the whole result (and ``horizon``) is NaN;
    - the uniform model is divided by pi only after the shaded result is formed, so
      ``0.8 * uniform + 0.2 * shadow`` uses the undivided uniform (0.8 pi + 0.2 on flat terrain);
    - ``shadow_az`` is matched to the direction key nearest to ``360 - shadow_az`` without wrap-around:
      ``shadow_az = 0`` picks the key 348.75 (32 directions), i.e. the same as ``shadow_az = 11.25``, while
      ``shadow_az = 360`` picks 0;
    - a tilted plane gives different values by facing direction (slope 26.6°, uniform model,
      ``max_fine_radius`` 20 px: about 1.060 facing east or west, 0.952 facing north or south), and values
      above 1, the flat-terrain value;
    - the pyramid is anchored at the top-left corner of the array, so a tile matches the full-raster
      result exactly only when its origin is a multiple of 2^3 = 8 px (3 coarse levels with the defaults).

    Implementation (bitwise identical to the original whole-array code, which is kept as the reference
    and the fallback):

    - level 0 is searched only on the unpadded extent: every level-0 shift is at most 20 px, the width of
      the padding, so the shifted windows are slices and ``np.roll`` never wraps there;
    - coarse levels keep ``np.roll`` semantics on their full extent (the spline reads all of them), via
      a wrap-padded copy whose slices equal the rolled arrays;
    - level search and the per-direction terms run in row blocks of about 32768 elements
      (``_sky_illum_block_rows``) with the same ufuncs, operand order and dtypes per pixel; the level
      search has numba kernels (``_sky_illum_search_kernel``, ``_sky_illum_search_literal_kernel``;
      ``RVT_NO_ACCEL=1`` switches them off);
    - ``_sky_illum_resample`` evaluates the linear spline without refitting it when the fitted
      coefficients are the data themselves.

    See Also
    --------
    shadow_horizon, horizon_generate_pyramids, slope_aspect
    """
    # Structure:
    #   1. checks, no_data -> NaN (in place), float32, ve_factor, sky model;
    #   2. slope and aspect (symmetric padding by 20 px), DEM pyramid with the shifts of every level;
    #   3. choose the level-0 window (inner0) and the blocked path; allocate accumulators; map shadow_az
    #      to the nearest direction key; prepare wrap-padded coarse levels for the blocked path;
    #   4. per direction: horizon search from the coarsest level to level 0, resampling max_slope to the
    #      next finer level with a linear spline;
    #   5. per direction: horizon of the sun direction (return early for shadow_horizon_only), then the
    #      uniform / overcast directional terms (blocked or whole-array);
    #   6. combine with slope: uniform, overcast (normalised by its maximum), shaded variants; uniform / pi;
    #   7. return the array selected by sky_model and compute_shadow.
    # standard pyramid settings
    pyramid_scale = 2
    max_pyramid_radius = 20

    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.sky_illumination: ve_factor must be between -10000 and 10000!")
    if shadow_az > 360 or shadow_az < 0:
        raise Exception("rvt.vis.sky_illumination: shadow_az must be between 0 and 360!")
    if shadow_el > 90 or shadow_el < 0:
        raise Exception("rvt.vis.sky_illumination: shadow_el must be between 0 and 90!")
    if resolution < 0:
        raise Exception("rvt.vis.sky_illumination: resolution must be a positive number!")

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    dem = dem * ve_factor

    if sky_model.lower() == "overcast":
        compute_overcast = True
        compute_uniform = False
    elif sky_model.lower() == "uniform":
        compute_overcast = False
        compute_uniform = True
    else:
        raise Exception("rvt.vis.sky_illumination: sky_model must be overcast or uniform!")

    # generate slope and aspect
    _ = slope_aspect(np.pad(dem, max_pyramid_radius, mode="symmetric"), resolution, resolution)
    slope = _["slope"]
    aspect = _["aspect"]

    # build DEM pyramids
    pyramid = horizon_generate_pyramids(dem,
                                        num_directions=num_directions,
                                        max_fine_radius=max_fine_radius,
                                        max_pyramid_radius=max_pyramid_radius,
                                        pyramid_scale=pyramid_scale, )
    n_levels = np.max([i for i in pyramid])

    # get the convolution window indices
    conv_to = int(np.floor(pyramid_scale / 2.))
    if (pyramid_scale % 2) == 0:
        conv_from = 1 - conv_to
    else:
        conv_from = -conv_to
    # directional halve-resolution for integration limits
    da = np.pi / num_directions

    # Level 0 is searched only on the unpadded extent (the padding is cropped from every output and
    # nothing tile-wide is taken before the crop). With |shift| <= max_pyramid_radius the shifted
    # windows stay inside the padded level-0 DEM, so np.roll never wraps there. Coarser levels are
    # searched on their full extent with np.roll semantics, because the spline reads all of them.
    pm = max_pyramid_radius
    shift0 = pyramid[0]["shift"]
    inner0 = all(abs(int(s[0])) <= pm and abs(int(s[1])) <= pm for d in shift0 for s in shift0[d]["shift"])
    p_in = pm if inner0 else None
    if inner0:
        work_shape = (pyramid[0]["dem"].shape[0] - 2 * pm, pyramid[0]["dem"].shape[1] - 2 * pm)
        slope = slope[pm:-pm, pm:-pm]
        aspect = aspect[pm:-pm, pm:-pm]
    else:
        work_shape = pyramid[0]["dem"].shape

    # init the intermediate results for uniform SI
    uniform_a = np.zeros(work_shape, dtype=np.float32)
    uniform_b = np.copy(uniform_a)
    # init the output for overcast SI
    if compute_overcast:
        overcast_out = np.zeros(dem.shape, dtype=np.float32)
        overcast_c = np.zeros(work_shape, dtype=np.float32)
        overcast_d = np.copy(overcast_c)
    else:
        overcast_out = None
    # init the output for shadows
    if compute_shadow:
        # use closest direction from pyramids as proxy for shadow azimuth
        # (just in case it is not the same as standard directions)
        _ = np.array([d for d in pyramid[0]["shift"]])
        i = np.argmin(np.abs(_ - (360 - shadow_az)))
        shadow_az = _[i]
        # binary shadows
        shadow_out = np.zeros(dem.shape, dtype=np.float32)
        # height of horizon in degrees
        horizon_out = np.zeros(dem.shape, dtype=np.float32)
        # overcast model + binary shadow
        if compute_overcast:
            overcast_sh_out = np.zeros(dem.shape, dtype=np.float32)
        else:
            overcast_sh_out = None
        # uniform model + binary shadow
        uniform_sh_out = np.zeros(dem.shape, dtype=np.float32)
    else:
        shadow_out = None
        horizon_out = None
        overcast_sh_out = None
        uniform_sh_out = None

    # Blocked level 0 and directional terms: every pixel is independent, so level 0 and the
    # per-direction terms are computed in row blocks that stay in cache (same ufuncs, operands,
    # operand order and dtypes per pixel). Used for the float32 level-0 DEM on the unpadded extent
    # (max_slope and the directional terms are then float64); otherwise the whole-array path below runs.
    height0 = pyramid[0]["dem"]
    blocked = (inner0 and height0.dtype == np.float32 and aspect.dtype in (np.float32, np.float64)
               and all(shift0[d]["distance"].dtype == np.float64 and len(shift0[d]["shift"]) > 0
                       for d in shift0))
    if blocked:
        n_w0, n_w1 = work_shape
        rows_blk = _sky_illum_block_rows(n_w1)
        blocks = [(r0, min(r0 + rows_blk, n_w0)) for r0 in range(0, n_w0, rows_blk)]
        blk_shape = (min(rows_blk, n_w0), n_w1)
        ang_b = np.empty(blk_shape, dtype=np.float64)
        dasp_b = np.empty(blk_shape, dtype=np.float64)
        buf_b = np.empty(blk_shape, dtype=np.float64)
        cos_b = np.empty(blk_shape, dtype=np.float64)

        def _shift_lists(move):
            """Return ``{direction: [(dy, dx, radius), ...]}`` from a ``horizon_shift_vector`` dict, shifts
            as Python ints and radii as the np.float64 elements of the distance array (the dtype the
            reference division uses)."""
            return {d: [(int(s[0]), int(s[1]), r) for s, r in zip(move[d]["shift"], move[d]["distance"])]
                    for d in move}

        shifts0 = _shift_lists(shift0)
        d_aspect_coef = -2 * np.sin(da)
        # Coarser levels keep np.roll semantics on their full extent: with |shift| <= p <= the level
        # size, np.roll(height, (dy, dx))[i, j] == np.pad(height, p, mode="wrap")[p + i - dy, p + j - dx].
        coarse_wrap = {}
        for i_level in range(1, n_levels + 1):
            h_c = pyramid[i_level]["dem"]
            move_c = pyramid[i_level]["shift"]
            p_c = max([max(abs(int(s[0])), abs(int(s[1]))) for d in move_c for s in move_c[d]["shift"]] + [0])
            if (h_c.ndim == 2 and p_c <= min(h_c.shape) and h_c.dtype.kind == "f"
                    and all(move_c[d]["distance"].dtype == np.float64 for d in move_c)):
                coarse_wrap[i_level] = (np.pad(h_c, p_c, mode="wrap"), p_c, _shift_lists(move_c))
            else:
                coarse_wrap[i_level] = None

    # search for horizon in each direction...
    for i_dir, direction in enumerate(pyramid[0]["shift"]):
        dir_rad = np.radians(direction)
        # reset maximum at each iteration (direction)
        top_shape = work_shape if n_levels == 0 else pyramid[n_levels]["dem"].shape
        max_slope = np.zeros(top_shape, dtype=np.float32) - 1000

        for i_level in reversed(range(n_levels + 1)):
            height = pyramid[i_level]["dem"]
            move = pyramid[i_level]["shift"]

            # ... and to the search radius (keep the largest slope)
            if i_level == 0 and blocked:
                # the zero sign of level 0 reaches the horizon of the shadow direction: literal there
                max_slope = _sky_illum_search_blocked(height0, pm, shifts0[direction], max_slope,
                                                      not (compute_shadow and direction == shadow_az))
            elif i_level > 0 and blocked and coarse_wrap[i_level] is not None:
                # coarse levels only reach the spline, whose values do not depend on the zero sign
                hp_c, p_c, shifts_c = coarse_wrap[i_level]
                max_slope = _sky_illum_search_blocked(hp_c, p_c, shifts_c[direction], max_slope, True)
            else:
                max_slope = _sky_illum_level_search(height, move, direction, max_slope,
                                                    p_in if i_level == 0 else None)

            # resample the max_slope to a lower pyramid level
            if i_level > 0:
                lin_fine = pyramid[i_level - 1]["i_lin"] + (
                        conv_from + max_pyramid_radius * pyramid_scale - max_pyramid_radius)
                col_fine = pyramid[i_level - 1]["i_col"] + (
                        conv_from + max_pyramid_radius * pyramid_scale - max_pyramid_radius)
                if i_level == 1 and inner0:
                    # the spline is evaluated point by point: a subset of the points gives the same values
                    lin_fine = lin_fine[pm:-pm]
                    col_fine = col_fine[pm:-pm]
                lin_coarse = pyramid[i_level]["i_lin"] * pyramid_scale
                col_coarse = pyramid[i_level]["i_col"] * pyramid_scale
                max_slope = _sky_illum_resample(lin_coarse, col_coarse, max_slope, lin_fine, col_fine)

        if blocked:
            if compute_shadow and (direction == shadow_az):
                horizon_out = np.degrees(np.arctan(max_slope))
                shadow_out = (horizon_out < shadow_el) * 1
                if shadow_horizon_only:
                    return {"shadow": shadow_out, "horizon": horizon_out}
            # float32 accumulators are promoted to float64 by the first acc + term (0. + term)
            if uniform_a.dtype != np.float64:
                uniform_a = uniform_a.astype(np.float64)
                uniform_b = uniform_b.astype(np.float64)
                if compute_overcast:
                    overcast_c = overcast_c.astype(np.float64)
                    overcast_d = overcast_d.astype(np.float64)
            for r0, r1 in blocks:
                n_r = r1 - r0
                # Numpy reference this block reproduces (the whole-array expressions below), not dead code:
                #   _ = np.arctan(max_slope)
                _ = np.arctan(max_slope[r0:r1], out=ang_b[:n_r])
                # reference: _d_aspect = -2 * np.sin(da) * np.cos(dir_rad - aspect), evaluated left to right,
                # so the constant -2 * np.sin(da) (d_aspect_coef) is formed first
                _d_aspect = np.subtract(dir_rad, aspect[r0:r1], out=dasp_b[:n_r])
                np.cos(_d_aspect, out=_d_aspect)
                np.multiply(d_aspect_coef, _d_aspect, out=_d_aspect)
                # the same sequence as the whole-array float64 branch below
                cos_ = np.cos(_, out=cos_b[:n_r])
                buf = np.square(cos_, out=buf_b[:n_r])
                np.add(uniform_a[r0:r1], buf, out=uniform_a[r0:r1])
                np.multiply(2., _, out=buf)
                np.sin(buf, out=buf)
                np.divide(buf, 4., out=buf)
                np.divide(_, 2., out=_)
                np.subtract(np.pi / 4., _, out=_)
                np.subtract(_, buf, out=_)
                np.multiply(_d_aspect, _, out=_)
                np.maximum(_, 0, out=_)
                np.add(uniform_b[r0:r1], _, out=uniform_b[r0:r1])
                if compute_overcast:
                    _cos3 = np.power(cos_, 3, out=buf)
                    np.maximum(_cos3, 0, out=_)
                    np.add(overcast_c[r0:r1], _, out=overcast_c[r0:r1])
                    np.subtract(2. / 3., cos_, out=cos_)
                    np.divide(_cos3, 3., out=_cos3)
                    np.add(cos_, _cos3, out=cos_)
                    np.multiply(_d_aspect, cos_, out=cos_)
                    np.maximum(cos_, 0, out=cos_)
                    np.add(overcast_d[r0:r1], cos_, out=overcast_d[r0:r1])
            continue

        # convert to angle in radians and compute directional output
        _ = np.arctan(max_slope)
        if compute_shadow and (direction == shadow_az):
            if inner0:
                horizon_out = np.degrees(_)
            else:
                horizon_out = np.degrees(
                    _[max_pyramid_radius:-max_pyramid_radius, max_pyramid_radius:-max_pyramid_radius])
            shadow_out = (horizon_out < shadow_el) * 1
            if shadow_horizon_only:
                return {"shadow": shadow_out, "horizon": horizon_out}
        _d_aspect = -2 * np.sin(da) * np.cos(dir_rad - aspect)
        if _.dtype == np.float64 and _d_aspect.dtype == np.float64:
            # same operations and operand order as the expressions below, in reused buffers
            cos_ = np.cos(_)
            buf = np.square(cos_)  # == cos_ ** 2
            uniform_a = _accumulate(uniform_a, buf)
            # reference expression reproduced in place: np.pi / 4. - _ / 2. - np.sin(2. * _) / 4.
            np.multiply(2., _, out=buf)
            np.sin(buf, out=buf)
            np.divide(buf, 4., out=buf)
            np.divide(_, 2., out=_)
            np.subtract(np.pi / 4., _, out=_)
            np.subtract(_, buf, out=_)
            np.multiply(_d_aspect, _, out=_)
            np.maximum(_, 0, out=_)
            uniform_b = _accumulate(uniform_b, _)
            if compute_overcast:
                _cos3 = np.power(cos_, 3, out=buf)  # == cos_ ** 3
                np.maximum(_cos3, 0, out=_)
                overcast_c = _accumulate(overcast_c, _)
                # reference expression reproduced in place: 2. / 3. - np.cos(_) + _cos3 / 3.
                np.subtract(2. / 3., cos_, out=cos_)
                np.divide(_cos3, 3., out=_cos3)
                np.add(cos_, _cos3, out=cos_)
                np.multiply(_d_aspect, cos_, out=cos_)
                np.maximum(cos_, 0, out=cos_)
                overcast_d = _accumulate(overcast_d, cos_)
        else:
            # Reference expressions of the original code (used for other dtypes; the two branches above
            # reproduce them bitwise for float64)
            uniform_a = uniform_a + (np.cos(_)) ** 2
            uniform_b = uniform_b + np.maximum(_d_aspect * (np.pi / 4. - _ / 2. - np.sin(2. * _) / 4.), 0)
            if compute_overcast:
                _cos3 = (np.cos(_)) ** 3
                overcast_c = overcast_c + np.maximum(_cos3, 0)
                overcast_d = overcast_d + np.maximum(_d_aspect * (2. / 3. - np.cos(_) + _cos3 / 3.), 0)

    # because of numeric stability check if the uniform_b is less then pi
    uniform_out = da * np.cos(slope) * uniform_a + np.sin(slope) * np.minimum(uniform_b, np.pi)
    if not inner0:
        uniform_out = uniform_out[max_pyramid_radius:-max_pyramid_radius, max_pyramid_radius:-max_pyramid_radius]

    if compute_overcast:
        overcast_out = (2. * da / 3.) * np.cos(slope) * overcast_c + np.sin(slope) * overcast_d
        if not inner0:
            overcast_out = overcast_out[max_pyramid_radius:-max_pyramid_radius,
                                        max_pyramid_radius:-max_pyramid_radius]
        overcast_out = 0.33 * uniform_out + 0.67 * overcast_out
        overcast_out = overcast_out / overcast_out.max()
    if compute_shadow:
        uniform_sh_out = (0.8 * uniform_out + 0.2 * shadow_out)
        if compute_overcast:
            overcast_sh_out = (0.8 * overcast_out + 0.2 * shadow_out)

    # normalize
    uniform_out = uniform_out / np.pi

    # output
    if compute_uniform and not compute_shadow:
        return uniform_out
    elif compute_uniform and compute_shadow:
        return uniform_sh_out
    elif compute_overcast and not compute_shadow:
        return overcast_out
    elif compute_overcast and compute_shadow:
        return overcast_sh_out


def shadow_horizon(dem,
                   resolution,
                   shadow_az=315,
                   shadow_el=35,
                   ve_factor=1,
                   no_data=None
                   ):
    """
    Compute the cast shadow and the horizon angle towards the sun.

    Calls ``sky_illumination(..., compute_shadow=True, shadow_horizon_only=True)`` with its default search
    (32 directions, 100 px), which stops after the direction nearest to the sun.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W).
    resolution : float
        Pixel size (m/px). Must not be negative. Does not affect the result (see ``sky_illumination``).
    shadow_az : int or float
        Sun azimuth, degrees clockwise from north, in [0, 360]; replaced by the nearest search direction.
        Default 315.
    shadow_el : int or float
        Sun elevation, degrees above the horizon, in [0, 90]. Default 35.
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**. Default None.

    Returns
    -------
    dict
        ``{"shadow": numpy.ndarray int64 (H, W), "horizon": numpy.ndarray float64 (H, W)}``: ``horizon`` is
        the horizon elevation angle in degrees in the sun direction; ``shadow`` is 1 where
        ``horizon < shadow_el`` (lit) and 0 where the pixel lies in cast shadow.

    Raises
    ------
    Exception
        If ``ve_factor`` is outside [-10000, 10000], ``shadow_az`` is outside [0, 360], ``shadow_el`` is
        outside [0, 90], or ``resolution`` is negative.

    Notes
    -----
    The known issues of ``sky_illumination`` apply (resolution not used for the angles; one NaN pixel makes
    the horizon NaN everywhere and the shadow 0 everywhere; ``shadow_az`` 0 and 360 differ; tile origin), see
    docs/KNOWN_ISSUES.md (KI-06, KI-07).

    See Also
    --------
    sky_illumination, hillshade
    """
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.shadow_horizon: ve_factor must be between -10000 and 10000!")
    if shadow_az > 360 or shadow_az < 0:
        raise Exception("rvt.vis.shadow_horizon: shadow_az must be between 0 and 360!")
    if shadow_el > 90 or shadow_el < 0:
        raise Exception("rvt.vis.shadow_horizon: shadow_el must be between 0 and 90!")
    if resolution < 0:
        raise Exception("rvt.vis.shadow_horizon: resolution must be a positive number!")

    return sky_illumination(dem=dem, resolution=resolution, compute_shadow=True,
                            shadow_horizon_only=True, shadow_el=shadow_el, shadow_az=shadow_az, ve_factor=ve_factor,
                            no_data=no_data)


def msrm(dem,
         resolution,
         feature_min,
         feature_max,
         scaling_factor,
         ve_factor=1,
         no_data=None
         ):
    """
    Compute the multi-scale relief model (MSRM).

    After Orengo and Petrie 2018 (docs/listofvis_msrm.rst). Mean-filtered surfaces L_k are computed with
    radii ``k ** scaling_factor`` px for k = i .. n, where
    ``i = floor(((feature_min - resolution) / (2 resolution)) ** (1 / scaling_factor))`` and
    ``n = ceil(((feature_max - resolution) / (2 resolution)) ** (1 / scaling_factor))``. The result is the
    mean of the differences of consecutive surfaces, ``sum_k (L_{k-1} - L_k) / (n - i)``.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W). Not checked for 2D.
    resolution : float
        Pixel size (m/px). Must not be negative.
    feature_min : float
        Minimum feature size to enhance, in m (not px); raised to ``resolution`` if smaller.
    feature_max : float
        Maximum feature size to enhance, in m, e.g. 20 m = 80 px at 0.25 m/px.
    scaling_factor : int
        Exponent of the radius sequence, truncated with ``int``. Larger values give a wider range of
        values (more contrast) with less sensitivity to intermediate feature sizes.
    ve_factor : int or float
        Vertical exaggeration, in [-10000, 10000]. Default 1.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**, before the float32 conversion.
        Default None.

    Returns
    -------
    numpy.ndarray
        float64 (H, W) in m (times ``ve_factor``); NaN where the input is NaN.

    Raises
    ------
    Exception
        If ``ve_factor`` is outside [-10000, 10000] or ``resolution`` is negative.

    Notes
    -----
    Edges: ``mean_filter`` pads in edge mode. Each difference ``L_{k-1} - L_k`` is taken in float32 and
    added to a float64 sum.

    Known issue (see docs/KNOWN_ISSUES.md): when i == n (e.g. ``feature_min == feature_max == resolution``,
    or any equal pair that gives an integer power) there is no
    difference to average and the sum is divided by 0: every pixel is NaN (RuntimeWarning).

    See Also
    --------
    mean_filter, slrm
    """
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.vis.msrm: ve_factor must be between -10000 and 10000!")
    if resolution < 0:
        raise Exception("rvt.vis.msrm: resolution must be a positive number!")

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    dem = dem * ve_factor

    if feature_min < resolution:  # feature_min can't be smaller than resolution
        feature_min = resolution

    scaling_factor = int(scaling_factor)  # has to be integer

    # calculation of i and n (from article)
    i = int(np.floor(((feature_min - resolution) / (2 * resolution)) ** (1 / scaling_factor)))
    n = int(np.ceil(((feature_max - resolution) / (2 * resolution)) ** (1 / scaling_factor)))

    # lpf = low pass filter
    relief_models_sum = np.zeros(dem.shape)  # sum of all substitution of 2 consecutive
    nr_relief_models = 0  # number of additions (substitutions of 2 consecutive surfaces)
    last_lpf_surface = 0

    # generation of filtered surfaces (lpf_surface)
    for ndx in range(i, n + 1, 1):
        kernel_radius = ndx ** scaling_factor
        # calculate mean filtered surface
        lpf_surface = mean_filter(dem=dem, kernel_radius=kernel_radius)
        if not ndx == i:  # if not first surface
            relief_models_sum += (last_lpf_surface - lpf_surface)  # substitution of 2 consecutive lpf_surface
            nr_relief_models += 1
        last_lpf_surface = lpf_surface

    msrm_out = relief_models_sum / nr_relief_models

    return msrm_out


def integral_image(dem, data_type=np.float64):
    """
    Return the integral image (summed-area table) of a 2D array, origin at the top-left corner.

    ``out[i, j] = sum(dem[:i + 1, :j + 1])``, computed as a cumulative sum along axis 0 and then axis 1.

    Parameters
    ----------
    dem : numpy.ndarray
        2D array, shape (H, W). Not modified.
    data_type : numpy dtype or type
        dtype the array is converted to before summing. Default ``np.float64``.

    Returns
    -------
    numpy.ndarray
        Shape (H, W), dtype ``data_type``; integer and bool types narrower than the platform integer are
        promoted by ``numpy.cumsum`` (int8 gives int64 on 64-bit platforms).

    Notes
    -----
    The box sum of rows r0 .. r1 and columns c0 .. c1 is
    ``I[r1, c1] - I[r0 - 1, c1] - I[r1, c0 - 1] + I[r0 - 1, c0 - 1]``. Summing float elevations in
    float64 keeps the differences of large sums exact enough for windows of hundreds of pixels.

    References
    ----------
    https://en.wikipedia.org/wiki/Summed-area_table

    Examples
    --------
    >>> integral_image(np.array([[7, 4, 7, 2], [6, 9, 9, 5], [6, 6, 7, 6]]))
    array([[ 7., 11., 18., 20.],
           [13., 26., 42., 49.],
           [19., 38., 61., 74.]])
    """
    dem = dem.astype(data_type)
    return dem.cumsum(axis=0).cumsum(axis=1)


def topographic_dev(dem, dem_i_nr_pixels, dem_i1, dem_i2, kernel_radius):
    """
    Compute the deviation from mean elevation, DEV(D) = (z0 - mean_D) / std_D, from summed-area tables.

    z0 is the centre pixel, mean_D and std_D are the mean and standard deviation of the
    (2D + 1) x (2D + 1) window with D = ``kernel_radius`` (Lindsay et al. 2015, docs/listofvis_mstp.rst).
    The window sums are four ``np.roll`` copies of each table, so the result is meaningful only away from
    the borders of the (padded) input.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m) padded by (D_max + 1, D_max) px, NaN replaced by 0, shape (H', W').
    dem_i_nr_pixels : numpy.ndarray
        Summed-area table of the non-NaN mask of ``dem`` (pixel counts), shape (H', W').
    dem_i1 : numpy.ndarray
        Summed-area table of ``dem``, shape (H', W').
    dem_i2 : numpy.ndarray
        Summed-area table of ``dem ** 2``, shape (H', W').
    kernel_radius : int
        Window radius D in px, e.g. D = 50 px = 12.5 m at 0.25 m/px. Truncated with ``int``.

    Returns
    -------
    numpy.ndarray
        Shape (H', W'), dtype of the tables (float64 for float64 tables). Element (i, j) belongs to the window
        centred on ``dem[i + 1, j + 1]`` (hence ``np.roll(dem, (-1, -1))``); for D <= 0 the input ``dem``
        itself is returned.

    Notes
    -----
    ``std = sqrt(|E[z^2] - mean^2|)``; the absolute value absorbs small negative variances from rounding,
    and 1e-6 is added to ``std`` to avoid division by 0. Windows without valid pixels give NaN (division
    warnings are suppressed).

    Not called inside rvt: ``max_elevation_deviation`` evaluates the same expressions on the output window
    only (``_box_sum``, ``_mstp_dev_max_kernel``). Kept as public API.

    See Also
    --------
    max_elevation_deviation, integral_image, mstp
    """
    radius_cell = int(kernel_radius)
    if radius_cell <= 0:
        return dem

    kernel_nr_pix_arr = (np.roll(dem_i_nr_pixels, (radius_cell, radius_cell), axis=(0, 1)) +
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (radius_cell, -radius_cell - 1), axis=(0, 1)))

    # sum
    dem_mean = (np.roll(dem_i1, (radius_cell, radius_cell), axis=(0, 1)) +
                np.roll(dem_i1, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                np.roll(dem_i1, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                np.roll(dem_i1, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    # divide with nr of pixels inside kernel
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warning for dividing by zero
        dem_mean = dem_mean / kernel_nr_pix_arr

    # std
    dem_std = (np.roll(dem_i2, (radius_cell, radius_cell), axis=(0, 1)) +
               np.roll(dem_i2, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
               np.roll(dem_i2, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
               np.roll(dem_i2, (radius_cell, -radius_cell - 1), axis=(0, 1)))

    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warning for dividing by zero
        dem_std = np.sqrt(np.abs(dem_std / kernel_nr_pix_arr - dem_mean ** 2))
        # returns nan values where division by zero happens

    dev_out = (np.roll(dem, (-1, -1), axis=(0, 1)) - dem_mean) / (dem_std + 1e-6)  # add 1e-6 to prevent division with 0

    return dev_out


@_accel.jit_kernel
def _mstp_dev_max_kernel(dem_i1, dem_i2, dem_i_nr_pixels, has_nr_pixels, dem_centre, radii, maximum_radius, out,
                         buf):
    """
    Numba kernel: DEV_max of ``max_elevation_deviation`` over all radii, for the output window.

    Inputs: ``dem_i1``, ``dem_i2`` the float64 summed-area tables of the padded DEM and of its squares;
    ``dem_i_nr_pixels`` the int64 table of the non-NaN mask, used only if ``has_nr_pixels`` (else a dummy
    and the count is ``(2r + 1)^2``); ``dem_centre`` (H, W) the centre elevations (float32 or float64);
    ``radii`` (int64, all > 0) in the order of ``range(minimum_radius, maximum_radius + 1, step)``;
    ``maximum_radius`` the padding offset. Output: ``out`` (H, W) float32, written in place; ``buf`` is a
    float64 scratch row of length W. Serial, so independent of the thread count.

    Equivalence with the sliced numpy path (``_box_sum``), bitwise, per pixel and radius:

    - box sums ``((a + b) - c) - d`` of the tables in the same term order;
    - the int64 pixel-count box sum, exact in int64, converted to float64 (numpy promotes it in the
      division);
    - ``mean = s1 / n``; ``var = s2 / n - mean * mean`` (numpy's ``mean ** 2`` is computed as
      ``mean * mean``); ``std = sqrt(|var|) + 1e-6``; ``dev = (float64(z) - mean) / std``;
    - ``np.divide`` is a plain IEEE division (0 / 0 gives NaN, no ZeroDivisionError), as in numpy;
    - the running maximum is replaced where ``not (|max| >= |dev|)``, radii in the same order, so ties
      and NaN behave as in the numpy loop;
    - the float64 maximum of a row is rounded to float32 once when copied into ``out``, as the final
      ``astype(np.float32)`` does.
    """
    n_rows, n_cols = out.shape
    n_radii = radii.shape[0]
    for i in range(n_rows):
        for k in range(n_radii):
            r = radii[k]
            lo = maximum_radius - r
            hi = maximum_radius + r + 1
            row_lo = lo + i
            row_hi = hi + i
            nr_pix_const = np.float64((2 * r + 1) ** 2)
            for j in range(n_cols):
                col_lo = lo + j
                col_hi = hi + j
                nr_pix = nr_pix_const
                if has_nr_pixels:
                    nr_pix = np.float64(dem_i_nr_pixels[row_lo, col_lo] + dem_i_nr_pixels[row_hi, col_hi]
                                        - dem_i_nr_pixels[row_hi, col_lo] - dem_i_nr_pixels[row_lo, col_hi])
                s1 = dem_i1[row_lo, col_lo] + dem_i1[row_hi, col_hi]
                s1 = s1 - dem_i1[row_hi, col_lo]
                s1 = s1 - dem_i1[row_lo, col_hi]
                mean = np.divide(s1, nr_pix)
                s2 = dem_i2[row_lo, col_lo] + dem_i2[row_hi, col_hi]
                s2 = s2 - dem_i2[row_hi, col_lo]
                s2 = s2 - dem_i2[row_lo, col_hi]
                var = np.divide(s2, nr_pix) - mean * mean
                std = np.sqrt(abs(var)) + 1e-6
                dev = np.divide(np.float64(dem_centre[i, j]) - mean, std)
                m = buf[j]
                buf[j] = dev if (k == 0 or not (abs(m) >= abs(dev))) else m
        for j in range(n_cols):
            out[i, j] = buf[j]


def _max_elevation_deviation_disk(dem, radii, kernel_shape):
    """
    Return DEV_max of ``max_elevation_deviation`` over ``radii`` (px) with disk windows, float32 (H, W).

    Radii >= ``DISK_MIN_RADIUS_PX`` get the disk of ``_window_radius(r, kernel_shape)``, smaller positive radii the
    square (2r + 1) x (2r + 1) footprint, all through one set of FFT spectra (``_disk_window_sums``). The DEM is
    padded in symmetric mode by floor(r_w) of the largest window. Per radius, with ``s1``, ``s2`` the window sums
    of ``z - zbar`` and its square and ``n`` the valid count: ``mean = s1 / n``, ``var = s2 / n - mean ** 2``,
    ``std = sqrt(|var|) + 1e-6``, ``dev = ((z - zbar) - mean) / std``, the expressions of the square path shifted by
    ``zbar``. The running maximum keeps ``max`` where ``|max| >= |dev|`` (radii in order), the tie rule of the square
    path; it is rounded to float32 once at the end. A radius <= 0 contributes, as on the square path, the elevation
    one pixel up and left of the centre with NaN as 0 (known issue, docs/KNOWN_ISSUES.md).
    """
    radii = [int(r) for r in radii]
    windows = [(_window_radius(r, kernel_shape), True) if r >= DISK_MIN_RADIUS_PX else (float(r), False)
               for r in radii if r > 0]
    pad = int(np.floor(max(r_w for r_w, _ in windows)))
    n_rows, n_cols = dem.shape
    idx_nan_dem = np.isnan(dem)
    dem_pad = np.pad(dem, pad, mode="symmetric")
    zbar, sums = _disk_window_sums(dem_pad, dem.shape, windows, True)
    # z - zbar at the centre pixels, as _disk_window_sums computes it
    dem_centre = dem_pad[pad:pad + n_rows, pad:pad + n_cols].astype(np.float64)
    dem_centre -= zbar
    dev_max_out = None
    for kernel_radius in radii:
        if kernel_radius <= 0:
            # the square path takes the padded dem (NaN -> 0) one pixel up and left (known issue)
            dev = dem_pad[pad - 1:pad - 1 + n_rows, pad - 1:pad - 1 + n_cols]
            dev = np.where(np.isnan(dev), dev.dtype.type(0), dev)
        else:
            s1, s2, count = next(sums)
            with np.errstate(divide="ignore", invalid="ignore"):  # windows without a valid pixel: output NaN
                dem_mean = s1 / count
                dem_std = s2 / count - dem_mean ** 2
                np.abs(dem_std, out=dem_std)
                np.sqrt(dem_std, out=dem_std)
            dem_std += 1e-6  # as on the square path: no division by 0 where the window is flat
            dev = (dem_centre - dem_mean) / dem_std
            del s1, s2, count, dem_mean, dem_std
        if dev_max_out is None:
            dev_max_out = dev
        else:
            dev_max_out = np.where(np.abs(dev_max_out) >= np.abs(dev), dev_max_out, dev)
        del dev
    dev_max_out = dev_max_out.astype(np.float32)
    dev_max_out[idx_nan_dem] = np.nan
    return dev_max_out


def max_elevation_deviation(dem, minimum_radius, maximum_radius, step, kernel_shape="square"):
    """
    Compute the maximum deviation from mean elevation (DEV_max) over a range of window radii.

    For every radius D in ``range(minimum_radius, maximum_radius + 1, step)`` the deviation
    ``(z0 - mean_D) / (std_D + 1e-6)`` of ``topographic_dev`` is computed, and per pixel the value with the
    largest absolute deviation is kept (Lindsay et al. 2015, docs/listofvis_mstp.rst). NaN pixels are left
    out of the window statistics.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W), float32 or float64 (NaN as no-data). Not modified.
    minimum_radius : int
        Smallest window radius in px, truncated with ``int``.
    maximum_radius : int
        Largest window radius in px, truncated with ``int``; also the padding width, e.g. 500 px = 125 m at
        0.25 m/px.
    step : int
        Radius step in px, truncated with ``int``; must be positive (``range`` raises ValueError for 0).
    kernel_shape : str
        "square" (default, (2D + 1) x (2D + 1) windows), "disk" (pixels with dx^2 + dy^2 <= D^2) or
        "disk_equal_area" (dx^2 + dy^2 <= r_w^2 with r_w = (2D + 1) / sqrt(pi), the disk with the square's area).
        When the largest radius is >= ``DISK_MIN_RADIUS_PX`` (8 px = 2 m at 0.25 m/px) every radius of the call
        goes through the FFT path, with the disk from 8 px and the square below; when all radii are < 8 px the
        square code runs unchanged (bitwise the default result).

    Returns
    -------
    numpy.ndarray
        float32 (H, W), signed deviation in standard deviations; NaN where ``dem`` is NaN.

    Raises
    ------
    ValueError
        If ``kernel_shape`` is not one of the three values.

    Notes
    -----
    Edges: the DEM is padded by ``maximum_radius + 1`` before and ``maximum_radius`` after each axis in
    symmetric mode, so the box sums of all radii are slices of the same summed-area tables. The tables of
    z and z^2 are float64 and the squares are taken in float64: float32 squares of elevations of hundreds
    of metres lose the small within-window variance. Without NaN the pixel count is the constant
    (2D + 1)^2; with NaN it comes from an int64 table of the non-NaN mask.

    Tie rule: the running maximum is replaced where ``not (|max| >= |dev|)``, so on equal absolute values
    the smaller radius is kept, and a NaN deviation replaces the maximum.

    Implementation: the original code called ``topographic_dev`` on the padded extent (12 ``np.roll``
    copies of the tables per radius) and cropped. Here the same float64 expressions, in the same order, are evaluated on
    the output window from slices (``_box_sum``), or per pixel by the numba kernel
    ``_mstp_dev_max_kernel`` when every radius is positive (``RVT_NO_ACCEL=1`` switches it off); both are
    bitwise identical to the original. The float64 running maximum is rounded to float32 once at the end.

    Known issue (see docs/KNOWN_ISSUES.md): a radius D <= 0 contributes the elevation itself (as
    ``topographic_dev`` returns the DEM), taken one pixel up and left of the centre (output (i, j) is
    ``z[i - 1, j - 1]``, with the symmetric padding at the border), not a deviation.

    Disk windows (``_max_elevation_deviation_disk``): the DEM is padded in symmetric mode by floor(r_w) of the
    largest window; the window sums of ``z - zbar`` and ``(z - zbar)^2`` (``zbar`` the float64 mean of the non-NaN
    padded pixels) and of the valid mask are FFT convolutions (``_disk_window_sums``, scipy.fft, float64,
    ``workers=1``, data spectra once per call). The deviation, the ``abs`` and ``+ 1e-6`` of the std, the tie rule
    and the final float32 rounding are those of the square path; removing ``zbar`` first keeps the rounding error
    at the scale of the relief, not of the elevation (DEV of a tile at +2 500 m within 1e-7 of the same tile at
    0 m, tests/test_disk_oracle.py). No numba kernel and no threads on this path. See docs/DISK_WINDOWS.md.

    See Also
    --------
    mstp, topographic_dev, integral_image
    """
    if _disk_path(kernel_shape, lambda: range(int(minimum_radius), int(maximum_radius) + 1, int(step)),
                  "rvt.vis.max_elevation_deviation", ValueError):
        return _max_elevation_deviation_disk(dem, range(int(minimum_radius), int(maximum_radius) + 1, int(step)),
                                             kernel_shape)
    minimum_radius = int(minimum_radius)
    maximum_radius = int(maximum_radius)
    step = int(step)

    # store positions of nan
    idx_nan_dem = np.isnan(dem)

    dem_pad = np.pad(dem, (maximum_radius + 1, maximum_radius), mode="symmetric")
    # store nans
    idx_nan_dem_pad = np.isnan(dem_pad)
    # change nan to 0
    dem_pad[idx_nan_dem_pad] = 0

    # number of pixels for summed area table; without nan the count inside a kernel is (2 * r + 1) ** 2
    if idx_nan_dem_pad.any():
        dem_i_nr_pixels = np.ones(dem_pad.shape)
        dem_i_nr_pixels[idx_nan_dem_pad] = 0
        dem_i_nr_pixels = integral_image(dem_i_nr_pixels, np.int64)
    else:
        dem_i_nr_pixels = None
    del idx_nan_dem_pad

    # Summed area tables of dem and of dem squared (squares in float64), as integral_image computes them, but
    # accumulated in place. This outputs float64, which is by design. Change final array to float32 at the end of the
    # function (at return)
    dem_i1 = dem_pad.astype(np.float64)
    np.cumsum(dem_i1, axis=0, out=dem_i1)
    np.cumsum(dem_i1, axis=1, out=dem_i1)
    dem_i2 = dem_pad.astype(np.float64)
    np.square(dem_i2, out=dem_i2)
    np.cumsum(dem_i2, axis=0, out=dem_i2)
    np.cumsum(dem_i2, axis=1, out=dem_i2)

    # topographic_dev evaluated only on the output window: with output row i = maximum_radius + k the rolled tables
    # of topographic_dev are slices starting at maximum_radius -/+ ..., and np.roll(dem_pad, (-1, -1)) is the
    # window starting at maximum_radius + 1 (no wrap-around reaches the output window)
    n_rows, n_cols = idx_nan_dem.shape
    dem_centre = dem_pad[maximum_radius + 1:maximum_radius + 1 + n_rows, maximum_radius + 1:maximum_radius + 1 + n_cols]
    dev_in_place = np.result_type(dem_centre.dtype, np.float64) == np.float64

    if (0 < minimum_radius <= maximum_radius and step > 0 and dem_centre.dtype in (np.float32, np.float64)
            and dem_i1.dtype == np.float64 and dem_i2.dtype == np.float64
            and (dem_i_nr_pixels is None or dem_i_nr_pixels.dtype == np.int64)):
        # accelerated path: all radii > 0, the float64 running max rounded to float32 as by astype at the return
        radii = np.array(range(minimum_radius, maximum_radius + 1, step), dtype=np.int64)
        has_nr_pixels = dem_i_nr_pixels is not None
        nr_pixels = dem_i_nr_pixels if has_nr_pixels else np.zeros((1, 1), dtype=np.int64)
        dev_max_out = np.empty((n_rows, n_cols), dtype=np.float32)
        if _accel.call(_mstp_dev_max_kernel, dem_i1, dem_i2, nr_pixels, has_nr_pixels, dem_centre, radii,
                       maximum_radius, dev_max_out, np.empty(n_cols, dtype=np.float64)):
            dev_max_out[idx_nan_dem] = np.nan
            return dev_max_out
        del dev_max_out  # fallback: numpy path below

    # Numpy path: topographic_dev (the reference) evaluated on the output window with _box_sum, the same
    # float64 expressions in the same order; also the fallback of the kernel and the path for radii <= 0.
    for kernel_radius in range(minimum_radius, maximum_radius + 1, step):
        if kernel_radius <= 0:
            # topographic_dev returns the padded dem itself
            dev = dem_pad[maximum_radius:-(maximum_radius + 1), maximum_radius:-(maximum_radius + 1)]
        else:
            lo = maximum_radius - kernel_radius
            hi = maximum_radius + kernel_radius + 1
            if dem_i_nr_pixels is None:
                kernel_nr_pix = float((2 * kernel_radius + 1) ** 2)
            else:
                kernel_nr_pix = _box_sum(dem_i_nr_pixels, lo, hi, n_rows, n_cols)
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warning for dividing by zero
                dem_mean = _box_sum(dem_i1, lo, hi, n_rows, n_cols)
                dem_mean /= kernel_nr_pix
                dem_std = _box_sum(dem_i2, lo, hi, n_rows, n_cols)
                dem_std /= kernel_nr_pix
                dem_std -= dem_mean ** 2
                np.abs(dem_std, out=dem_std)
                np.sqrt(dem_std, out=dem_std)
            dem_std += 1e-6  # add 1e-6 to prevent division with 0
            if dev_in_place:
                dev = np.subtract(dem_centre, dem_mean, out=dem_mean)
            else:
                dev = dem_centre - dem_mean
            dev /= dem_std
            del dem_mean, dem_std
        if kernel_radius == minimum_radius:
            dev_max_out = dev
            dev_max_owned = kernel_radius > 0
        elif dev_max_owned and dev_max_out.dtype == np.result_type(dev_max_out, dev):
            replace = np.abs(dev_max_out) >= np.abs(dev)
            np.logical_not(replace, out=replace)
            np.copyto(dev_max_out, dev, where=replace)
        else:
            dev_max_out = np.where(np.abs(dev_max_out) >= np.abs(dev), dev_max_out, dev)
            dev_max_owned = True
        del dev

    # change where dem nan back to nan
    dev_max_out[idx_nan_dem] = np.nan

    return dev_max_out.astype(np.float32)


def mstp(dem,
         local_scale=(1, 5, 1),
         meso_scale=(5, 50, 5),
         broad_scale=(50, 500, 50),
         lightness=1.2,
         ve_factor=1,
         no_data=None,
         kernel_shape="square"
         ):
    """
    Compute multi-scale topographic position (MSTP) as an RGB image.

    After Lindsay et al. 2015 (docs/listofvis_mstp.rst). ``max_elevation_deviation`` is computed for the
    broad, meso and local radius ranges, and each band is ``1 - exp(-lightness * |DEV_max|)``, clipped
    to [0, 1].

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m), shape (H, W). Not checked for 2D.
    local_scale : tuple of int
        (minimum radius, maximum radius, step) in px of the local scale (blue). Default (1, 5, 1).
    meso_scale : tuple of int
        (minimum radius, maximum radius, step) in px of the meso scale (green). Default (5, 50, 5).
    broad_scale : tuple of int
        (minimum radius, maximum radius, step) in px of the broad scale (red). Default (50, 500, 50), i.e. up
        to 500 px = 125 m at 0.25 m/px; the DEM is padded by the largest radius.
    lightness : float
        Factor in the exponent; larger values give a lighter image. Default 1.2.
    ve_factor : int or float
        Vertical exaggeration, in [-1000, 10000] (see Raises). Default 1.
    no_data : int or float or None
        Pixels equal to it are set to NaN **in the caller's array**, before the float32 conversion.
        Default None.
    kernel_shape : str
        Window shape of the three ``max_elevation_deviation`` calls: "square" (default), "disk" or
        "disk_equal_area". A scale whose largest radius is >= 8 px (``DISK_MIN_RADIUS_PX``) uses the disk from 8 px
        and the square below it; a scale whose radii are all < 8 px (the default local scale 1..5 px) stays on the
        square code. With the defaults the meso band changes from 10 px, the broad band at every radius.

    Returns
    -------
    numpy.ndarray
        float32, shape (3, H, W): bands red (broad), green (meso), blue (local), values in [0, 1]; NaN where
        the input is NaN.

    Raises
    ------
    Exception
        If a scale has minimum > maximum, a step larger than maximum - minimum, ``ve_factor`` is outside
        [-1000, 10000], or ``kernel_shape`` is not "square", "disk" or "disk_equal_area".

    Notes
    -----
    The deviation (z - mean) / std is invariant to scaling z except for the 1e-6 added to std, and the sign
    is removed by ``abs``, so ``ve_factor`` changes the result only through that term (by about 2e-7 for
    ``ve_factor`` 5 on a 0-10 m test DEM). The lower ``ve_factor`` limit is -1000, unlike the -10000 of the
    other functions; known issue, see docs/KNOWN_ISSUES.md.

    See Also
    --------
    max_elevation_deviation, topographic_dev
    """
    # Structure:
    #   1. checks of the three scales and ve_factor; no_data -> NaN (in place), float32, ve_factor;
    #   2. DEV_max for the local, meso and broad radius ranges (max_elevation_deviation);
    #   3. per scale 1 - exp(-lightness * |DEV_max|), clipped to [0, 1]; stacked as (broad, meso, local).
    if local_scale[0] > local_scale[1] or meso_scale[0] > meso_scale[1] or broad_scale[0] > broad_scale[1]:
        raise Exception("rvt.vis.mstp: local_scale, meso_scale, broad_scale min has to be smaller than max!")
    if (local_scale[1] - local_scale[0] < local_scale[2]) or (meso_scale[1] - meso_scale[0] < meso_scale[2]) or \
            (broad_scale[1] - broad_scale[0] < broad_scale[2]):
        raise Exception("rvt.vis.mstp: local_scale, meso_scale, broad_scale step has"
                        " to be within min and max!")
    if not (10000 >= ve_factor >= -1000):
        raise Exception("rvt.vis.mstp: ve_factor must be between -1000 and 10000!")
    _check_kernel_shape(kernel_shape, "rvt.vis.mstp", Exception)

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    dem = dem * ve_factor

    local_dev = max_elevation_deviation(dem=dem, minimum_radius=local_scale[0], maximum_radius=local_scale[1],
                                        step=local_scale[2], kernel_shape=kernel_shape)
    meso_dev = max_elevation_deviation(dem=dem, minimum_radius=meso_scale[0], maximum_radius=meso_scale[1],
                                       step=meso_scale[2], kernel_shape=kernel_shape)
    broad_dev = max_elevation_deviation(dem=dem, minimum_radius=broad_scale[0], maximum_radius=broad_scale[1],
                                        step=broad_scale[2], kernel_shape=kernel_shape)

    cutoff = lightness
    # RGB order - broad, meso, local
    red = 1 - np.exp(-cutoff * np.abs(broad_dev))
    green = 1 - np.exp(-cutoff * np.abs(meso_dev))
    blue = 1 - np.exp(-cutoff * np.abs(local_dev))

    red[red < 0] = 0
    green[green < 0] = 0
    blue[blue < 0] = 0

    red[red > 1] = 1
    green[green > 1] = 1
    blue[blue > 1] = 1

    return np.asarray([red, green, blue])  # RGB float32 (3 x 32bit)


def fill_where_nan(dem, method="idw"):
    """
    Fill NaN pixels by interpolation from the valid pixels.

    Parameters
    ----------
    dem : numpy.ndarray
        2D array (m), shape (H, W), floating dtype. Not modified.
    method : str
        One of:

        - ``"idw"`` (default) or ``"idw_<r>_<p>"``: inverse distance weighting over the valid pixels of
          the (2r + 1) x (2r + 1) window around each NaN pixel (clipped at the raster edge), weights
          ``1 / distance ** p``; ``"idw"`` alone (or with other than two suffixes) uses r = 20 px, p = 2.
          Example: ``"idw_5_2"``. Windows with no valid pixel stay NaN; values filled earlier are not used;
        - ``"linear_row"``: the array is flattened row by row and NaN is interpolated linearly along that
          1D sequence (``np.interp``, constant beyond the ends). Fast but crosses row ends;
        - ``"kd_tree"``: value of the nearest valid pixel (``scipy.spatial.cKDTree``, Euclidean distance in
          px);
        - ``"nearest_neighbour"`` or ``"nearest_neighbor"``: the same with
          ``scipy.interpolate.griddata(method="nearest")``.

    Returns
    -------
    numpy.ndarray
        Same shape and dtype as ``dem``. Without NaN the input array itself is returned (same object);
        otherwise a filled copy.

    Raises
    ------
    Exception
        If ``method`` is not one of the names above (checked only when ``dem`` contains NaN).
    IndexError
        ``"idw"`` on an array wider than it is tall, in some cases (see Notes).

    Notes
    -----
    Known issue (see docs/KNOWN_ISSUES.md): for ``"idw"`` the window end column is clipped to
    ``dem.shape[0]`` (the number of rows) instead of ``dem.shape[1]``. On an array with more columns than
    rows, windows that reach past the right edge are cut at column H, so valid pixels beyond it are
    ignored; a NaN pixel at column >= H gets a window that does not contain its own column, which raises
    IndexError, or an empty window, which leaves it NaN. On an array with more rows than columns the clip has no effect (NumPy slicing stops
    at the edge).

    See Also
    --------
    scipy.spatial.cKDTree, scipy.interpolate.griddata
    """
    if np.all(~np.isnan(dem)):  # if there is no nan return dem
        return dem

    dem_out = np.copy(dem)
    mask = np.isnan(dem_out)

    if method == "linear_row":
        # 1D row linear interpolation
        dem_out[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), dem_out[~mask])

    elif method.split("_")[0] == "idw":
        radius = 20
        power = 2
        if len(method.split("_")) == 3:
            radius = int(method.split("_")[1])
            power = float(method.split("_")[2])
        nan_idx = zip(*np.where(mask))  # find nan positions
        for i_row, i_column in nan_idx:  # iterate through nans
            # nan surrounding array based on radius
            i_row_start = i_row - radius  # start row idx of nan surrounding
            i_column_start = i_column - radius  # start col idx of nan surrounding
            i_row_end = i_row + radius + 1  # end row idx of nan surrounding
            i_column_end = i_column + radius + 1  # end col idx of nan surrounding
            # row idx center pixel (nan pixel to idw) of nan surrounding array
            i_row_center = i_row - i_row_start
            # col idx center pixel (nan pixel to idw) of nan surrounding array
            i_column_center = i_column - i_column_start
            if i_row_start < 0:  # edge
                i_row_center = i_row
                i_row_start = 0
            if i_column_start < 0:  # edge
                i_column_center = i_column
                i_column_start = 0
            if i_row_end > dem.shape[0]:  # edge
                i_row_end = dem.shape[0]
            if i_column_end > dem.shape[1]:  # edge
                i_column_end = dem.shape[0]
            nan_surrounding_arr = dem[i_row_start:i_row_end, i_column_start:i_column_end]
            if np.all(np.isnan(nan_surrounding_arr)):  # whole surrounding array is nan
                dem_out[i_row, i_column] = np.nan
            else:
                # calculate distance array (weight matrix)
                dist_arr = np.ones(nan_surrounding_arr.shape)  # all ones
                # center pixel is 0 to calc distance matrix around 0 pixel with distance_transform_edt
                dist_arr[i_row_center, i_column_center] = 0
                dist_arr = distance_transform_edt(dist_arr)
                dist_arr[dist_arr == 0] = np.nan  # can't divide with zero
                dist_arr = 1 / dist_arr ** power
                nan_mask = np.isnan(nan_surrounding_arr)
                dist_arr[nan_mask] = 0  # where nan weight matrix is zero
                # calculate idw for one nan value
                dem_out[i_row, i_column] = np.nansum(nan_surrounding_arr * dist_arr) / np.nansum(dist_arr)

    elif method == "kd_tree" or method == "nearest_neighbour" or method == "nearest_neighbor":
        x, y = np.mgrid[0:dem_out.shape[0], 0:dem_out.shape[1]]
        xy_good = np.array((x[~mask], y[~mask])).T
        xy_bad = np.array((x[mask], y[mask])).T

        # cKD-Tree (K-D Tree) interpolation
        # https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
        if method == "kd_tree":
            leaf_size = 1000
            dem_out[mask] = dem_out[~mask][cKDTree(data=xy_good, leafsize=leaf_size).query(xy_bad)[1]]

        # Nearest neighbour interpolation
        elif method == "nearest_neighbour" or method == "nearest_neighbor":
            dem_out[mask] = griddata(xy_good, dem_out[~mask], xy_bad, method='nearest')

    else:
        raise Exception("rvt.vis.fill_where_nan: Wrong method!")

    return dem_out
