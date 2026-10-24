"""
Relief Visualization Toolbox – blend modes, normalization, opacity and color ramps (array level).

This module holds the per-array operations of the blender. `rvt.blend` builds the layer model on top of it
(BlenderLayer, BlenderCombination, e3mstp, ...); `rvt.default.DefaultValues.float_to_8bit` uses the normalization
functions (normalize_image, normalize_lin, normalize_perc) to write 8-bit rasters. Nothing here computes a
visualization from a DEM; that is `rvt.vis`.

Function groups
---------------
Color ramps
    gray_scale_to_color_ramp, truncate_colormap (matplotlib colormaps).
Normalization (stretch to [0, 1])
    normalize_lin (linear between two values), normalize_perc and lin_cutoff_calc_from_perc (between two
    percentiles), advanced_normalization (dispatch by "value" / "perc"), normalize_image (the blender's entry point,
    adds the rescale to [0, 1] and the inversion of slope and negative openness), cut_off_normalize.
Rescaling of arrays outside [0, 1]
    scale_0_to_1 (dispatch), scale_within_0_and_1, scale_strict_0_to_1.
Blend modes
    blend_normal, blend_screen, blend_multiply, blend_overlay, blend_soft_light, blend_luminosity (with lum,
    channel_min, channel_max, clip_color, matrix_eq_min_lt_zero, matrix_eq_max_gt_one); the dispatchers
    equation_blend, blend_multi_dim_images and blend_images.
Opacity
    apply_opacity, render_images.

Conventions
-----------
- Images are 2D ``(H, W)`` gray arrays or 3D ``(3, H, W)`` RGB arrays with the band axis first.
- Blend and opacity functions expect values in [0, 1]; render_images rescales an operand outside [0, 1] first.
- NaN marks no-data pixels; how each function treats NaN is stated in its docstring.
- Opacity is a percentage 0-100; a value <= 1 is taken as a fraction (see apply_opacity).
- In-place effects on arguments: normalize_lin (and everything that calls it) clips its argument to
  [minimum, maximum] in place; blend_overlay and blend_soft_light write their result into ``background``;
  scale_within_0_and_1 and scale_strict_0_to_1 overwrite NaN in their argument.

Blocked evaluation (optimized paths)
------------------------------------
normalize_lin, blend_overlay, blend_soft_light, blend_luminosity (with clip_color), blend_multi_dim_images and
render_images evaluate their elementwise expressions on row blocks of about ``_BLOCK_ELEMS`` = 32768 elements
instead of on whole images (commit 8e81581, "K-11"). Every element goes through the same IEEE operations, in the same
order, with the same operand dtypes as in the original whole-array code, so the results are bitwise identical; only
the size of the temporaries changes. The blocked path is taken only for plain float32/float64 ndarrays of one shape
without memory overlap (see ``_float_arrays``); every other input runs the original code, which is kept next to the
blocked code (``_blend_overlay_ref``, ``_blend_soft_light_ref``, the second half of normalize_lin, the non-blocked
branch of ``_assign_elementwise``). Floating point warnings are reproduced with ``_fp_strict``. The results do not
depend on the thread count (numpy only, no threads).

Known issues
------------
Several functions behave in ways that look unintended (bounds of scale_within_0_and_1, chained comparison in
scale_strict_0_to_1, case-sensitive mode checks, in-place writes of overlay / soft light). They are described in
the Notes of each function and listed in docs/KNOWN_ISSUES.md; the behaviour is kept as it is.

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

# TODO: more testing, find and fix bugs if they exists

import warnings

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

from rvt._blocking import BLOCK_ELEMS as _BLOCK_ELEMS

# Blocked elementwise evaluation (pure data-flow change). The blend kernels below evaluate exactly the same
# expressions on the same element values, only on row blocks of about _BLOCK_ELEMS elements instead of whole
# images, so temporaries stay cache-resident and full-size temporaries / fancy indexing are avoided. Every element
# goes through the same IEEE operations in the same order and with the same dtypes, so results are bitwise equal.
# _BLOCK_ELEMS = 32768 is imported from rvt/_blocking.py, which explains the value (cache size of the block
# temporaries) and is shared with the block loops in vis.py. The results do not depend on it, because every
# operation is elementwise.
_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_REAL_SCALARS = (int, float, np.integer, np.floating)


def _row_slices(n_rows, row_elems):
    """
    Return slices that split axis 0 into blocks of about ``_BLOCK_ELEMS`` elements.

    Parameters
    ----------
    n_rows : int
        Length of axis 0.
    row_elems : int
        Number of elements per index of axis 0 (row length; 1 for a flat view).

    Returns
    -------
    list of slice
        ``slice(r, r + step)`` for ``r = 0, step, 2 * step, ...`` with ``step = max(1, _BLOCK_ELEMS // row_elems)``
        (a row longer than ``_BLOCK_ELEMS`` makes one-row blocks). The last slice may extend past ``n_rows``,
        which numpy slicing clips.
    """
    step = max(1, _BLOCK_ELEMS // max(1, row_elems))
    return [slice(r, r + step) for r in range(0, n_rows, step)]


def _float_arrays(*arrays):
    """
    Return True if all arguments are plain float32 / float64 ndarrays of one non-empty shape with ndim >= 1.

    This is the entry condition of the blocked paths. ``type(arr) is np.ndarray`` excludes subclasses (masked
    arrays, matrices), whose operators can behave differently per block; integer and other dtypes are excluded
    because the reference code casts them in ways the blocked code does not repeat; equal shapes exclude
    broadcasting. Any other input runs the reference code.

    Parameters
    ----------
    *arrays : object
        Candidate operands (any objects).

    Returns
    -------
    bool
        True if the blocked path may be used for these operands.
    """
    shape = arrays[0].shape if isinstance(arrays[0], np.ndarray) else None
    if shape is None or len(shape) == 0 or arrays[0].size == 0:
        return False
    for arr in arrays:
        if type(arr) is not np.ndarray or arr.dtype not in _FLOAT_DTYPES or arr.shape != shape:
            return False
    return True


def _fp_strict():
    """
    Return the caller's numpy error state with every category that is not "ignore" turned into "raise".

    Why it exists: the blocked overlay / soft light evaluate *both* branch expressions for every element of a
    block, while the reference evaluates each branch only on the elements of its mask. Elements of the other
    branch can flag floating point errors that the reference never computes (e.g. ``sqrt`` of a negative
    background where ``active < 0.5``), which would add warnings. Under the returned state such a flag raises
    FloatingPointError; the caller then recomputes the block with the reference code under the caller's own state,
    so the same kinds of warnings are issued as by the reference. A block without any flag cannot produce a
    warning in the reference either (it computes a subset of the same operations). Categories set to "ignore"
    stay ignored, since they give no warning in either code.

    Returns
    -------
    dict or None
        Keyword arguments for ``numpy.errstate``, or None if any category of the caller's state is "raise",
        "call" or "log". Then the blocked code is not used and the reference runs on the whole array, so an
        exception leaves the inputs in the same partially updated state as the reference does.
    """
    err = np.geterr()
    if any(val not in ("ignore", "warn", "print") for val in err.values()):
        return None
    return {key: ("ignore" if val == "ignore" else "raise") for key, val in err.items()}


def gray_scale_to_color_ramp(gray_scale, colormap, min_colormap_cut=None, max_colormap_cut=None, alpha=False,
                             output_8bit=True):
    """
    Map a normalized gray-scale image to RGB or RGBA with a matplotlib colormap.

    The colormap is optionally truncated to the part between ``min_colormap_cut`` and ``max_colormap_cut``, then
    applied to every pixel. Values below 0 or above 1 get the colors of the ends of the colormap; NaN gets the
    colormap's "bad" color, which is (0, 0, 0, 0) for the matplotlib colormaps.

    Parameters
    ----------
    gray_scale : numpy.ndarray
        2D ``(H, W)`` image with values in [0, 1]. Not modified.
    colormap : str or matplotlib.colors.Colormap
        Name of a matplotlib colormap (https://matplotlib.org/stable/gallery/color/colormap_reference.html), or a
        colormap object.
    min_colormap_cut : float or None
        Lower end of the part of the colormap that is used, in [0, 1]; 0.2 drops the lowest 20 % of the colors.
        None means 0 when ``max_colormap_cut`` is given, otherwise no truncation.
    max_colormap_cut : float or None
        Upper end of the part of the colormap that is used, in [0, 1]; 0.8 drops the highest 20 % of the colors.
        None means 1 when ``min_colormap_cut`` is given, otherwise no truncation.
    alpha : bool
        True returns 4 bands (RGBA), False returns 3 bands (RGB).
    output_8bit : bool
        True returns uint8 values 0-255 (NaN pixels set to 0), False returns float64 values in [0, 1].

    Returns
    -------
    numpy.ndarray
        ``(3, H, W)`` (``alpha=False``) or ``(4, H, W)`` (``alpha=True``), band axis first; uint8 if
        ``output_8bit`` else float64. The array is a transposed view of the matplotlib result.

    Raises
    ------
    Exception
        If a cut lies outside [0, 1], or if ``min_colormap_cut >= max_colormap_cut``.
    TypeError
        If a cut is a string (comparison of str with int), see Notes.

    Notes
    -----
    The 8-bit conversion is ``np.uint8(rgba * 255)``, a truncation, not a rounding.

    Known issue: BlenderCombination.read_from_json stores the cuts read from JSON as strings, which makes this
    function raise TypeError; see docs/KNOWN_ISSUES.md.

    See Also
    --------
    truncate_colormap
    """
    cm = colormaps.get_cmap(colormap)

    # Truncate colormap if required
    if min_colormap_cut is not None or max_colormap_cut is not None:
        if min_colormap_cut is None:
            min_colormap_cut = 0.0
        if max_colormap_cut is None:
            max_colormap_cut = 1.0
        if min_colormap_cut > 1 or min_colormap_cut < 0 or max_colormap_cut > 1 or max_colormap_cut < 0:
            raise Exception("rvt.blend_func.gray_scale_to_color_ramp: min_colormap_cut and max_colormap_cut must be"
                            " between 0 and 1!")
        if min_colormap_cut >= max_colormap_cut:
            raise Exception("rvt.blend_func.gray_scale_to_color_ramp: min_colormap_cut has to be smaller than"
                            " max_colormap_cut!")
        cm = truncate_colormap(cmap=cm, minval=min_colormap_cut, maxval=max_colormap_cut)

    # Compute normalized RGBA
    rgba_mtpl_out = cm(gray_scale)

    if output_8bit:
        nan_mask = np.isnan(gray_scale)
        rgba_mtpl_out[nan_mask] = 0  # Change nan to 0
        rgba_mtpl_out = np.uint8(rgba_mtpl_out * 255)  # 0-1 scale to 0-255 and change type to uint8

    # Move array axes to correct positions, i.e. (x, y, bands) to (bands, x, y)
    rgba_out = rgba_mtpl_out.transpose(2, 0, 1)

    # Discard 4th band if not using Alpha
    if not alpha:
        rgba_out = rgba_out[:3, ...]

    return rgba_out


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Return a colormap made of the part of ``cmap`` between ``minval`` and ``maxval``.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Source colormap (an object, not a name).
    minval : float
        Start of the used part, in [0, 1].
    maxval : float
        End of the used part, in [0, 1].
    n : int
        Number of colors sampled evenly from ``minval`` to ``maxval`` (default 100).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        New colormap named ``"trunc(<name>,<minval>,<maxval>)"`` that interpolates linearly between the ``n``
        sampled colors.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def normalize_lin(image, minimum, maximum):
    """
    Clip an image to [minimum, maximum] and stretch it linearly to [0, 1].

    Parameters
    ----------
    image : numpy.ndarray
        Image of any shape, usually ``(H, W)`` or ``(3, H, W)``. **Clipped to [minimum, maximum] in place**; NaN
        stays NaN.
    minimum : float
        Value mapped to 0 (same unit as ``image``).
    maximum : float
        Value mapped to 1.

    Returns
    -------
    numpy.ndarray
        float32 array of the shape of ``image``, ``(image - minimum) / (maximum - minimum)`` clamped to [0, 1];
        NaN where ``image`` is NaN. Always a new array.

    Notes
    -----
    ``minimum == maximum`` gives 0 / 0 = NaN at every non-NaN pixel (RuntimeWarning); ``minimum > maximum`` gives
    -0.0 everywhere. There is no check here; advanced_normalization checks the limits for "value".

    The function has two code paths with the same result. The first (blocked) path runs for C-contiguous,
    writeable float32 / float64 ndarrays with real scalar limits: it clips the whole array in place first, exactly
    as the reference does, then computes the stretch per block of ``_BLOCK_ELEMS`` = 32768 elements and writes it
    into a preallocated float32 output, so the float64 temporaries of the stretch are block-sized instead of
    image-sized. The second path is the original code. It is kept for every other input (integer arrays,
    non-contiguous views, read-only arrays, array-valued limits), where the blocked path would need extra rules.
    Both paths evaluate the same expressions with the same operand dtypes per element, so results are bitwise
    identical (commit 8e81581).

    See Also
    --------
    normalize_perc, advanced_normalization
    """
    if (_float_arrays(image) and image.flags.c_contiguous and image.flags.writeable
            and isinstance(minimum, _REAL_SCALARS) and isinstance(maximum, _REAL_SCALARS)):
        # Same steps per row block: the cut-off is still applied to the whole `image` in place first, the stretched
        # result is cast to float32 exactly as np.float32() casts the whole array.
        # The image is C-contiguous, so blocks are taken from its flat view.
        flat = image.reshape(-1)
        slices = _row_slices(flat.size, 1)
        for sl in slices:
            blk = flat[sl]
            blk[blk > maximum] = maximum
            blk[blk < minimum] = minimum
        out = np.empty(image.shape, dtype=np.float32)
        out_flat = out.reshape(-1)
        for sl in slices:
            stretched = (flat[sl] - minimum) / (maximum - minimum)
            stretched[stretched > 1] = 1
            stretched[stretched < 0] = 0
            out_flat[sl] = stretched
        return out

    # linear cut off
    image[image > maximum] = maximum
    image[image < minimum] = minimum

    # stretch to 0.0 - 1.0 interval
    image = (image - minimum) / (maximum - minimum)
    image[image > 1] = 1
    image[image < 0] = 0
    return np.float32(image)


def lin_cutoff_calc_from_perc(image, minimum, maximum):
    """
    Return the values at which a percent cut-off clips an image.

    Parameters
    ----------
    image : numpy.ndarray
        Image of any shape; NaN is ignored. Not modified.
    minimum : float
        Percentage of the lowest values to cut off, in [0, 100].
    maximum : float
        Percentage of the highest values to cut off, in [0, 100]; ``minimum + maximum <= 100``.

    Returns
    -------
    dict
        ``{"min_lin": float, "max_lin": float}``: the ``minimum``-th and the ``(100 - maximum)``-th percentile of
        ``image`` (``numpy.nanpercentile``, linear interpolation), as numpy float64 for float input. If the two
        are equal, ``numpy.nanmin`` and ``numpy.nanmax`` of the image are returned instead.

    Raises
    ------
    Exception
        If ``minimum`` or ``maximum`` is outside [0, 100], or ``minimum + maximum > 100``.
    """
    if minimum < 0 or maximum < 0 or minimum > 100 or maximum > 100:
        raise Exception("rvt.blend_func.lin_cutoff_calc_from_perc: minimum, maximum are percent and have to be in "
                        "range 0-100!")
    if minimum + maximum > 100:
        raise Exception("rvt.blend_func.lin_cutoff_calc_from_perc: if minimum + maximum > 100% then there are no"
                        " values left! You can't cutoff whole image!")
    distribution = np.nanpercentile(a=image, q=np.array([minimum, 100 - maximum]))
    min_lin = distribution[0]
    max_lin = distribution[1]
    if min_lin == max_lin:
        min_lin = np.nanmin(image)
        max_lin = np.nanmax(image)
    return {"min_lin": min_lin, "max_lin": max_lin}


def normalize_perc(image, minimum, maximum):
    """
    Clip an image at two percentiles and stretch it linearly to [0, 1].

    Parameters
    ----------
    image : numpy.ndarray
        Image of any shape. **Clipped in place** (see normalize_lin); NaN stays NaN.
    minimum : float
        Percentage of the lowest values to cut off, in [0, 100].
    maximum : float
        Percentage of the highest values to cut off, in [0, 100]; ``minimum + maximum <= 100``.

    Returns
    -------
    numpy.ndarray
        float32 array of the shape of ``image`` with values in [0, 1].

    Raises
    ------
    Exception
        From lin_cutoff_calc_from_perc, if the percentages are out of range.

    See Also
    --------
    lin_cutoff_calc_from_perc, normalize_lin
    """
    min_max_lin_dict = lin_cutoff_calc_from_perc(image, minimum, maximum)
    min_lin = min_max_lin_dict["min_lin"]
    max_lin = min_max_lin_dict["max_lin"]
    return normalize_lin(image, min_lin, max_lin)


def advanced_normalization(image, minimum, maximum, normalization):
    """
    Normalize an image to [0, 1] by values ("value") or by percentiles ("perc").

    Parameters
    ----------
    image : numpy.ndarray
        Image of any shape. **Clipped in place** by normalize_lin.
    minimum : float
        "value": value mapped to 0. "perc": percentage of the lowest values cut off, in [0, 100].
    maximum : float
        "value": value mapped to 1. "perc": percentage of the highest values cut off, in [0, 100].
    normalization : str
        "value" or "perc", case-insensitive for the dispatch.

    Returns
    -------
    numpy.ndarray
        float32 array of the shape of ``image`` with values in [0, 1] (NaN kept).

    Raises
    ------
    Exception
        If ``normalization == "value"`` and ``minimum == maximum`` or ``minimum > maximum``; for an unknown
        normalization name; from lin_cutoff_calc_from_perc for invalid percentages.
    AttributeError
        If ``normalization`` is None (see Notes).

    Notes
    -----
    Known issue: the two limit checks compare ``normalization`` with "value" case-sensitively, so "Value" skips
    them (``minimum == maximum`` then gives all NaN, ``minimum > maximum`` all -0.0). The branch for
    ``normalization is None``, which would return the image unchanged, comes after ``normalization.lower()`` and
    is never reached: None raises AttributeError. See docs/KNOWN_ISSUES.md.
    """

    # Perform checks if correct values were given
    if minimum == maximum and normalization == "value":
        raise Exception("rvt.blend_func.advanced_normalization: If normalization == value, min and max cannot be the"
                        " same!")

    if minimum > maximum and normalization == "value":
        raise Exception("rvt.blend_func.advanced_normalization: If normalization == value, max can't be smaller"
                        " than min!")

    # Select normalization type
    if normalization.lower() == "value":
        equ_image = normalize_lin(image=image, minimum=minimum, maximum=maximum)
    elif normalization.lower() == "perc":
        equ_image = normalize_perc(image=image, minimum=minimum, maximum=maximum)
    elif normalization is None:
        equ_image = image
    else:
        raise Exception(f"rvt.blend_func.advanced_normalization: Unknown normalization type: {normalization}")

    return equ_image


def lum(img):
    """
    Return the luminosity of an RGB image, or a gray image unchanged.

    Parameters
    ----------
    img : numpy.ndarray
        ``(3, H, W)`` RGB image (band axis first) or ``(H, W)`` gray image.

    Returns
    -------
    numpy.ndarray
        For 3D input a new float32 ``(H, W)`` array ``0.3 R + 0.59 G + 0.11 B``; for any other input ``img``
        itself (same object).

    Notes
    -----
    The weights are those of the "Luminosity" blend mode of the PDF / W3C compositing specification (Lum()).
    Only bands 0-2 are read, so a 3D array with more than 3 bands is accepted.
    """
    if len(img.shape) == 3:
        r = img[0]
        g = img[1]
        b = img[2]
        lum_img = np.float32((0.3 * r) + (0.59 * g) + (0.11 * b))
    else:
        lum_img = img

    return lum_img


def matrix_eq_min_lt_zero(r: np.ndarray, idx_min_lt_zero, lum_c, min_c):
    """
    Apply the lower branch of ClipColor to one channel, in place.

    At the selected pixels ``r = lum + (r - lum) * lum / (lum - min)``, which moves a color with a channel below 0
    towards its luminosity until the lowest channel is 0 (PDF / W3C compositing specification, ClipColor()).

    Parameters
    ----------
    r : numpy.ndarray
        One channel ``(H, W)``. **Modified in place** at the selected pixels.
    idx_min_lt_zero : tuple of numpy.ndarray or numpy.ndarray
        Index of the pixels to correct (``numpy.where(min_c < 0)``).
    lum_c : numpy.ndarray
        Luminosity ``(H, W)`` of the color (lum()).
    min_c : numpy.ndarray
        Lowest channel ``(H, W)`` of the color (channel_min()).

    Returns
    -------
    numpy.ndarray
        ``r`` itself.

    Notes
    -----
    For a gray pixel (``lum == min``) the expression is 0 / 0 and the pixel becomes NaN (RuntimeWarning); see
    clip_color and docs/KNOWN_ISSUES.md.
    """
    r[idx_min_lt_zero] = lum_c[idx_min_lt_zero] + (((r[idx_min_lt_zero] - lum_c[idx_min_lt_zero]) *
                                                    lum_c[idx_min_lt_zero]) / (lum_c[idx_min_lt_zero] -
                                                                               min_c[idx_min_lt_zero]))
    return r


def matrix_eq_max_gt_one(r: np.ndarray, idx_max_c_gt_one, lum_c, max_c):
    """
    Apply the upper branch of ClipColor to one channel, in place.

    At the selected pixels ``r = lum + (r - lum) * (1 - lum) / (max - lum)``, which moves a color with a channel
    above 1 towards its luminosity until the highest channel is 1 (PDF / W3C compositing specification).

    Parameters
    ----------
    r : numpy.ndarray
        One channel ``(H, W)``. **Modified in place** at the selected pixels.
    idx_max_c_gt_one : tuple of numpy.ndarray or numpy.ndarray
        Index of the pixels to correct (``numpy.where(max_c > 1)``).
    lum_c : numpy.ndarray
        Luminosity ``(H, W)`` of the color (lum()).
    max_c : numpy.ndarray
        Highest channel ``(H, W)`` of the color (channel_max()).

    Returns
    -------
    numpy.ndarray
        ``r`` itself.
    """
    r[idx_max_c_gt_one] = lum_c[idx_max_c_gt_one] + (((r[idx_max_c_gt_one] - lum_c[idx_max_c_gt_one]) *
                                                      (1.0 - lum_c[idx_max_c_gt_one])) / (max_c[idx_max_c_gt_one]
                                                                                          - lum_c[idx_max_c_gt_one]))
    return r


def channel_min(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    """
    Return the per-pixel minimum of three channels.

    Parameters
    ----------
    r : numpy.ndarray
        Red channel ``(H, W)``. Not modified.
    g : numpy.ndarray
        Green channel, same shape.
    b : numpy.ndarray
        Blue channel, same shape.

    Returns
    -------
    numpy.ndarray
        New array (``r * 1.0``, so the dtype of ``r`` for float input) holding the minimum.

    Notes
    -----
    Known issue: the minimum starts as ``r`` and is replaced where ``g < min`` and then where ``b < min``. A
    comparison with NaN is False, so NaN in ``r`` is kept while NaN in ``g`` or ``b`` is ignored (``np.minimum``
    would give NaN in both cases). See docs/KNOWN_ISSUES.md.
    """
    min_c = r * 1.0
    idx_min = np.where(g < min_c)
    min_c[idx_min] = g[idx_min]
    idx_min = np.where(b < min_c)
    min_c[idx_min] = b[idx_min]
    return min_c


def channel_max(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    """
    Return the per-pixel maximum of three channels.

    Parameters
    ----------
    r : numpy.ndarray
        Red channel ``(H, W)``. Not modified.
    g : numpy.ndarray
        Green channel, same shape.
    b : numpy.ndarray
        Blue channel, same shape.

    Returns
    -------
    numpy.ndarray
        New array (``r * 1.0``) holding the maximum.

    Notes
    -----
    Known issue: as in channel_min, NaN in ``r`` is kept and NaN in ``g`` or ``b`` is ignored; see
    docs/KNOWN_ISSUES.md.
    """
    max_c = r * 1.0
    idx_max = np.where(g > max_c)
    max_c[idx_max] = g[idx_max]
    idx_max = np.where(b > max_c)
    max_c[idx_max] = b[idx_max]
    return max_c


def clip_color(c, min_c=None, max_c=None):
    """
    Bring an RGB image back into [0, 1] while keeping the luminosity of each pixel (ClipColor).

    Pixels whose lowest channel is below 0 are moved towards their luminosity with matrix_eq_min_lt_zero, then
    pixels whose highest channel is above 1 with matrix_eq_max_gt_one (PDF / W3C compositing specification).
    Both corrections use the luminosity, minimum and maximum computed before any correction.

    Parameters
    ----------
    c : numpy.ndarray
        ``(3, H, W)`` RGB image (band axis first). Not modified.
    min_c : numpy.ndarray or None
        Per-pixel minimum channel ``(H, W)``. Computed from ``c`` (float32 channels) when both ``min_c`` and
        ``max_c`` are None.
    max_c : numpy.ndarray or None
        Per-pixel maximum channel ``(H, W)``; see ``min_c``.

    Returns
    -------
    numpy.ndarray
        New float64 ``(3, H, W)`` array. The channels are computed in float32 (``np.float32(c[i])``) and widened.

    Raises
    ------
    TypeError
        If only one of ``min_c`` and ``max_c`` is given (the other stays None and is compared with a number).

    Notes
    -----
    Known issues (docs/KNOWN_ISSUES.md): a gray pixel (R = G = B) below 0 has ``lum == min``; its correction is
    0 / 0 and the pixel becomes NaN in all bands instead of 0. NaN handling follows channel_min / channel_max.
    Giving only one of ``min_c`` / ``max_c`` raises TypeError.

    The blocked equivalent used by blend_luminosity is ``_luminosity_clip_blocked``.
    """
    lum_c = lum(c)

    r = np.float32(c[0])
    g = np.float32(c[1])
    b = np.float32(c[2])

    if min_c is None and max_c is None:
        min_c = channel_min(r, g, b)
        max_c = channel_max(r, g, b)

    idx_min_lt_zero = np.where(min_c < 0)
    r = matrix_eq_min_lt_zero(r, idx_min_lt_zero, lum_c, min_c)
    g = matrix_eq_min_lt_zero(g, idx_min_lt_zero, lum_c, min_c)
    b = matrix_eq_min_lt_zero(b, idx_min_lt_zero, lum_c, min_c)

    idx_max_c_gt_one = np.where(max_c > 1)
    r = matrix_eq_max_gt_one(r, idx_max_c_gt_one, lum_c, max_c)
    g = matrix_eq_max_gt_one(g, idx_max_c_gt_one, lum_c, max_c)
    b = matrix_eq_max_gt_one(b, idx_max_c_gt_one, lum_c, max_c)

    c_out = np.zeros(c.shape)
    c_out[0, :, :] = r
    c_out[1, :, :] = g
    c_out[2, :, :] = b
    return c_out


def _luminosity_clip_blocked(active, background):
    """
    Compute ``blend_luminosity(active, background)`` for an RGB float background, per row block.

    Equivalence contract with the reference (blend_luminosity followed by clip_color): per block the same
    expressions in the same order and dtypes: ``luminosity = lum(active) - lum(background)``;
    ``c[i] = background[i] + luminosity`` computed in the operand dtype and stored in a float64 array (as the
    reference's ``np.zeros`` array); ``lum_c = lum(c)`` (float32); float32 channels; channel minimum / maximum
    with ``np.copyto(where=...)``, which selects the same elements as the ``np.where`` updates of channel_min /
    channel_max (NaN comparisons are False in both); then the ``min < 0`` and the ``max > 1`` corrections on the
    same element subsets, channel by channel (r, g, b), with ``lum_c``, ``min_c`` and ``max_c`` from before the
    corrections. Every element is computed independently, so splitting into blocks does not change any value.

    Parameters
    ----------
    active : numpy.ndarray
        float32 / float64, ``(3, H, W)`` or ``(H, W)``.
    background : numpy.ndarray
        float32 / float64, ``(3, H, W)``.

    Returns
    -------
    numpy.ndarray
        New float64 ``(3, H, W)`` array, bitwise equal to the reference result.
    """
    c_out = np.zeros(background.shape)
    rows = background.shape[1]
    for sl in _row_slices(rows, background[0].size // rows):
        luminosity = lum(active[:, sl] if active.ndim == 3 else active[sl]) - lum(background[:, sl])
        c = np.empty((3,) + luminosity.shape)
        c[0] = background[0, sl] + luminosity
        c[1] = background[1, sl] + luminosity
        c[2] = background[2, sl] + luminosity
        lum_c = lum(c)
        chans = [np.float32(c[0]), np.float32(c[1]), np.float32(c[2])]
        r, g, b = chans
        min_c = r * 1.0
        np.copyto(min_c, g, where=g < min_c)
        np.copyto(min_c, b, where=b < min_c)
        max_c = r * 1.0
        np.copyto(max_c, g, where=g > max_c)
        np.copyto(max_c, b, where=b > max_c)
        lum_flat = lum_c.reshape(-1)
        idx = np.flatnonzero(min_c < 0)
        if idx.size:
            lum_i = lum_flat[idx]
            min_i = min_c.reshape(-1)[idx]
            for ch in chans:
                ch_flat = ch.reshape(-1)
                ch_flat[idx] = lum_i + (((ch_flat[idx] - lum_i) * lum_i) / (lum_i - min_i))
        idx = np.flatnonzero(max_c > 1)
        if idx.size:
            lum_i = lum_flat[idx]
            max_i = max_c.reshape(-1)[idx]
            for ch in chans:
                ch_flat = ch.reshape(-1)
                ch_flat[idx] = lum_i + (((ch_flat[idx] - lum_i) * (1.0 - lum_i)) / (max_i - lum_i))
        c_out[0, sl] = r
        c_out[1, sl] = g
        c_out[2, sl] = b
    return c_out


def blend_normal(active, background):
    """
    Return the active layer (blend mode "normal").

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer, values in [0, 1].
    background : numpy.ndarray
        Lower layer; not used.

    Returns
    -------
    numpy.ndarray
        ``active`` itself (same object).
    """
    return active


def blend_screen(active, background):
    """
    Blend two layers with "screen": ``1 - (1 - active) * (1 - background)``.

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer, values in [0, 1]. Not modified.
    background : numpy.ndarray
        Lower layer, values in [0, 1], broadcastable with ``active``. Not modified.

    Returns
    -------
    numpy.ndarray
        New array, broadcast shape, numpy dtype promotion of the operands; NaN where either operand is NaN.
    """
    return 1 - (1 - active) * (1 - background)


def blend_multiply(active, background):
    """
    Blend two layers with "multiply": ``active * background``.

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer, values in [0, 1]. Not modified.
    background : numpy.ndarray
        Lower layer, values in [0, 1], broadcastable with ``active``. Not modified.

    Returns
    -------
    numpy.ndarray
        New array, broadcast shape, numpy dtype promotion of the operands; NaN where either operand is NaN.
    """
    return active * background


def _blend_in_place_blocked(active, background, masks, branches, reference):
    """
    Run an in-place two-branch blend (overlay, soft light) per row block; shared driver of both.

    Per block: compute both masks before any write (as the reference computes both ``np.where`` index sets before
    its first assignment), evaluate both branch expressions for all elements under ``_fp_strict()``, and copy each
    into ``background`` where its mask holds (``casting="unsafe"``, the same cast as a fancy-index assignment).
    Elements outside both masks (NaN in the mask operand) keep their value, as in the reference. A block whose
    branch evaluation flags a floating point error is recomputed by ``reference`` on the block views under the
    caller's error state.

    The blocked path is used only if ``_fp_strict()`` is not None, both operands pass ``_float_arrays``,
    ``background`` is writeable and the operands do not share memory (with overlap, writes to one block of
    ``background`` would change ``active`` values read by a later block). Otherwise ``reference`` runs on the
    whole arrays.

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer.
    background : numpy.ndarray
        Lower layer, overwritten in place.
    masks : callable
        ``masks(a, b) -> (m1, m2)``: boolean masks of the two branches.
    branches : callable
        ``branches(a, b) -> (e1, e2)``: values of the two branch expressions for all elements.
    reference : callable
        ``reference(a, b)``: the original whole-array code, writing into ``b`` and returning it.

    Returns
    -------
    numpy.ndarray
        ``background`` (the return value of ``reference`` on the fallback path, which is also ``background``).
    """
    strict = _fp_strict()
    if not (strict is not None and _float_arrays(active, background) and background.flags.writeable
            and not np.may_share_memory(active, background)):
        return reference(active, background)
    for sl in _row_slices(background.shape[0], background.size // background.shape[0]):
        a = active[sl]
        b = background[sl]
        m1, m2 = masks(a, b)
        try:
            with np.errstate(**strict):
                e1, e2 = branches(a, b)
        except FloatingPointError:
            reference(a, b)
            continue
        np.copyto(b, e1, where=m1, casting="unsafe")
        np.copyto(b, e2, where=m2, casting="unsafe")
    return background


def _overlay_masks(active, background):
    """Return the overlay branch masks ``(background > 0.5, background <= 0.5)`` (as in _blend_overlay_ref)."""
    return background > 0.5, background <= 0.5


def _overlay_branches(active, background):
    """Return both overlay branch expressions for all elements, written exactly as in _blend_overlay_ref."""
    return (1 - (1 - 2 * (background - 0.5)) * (1 - active)), ((2 * background) * active)


def _blend_overlay_ref(active, background):
    """
    Overlay blend, original whole-array code: reference and fallback of blend_overlay.

    Writes the result into ``background`` at the ``np.where`` index sets of ``background > 0.5`` and
    ``background <= 0.5`` (both taken before the first write) and returns ``background``.
    """
    idx1 = np.where(background > 0.5)
    idx2 = np.where(background <= 0.5)
    background[idx1] = (1 - (1 - 2 * (background[idx1] - 0.5)) * (1 - active[idx1]))
    background[idx2] = ((2 * background[idx2]) * active[idx2])
    return background


def blend_overlay(active, background):
    """
    Blend two layers with "overlay", writing the result into ``background``.

    The branch is chosen by the background: where ``background > 0.5`` the result is
    ``1 - (1 - 2 * (background - 0.5)) * (1 - active)`` (screen-like), where ``background <= 0.5`` it is
    ``2 * background * active`` (multiply-like).

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer, values in [0, 1], same shape as ``background``. Not modified.
    background : numpy.ndarray
        Lower layer, values in [0, 1]. **Overwritten in place with the result.**

    Returns
    -------
    numpy.ndarray
        ``background`` itself, holding the result in its own dtype. NaN in ``background`` matches neither branch
        and stays NaN; NaN in ``active`` gives NaN.

    Notes
    -----
    Known issue: because the result replaces the background, the caller's background array changes, and in
    BlenderCombination.render_all_images the opacity of an overlay layer has no visible effect (the opacity mix
    is computed between two arrays with the same values). See docs/KNOWN_ISSUES.md.

    Accelerated path: for plain float32 / float64 ndarrays of one shape without memory overlap, both branch
    expressions are evaluated per row block and copied where their mask holds (``_blend_in_place_blocked``);
    otherwise ``_blend_overlay_ref`` (the original code) runs. Results are bitwise identical.
    """
    # the in-place driver writes into a copy, so the result is a new array and the caller's background is unchanged
    return _blend_in_place_blocked(active, background.copy(), _overlay_masks, _overlay_branches, _blend_overlay_ref)


def _soft_light_masks(active, background):
    """Return the soft light branch masks ``(active < 0.5, active >= 0.5)`` (as in _blend_soft_light_ref)."""
    return active < 0.5, active >= 0.5


def _soft_light_branches(active, background):
    """Return both soft light branch expressions for all elements, written exactly as in _blend_soft_light_ref."""
    return (2 * background * active + background ** 2 * (1.0 - 2 * active),
            2 * background * (1.0 - active) + np.sqrt(background) * (2 * active - 1.0))


def blend_soft_light(active, background):
    """
    Blend two layers with "soft light", writing the result into ``background``.

    The branch is chosen by the active layer: where ``active < 0.5`` the result is
    ``2 * background * active + background ** 2 * (1 - 2 * active)``, where ``active >= 0.5`` it is
    ``2 * background * (1 - active) + sqrt(background) * (2 * active - 1)`` (the "Photoshop" soft light formula).

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer, values in [0, 1], same shape as ``background``. Not modified.
    background : numpy.ndarray
        Lower layer, values in [0, 1]. **Overwritten in place with the result.**

    Returns
    -------
    numpy.ndarray
        ``background`` itself, holding the result in its own dtype. NaN in ``background`` gives NaN; a negative
        background value in the second branch gives NaN (square root, RuntimeWarning).

    Notes
    -----
    Known issues (docs/KNOWN_ISSUES.md): NaN in ``active`` matches neither branch, so the background value is
    kept instead of NaN. As with blend_overlay, the in-place write makes the opacity of a soft light layer
    ineffective in BlenderCombination.render_all_images.

    Accelerated path: as blend_overlay, with ``_blend_soft_light_ref`` as the original code.
    """
    # the in-place driver writes into a copy, so the result is a new array and the caller's background is unchanged
    return _blend_in_place_blocked(active, background.copy(), _soft_light_masks, _soft_light_branches,
                                   _blend_soft_light_ref)


def _blend_soft_light_ref(active, background):
    """
    Soft light blend, original whole-array code: reference and fallback of blend_soft_light.

    Writes the result into ``background`` at the ``np.where`` index sets of ``active < 0.5`` and
    ``active >= 0.5`` (both taken before the first write) and returns ``background``.
    """
    idx1 = np.where(active < 0.5)
    idx2 = np.where(active >= 0.5)
    background[idx1] = 2 * background[idx1] * active[idx1] + background[idx1] ** 2 * (1.0 - 2 * active[idx1])
    background[idx2] = 2 * background[idx2] * (1.0 - active[idx2]) + np.sqrt(background[idx2]) * (
                2 * active[idx2] - 1.0)
    return background


def blend_luminosity(active, background, min_c=None, max_c=None):
    """
    Blend two layers with "luminosity": the hue and saturation of the background with the luminosity of the
    active layer.

    For an RGB background each band becomes ``background[i] + (lum(active) - lum(background))`` (SetLum of the
    PDF / W3C compositing specification); the result is brought back into [0, 1] by clip_color. For a gray
    background the result is ``lum(active)``.

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer, ``(H, W)`` or ``(3, H, W)``, values in [0, 1]. Not modified.
    background : numpy.ndarray
        Lower layer, ``(H, W)`` or ``(3, H, W)``, values in [0, 1]. Not modified.
    min_c : numpy.ndarray or None
        Passed to clip_color; None computes it there.
    max_c : numpy.ndarray or None
        Passed to clip_color; None computes it there.

    Returns
    -------
    numpy.ndarray
        RGB background: new float64 ``(3, H, W)`` array. Gray background: ``lum(active)``, i.e. ``active`` itself
        for a gray active layer, or a new float32 ``(H, W)`` array for an RGB active layer.

    Notes
    -----
    Known issue: a gray background pixel whose luminosity result goes below 0 becomes NaN (clip_color divides
    0 by 0); see docs/KNOWN_ISSUES.md.

    Accelerated path: with ``min_c`` and ``max_c`` None, a float32 / float64 ``(3, H, W)`` ndarray background and a
    float ndarray active layer of shape ``(3, H, W)`` or ``(H, W)``, ``_luminosity_clip_blocked`` evaluates the
    same expressions per row block. Results are bitwise identical.
    """
    if (min_c is None and max_c is None and _float_arrays(background) and background.ndim == 3
            and background.shape[0] == 3 and type(active) is np.ndarray and active.dtype in _FLOAT_DTYPES
            and (active.shape == background.shape or active.shape == background.shape[1:])):
        return _luminosity_clip_blocked(active, background)

    lum_active = lum(active)
    lum_background = lum(background)
    luminosity = lum_active - lum_background

    if len(background.shape) < 3:
        return lum_active

    r = background[0] + luminosity
    g = background[1] + luminosity
    b = background[2] + luminosity

    c = np.zeros(background.shape)
    c[0, :, :] = r
    c[1, :, :] = g
    c[2, :, :] = b

    clipped_image = clip_color(c, min_c, max_c)

    return clipped_image


def equation_blend(blend_mode, active, background):
    """
    Apply one of the elementwise blend modes screen, multiply, overlay or soft_light.

    Parameters
    ----------
    blend_mode : str
        "screen", "multiply", "overlay" or "soft_light", case-insensitive.
    active : numpy.ndarray
        Upper layer, values in [0, 1].
    background : numpy.ndarray
        Lower layer, values in [0, 1]. Overwritten in place by "overlay" and "soft_light".

    Returns
    -------
    numpy.ndarray or None
        Result of blend_screen, blend_multiply, blend_overlay or blend_soft_light; None for any other name.

    Notes
    -----
    Known issue: "normal" and "luminosity" are handled by blend_images only; here they (and unknown names) return
    None without an error. See docs/KNOWN_ISSUES.md.
    """
    if blend_mode.lower() == "screen":
        return blend_screen(active, background)
    elif blend_mode.lower() == "multiply":
        return blend_multiply(active, background)
    elif blend_mode.lower() == "overlay":
        return blend_overlay(active, background)
    elif blend_mode.lower() == "soft_light":
        return blend_soft_light(active, background)


def _assign_elementwise(out, func, active, background):
    """
    Assign ``out[...] = func(active, background)`` for an elementwise ``func``, per row block when possible.

    The blocked path runs when both operands are plain ndarrays of ``out``'s shape (no broadcasting), ``out`` is
    not empty, the operands do not share memory and the caller's error state is not raising (``_fp_strict()``
    is not None). Because ``func`` is elementwise, ``func`` on a block gives the same values as the block of
    ``func`` on the whole arrays, and its in-place effects on ``background`` (overlay, soft light) are the same.
    With a raising error state the whole array is processed at once, so an exception leaves ``out`` and
    ``background`` in the same state as the original code.

    Parameters
    ----------
    out : numpy.ndarray
        Target array (a band view of the result); written in place.
    func : callable
        Elementwise function ``func(active, background)``.
    active : numpy.ndarray or object
        First operand.
    background : numpy.ndarray or object
        Second operand; may be modified in place by ``func``.
    """
    if (type(active) is np.ndarray and type(background) is np.ndarray and active.ndim >= 1 and out.size > 0
            and active.shape == out.shape and background.shape == out.shape and _fp_strict() is not None
            and not np.may_share_memory(active, background)):
        for sl in _row_slices(out.shape[0], out.size // out.shape[0]):
            out[sl] = func(active[sl], background[sl])
    else:
        out[...] = func(active, background)


def blend_multi_dim_images(blend_mode, active, background):
    """
    Apply an elementwise blend mode to gray or RGB layers, band by band.

    When either layer is RGB, band ``i`` of the result is ``equation_blend(blend_mode, a_i, b_i)``, where a gray
    layer is used for every band. Two gray layers are blended directly.

    Parameters
    ----------
    blend_mode : str
        "screen", "multiply", "overlay" or "soft_light", case-insensitive.
    active : numpy.ndarray
        Upper layer, ``(H, W)`` or ``(3, H, W)``, values in [0, 1]. Not modified.
    background : numpy.ndarray
        Lower layer, ``(H, W)`` or ``(3, H, W)``, values in [0, 1]. Overwritten in place by "overlay" and
        "soft_light" (for an RGB background through its band views).

    Returns
    -------
    numpy.ndarray or None
        At least one RGB layer: new float64 ``(3, H, W)`` array. Two gray layers: the result of equation_blend
        (dtype of the operands; for overlay / soft_light ``background`` itself). For an unknown mode: None for two
        gray layers, an all-NaN float64 array otherwise (None assigned into a float array).

    Notes
    -----
    Known issue (docs/KNOWN_ISSUES.md): with "overlay" or "soft_light", an RGB active layer and a gray
    background, each band writes its result into the same gray background, so band 1 is blended onto the result
    of band 0 and band 2 onto that of band 1 (overlay of 0.25 on 0.25 gives 0.125, 0.0625, 0.03125 instead of
    0.125 in every band).

    Accelerated path: each band is assigned through ``_assign_elementwise`` (row blocks); bitwise identical.
    """
    a_rgb = len(active.shape) == 3  # bool, is active rgb
    b_rgb = len(background.shape) == 3  # bool, is background rgb
    blended_image = None

    def func(act, back):
        return equation_blend(blend_mode, act, back)

    if a_rgb and b_rgb:
        blended_image = np.zeros(background.shape)
        for i in range(3):
            _assign_elementwise(blended_image[i, :, :], func, active[i, :, :], background[i, :, :])
    if a_rgb and not b_rgb:
        blended_image = np.zeros(active.shape)
        for i in range(3):
            _assign_elementwise(blended_image[i, :, :], func, active[i, :, :], background)
    if not a_rgb and b_rgb:
        blended_image = np.zeros(background.shape)
        for i in range(3):
            _assign_elementwise(blended_image[i, :, :], func, active, background[i, :, :])
    if not a_rgb and not b_rgb:
        blended_image = equation_blend(blend_mode, active, background)

    return blended_image


def blend_images(blend_mode, active, background, min_c=None, max_c=None):
    """
    Blend an active layer onto a background with the named blend mode (no opacity).

    Parameters
    ----------
    blend_mode : str
        "normal", "multiply", "screen", "overlay", "soft_light" or "luminosity", case-insensitive. Any other name
        is treated as "normal".
    active : numpy.ndarray
        Upper layer, ``(H, W)`` or ``(3, H, W)``, values in [0, 1].
    background : numpy.ndarray
        Lower layer, ``(H, W)`` or ``(3, H, W)``, values in [0, 1]. Overwritten in place by "overlay" and
        "soft_light".
    min_c : numpy.ndarray or None
        Only for "luminosity", passed to blend_luminosity.
    max_c : numpy.ndarray or None
        Only for "luminosity", passed to blend_luminosity.

    Returns
    -------
    numpy.ndarray
        "normal" (and unknown names): ``active`` itself. "luminosity": see blend_luminosity. Other modes: see
        blend_multi_dim_images.

    See Also
    --------
    render_images : applies the opacity afterwards.
    """
    if blend_mode.lower() == "multiply" or blend_mode.lower() == "overlay" or blend_mode.lower() == "screen" \
            or blend_mode.lower() == "soft_light":
        return blend_multi_dim_images(blend_mode, active, background)
    elif blend_mode.lower() == "luminosity":
        return blend_luminosity(active, background, min_c, max_c)
    else:
        return blend_normal(active, background)


def _outside_0_1(image):
    """
    Return True if any non-NaN value of ``image`` is below 0 or above 1 (the range test before blending).

    The test is a pure function of the array values; render_all_images reuses its result for an array that has
    not been modified since. An all-NaN array gives False (``np.nanmin`` returns NaN with a RuntimeWarning).

    Parameters
    ----------
    image : numpy.ndarray
        Image of any shape.

    Returns
    -------
    bool
        ``nanmin(image) < 0 or nanmax(image) > 1``.
    """
    return bool(np.nanmin(image) < 0 or np.nanmax(image) > 1)


def render_images(active, background, opacity):
    """
    Mix a blended layer with the background according to the opacity.

    An operand with any non-NaN value outside [0, 1] is first rescaled with scale_0_to_1. Then
    ``active * opacity + background * (1 - opacity)`` is computed (apply_opacity), band by band when either
    operand is RGB.

    Parameters
    ----------
    active : numpy.ndarray
        Blended layer (result of blend_images), ``(H, W)`` or ``(3, H, W)``.
    background : numpy.ndarray
        Background, ``(H, W)`` or ``(3, H, W)``.
    opacity : float
        Opacity of ``active`` in percent, 0-100; a value <= 1 is used as a fraction (see apply_opacity).

    Returns
    -------
    numpy.ndarray
        At least one RGB operand: new float64 ``(3, H, W)`` array. Two gray operands: the result of
        apply_opacity (dtype of the operands, e.g. float32 for float32 input).

    Notes
    -----
    The rescale can modify an operand in place (scale_0_to_1 replaces NaN; see its Notes). The values are not
    clipped, so an opacity above 100 extrapolates beyond [0, 1].

    Accelerated path: RGB bands are assigned through ``_assign_elementwise`` (row blocks); bitwise identical.
    """
    return _render_images(active, background, opacity)


def _render_images(active, background, opacity, active_outside=None, background_outside=None):
    """
    Implement render_images, optionally with range tests already evaluated by the caller.

    Contract: a caller that has evaluated ``_outside_0_1()`` on the very same array object, and has not modified
    the array since, may pass the result as ``active_outside`` / ``background_outside``; the result is then
    identical to recomputing the test. None means "compute here". render_all_images uses this to avoid
    repeating the full-image ``nanmin`` / ``nanmax`` passes.

    Parameters
    ----------
    active : numpy.ndarray
        Blended layer, ``(H, W)`` or ``(3, H, W)``.
    background : numpy.ndarray
        Background, ``(H, W)`` or ``(3, H, W)``.
    opacity : float
        Opacity in percent (see apply_opacity).
    active_outside : bool or None
        ``_outside_0_1(active)`` if known, else None.
    background_outside : bool or None
        ``_outside_0_1(background)`` if known, else None.

    Returns
    -------
    numpy.ndarray
        As render_images.
    """

    # Both active and background image have to be between 0 and 1, clip if not (new array; NaN and dtype kept)
    if active_outside is None:
        active_outside = _outside_0_1(active)
    if active_outside:
        active = np.clip(active, 0, 1)
    if background_outside is None:
        background_outside = _outside_0_1(background)
    if background_outside:
        background = np.clip(background, 0, 1)

    # True if image has 3 bands (RGB), false if single band
    a_rgb = len(active.shape) == 3
    b_rgb = len(background.shape) == 3

    def func(act, back):
        return apply_opacity(act, back, opacity)

    # Apply opacity
    if a_rgb and b_rgb:
        # Both images 3 bands
        render_image = np.zeros(background.shape)
        for i in range(3):
            _assign_elementwise(render_image[i, :, :], func, active[i, :, :], background[i, :, :])
    elif a_rgb and not b_rgb:
        # Active image 3 bands
        render_image = np.zeros(active.shape)
        for i in range(3):
            _assign_elementwise(render_image[i, :, :], func, active[i, :, :], background)
    elif not a_rgb and b_rgb:
        # Background image 3 bands
        render_image = np.zeros(background.shape)
        for i in range(3):
            _assign_elementwise(render_image[i, :, :], func, active, background[i, :, :])
    else:
        render_image = apply_opacity(active, background, opacity)

    return render_image


def scale_within_0_and_1(numeric_value):
    """
    Rescale an array with a value range wider than 0.3 for blending (intended: into [0, 1]).

    Steps as implemented:

    1. If all non-NaN values lie in [0, 1], return the argument unchanged (same object).
    2. Replace NaN by the minimum, **in the caller's array**.
    3. If ``1 < max <= 255``, ``max - min > 30`` and ``min > 0`` (data that look like 8-bit), return
       ``numeric_value / 255``.
    4. Otherwise return ``(numeric_value - lo) / (hi - lo)`` with ``lo = 0`` and ``hi = 1`` (see Notes), and set
       values in (-0.01, 0) to 0 when no value is below -0.01.

    Parameters
    ----------
    numeric_value : numpy.ndarray
        Array of any shape. NaN is overwritten in place (step 2).

    Returns
    -------
    numpy.ndarray
        ``numeric_value`` itself (step 1) or a new array of its shape and float dtype (steps 3, 4).

    Notes
    -----
    Known issues (docs/KNOWN_ISSUES.md):

    - ``np.array(0, actual_min)`` and ``np.array(1, actual_max)`` pass the minimum / maximum as the *dtype*
      argument (a numpy scalar is accepted as its dtype), so the stretch limits are always 0 and 1 and step 4
      returns the values unchanged instead of stretching them to [0, 1].
    - The NaN mask is lost: NaN pixels come back as the minimum, and the argument is modified.
    """
    if np.nanmin(numeric_value) >= 0 and np.nanmax(numeric_value) <= 1:
        return numeric_value

    # NaN is not filled: it stays NaN in the result, and the argument is not modified
    actual_min = np.nanmin(numeric_value)
    norm_min_value = actual_min

    actual_max = np.nanmax(numeric_value)
    norm_max_value = actual_max

    # Values that look like 8-bit data (1 < max <= 255, max - min > 30, min > 0) are divided by 255; the dtype is
    # not checked
    if 255 >= actual_max > 1:
        if actual_max - actual_min > 30 and actual_min > 0:
            scaled = numeric_value / 255
            return scaled

    scaled = (numeric_value - norm_min_value) / (norm_max_value - norm_min_value)

    if np.nanmin(scaled) > -0.01:
        scaled[(0 > scaled) & (scaled > -0.01)] = 0

    return scaled


def scale_strict_0_to_1(numeric_value):
    """
    Stretch an array min-max to [0, 1] (intended for arrays with a value range up to 0.3).

    If the minimum is 0 and the maximum 1, the argument is returned unchanged. Otherwise NaN is replaced by 0
    **in the caller's array**, the array is stretched with ``(x - min) / (max - min)``, and values in (-0.01, 0)
    are meant to be set to 0.

    Parameters
    ----------
    numeric_value : numpy.ndarray
        Array of any shape. NaN is overwritten with 0 in place.

    Returns
    -------
    numpy.ndarray
        ``numeric_value`` itself (minimum 0 and maximum 1), or, for a constant array (including any one-element
        array), a new all-NaN array (0 / 0, RuntimeWarning).

    Raises
    ------
    ValueError
        For every non-constant array with more than one element, see Notes.

    Notes
    -----
    Known issue: ``scaled[0 > scaled > -0.01] = 0`` is a chained comparison, evaluated as
    ``(0 > scaled) and (scaled > -0.01)``; ``and`` needs the truth value of an array, which raises ValueError for
    arrays with more than one element. After the stretch the minimum is 0, so the comparison is always reached
    for non-constant input; NaN has already been replaced by 0 in the argument at that point. See
    docs/KNOWN_ISSUES.md.
    """
    if np.nanmin(numeric_value) == 0 and np.nanmax(numeric_value) == 1:
        return numeric_value

    # NaN is not filled: it stays NaN in the result, and the argument is not modified
    min_value = np.nanmin(numeric_value)
    max_value = np.nanmax(numeric_value)

    scaled = (numeric_value - min_value) / (max_value - min_value)

    if np.nanmin(scaled) > -0.01:
        scaled[(0 > scaled) & (scaled > -0.01)] = 0

    return scaled


def scale_0_to_1(numeric_value):
    """
    Rescale an array for blending (intended: into [0, 1]); dispatch to the two scaling functions.

    - Maximum in (0.9, 1] and minimum 0: return the argument unchanged.
    - Range ``max - min > 0.3``: scale_within_0_and_1.
    - Otherwise: scale_strict_0_to_1.

    Parameters
    ----------
    numeric_value : numpy.ndarray
        Array of any shape; NaN may be overwritten in place by the called function.

    Returns
    -------
    numpy.ndarray
        The argument itself or a new array, see the called functions.

    Raises
    ------
    ValueError
        From scale_strict_0_to_1, for non-constant arrays with a range up to 0.3.

    Notes
    -----
    Known issue: through the issues of the two called functions, an array outside [0, 1] with a range above 0.3
    is returned with unchanged values (except the 8-bit case, divided by 255), and a non-constant array with a
    range up to 0.3 raises ValueError. See docs/KNOWN_ISSUES.md.
    """
    if 1 >= np.nanmax(numeric_value) > 0.9 and np.nanmin(numeric_value) == 0:
        return numeric_value
    elif np.nanmax(numeric_value) - np.nanmin(numeric_value) > 0.3:
        return scale_within_0_and_1(numeric_value)
    else:
        return scale_strict_0_to_1(numeric_value)


def apply_opacity(active, background, opacity):
    """
    Mix two layers: ``active * opacity + background * (1 - opacity)``.

    Parameters
    ----------
    active : numpy.ndarray
        Upper layer. Not modified.
    background : numpy.ndarray
        Lower layer, broadcastable with ``active``. Not modified.
    opacity : float
        Opacity of ``active`` in percent, 0-100. A value above 1 is divided by 100; a value <= 1 is used as a
        fraction as is.

    Returns
    -------
    numpy.ndarray
        New array, broadcast shape, numpy dtype promotion of the operands (float32 operands give float32). The
        result is not clipped.

    Notes
    -----
    Known issue: because of the "> 1" limit, ``opacity=1`` means 100 %, not 1 %, and there is no way to ask for
    1 %. Values above 100 (or below 0) are not rejected and extrapolate. See docs/KNOWN_ISSUES.md.
    """
    if opacity > 1:
        opacity = opacity / 100
    return active * opacity + background * (1 - opacity)


def normalize_image(visualization, image, min_norm, max_norm, normalization):
    """
    Normalize a visualization to [0, 1] for blending; invert slope and negative openness.

    This is the normalization entry point of BlenderCombination.render_all_images and of
    DefaultValues.float_to_8bit. Steps: "percent" is renamed to "perc"; advanced_normalization; a rescale with
    scale_0_to_1 if the result has values above 1 or below 0; ``1 - x`` for slope and negative openness (high
    slope and high negative openness become dark).

    Parameters
    ----------
    visualization : str or None
        Visualization name. Inverted: "slope gradient" and "openness - negative" (case-insensitive), "slp" and
        "neg_opns" (case-sensitive). None returns None.
    image : numpy.ndarray
        Visualization ``(H, W)`` or ``(3, H, W)``. **Clipped in place** by normalize_lin.
    min_norm : float
        Lower limit: a value for "value", a percentage (0-100) cut off for "perc".
    max_norm : float
        Upper limit: a value for "value", a percentage (0-100) cut off for "perc".
    normalization : str
        "value" or "perc" (case-insensitive) or "percent" (lower case only).

    Returns
    -------
    numpy.ndarray or None
        float32 array of the shape of ``image`` with values in [0, 1], NaN kept; None if ``visualization`` is None.

    Raises
    ------
    Exception
        From advanced_normalization (limits, unknown normalization name such as "Percent").

    Notes
    -----
    The output of normalize_lin is already clamped to [0, 1], so with the existing normalizations the two rescale
    branches (and their warnings "unexpected values!") are not reached; they are kept as they are.

    Known issue: the abbreviations "slp", "neg_opns" (and "mhs") and the rename of "percent" are compared
    case-sensitively, unlike the long names; "SLP" is not inverted and "Percent" raises. See
    docs/KNOWN_ISSUES.md.
    """
    if visualization is None:
        return None

    if normalization == "percent":
        normalization = "perc"

    norm_image = advanced_normalization(image=image, minimum=min_norm, maximum=max_norm, normalization=normalization)

    # Make sure it scales 0 to 1
    if np.nanmax(norm_image) > 1:
        if visualization.lower() == "multiple directions hillshade" or visualization == "mhs":
            norm_image = scale_0_to_1(norm_image)
        else:
            norm_image = scale_0_to_1(norm_image)
            warnings.warn("rvt.blend_func.normalize_image: unexpected values! max > 1")

    if np.nanmin(norm_image) < 0:
        norm_image = scale_0_to_1(norm_image)
        warnings.warn("rvt.blend_func.normalize_image: unexpected values! min < 0")

    # For slope invert scale (high slopes will be black)
    if visualization.lower() == "slope gradient" or visualization.lower() == "openness - negative" or \
            visualization == "slp" or visualization == "neg_opns":
        norm_image = 1 - norm_image

    return norm_image


def cut_off_normalize(image, mode, cutoff_min=None, cutoff_max=None, bool_norm=True):
    """
    Clip a one-band image by values or percentiles, and optionally stretch it to [0, 1].

    Examples: mode "perc", min 2, max 3 cuts off the lowest 2 % and the highest 3 % of the values; mode "value",
    min 10, max 60 clips the image to 10-60.

    Parameters
    ----------
    image : numpy.ndarray
        2D ``(H, W)`` image. **Clipped in place** in the "value" and "perc" modes.
    mode : str
        "value" or "perc" / "percent" (case-insensitive for the dispatch).
    cutoff_min : float or None
        "value": lower clip value, None = ``np.amin(image)``. "perc": percentage of the lowest values to cut off
        (0-100), None = 0.
    cutoff_max : float or None
        "value": upper clip value, None = ``np.amax(image)``. "perc": percentage of the highest values to cut off
        (0-100), None = 0.
    bool_norm : bool
        True: clip and stretch to [0, 1] (normalize_lin / normalize_perc). False: clip only.

    Returns
    -------
    numpy.ndarray
        ``bool_norm=True``: new float32 array with values in [0, 1]. ``bool_norm=False`` or an unknown mode:
        ``image`` itself (clipped, or unchanged for an unknown mode).

    Raises
    ------
    Exception
        If both limits are given, ``mode == "value"`` and ``cutoff_min == cutoff_max`` or
        ``cutoff_min > cutoff_max``; from lin_cutoff_calc_from_perc for invalid percentages.

    Notes
    -----
    The default limits of "value" use ``np.amin`` / ``np.amax``, so an image with NaN gets NaN limits and an
    all-NaN result.

    Known issues (docs/KNOWN_ISSUES.md): the limit checks compare ``mode`` with "value" case-sensitively, so
    "Value" skips them (min == max gives all NaN); an unknown mode raises nothing and returns ``image`` unchanged.
    """
    if cutoff_min is not None and cutoff_max is not None:
        if cutoff_min == cutoff_max and mode == "value":
            raise Exception("rvt.blend_func.cut_off_normalize: If normalization == value, min and max cannot be the"
                            " same!")
        if cutoff_min > cutoff_max and mode == "value":
            raise Exception("rvt.blend_func.cut_off_normalize: If normalization == value, max can't be smaller"
                            " than min!")

    cut_off_arr = image
    if cutoff_min is None and mode.lower() == "value":
        cutoff_min = np.amin(image)
    if cutoff_max is None and mode.lower() == "value":
        cutoff_max = np.amax(image)
    if cutoff_min is None and (mode.lower() == "perc" or mode.lower() == "percent"):
        cutoff_min = 0
    if cutoff_max is None and (mode.lower() == "perc" or mode.lower() == "percent"):
        cutoff_max = 0
    if bool_norm:
        if mode.lower() == "value":
            cut_off_arr = normalize_lin(cut_off_arr, cutoff_min, cutoff_max)
        elif mode.lower() == "perc" or mode.lower() == "percent":
            cut_off_arr = normalize_perc(cut_off_arr, cutoff_min, cutoff_max)
    else:
        if mode.lower() == "value":
            cut_off_arr[cut_off_arr > cutoff_max] = cutoff_max
            cut_off_arr[cut_off_arr < cutoff_min] = cutoff_min
        elif mode.lower() == "perc" or mode.lower() == "percent":
            min_max_value_dict = lin_cutoff_calc_from_perc(cut_off_arr, cutoff_min, cutoff_max)
            min_value = min_max_value_dict["min_lin"]
            max_value = min_max_value_dict["max_lin"]
            cut_off_arr[cut_off_arr > max_value] = max_value
            cut_off_arr[cut_off_arr < min_value] = min_value
    return cut_off_arr
