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

# Blocked elementwise evaluation (pure data-flow change). The blend kernels below evaluate exactly the same
# expressions on the same element values, only on row blocks of about _BLOCK_ELEMS elements instead of whole
# images, so temporaries stay cache-resident and full-size temporaries / fancy indexing are avoided. Every element
# goes through the same IEEE operations in the same order and with the same dtypes, so results are bitwise equal.
_BLOCK_ELEMS = 32768
_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_REAL_SCALARS = (int, float, np.integer, np.floating)


def _row_slices(n_rows, row_elems):
    """Row-block slices along axis 0 with about _BLOCK_ELEMS elements per block."""
    step = max(1, _BLOCK_ELEMS // max(1, row_elems))
    return [slice(r, r + step) for r in range(0, n_rows, step)]


def _float_arrays(*arrays):
    """True if all arguments are plain float32/float64 ndarrays of one non-empty shape with ndim >= 1."""
    shape = arrays[0].shape if isinstance(arrays[0], np.ndarray) else None
    if shape is None or len(shape) == 0 or arrays[0].size == 0:
        return False
    for arr in arrays:
        if type(arr) is not np.ndarray or arr.dtype not in _FLOAT_DTYPES or arr.shape != shape:
            return False
    return True


def _fp_strict():
    """Current numpy error state with every non-ignored category turned into 'raise'. Branch expressions are
    evaluated for all elements of a block under this state; if any floating point error is flagged, the block is
    recomputed with the reference code under the caller's error state, so the same kinds of warnings are issued.
    Returns None if the caller's state raises or calls a handler (then the reference code runs on the whole array,
    so an exception leaves the inputs in exactly the same partially updated state)."""
    err = np.geterr()
    if any(val not in ("ignore", "warn", "print") for val in err.values()):
        return None
    return {key: ("ignore" if val == "ignore" else "raise") for key, val in err.items()}


def gray_scale_to_color_ramp(gray_scale, colormap, min_colormap_cut=None, max_colormap_cut=None, alpha=False,
                             output_8bit=True):
    """
    Turns normalized gray scale np.array to rgba (np.array of 4 np.arrays r, g, b, a).

    Parameters
    ----------
    gray_scale : np.array (2D)
        Normalized gray_scale img as np.array (0-1)
    colormap : str
        Colormap form matplotlib (https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html)
    min_colormap_cut : float
        What lower part of colormap to cut to select part of colormap.
        Valid values are between 0 and 1, if 0.2 it cuts off (deletes) 20% of lower colors in colormap.
        If None cut is not applied.
    max_colormap_cut : float
        What upper part of colormap to cut to select part of colormap.
        Valid values are between 0 and 1, if 0.8 it cuts off (deletes) 20% of upper colors in colormap.
        If None cut is not applied.
    alpha : bool
        If True outputs 4D array RGBA, if False outputs 3D array RGB
    output_8bit : bool
        If true output values will be int 0-255 instead of normalized values.
    Returns
    -------
    rgba_out : np.array (3D: red 0-255, green 0-255, blue 0-255)
            If alpha False: np.array (4D: red 0-255, green 0-255, blue 0-255, alpha 0-255)
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
            raise Exception("rvt.blend_func.gray_scale_to_color_ramp: min_colormap_cut can't be smaller than"
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
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def normalize_lin(image, minimum, maximum):
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
    """Minimum cutoff in percent, maximum cutoff in percent (0%-100%). Returns min and max values for linear
    stretch (cut-off)."""
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
    min_max_lin_dict = lin_cutoff_calc_from_perc(image, minimum, maximum)
    min_lin = min_max_lin_dict["min_lin"]
    max_lin = min_max_lin_dict["max_lin"]
    return normalize_lin(image, min_lin, max_lin)


def advanced_normalization(image, minimum, maximum, normalization):
    """Runs normalization based on the selected normalization type: value or percent."""

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
    if len(img.shape) == 3:
        r = img[0]
        g = img[1]
        b = img[2]
        lum_img = np.float32((0.3 * r) + (0.59 * g) + (0.11 * b))
    else:
        lum_img = img

    return lum_img


def matrix_eq_min_lt_zero(r: np.ndarray, idx_min_lt_zero, lum_c, min_c):
    r[idx_min_lt_zero] = lum_c[idx_min_lt_zero] + (((r[idx_min_lt_zero] - lum_c[idx_min_lt_zero]) *
                                                    lum_c[idx_min_lt_zero]) / (lum_c[idx_min_lt_zero] -
                                                                               min_c[idx_min_lt_zero]))
    return r


def matrix_eq_max_gt_one(r: np.ndarray, idx_max_c_gt_one, lum_c, max_c):
    r[idx_max_c_gt_one] = lum_c[idx_max_c_gt_one] + (((r[idx_max_c_gt_one] - lum_c[idx_max_c_gt_one]) *
                                                      (1.0 - lum_c[idx_max_c_gt_one])) / (max_c[idx_max_c_gt_one]
                                                                                          - lum_c[idx_max_c_gt_one]))
    return r


def channel_min(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    min_c = r * 1.0
    idx_min = np.where(g < min_c)
    min_c[idx_min] = g[idx_min]
    idx_min = np.where(b < min_c)
    min_c[idx_min] = b[idx_min]
    return min_c


def channel_max(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    max_c = r * 1.0
    idx_max = np.where(g > max_c)
    max_c[idx_max] = g[idx_max]
    idx_max = np.where(b > max_c)
    max_c[idx_max] = b[idx_max]
    return max_c


def clip_color(c, min_c=None, max_c=None):
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
    """blend_luminosity(active, background) followed by clip_color(c) for a 3-band float background, evaluated per
    row block with the same expressions: c = background + luminosity (stored as float64), lum_c = lum(c),
    r, g, b = float32(c), channel min / max, then the min < 0 and max > 1 corrections on the same subsets."""
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
    return active


def blend_screen(active, background):
    return 1 - (1 - active) * (1 - background)


def blend_multiply(active, background):
    return active * background


def _blend_in_place_blocked(active, background, masks, branches, reference):
    """Shared driver of blend_overlay / blend_soft_light: per row block, evaluate both branch expressions for all
    elements and copy each into `background` where its mask holds (masks are taken before any write, as with
    np.where in the reference). Elements outside both masks (NaN) keep their value. A block whose branch evaluation
    flags a floating point error is recomputed by `reference` under the caller's error state instead."""
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
    return background > 0.5, background <= 0.5


def _overlay_branches(active, background):
    return (1 - (1 - 2 * (background - 0.5)) * (1 - active)), ((2 * background) * active)


def _blend_overlay_ref(active, background):
    idx1 = np.where(background > 0.5)
    idx2 = np.where(background <= 0.5)
    background[idx1] = (1 - (1 - 2 * (background[idx1] - 0.5)) * (1 - active[idx1]))
    background[idx2] = ((2 * background[idx2]) * active[idx2])
    return background


def blend_overlay(active, background):
    return _blend_in_place_blocked(active, background, _overlay_masks, _overlay_branches, _blend_overlay_ref)


def _soft_light_masks(active, background):
    return active < 0.5, active >= 0.5


def _soft_light_branches(active, background):
    return (2 * background * active + background ** 2 * (1.0 - 2 * active),
            2 * background * (1.0 - active) + np.sqrt(background) * (2 * active - 1.0))


def blend_soft_light(active, background):
    return _blend_in_place_blocked(active, background, _soft_light_masks, _soft_light_branches,
                                   _blend_soft_light_ref)


def _blend_soft_light_ref(active, background):
    # idx1 = np.where(active > 0.5)
    # idx2 = np.where(active <= 0.5)
    # background[idx1] = 1 - (1-background[idx1]) * (1-(active[idx1]-0.5))
    # background[idx2] = background[idx2] * (active[idx2]+0.5)
    idx1 = np.where(active < 0.5)
    idx2 = np.where(active >= 0.5)
    background[idx1] = 2 * background[idx1] * active[idx1] + background[idx1] ** 2 * (1.0 - 2 * active[idx1])
    background[idx2] = 2 * background[idx2] * (1.0 - active[idx2]) + np.sqrt(background[idx2]) * (
                2 * active[idx2] - 1.0)
    return background


def blend_luminosity(active, background, min_c=None, max_c=None):
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
    if blend_mode.lower() == "screen":
        return blend_screen(active, background)
    elif blend_mode.lower() == "multiply":
        return blend_multiply(active, background)
    elif blend_mode.lower() == "overlay":
        return blend_overlay(active, background)
    elif blend_mode.lower() == "soft_light":
        return blend_soft_light(active, background)


def _assign_elementwise(out, func, active, background):
    """out[...] = func(active, background) for an elementwise func, evaluated per row block when both operands are
    plain, non-overlapping ndarrays of out's shape (no broadcasting); in-place effects of func on background are
    the same (also when an error is raised: with a raising numpy error state the whole array is processed at once)."""
    if (type(active) is np.ndarray and type(background) is np.ndarray and active.ndim >= 1 and out.size > 0
            and active.shape == out.shape and background.shape == out.shape and _fp_strict() is not None
            and not np.may_share_memory(active, background)):
        for sl in _row_slices(out.shape[0], out.size // out.shape[0]):
            out[sl] = func(active[sl], background[sl])
    else:
        out[...] = func(active, background)


def blend_multi_dim_images(blend_mode, active, background):
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
    if blend_mode.lower() == "multiply" or blend_mode.lower() == "overlay" or blend_mode.lower() == "screen" \
            or blend_mode.lower() == "soft_light":
        return blend_multi_dim_images(blend_mode, active, background)
    elif blend_mode.lower() == "luminosity":
        return blend_luminosity(active, background, min_c, max_c)
    else:
        return blend_normal(active, background)


def _outside_0_1(image):
    """The range test used before blending: any non-NaN value below 0 or above 1."""
    return bool(np.nanmin(image) < 0 or np.nanmax(image) > 1)


def render_images(active, background, opacity):
    return _render_images(active, background, opacity)


def _render_images(active, background, opacity, active_outside=None, background_outside=None):
    """render_images; a caller that has just evaluated _outside_0_1() on the very same unchanged array may pass
    the result as active_outside / background_outside instead of having it recomputed (None = compute)."""

    # Both active and background image have to be between 0 and 1, scale if not
    if active_outside is None:
        active_outside = _outside_0_1(active)
    if active_outside:
        active = scale_0_to_1(active)
    if background_outside is None:
        background_outside = _outside_0_1(background)
    if background_outside:
        background = scale_0_to_1(background)

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
    if np.nanmin(numeric_value) >= 0 and np.nanmax(numeric_value) <= 1:
        return numeric_value

    # Create mask for NaN values
    # nan_mask = np.isnan(numeric_value)

    numeric_value[np.isnan(numeric_value)] = np.nanmin(numeric_value)  # nan change to nanmin

    actual_min = np.nanmin(numeric_value)
    norm_min_value = np.nanmax(np.array(0, actual_min))

    actual_max = np.nanmax(numeric_value)
    norm_max_value = np.nanmin(np.array(1, actual_max))

    # Do not scale values where max is between 1 and 255 if the max-min values difference is at least 30 and min >0
    # and numeric values are integer type
    if 255 >= actual_max > 1:
        if actual_max - actual_min > 30 and actual_min > 0:
            scaled = numeric_value / 255
            return scaled

    scaled = (numeric_value - norm_min_value) / (norm_max_value - norm_min_value)

    if np.nanmin(scaled) > -0.01:
        scaled[(0 > scaled) & (scaled > -0.01)] = 0

    # scaled[nan_mask] = np.nan

    return scaled


def scale_strict_0_to_1(numeric_value):
    if np.nanmin(numeric_value) == 0 and np.nanmax(numeric_value) == 1:
        return numeric_value

    numeric_value[np.isnan(numeric_value)] = 0  # nan change to 0

    min_value = np.nanmin(numeric_value)
    max_value = np.nanmax(numeric_value)

    scaled = (numeric_value - min_value) / (max_value - min_value)

    if np.nanmin(scaled) > -0.01:
        scaled[0 > scaled > -0.01] = 0

    return scaled


def scale_0_to_1(numeric_value):
    if 1 >= np.nanmax(numeric_value) > 0.9 and np.nanmin(numeric_value) == 0:
        return numeric_value
    elif np.nanmax(numeric_value) - np.nanmin(numeric_value) > 0.3:
        return scale_within_0_and_1(numeric_value)
    else:
        return scale_strict_0_to_1(numeric_value)


def apply_opacity(active, background, opacity):
    if opacity > 1:
        opacity = opacity / 100
    return active * opacity + background * (1 - opacity)


def normalize_image(visualization, image, min_norm, max_norm, normalization):
    """Main function for normalization. Runs advanced normalization on the array and performs special operations for
    some visualization types (e.g. invert scale for slope, scale for mhs, etc.).
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
    One band image cut-off or normalization or both. Image is 2D np.ndarray of raster, mode is perc or value
    (min and max units), min and max are minimum value to cutoff and maximum value to cutoff.
    (e.x. percent min=2 and max=3 -> cutoff lower 2% values and higher 3% values;
     e.x. value min=10 and max=60 -> cutoff below 10 and above 60, image values will be 10-60)
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
