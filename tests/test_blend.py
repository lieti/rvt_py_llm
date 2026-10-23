"""
Blend / render path of e3MSTP and e4MSTP: rvt.blend_func (blocked normalize_lin, blend_overlay, blend_soft_light,
blend_luminosity + clip_color, blend_multi_dim_images, render_images) and the layer loop of
rvt.blend.BlenderCombination.render_all_images (range tests reused) against the np.where / fancy indexing
formulation of tag baseline-fixed. Unchanged helpers (scale_0_to_1, lin_cutoff_calc_from_perc,
gray_scale_to_color_ramp, blend_screen, blend_multiply) are taken from rvt.
"""

import copy
import warnings

import numpy as np
import pytest

import rvt.blend
import rvt.blend_func as bf
import rvt.vis
from conftest import assert_bitwise, assert_same, synthetic_dem


# ----------------------------------------------------------------------------------------------------------------------
# reference implementation (baseline-fixed, transcribed)
# ----------------------------------------------------------------------------------------------------------------------

def ref_normalize_lin(image, minimum, maximum):
    image[image > maximum] = maximum
    image[image < minimum] = minimum
    image = (image - minimum) / (maximum - minimum)
    image[image > 1] = 1
    image[image < 0] = 0
    return np.float32(image)


def ref_normalize_perc(image, minimum, maximum):
    d = bf.lin_cutoff_calc_from_perc(image, minimum, maximum)
    return ref_normalize_lin(image, d["min_lin"], d["max_lin"])


def ref_advanced_normalization(image, minimum, maximum, normalization):
    if normalization.lower() == "value":
        return ref_normalize_lin(image=image, minimum=minimum, maximum=maximum)
    elif normalization.lower() == "perc":
        return ref_normalize_perc(image=image, minimum=minimum, maximum=maximum)
    raise ValueError(normalization)


def ref_normalize_image(visualization, image, min_norm, max_norm, normalization):
    if visualization is None:
        return None
    if normalization == "percent":
        normalization = "perc"
    norm_image = ref_advanced_normalization(image, min_norm, max_norm, normalization)
    if np.nanmax(norm_image) > 1:
        if visualization.lower() == "multiple directions hillshade" or visualization == "mhs":
            norm_image = bf.scale_0_to_1(norm_image)
        else:
            norm_image = bf.scale_0_to_1(norm_image)
            warnings.warn("rvt.blend_func.normalize_image: unexpected values! max > 1")
    if np.nanmin(norm_image) < 0:
        norm_image = bf.scale_0_to_1(norm_image)
        warnings.warn("rvt.blend_func.normalize_image: unexpected values! min < 0")
    if visualization.lower() == "slope gradient" or visualization.lower() == "openness - negative" or \
            visualization == "slp" or visualization == "neg_opns":
        norm_image = 1 - norm_image
    return norm_image


def ref_lum(img):
    if len(img.shape) == 3:
        return np.float32((0.3 * img[0]) + (0.59 * img[1]) + (0.11 * img[2]))
    return img


def ref_matrix_eq_min_lt_zero(r, idx, lum_c, min_c):
    r[idx] = lum_c[idx] + (((r[idx] - lum_c[idx]) * lum_c[idx]) / (lum_c[idx] - min_c[idx]))
    return r


def ref_matrix_eq_max_gt_one(r, idx, lum_c, max_c):
    r[idx] = lum_c[idx] + (((r[idx] - lum_c[idx]) * (1.0 - lum_c[idx])) / (max_c[idx] - lum_c[idx]))
    return r


def ref_channel_min(r, g, b):
    min_c = r * 1.0
    idx = np.where(g < min_c)
    min_c[idx] = g[idx]
    idx = np.where(b < min_c)
    min_c[idx] = b[idx]
    return min_c


def ref_channel_max(r, g, b):
    max_c = r * 1.0
    idx = np.where(g > max_c)
    max_c[idx] = g[idx]
    idx = np.where(b > max_c)
    max_c[idx] = b[idx]
    return max_c


def ref_clip_color(c, min_c=None, max_c=None):
    lum_c = ref_lum(c)
    r = np.float32(c[0])
    g = np.float32(c[1])
    b = np.float32(c[2])
    if min_c is None and max_c is None:
        min_c = ref_channel_min(r, g, b)
        max_c = ref_channel_max(r, g, b)
    idx = np.where(min_c < 0)
    r = ref_matrix_eq_min_lt_zero(r, idx, lum_c, min_c)
    g = ref_matrix_eq_min_lt_zero(g, idx, lum_c, min_c)
    b = ref_matrix_eq_min_lt_zero(b, idx, lum_c, min_c)
    idx = np.where(max_c > 1)
    r = ref_matrix_eq_max_gt_one(r, idx, lum_c, max_c)
    g = ref_matrix_eq_max_gt_one(g, idx, lum_c, max_c)
    b = ref_matrix_eq_max_gt_one(b, idx, lum_c, max_c)
    c_out = np.zeros(c.shape)
    c_out[0, :, :] = r
    c_out[1, :, :] = g
    c_out[2, :, :] = b
    return c_out


def ref_blend_overlay(active, background):
    idx1 = np.where(background > 0.5)
    idx2 = np.where(background <= 0.5)
    background[idx1] = (1 - (1 - 2 * (background[idx1] - 0.5)) * (1 - active[idx1]))
    background[idx2] = ((2 * background[idx2]) * active[idx2])
    return background


def ref_blend_soft_light(active, background):
    idx1 = np.where(active < 0.5)
    idx2 = np.where(active >= 0.5)
    background[idx1] = 2 * background[idx1] * active[idx1] + background[idx1] ** 2 * (1.0 - 2 * active[idx1])
    background[idx2] = 2 * background[idx2] * (1.0 - active[idx2]) + np.sqrt(background[idx2]) * (
            2 * active[idx2] - 1.0)
    return background


def ref_blend_luminosity(active, background, min_c=None, max_c=None):
    lum_active = ref_lum(active)
    lum_background = ref_lum(background)
    luminosity = lum_active - lum_background
    if len(background.shape) < 3:
        return lum_active
    c = np.zeros(background.shape)
    c[0, :, :] = background[0] + luminosity
    c[1, :, :] = background[1] + luminosity
    c[2, :, :] = background[2] + luminosity
    return ref_clip_color(c, min_c, max_c)


def ref_equation_blend(blend_mode, active, background):
    if blend_mode.lower() == "screen":
        return 1 - (1 - active) * (1 - background)
    elif blend_mode.lower() == "multiply":
        return active * background
    elif blend_mode.lower() == "overlay":
        return ref_blend_overlay(active, background)
    elif blend_mode.lower() == "soft_light":
        return ref_blend_soft_light(active, background)


def ref_blend_multi_dim_images(blend_mode, active, background):
    a_rgb = len(active.shape) == 3
    b_rgb = len(background.shape) == 3
    blended_image = None
    if a_rgb and b_rgb:
        blended_image = np.zeros(background.shape)
        for i in range(3):
            blended_image[i, :, :] = ref_equation_blend(blend_mode, active[i, :, :], background[i, :, :])
    if a_rgb and not b_rgb:
        blended_image = np.zeros(active.shape)
        for i in range(3):
            blended_image[i, :, :] = ref_equation_blend(blend_mode, active[i, :, :], background)
    if not a_rgb and b_rgb:
        blended_image = np.zeros(background.shape)
        for i in range(3):
            blended_image[i, :, :] = ref_equation_blend(blend_mode, active, background[i, :, :])
    if not a_rgb and not b_rgb:
        blended_image = ref_equation_blend(blend_mode, active, background)
    return blended_image


def ref_blend_images(blend_mode, active, background, min_c=None, max_c=None):
    if blend_mode.lower() in ("multiply", "overlay", "screen", "soft_light"):
        return ref_blend_multi_dim_images(blend_mode, active, background)
    elif blend_mode.lower() == "luminosity":
        return ref_blend_luminosity(active, background, min_c, max_c)
    return active


def ref_apply_opacity(active, background, opacity):
    if opacity > 1:
        opacity = opacity / 100
    return active * opacity + background * (1 - opacity)


def ref_render_images(active, background, opacity):
    if np.nanmin(active) < 0 or np.nanmax(active) > 1:
        active = bf.scale_0_to_1(active)
    if np.nanmin(background) < 0 or np.nanmax(background) > 1:
        background = bf.scale_0_to_1(background)
    a_rgb = len(active.shape) == 3
    b_rgb = len(background.shape) == 3
    if a_rgb and b_rgb:
        render_image = np.zeros(background.shape)
        for i in range(3):
            render_image[i, :, :] = ref_apply_opacity(active[i, :, :], background[i, :, :], opacity)
    elif a_rgb and not b_rgb:
        render_image = np.zeros(active.shape)
        for i in range(3):
            render_image[i, :, :] = ref_apply_opacity(active[i, :, :], background, opacity)
    elif not a_rgb and b_rgb:
        render_image = np.zeros(background.shape)
        for i in range(3):
            render_image[i, :, :] = ref_apply_opacity(active, background[i, :, :], opacity)
    else:
        render_image = ref_apply_opacity(active, background, opacity)
    return render_image


def ref_render_layers(layers):
    """The layer loop of BlenderCombination.render_all_images for layers with an image given as an array."""
    rendered_image = None
    for layer in reversed(layers):
        image = layer.image
        norm_image = ref_normalize_image(layer.vis, image, layer.min, layer.max, layer.normalization)
        if layer.colormap is not None and len(image.shape) < 3:
            norm_image = bf.gray_scale_to_color_ramp(gray_scale=norm_image, colormap=layer.colormap,
                                                     output_8bit=False, min_colormap_cut=layer.min_colormap_cut,
                                                     max_colormap_cut=layer.max_colormap_cut)
        if rendered_image is None:
            rendered_image = norm_image
            continue
        active = norm_image
        background = rendered_image
        if np.nanmin(active) < 0 or np.nanmax(active) > 1:
            active = bf.scale_0_to_1(active)
        if np.nanmin(background) < 0 or np.nanmax(background) > 1:
            background = bf.scale_0_to_1(background)
        top = ref_blend_images(layer.blend_mode, active, background)
        rendered_image = ref_render_images(top, background, layer.opacity)
        if np.nanmin(rendered_image) < 0 or np.nanmax(rendered_image) > 1:
            warnings.warn("rvt.blend.BlenderCombination.render_all_images: Rendered image scale distorted")
    return rendered_image


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------

def _images(shape, seed, lo=0.0, hi=1.0, nan=True, dtype=np.float64):
    rng = np.random.default_rng(seed)
    img = rng.uniform(lo, hi, size=shape)
    img[..., 0, 0] = lo
    img[..., -1, -1] = hi
    if nan:
        img[..., 3:5, 7:9] = np.nan
    return img.astype(dtype)


def _same_call(func, ref_func, *args):
    """func and ref_func on deep copies of args: same result, same in-place effects on the arguments, same warnings,
    or the same exception type."""
    args_a = copy.deepcopy(args)
    args_b = copy.deepcopy(args)
    with warnings.catch_warnings(record=True) as w_a:
        warnings.simplefilter("always")
        try:
            out_a, exc_a = func(*args_a), None
        except Exception as exc:  # noqa: BLE001
            out_a, exc_a = None, type(exc)
    with warnings.catch_warnings(record=True) as w_b:
        warnings.simplefilter("always")
        try:
            out_b, exc_b = ref_func(*args_b), None
        except Exception as exc:  # noqa: BLE001
            out_b, exc_b = None, type(exc)
    assert exc_a == exc_b
    if exc_a is None:
        assert_same(out_a, out_b)
    for a, b in zip(args_a, args_b):
        if isinstance(a, np.ndarray):
            assert_same(a, b)
    # the same kinds of warnings (a blocked path may repeat a warning once per block)
    assert {(type(x.message), str(x.message)) for x in w_a} == {(type(x.message), str(x.message)) for x in w_b}
    return out_a


SHAPE = (250, 300)  # 3 row blocks of 32768 elements


# ----------------------------------------------------------------------------------------------------------------------
# blend_func
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("limits", [(0, 1), (-0.5, 0.5), (0.7, 1), (-15, 15), (0, 55), (0.25, 0.25001)])
def test_normalize_lin(dtype, limits):
    img = _images(SHAPE, 1, -20, 60, dtype=dtype)
    _same_call(bf.normalize_lin, ref_normalize_lin, img, *limits)
    _same_call(bf.normalize_lin, ref_normalize_lin, img[:, ::2], *limits)  # non-contiguous: whole-array code
    _same_call(bf.normalize_lin, ref_normalize_lin, np.nan_to_num(img).astype(np.int32), np.int64(limits[0]), limits[1] + 1)
    _same_call(bf.normalize_lin, ref_normalize_lin, np.stack([img, img[::-1], img[:, ::-1]]), *limits)
    _same_call(bf.normalize_lin, ref_normalize_lin, img, np.float32(limits[0]), np.float64(limits[1]))


@pytest.mark.parametrize("mode", ["overlay", "soft_light", "screen", "multiply"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blend_modes_2d(mode, dtype):
    active = _images(SHAPE, 2, dtype=dtype)
    background = _images(SHAPE, 3, dtype=dtype)
    background[100:102, :] = 0.5  # the branch boundary of overlay
    active[:, 50:52] = 0.5  # the branch boundary of soft light
    _same_call(lambda a, b: bf.blend_images(mode, a, b), lambda a, b: ref_blend_images(mode, a, b),
               active, background)
    # mixed dtypes and values outside 0..1 (soft light takes sqrt of negative values: invalid -> NaN + warning)
    _same_call(lambda a, b: bf.blend_images(mode, a, b), lambda a, b: ref_blend_images(mode, a, b),
               _images(SHAPE, 4, -0.5, 1.5, dtype=np.float32), _images(SHAPE, 5, -0.5, 1.5))


@pytest.mark.parametrize("mode", ["overlay", "soft_light", "screen", "multiply", "luminosity", "normal"])
@pytest.mark.parametrize("dims", [(3, 3), (3, 2), (2, 3)])
def test_blend_modes_rgb(mode, dims):
    active = _images((3,) + SHAPE if dims[0] == 3 else SHAPE, 6)
    background = _images((3,) + SHAPE if dims[1] == 3 else SHAPE, 7)
    _same_call(lambda a, b: bf.blend_images(mode, a, b), lambda a, b: ref_blend_images(mode, a, b),
               active, background)


def test_blend_luminosity_clipping():
    # luminosity shifts that push channels below 0 and above 1 (both clip_color corrections), float32 background
    active = _images((3,) + SHAPE, 8, -0.8, 1.8)
    background = _images((3,) + SHAPE, 9, 0.0, 1.0, dtype=np.float32)
    _same_call(bf.blend_luminosity, ref_blend_luminosity, active, background)
    _same_call(bf.blend_luminosity, ref_blend_luminosity, active[0], background)
    _same_call(bf.blend_luminosity, ref_blend_luminosity, active[0], background[0])


@pytest.mark.parametrize("mode", ["overlay", "soft_light"])
def test_blend_modes_raise_errstate(mode):
    # under a raising numpy error state both versions raise; background is modified the same way before the raise
    active = _images(SHAPE, 10, 0.0, 1.0, nan=False)
    background = _images(SHAPE, 11, -1.0, 1.0, nan=False)
    with np.errstate(invalid="raise"):
        _same_call(lambda a, b: bf.blend_images(mode, a, b), lambda a, b: ref_blend_images(mode, a, b),
                   active, background)


@pytest.mark.parametrize("opacity", [100, 90, 70, 50, 25, 0.3])
@pytest.mark.parametrize("dims", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_render_images(opacity, dims):
    active = _images((3,) + SHAPE if dims[0] == 3 else SHAPE, 12)
    background = _images((3,) + SHAPE if dims[1] == 3 else SHAPE, 13, -2.0, 3.0)  # scaled first
    _same_call(bf.render_images, ref_render_images, active, background, opacity)
    _same_call(bf.render_images, ref_render_images, background, active, opacity)


# ----------------------------------------------------------------------------------------------------------------------
# BlenderCombination.render_all_images (e3MSTP / e4MSTP / CRIM layer stacks)
# ----------------------------------------------------------------------------------------------------------------------

def _layer_stacks():
    dem = synthetic_dem(SHAPE, seed=14, relief=10.0, nan_patch=True)
    slope = rvt.vis.slope_aspect(dem, 0.5, 0.5)["slope"]
    rgb = np.stack([_images(SHAPE, 15), _images(SHAPE, 16), _images(SHAPE, 17)]).astype(np.float32)
    g1 = _images(SHAPE, 18, -0.8, 0.8, dtype=np.float32)
    g2 = _images(SHAPE, 19, 0.5, 1.0, dtype=np.float32)
    g3 = _images(SHAPE, 20, -20, 20, dtype=np.float32)
    L = rvt.blend.BlenderLayer
    return {
        "e3mstp": [L("slrm", "value", -0.5, 0.5, "screen", 25, image=g1),
                   L("crim_red", "value", 0, 1, "soft_light", 70, image=rgb[::-1].copy()),
                   L("mstp", "value", 0, 1, "normal", 100, image=rgb)],
        "e4mstp": [L("mstp", "value", 0, 1, "overlay", 90, image=rgb),
                   L("Comb svf", "value", -0.5, 0.5, "multiply", 25, image=g2),
                   L("Comb openness LD", "value", 0, 1, "multiply", 100, image=g1),
                   L("Slope gradient", "value", 0, 55, "normal", 100, colormap="Reds_r", min_colormap_cut=0,
                     max_colormap_cut=1, image=slope)],
        "crim": [L("Openness_Pos-Neg", "value", -28, 28, "overlay", 50, image=g3),
                 L("Openness_Pos-Neg", "value", -28, 28, "luminosity", 50, image=g3.copy()),
                 L("slope gradient rad", "value", 0, 0.8, "normal", 100, colormap="OrRd", min_colormap_cut=0,
                   max_colormap_cut=1, image=np.radians(slope))],
        "comb_svf": [L("Sky-view factor", "Value", 0.7, 1, "normal", 50, image=g2),
                     L("Sky-view factor", "Value", 0.9, 1, "normal", 100, image=g2[::-1].copy())],
        "gray_modes": [L("a", "value", 0, 1, "soft_light", 60, image=g2),
                       L("b", "perc", 2, 3, "overlay", 80, image=g3),
                       L("c", "value", -1, 1, "screen", 100, image=g1),
                       L("Openness - Negative", "value", 60, 110, "normal", 100, image=g3 + 90)],
        # opacity above 100 % extrapolates: rendered images outside 0..1 (warning, rescaled as background of the
        # next layer, range test of the rendered image reused)
        "distorted": [L("a", "value", 0, 1, "normal", 100, image=g2.copy()),
                      L("b", "value", 0, 1, "multiply", 160, image=rgb[1].copy()),
                      L("c", "value", -1, 1, "soft_light", 130, image=g1.copy()),
                      L("d", "value", 0, 1, "overlay", 150, image=rgb.copy()),
                      L("e", "value", 0, 1, "normal", 100, image=g2[::-1].copy())],
    }


@pytest.mark.parametrize("name", ["e3mstp", "e4mstp", "crim", "comb_svf", "gray_modes", "distorted"])
def test_render_all_images_matches_reference(name):
    layers = _layer_stacks()[name]

    def run_rvt(layer_list):
        comb = rvt.blend.BlenderCombination()
        for layer in layer_list:
            comb.add_layer(layer)
        return comb.render_all_images()

    out = _same_call(run_rvt, ref_render_layers, layers)
    assert np.isfinite(out).any()


def test_e3mstp_e4mstp_deterministic_and_accel_equals_numpy(run_both):
    # the full pipelines on a small DEM: two runs bitwise equal, accelerated path == numpy path (when numba exists)
    dem = synthetic_dem((48, 52), seed=21, relief=6.0)
    out_a = run_both(lambda: (rvt.blend.e3mstp(dem.copy(), 1.0, default=rvt.default.DefaultValues()),
                              rvt.blend.e4mstp(dem.copy(), 1.0, default=rvt.default.DefaultValues())))
    out_b = run_both(lambda: (rvt.blend.e3mstp(dem.copy(), 1.0, default=rvt.default.DefaultValues()),
                              rvt.blend.e4mstp(dem.copy(), 1.0, default=rvt.default.DefaultValues())))
    for i in range(2):
        assert_bitwise(out_a[0][i], out_a[1][i])
        assert_bitwise(out_a[0][i], out_b[0][i])
        assert_bitwise(out_a[1][i], out_b[1][i])


def test_e3mstp_e4mstp_deterministic():
    dem = synthetic_dem((40, 45), seed=22, relief=6.0)
    for func in (rvt.blend.e3mstp, rvt.blend.e4mstp):
        assert_bitwise(func(dem.copy(), 0.5, default=rvt.default.DefaultValues()),
                       func(dem.copy(), 0.5, default=rvt.default.DefaultValues()))
