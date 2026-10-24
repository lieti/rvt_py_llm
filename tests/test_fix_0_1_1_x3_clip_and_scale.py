"""
0.1.1 fix X3 (docs/KNOWN_ISSUES.md KI-19, KI-20, KI-21; review item 4, C-12): operands outside [0, 1] are clipped,
and the scale_* functions do what their docstrings say.

At v0.1.0 render_images (and the layer loop of BlenderCombination.render_all_images) rescaled an operand with any value
outside [0, 1] through scale_0_to_1, which filled NaN with the minimum in the caller's array (KI-20), returned values
outside [0, 1] unchanged (KI-19: the stretch limits were always 0 and 1) or raised ValueError for a narrow range
(KI-21). The reading of FIX_BRIEF_LLM.md section 1: render_images and render_all_images clip such an operand to [0, 1]
(NaN stays NaN) instead of rescaling it; scale_within_0_and_1, scale_strict_0_to_1 and scale_0_to_1 stretch with the
actual minimum and maximum, compare elementwise, restore NaN and do not modify their argument (the 8-bit branch / 255
and the snapping of values in (-0.01, 0) stay).

Expected values: hand-written arrays (dyadic values, exact in float32 and float64), the min-max stretch and the clip
written with numpy, the opacity mix of apply_opacity written out (``a * op + b * (1 - op)``, same order and dtypes).
blend_luminosity, which the fix does not change, builds the operand that leaves [0, 1] in the layer-loop test.
Synthetic data only.
"""

import warnings

import numpy as np
import pytest

import rvt.blend as rb
import rvt.blend_func as bf
from conftest import assert_same


def _clip(x):
    """Clip to [0, 1] with NaN kept, in the dtype of x."""
    return np.where(np.isnan(x), x, np.where(x < 0, 0, np.where(x > 1, 1, x))).astype(x.dtype)


def _mix(active, background, opacity):
    op = opacity / 100 if opacity > 1 else opacity
    return active * op + background * (1 - op)


def _stretch(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo)


# ----------------------------------------------------------------------------------------------------------------------
# render_images
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("opacity", [100, 75, 50, 0.25])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_render_images_clips_operands_outside_0_1_and_keeps_nan(dtype, opacity):
    a = np.array([[-0.5, 0.25, 1.5, np.nan, 0.75, 3.0]], dtype=dtype)
    b = np.array([[0.5, 2.0, 0.5, 0.5, np.nan, -1.0]], dtype=dtype)
    a0, b0 = a.copy(), b.copy()
    out = bf.render_images(a, b, opacity)
    assert out.dtype == dtype
    assert_same(out, _mix(_clip(a0), _clip(b0), opacity))
    assert np.isnan(out[0, 3]) and np.isnan(out[0, 4]) and not np.isnan(out[0, [0, 1, 2, 5]]).any()
    assert_same(a, a0)  # the caller's arrays are not modified
    assert_same(b, b0)


def test_render_images_clips_rgb_and_gray_operands():
    rng = np.random.default_rng(91)
    a = rng.uniform(-1.0, 2.0, size=(3, 6, 7))
    a[1, 2, 3] = np.nan
    b = rng.uniform(0.0, 1.0, size=(6, 7)).astype(np.float32)
    b[4, 4] = np.nan
    a0 = a.copy()
    out = bf.render_images(a, b, 60)
    assert out.shape == (3, 6, 7) and out.dtype == np.float64
    for i in range(3):
        assert_same(out[i], _mix(_clip(a0[i]), b, 60))
    assert_same(a, a0)
    # gray active outside [0, 1] over an RGB background
    out = bf.render_images(a0[0].astype(np.float32), np.clip(a0, 0, 1), 30)
    for i in range(3):
        assert_same(out[i], _mix(_clip(a0[0].astype(np.float32)), np.clip(a0[i], 0, 1), 30))


def test_render_images_operands_inside_0_1_are_used_as_they_are():
    a = np.array([[0.0, 0.25, 1.0, np.nan]])
    b = np.array([[1.0, 0.5, 0.0, 0.5]])
    assert_same(bf.render_images(a, b, 40), _mix(a, b, 40))


def _render_combination(layers):
    comb = rb.BlenderCombination()
    for i, (image, mode, opacity) in enumerate(layers):
        comb.create_layer(vis_method="layer {}".format(i), normalization="value", minimum=0, maximum=1,
                          blend_mode=mode, opacity=opacity, image=image)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return comb.render_all_images()


def test_render_all_images_clips_a_blend_result_outside_0_1_and_keeps_nan():
    # a luminosity blend lands slightly outside [0, 1] (rounding of the float32 luminosity); at v0.1.0 the rescale
    # through scale_within_0_and_1 filled its NaN pixels with the minimum, so the no-data pixels of the luminosity
    # layer were lost
    rng = np.random.default_rng(92)
    b = rng.uniform(0, 1, size=(3, 60, 80)).astype(np.float32)
    a = rng.uniform(0, 1, size=(60, 80)).astype(np.float32)
    c = rng.uniform(0, 1, size=(60, 80)).astype(np.float32)
    a[10:12, 20:23] = np.nan
    b[2, 40, 50] = np.nan
    c[0, 0] = np.nan
    top = bf.blend_luminosity(a, b)
    assert np.nanmin(top) < 0 or np.nanmax(top) > 1  # precondition: the blend result leaves [0, 1]
    rendered = _mix(_clip(top), b, 60)
    expected = _mix(c, rendered, 50)
    out = _render_combination([(c.copy(), "normal", 50), (a.copy(), "luminosity", 60), (b.copy(), "normal", 100)])
    assert_same(out, expected)
    nan_mask = np.isnan(a) | np.isnan(b).any(axis=0) | np.isnan(c)
    assert np.array_equal(np.isnan(out), np.broadcast_to(nan_mask, out.shape))


# ----------------------------------------------------------------------------------------------------------------------
# scale_within_0_and_1, scale_strict_0_to_1, scale_0_to_1
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scale_within_stretches_with_the_actual_min_and_max(dtype):
    img = np.array([[-1.0, 0.5, np.nan], [2.0, 3.0, 1.0]], dtype=dtype)
    img0 = img.copy()
    out = bf.scale_within_0_and_1(img)
    assert out.dtype == dtype
    assert_same(out, np.array([[0.0, 0.375, np.nan], [0.75, 1.0, 0.5]], dtype=dtype))
    assert_same(img, img0)


def test_scale_within_byte_like_branch_restores_nan():
    img = np.array([[40.0, np.nan], [200.0, 255.0]])
    img0 = img.copy()
    out = bf.scale_within_0_and_1(img)
    assert_same(out, img0 / 255.0)
    assert_same(img, img0)


def test_scale_within_inside_0_1_returns_the_argument():
    img = np.array([[0.0, 0.5, np.nan], [0.25, 1.0, 0.75]])
    assert bf.scale_within_0_and_1(img) is img


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scale_strict_compares_elementwise_and_restores_nan(dtype):
    img = np.array([[1.25, 1.5, np.nan], [1.375, 1.3125, 1.25]], dtype=dtype)
    img0 = img.copy()
    out = bf.scale_strict_0_to_1(img)
    assert out.dtype == dtype
    assert_same(out, np.array([[0.0, 1.0, np.nan], [0.5, 0.25, 0.0]], dtype=dtype))
    assert_same(img, img0)


def test_scale_strict_unit_range_and_constant_input():
    img = np.array([[0.0, 0.5], [np.nan, 1.0]])
    assert bf.scale_strict_0_to_1(img) is img
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = bf.scale_strict_0_to_1(np.full((2, 2), 3.0))
    assert np.isnan(out).all()  # 0 / 0


@pytest.mark.parametrize("img", [
    np.array([[-2.0, 3.0, np.nan, 0.5]]),  # wide range: scale_within_0_and_1
    np.array([[0.2, 0.3, np.nan, 0.25, 0.4]]),  # narrow range inside 0..1: scale_strict_0_to_1
    np.array([[1.2, 1.3, 1.25, np.nan, 1.4]]),  # narrow range above 1: scale_strict_0_to_1
], ids=["wide", "narrow_inside", "narrow_above"])
def test_scale_0_to_1_stretches_to_0_1_and_restores_nan(img):
    img0 = img.copy()
    out = bf.scale_0_to_1(img)
    assert_same(out, _stretch(img0))
    assert np.nanmin(out) == 0.0 and np.nanmax(out) == 1.0
    assert_same(img, img0)


def test_scale_0_to_1_unit_range_returns_the_argument():
    img = np.array([[0.0, 0.95], [np.nan, 0.5]])
    assert bf.scale_0_to_1(img) is img
