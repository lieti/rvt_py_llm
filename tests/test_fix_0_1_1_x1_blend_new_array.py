"""
0.1.1 fix X1 (docs/KNOWN_ISSUES.md KI-33, KI-34): blend_overlay and blend_soft_light return a new array.

At v0.1.0 both functions wrote their result into ``background`` and returned it. BlenderCombination.render_all_images
then mixed two arrays holding the same values, so the opacity of an overlay or soft light layer acted as 100 % (KI-33),
and an RGB active layer over a gray background blended band 1 onto the result of band 0 (KI-34). The reading of
FIX_BRIEF_LLM.md section 1: the same formulas in the same order, the same NaN behaviour (KI-31 stays), the dtype of
``background``, the result in a new array, the caller's arrays unchanged; the opacity mix is then
``o * B(a, b) + (1 - o) * b`` with ``b`` the original background, as for normal, multiply and screen.

Expected values: ``_old_overlay_in_place`` / ``_old_soft_light_in_place`` below, the whole-array code of v0.1.0
(``_blend_overlay_ref``, ``_blend_soft_light_ref``, unchanged since tag baseline) transcribed and applied to a copy of
the background; the opacity mix written out as in apply_opacity (``top * op + b * (1 - op)``, the same operations in
the same order and dtypes). Synthetic data only.
"""

import warnings

import numpy as np
import pytest

import rvt.blend as rb
import rvt.blend_func as bf
from conftest import assert_bitwise, assert_same


# ----------------------------------------------------------------------------------------------------------------------
# reference: the in-place code of v0.1.0, transcribed (writes into `background` and returns it)
# ----------------------------------------------------------------------------------------------------------------------

def _old_overlay_in_place(active, background):
    idx1 = np.where(background > 0.5)
    idx2 = np.where(background <= 0.5)
    background[idx1] = (1 - (1 - 2 * (background[idx1] - 0.5)) * (1 - active[idx1]))
    background[idx2] = ((2 * background[idx2]) * active[idx2])
    return background


def _old_soft_light_in_place(active, background):
    idx1 = np.where(active < 0.5)
    idx2 = np.where(active >= 0.5)
    background[idx1] = 2 * background[idx1] * active[idx1] + background[idx1] ** 2 * (1.0 - 2 * active[idx1])
    background[idx2] = 2 * background[idx2] * (1.0 - active[idx2]) + np.sqrt(background[idx2]) * (
            2 * active[idx2] - 1.0)
    return background


OLD = {"overlay": _old_overlay_in_place, "soft_light": _old_soft_light_in_place}
NEW = {"overlay": bf.blend_overlay, "soft_light": bf.blend_soft_light}
MODES = ["overlay", "soft_light"]


def _old_blend(mode, active, background):
    """The blend of v0.1.0 on a copy of the background: the values the fixed function must return."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # sqrt of negative values (soft light outside 0..1)
        return OLD[mode](active, np.array(background, copy=True))


def _images(shape, seed, lo=0.0, hi=1.0, dtype=np.float64, nan=True):
    """Seeded uniform values with the edge values 0, 0.5, 1 (branch boundaries) and, optionally, NaN pixels."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(lo, hi, size=shape)
    flat = img.reshape(-1)
    flat[:3] = [0.0, 0.5, 1.0]
    if nan:
        flat[5::97] = np.nan
    return img.astype(dtype)


def _mix(top, background, opacity):
    """The opacity mix of apply_opacity, written out: percent above 1, fraction otherwise."""
    op = opacity / 100 if opacity > 1 else opacity
    return top * op + background * (1 - op)


# ----------------------------------------------------------------------------------------------------------------------
# blend_overlay / blend_soft_light: new array, same bits, inputs unchanged
# ----------------------------------------------------------------------------------------------------------------------

# (130, 600) = 78,000 elements: several row blocks of the accelerated path; (7, 9): one block; values outside 0..1
CASES = {
    "blocks_f64": ((130, 600), np.float64, np.float64, 0.0, 1.0),
    "blocks_f32": ((130, 600), np.float32, np.float32, 0.0, 1.0),
    "small_f32_on_f64": ((7, 9), np.float32, np.float64, 0.0, 1.0),
    "f64_on_f32": ((40, 50), np.float64, np.float32, 0.0, 1.0),
    "outside_0_1": ((40, 50), np.float64, np.float64, -0.5, 1.5),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("mode", MODES)
def test_blend_returns_a_new_array_with_the_old_bits_and_keeps_the_inputs(mode, case):
    shape, dtype_a, dtype_b, lo, hi = CASES[case]
    a = _images(shape, 1, lo, hi, dtype_a)
    b = _images(shape, 2, lo, hi, dtype_b)
    a0, b0 = a.copy(), b.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = NEW[mode](a, b)
    assert out is not a and out is not b
    assert not np.shares_memory(out, a) and not np.shares_memory(out, b)
    assert out.dtype == dtype_b  # the dtype of the background, as the old in-place write
    assert_bitwise(out, _old_blend(mode, a0, b0))
    assert_bitwise(a, a0)
    assert_bitwise(b, b0)


@pytest.mark.parametrize("mode", MODES)
def test_blend_accepts_read_only_and_strided_backgrounds(mode):
    a = _images((60, 80), 3)
    b = _images((60, 160), 4)[:, ::2]  # a strided view
    b0 = b.copy()
    out = NEW[mode](a, b)
    assert_bitwise(out, _old_blend(mode, a, b0))
    assert_bitwise(b, b0)
    ro = b0.copy()
    ro.flags.writeable = False  # the old code raised here (assignment into a read-only array)
    assert_bitwise(NEW[mode](a, ro), _old_blend(mode, a, b0))


@pytest.mark.parametrize("mode", MODES)
def test_equation_blend_and_blend_images_keep_the_background(mode):
    a = _images((20, 30), 5)
    b = _images((20, 30), 6)
    b0 = b.copy()
    expected = _old_blend(mode, a, b0)
    for func in (bf.equation_blend, bf.blend_images, bf.blend_multi_dim_images):
        out = func(mode, a, b)
        assert_bitwise(out, expected)
        assert_bitwise(b, b0)


def test_known_issue_repro_overlay_then_render_images_mixes_with_the_original_background():
    # KI-33 repro: overlay of 0.25 on 0.25 is 0.125; at 30 % opacity 0.3 * 0.125 + 0.7 * 0.25 (0.2125)
    b = np.full((1, 2), 0.25)
    out = bf.blend_overlay(np.full((1, 2), 0.25), b)
    assert out is not b
    assert_same(b, np.full((1, 2), 0.25))
    assert_same(out, np.full((1, 2), 0.125))
    assert_same(bf.render_images(out, b, 30), _mix(np.full((1, 2), 0.125), np.full((1, 2), 0.25), 30))
    np.testing.assert_allclose(bf.render_images(out, b, 30), 0.2125, rtol=0, atol=1e-15)


# ----------------------------------------------------------------------------------------------------------------------
# KI-34: an RGB active layer over a gray background, band by band onto the original gray
# ----------------------------------------------------------------------------------------------------------------------

def test_known_issue_repro_rgb_over_gray_gives_the_same_value_in_every_band():
    a = np.full((3, 1, 1), 0.25)
    b = np.full((1, 1), 0.25)
    out = bf.blend_multi_dim_images("overlay", a, b)
    assert_same(out, np.full((3, 1, 1), 0.125))  # v0.1.0: 0.125, 0.0625, 0.03125
    assert_same(b, np.full((1, 1), 0.25))


@pytest.mark.parametrize("dtype_b", [np.float32, np.float64])
@pytest.mark.parametrize("mode", MODES)
def test_rgb_over_gray_blends_every_band_onto_the_original_gray(mode, dtype_b):
    a = np.stack([_images((50, 70), 10 + i) for i in range(3)])
    b = _images((50, 70), 13, dtype=dtype_b)
    b0 = b.copy()
    out = bf.blend_multi_dim_images(mode, a, b)
    assert out.shape == (3, 50, 70) and out.dtype == np.float64
    for i in range(3):
        assert_bitwise(out[i], _old_blend(mode, a[i], b0).astype(np.float64))
    assert_bitwise(b, b0)
    # band independence: three equal bands give three equal result bands
    same = bf.blend_multi_dim_images(mode, np.stack([a[0]] * 3), b)
    assert_bitwise(same[1], same[0])
    assert_bitwise(same[2], same[0])


@pytest.mark.parametrize("mode", MODES)
def test_gray_over_rgb_and_rgb_over_rgb_keep_the_background(mode):
    a = _images((30, 40), 20)
    b = np.stack([_images((30, 40), 21 + i) for i in range(3)])
    b0 = b.copy()
    out = bf.blend_multi_dim_images(mode, a, b)
    for i in range(3):
        assert_bitwise(out[i], _old_blend(mode, a, b0[i]))
    assert_bitwise(b, b0)
    a3 = np.stack([a, a[::-1], a[:, ::-1]])
    out = bf.blend_multi_dim_images(mode, a3, b)
    for i in range(3):
        assert_bitwise(out[i], _old_blend(mode, a3[i], b0[i]))
    assert_bitwise(b, b0)


# ----------------------------------------------------------------------------------------------------------------------
# BlenderCombination.render_all_images: the opacity of overlay / soft light layers
# ----------------------------------------------------------------------------------------------------------------------

def _render(layers):
    """layers: (image, blend_mode, opacity), top first; "value" normalization 0..1 (in-range float32 images keep
    their values)."""
    comb = rb.BlenderCombination()
    for i, (image, mode, opacity) in enumerate(layers):
        comb.create_layer(vis_method="layer {}".format(i), normalization="value", minimum=0, maximum=1,
                          blend_mode=mode, opacity=opacity, image=image)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return comb.render_all_images()


@pytest.mark.parametrize("opacity", [90, 70, 50, 25, 0])
@pytest.mark.parametrize("mode", MODES)
def test_render_all_images_mixes_overlay_and_soft_light_by_opacity(mode, opacity):
    a = _images((40, 50), 30, dtype=np.float32)
    b = _images((40, 50), 31, dtype=np.float32)
    out = _render([(a.copy(), mode, opacity), (b.copy(), "normal", 100)])
    expected = _mix(_old_blend(mode, a, b), b, opacity)
    assert out.dtype == np.float32
    assert_same(out, expected)


@pytest.mark.parametrize("mode", MODES)
def test_render_all_images_rgb_layer_over_gray_background(mode):
    # KI-34 through the combination: an RGB layer (3, H, W) at 50 % over a gray bottom layer
    a = np.stack([_images((30, 40), 40 + i, dtype=np.float32) for i in range(3)])
    b = _images((30, 40), 43, dtype=np.float32)
    out = _render([(a.copy(), mode, 50), (b.copy(), "normal", 100)])
    assert out.shape == (3, 30, 40) and out.dtype == np.float64
    for i in range(3):
        top = _old_blend(mode, a[i], b).astype(np.float64)  # the float64 target of blend_multi_dim_images
        assert_same(out[i], _mix(top, b, 50))


@pytest.mark.parametrize("mode", MODES)
def test_render_all_images_gray_layer_over_rgb_background(mode):
    # the CRIM case: a gray layer over an RGB rendered image
    a = _images((30, 40), 50, dtype=np.float32)
    b = np.stack([_images((30, 40), 51 + i, dtype=np.float32) for i in range(3)])
    out = _render([(a.copy(), mode, 50), (b.copy(), "normal", 100)])
    for i in range(3):
        top = _old_blend(mode, a, b[i]).astype(np.float64)
        assert_same(out[i], _mix(top, b[i], 50))


@pytest.mark.parametrize("mode", MODES)
def test_opacity_100_gives_the_bits_of_the_old_in_place_code(mode):
    # property: at 100 % the fix changes nothing. The old pipeline: the in-place blend writes into the background,
    # which is then also the second operand of the opacity mix (top * 1.0 + top * 0.0).
    for seed, shape in ((60, (40, 50)), (61, (130, 600))):
        a = _images(shape, seed, dtype=np.float32)
        b = _images(shape, seed + 100, dtype=np.float32)
        background = b.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            top = OLD[mode](a, background)
        assert top is background
        old = _mix(top, background, 100)
        assert_bitwise(_render([(a.copy(), mode, 100), (b.copy(), "normal", 100)]), old)
        assert_bitwise(bf.render_images(NEW[mode](a, b), b, 100), old)
