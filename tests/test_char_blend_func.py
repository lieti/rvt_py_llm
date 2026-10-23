"""
Characterization tests for rvt.blend_func: the public functions that tests/test_blend.py does not assert.

tests/test_blend.py compares the blocked blend kernels (normalize_lin, blend_overlay, blend_soft_light,
blend_luminosity, clip_color, blend_multi_dim_images, render_images) with the code of tag baseline-fixed. This file
adds, for every other public function of rvt.blend_func:

- analytic results (multiply by 1, screen formula, linear stretch of a known range, ...);
- properties (range [0, 1], shape (3, H, W), dtype, NaN mask, identity of the returned object, in-place effects);
- a plain scalar or numpy reference kept in this file (never produced by calling the function under test);
- pins of behaviour that looks wrong, marked "# pins current behaviour, KNOWN_ISSUES candidate".

Tolerances are stated where they are used. All inputs are small, seeded or analytic; no file is written.
"""

import warnings

import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap

import rvt.blend_func as bf
from conftest import assert_same


# ----------------------------------------------------------------------------------------------------------------------
# gray_scale_to_color_ramp, truncate_colormap
# ----------------------------------------------------------------------------------------------------------------------

# matplotlib samples a colormap on a 256-entry look-up table: the value of "gray" at x in [0, 1) is
# floor(256 * x) / 255, so |gray(x) - x| < 1 / 128 everywhere on [0, 1].
LUT_ATOL = 1.0 / 128.0


def _ramp_input():
    """Gray scale with the edge values 0 and 1, a mid value and one NaN."""
    return np.array([[0.0, 0.25, 0.5],
                     [0.75, 1.0, np.nan]])


def test_ramp_gray_float_output():
    gray = _ramp_input()
    out = bf.gray_scale_to_color_ramp(gray, "gray", output_8bit=False)
    assert out.shape == (3,) + gray.shape  # bands first, alpha dropped
    assert out.dtype == np.float64
    valid = ~np.isnan(gray)
    for band in range(3):  # "gray" has R == G == B == x (up to the look-up table step)
        np.testing.assert_allclose(out[band][valid], gray[valid], atol=LUT_ATOL, rtol=0)
    # matplotlib maps NaN to its "bad" color (0, 0, 0, 0): black, not NaN
    assert np.all(out[:, ~valid] == 0)
    assert not np.isnan(out).any()
    # 0 and 1 are the ends of the ramp exactly
    assert np.all(out[:, 0, 0] == 0.0)
    assert np.all(out[:, 1, 1] == 1.0)


def test_ramp_gray_8bit_output():
    gray = _ramp_input()
    out = bf.gray_scale_to_color_ramp(gray, "gray")  # defaults: output_8bit=True, alpha=False
    assert out.shape == (3,) + gray.shape
    assert out.dtype == np.uint8
    valid = ~np.isnan(gray)
    for band in range(3):
        # uint8(x * 255) truncates: within one count of 255 * x plus the look-up table step
        assert np.all(np.abs(out[band][valid].astype(float) - 255.0 * gray[valid]) <= 255.0 * LUT_ATOL + 1.0)
    assert np.all(out[:, 0, 0] == 0)
    assert np.all(out[:, 1, 1] == 255)
    assert np.all(out[:, ~valid] == 0)  # NaN pixels are set to 0 in all bands
    # 8-bit values are monotonic in the input for a monotonic colormap
    order = np.argsort(gray[valid])
    assert np.all(np.diff(out[0][valid][order].astype(int)) >= 0)


def test_ramp_alpha_band():
    gray = _ramp_input()
    out_f = bf.gray_scale_to_color_ramp(gray, "gray", alpha=True, output_8bit=False)
    assert out_f.shape == (4,) + gray.shape
    valid = ~np.isnan(gray)
    assert np.all(out_f[3][valid] == 1.0)  # opaque where defined
    assert np.all(out_f[3][~valid] == 0.0)  # transparent at NaN (matplotlib "bad" color)
    out_8 = bf.gray_scale_to_color_ramp(gray, "gray", alpha=True, output_8bit=True)
    assert out_8.shape == (4,) + gray.shape and out_8.dtype == np.uint8
    assert np.all(out_8[3][valid] == 255) and np.all(out_8[3][~valid] == 0)


def test_ramp_reversed_colormap_is_mirrored():
    # "gray_r" is "gray" read backwards: value(x) = 1 - value(1 - x) up to the look-up table step
    gray = np.linspace(0, 1, 11).reshape(1, 11)
    fwd = bf.gray_scale_to_color_ramp(gray, "gray", output_8bit=False)
    rev = bf.gray_scale_to_color_ramp(gray, "gray_r", output_8bit=False)
    np.testing.assert_allclose(rev[0, 0], 1.0 - fwd[0, 0], atol=2 * LUT_ATOL, rtol=0)
    assert rev[0, 0, 0] == 1.0 and rev[0, 0, -1] == 0.0


def test_ramp_values_above_one_use_the_top_color():
    gray = np.array([[1.0, 2.0, 50.0]])
    out = bf.gray_scale_to_color_ramp(gray, "gray", output_8bit=False)
    assert np.all(out[:, 0, :] == 1.0)  # matplotlib "over" color = last color of the ramp


def test_ramp_dtype_float32_input():
    gray = _ramp_input().astype(np.float32)
    out = bf.gray_scale_to_color_ramp(gray, "gray", output_8bit=False)
    assert out.dtype == np.float64  # matplotlib returns float64 RGBA for float input
    assert out.shape == (3,) + gray.shape


def test_ramp_colormap_cuts():
    gray = np.array([[0.0, 0.5, 1.0]])
    # keep the colors of "gray" between 0.25 and 0.75: the ramp now runs from about 0.25 to about 0.75
    out = bf.gray_scale_to_color_ramp(gray, "gray", min_colormap_cut=0.25, max_colormap_cut=0.75, output_8bit=False)
    assert abs(out[0, 0, 0] - 0.25) < 0.01
    assert abs(out[0, 0, 1] - 0.50) < 0.01
    assert abs(out[0, 0, 2] - 0.75) < 0.01
    # only the lower cut given: the upper cut defaults to 1
    out = bf.gray_scale_to_color_ramp(gray, "gray", min_colormap_cut=0.5, output_8bit=False)
    assert abs(out[0, 0, 0] - 0.5) < 0.01 and abs(out[0, 0, 2] - 1.0) < 0.01
    # only the upper cut given: the lower cut defaults to 0
    out = bf.gray_scale_to_color_ramp(gray, "gray", max_colormap_cut=0.5, output_8bit=False)
    assert abs(out[0, 0, 0] - 0.0) < 0.01 and abs(out[0, 0, 2] - 0.5) < 0.01


@pytest.mark.parametrize("cuts", [(-0.1, 0.5), (0.0, 1.1), (1.5, 2.0), (0.5, 0.5), (0.8, 0.2)])
def test_ramp_invalid_cuts_raise(cuts):
    with pytest.raises(Exception, match="gray_scale_to_color_ramp"):
        bf.gray_scale_to_color_ramp(np.zeros((2, 2)), "gray", min_colormap_cut=cuts[0], max_colormap_cut=cuts[1])


def test_ramp_unknown_colormap_raises():
    with pytest.raises(ValueError):
        bf.gray_scale_to_color_ramp(np.zeros((2, 2)), "no_such_colormap_name")


def test_truncate_colormap():
    from matplotlib import colormaps
    cm = colormaps.get_cmap("gray")
    # defaults: the whole ramp (resampled to 100 colors and interpolated back to 256)
    full = bf.truncate_colormap(cm)
    assert isinstance(full, LinearSegmentedColormap)
    x = np.linspace(0, 1, 21)
    np.testing.assert_allclose(full(x)[:, 0], x, atol=0.01, rtol=0)
    # range 0.2 - 0.8: ends at about 0.2 and 0.8, linear in between
    part = bf.truncate_colormap(cm, minval=0.2, maxval=0.8)
    np.testing.assert_allclose(part(x)[:, 0], 0.2 + 0.6 * x, atol=0.01, rtol=0)
    # the name encodes the source name and both limits with two decimals
    assert part.name == "trunc(gray,0.20,0.80)"
    # n = 2 colors: linear interpolation between the two end colors
    two = bf.truncate_colormap(cm, 0.0, 1.0, n=2)
    np.testing.assert_allclose(two(x)[:, 0], x, atol=0.01, rtol=0)
    # the source colormap is not changed
    assert cm.name == "gray" and abs(cm(0.0)[0]) == 0.0


# ----------------------------------------------------------------------------------------------------------------------
# normalize_lin (analytic complement), lin_cutoff_calc_from_perc, normalize_perc
# ----------------------------------------------------------------------------------------------------------------------

def test_normalize_lin_known_range():
    img = np.array([[-5.0, 0.0, 2.5], [5.0, 10.0, 15.0]])
    expected = np.array([[0.0, 0.0, 0.25], [0.5, 1.0, 1.0]], dtype=np.float32)  # (x - 0) / 10, clipped to [0, 1]
    out = bf.normalize_lin(img, 0, 10)
    assert_same(out, expected)
    # in-place effect: the argument is cut off to [minimum, maximum] (float arrays on the blocked path)
    assert_same(img, np.array([[0.0, 0.0, 2.5], [5.0, 10.0, 10.0]]))


def test_normalize_lin_nan_stays_nan_and_offset_invariance():
    img = np.array([[1.0, np.nan], [3.0, 5.0]])
    out = bf.normalize_lin(img.copy(), 1, 5)
    assert_same(out, np.array([[0.0, np.nan], [0.5, 1.0]], dtype=np.float32))
    out2 = bf.normalize_lin(img.copy() + 100.0, 101, 105)  # shifting data and limits together changes nothing
    assert_same(out2, out)


def test_normalize_lin_integer_input():
    img = np.array([[0, 5], [10, 20]], dtype=np.int32)
    out = bf.normalize_lin(img, 0, 10)
    assert_same(out, np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float32))


def test_lin_cutoff_calc_from_perc_known_values():
    img = np.arange(101, dtype=np.float64)  # the q-th percentile of 0..100 is q
    res = bf.lin_cutoff_calc_from_perc(img, 2, 3)
    assert sorted(res) == ["max_lin", "min_lin"]
    assert res["min_lin"] == 2.0 and res["max_lin"] == 97.0
    res = bf.lin_cutoff_calc_from_perc(img, 0, 0)
    assert res["min_lin"] == 0.0 and res["max_lin"] == 100.0
    # percentile interpolates linearly between the two neighbours: 0.5 % of 0..100 is 0.5
    assert bf.lin_cutoff_calc_from_perc(img, 0.5, 0.5)["min_lin"] == 0.5


def test_lin_cutoff_calc_from_perc_ignores_nan_and_does_not_modify():
    img = np.arange(101, dtype=np.float64)
    img[[3, 40, 77]] = np.nan  # NaN is skipped (np.nanpercentile)
    before = img.copy()
    res = bf.lin_cutoff_calc_from_perc(img, 10, 10)
    valid = np.sort(img[~np.isnan(img)])
    # plain reference: linear interpolation between order statistics (numpy default method), written out
    def percentile(sorted_values, q):
        pos = (len(sorted_values) - 1) * q / 100.0
        lo = int(np.floor(pos))
        hi = min(lo + 1, len(sorted_values) - 1)
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
    assert res["min_lin"] == pytest.approx(percentile(valid, 10), abs=1e-12)
    assert res["max_lin"] == pytest.approx(percentile(valid, 90), abs=1e-12)
    assert_same(img, before)


def test_lin_cutoff_calc_from_perc_equal_percentiles_fall_back_to_min_max():
    img = np.array([1.0] + [5.0] * 18 + [9.0])
    # 10 % and 90 % percentiles are both 5 -> the function returns the true minimum and maximum instead
    res = bf.lin_cutoff_calc_from_perc(img, 10, 10)
    assert res["min_lin"] == 1.0 and res["max_lin"] == 9.0
    # minimum + maximum == 100 is allowed (check is "> 100"); both percentiles are the median -> same fallback
    res = bf.lin_cutoff_calc_from_perc(np.arange(11.0), 50, 50)
    assert res["min_lin"] == 0.0 and res["max_lin"] == 10.0


@pytest.mark.parametrize("limits", [(-1, 5), (5, -1), (101, 0), (0, 101), (60, 41), (100, 100)])
def test_lin_cutoff_calc_from_perc_invalid_limits_raise(limits):
    with pytest.raises(Exception, match="lin_cutoff_calc_from_perc"):
        bf.lin_cutoff_calc_from_perc(np.arange(10.0), *limits)


def test_normalize_perc_known_values():
    img = np.arange(101, dtype=np.float64)
    out = bf.normalize_perc(img.copy(), 2, 3)
    # cut off at the 2 % and 97 % values (2 and 97), then stretch (x - 2) / 95; same float64 operations, cast to float32
    expected = (np.clip(img, 2.0, 97.0) - 2.0) / (97.0 - 2.0)
    assert_same(out, expected.astype(np.float32))
    assert out.min() == 0.0 and out.max() == 1.0
    # in place the argument is cut off to the percentile limits
    arg = img.copy()
    bf.normalize_perc(arg, 2, 3)
    assert arg.min() == 2.0 and arg.max() == 97.0
    # 0 % / 0 %: the plain min-max stretch
    assert_same(bf.normalize_perc(img.copy(), 0, 0), (img / 100.0).astype(np.float32))


# ----------------------------------------------------------------------------------------------------------------------
# advanced_normalization, normalize_image
# ----------------------------------------------------------------------------------------------------------------------

def test_advanced_normalization_value_and_perc():
    img = np.arange(101, dtype=np.float64)
    assert_same(bf.advanced_normalization(img.copy(), 20, 70, "value"),
                ((np.clip(img, 20, 70) - 20) / 50).astype(np.float32))
    assert_same(bf.advanced_normalization(img.copy(), 2, 3, "perc"), bf.normalize_perc(img.copy(), 2, 3))
    # the type is matched case-insensitively
    assert_same(bf.advanced_normalization(img.copy(), 20, 70, "VALUE"), bf.advanced_normalization(img.copy(), 20, 70, "value"))
    assert_same(bf.advanced_normalization(img.copy(), 2, 3, "Perc"), bf.advanced_normalization(img.copy(), 2, 3, "perc"))


def test_advanced_normalization_errors():
    img = np.arange(10, dtype=np.float64)
    with pytest.raises(Exception, match="min and max cannot be the same"):
        bf.advanced_normalization(img, 3, 3, "value")
    with pytest.raises(Exception, match="max can't be smaller than min"):
        bf.advanced_normalization(img, 5, 3, "value")
    with pytest.raises(Exception, match="Unknown normalization type"):
        bf.advanced_normalization(img, 0, 1, "percent")  # "percent" is translated only by normalize_image
    with pytest.raises(Exception, match="Unknown normalization type"):
        bf.advanced_normalization(img, 0, 1, "something")


def test_advanced_normalization_none_raises_attribute_error():
    # pins current behaviour, KNOWN_ISSUES candidate: the branch "normalization is None" (returns the image) comes
    # after normalization.lower(), so None raises AttributeError instead of returning the image.
    with pytest.raises(AttributeError):
        bf.advanced_normalization(np.arange(4.0), 0, 1, None)


def test_advanced_normalization_equality_checks_are_case_sensitive():
    img = np.array([0.0, 1.0, 2.0, 3.0])
    # pins current behaviour, KNOWN_ISSUES candidate: the min == max and min > max checks compare with "value" only;
    # "Value" skips them. min == max divides 0 by 0 after the cut-off -> all NaN (RuntimeWarning).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = bf.advanced_normalization(img.copy(), 1, 1, "Value")
    assert out.dtype == np.float32 and np.isnan(out).all()
    # min > max: cut-off collapses every value to min, the stretch gives (min - min) / negative = -0 -> all zero
    out = bf.advanced_normalization(img.copy(), 2, 1, "Value")
    assert out.dtype == np.float32 and np.all(out == 0)


def test_normalize_image_none_visualization():
    assert bf.normalize_image(None, np.arange(4.0), 0, 1, "value") is None


@pytest.mark.parametrize("vis", ["Hillshade", "hillshade", "Multiple directions hillshade", "mhs", "Sky-view factor",
                                 "Openness - Positive", "slrm", "something else"])
def test_normalize_image_no_inversion(vis):
    img = np.linspace(0.0, 10.0, 11).reshape(1, 11)
    out = bf.normalize_image(vis, img.copy(), 0, 10, "value")
    assert_same(out, (img / 10.0).astype(np.float32))


@pytest.mark.parametrize("vis", ["Slope gradient", "slope gradient", "SLOPE GRADIENT", "Openness - Negative",
                                 "openness - negative", "slp", "neg_opns"])
def test_normalize_image_inverts_slope_and_negative_openness(vis):
    img = np.linspace(0.0, 10.0, 11).reshape(1, 11)
    out = bf.normalize_image(vis, img.copy(), 0, 10, "value")
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 1.0 - img / 10.0, atol=1e-7, rtol=0)
    assert out[0, 0] == 1.0 and out[0, -1] == 0.0  # steep = dark


def test_normalize_image_abbreviation_inversion_is_case_sensitive():
    img = np.linspace(0.0, 10.0, 11).reshape(1, 11)
    # pins current behaviour, KNOWN_ISSUES candidate: the abbreviations "slp" and "neg_opns" are compared without
    # lower(); "SLP" is not inverted while "Slope gradient" and "SLOPE GRADIENT" are.
    out = bf.normalize_image("SLP", img.copy(), 0, 10, "value")
    assert_same(out, (img / 10.0).astype(np.float32))


def test_normalize_image_percent_is_perc():
    img = np.arange(101, dtype=np.float64)
    assert_same(bf.normalize_image("Hillshade", img.copy(), 2, 3, "percent"),
                bf.normalize_perc(img.copy(), 2, 3))
    assert_same(bf.normalize_image("Hillshade", img.copy(), 2, 3, "perc"),
                bf.normalize_perc(img.copy(), 2, 3))


def test_normalize_image_range_nan_and_dtype():
    rng = np.random.default_rng(5)
    img = rng.uniform(-30, 90, size=(20, 25)).astype(np.float32)
    img[4:6, 7:9] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "unexpected values" warning for a normalized image
        out = bf.normalize_image("Hillshade", img.copy(), 0, 60, "value")
    assert out.dtype == np.float32 and out.shape == img.shape
    assert np.array_equal(np.isnan(out), np.isnan(img))
    assert np.nanmin(out) >= 0 and np.nanmax(out) <= 1
    # plain reference of the stretch: np.clip, then (x - 0) / 60, cast to float32; one float32 ulp at 1 is 6e-8
    ref = np.clip(img, 0, 60)
    ref = ((ref - 0) / (60 - 0)).astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=1e-7, rtol=0, equal_nan=True)


def test_normalize_image_three_band_image():
    # multiple directions hillshade is a (3, H, W) array; every band is normalized with the same limits
    rng = np.random.default_rng(6)
    img = rng.uniform(0, 255, size=(3, 8, 9))
    out = bf.normalize_image("Multiple directions hillshade", img.copy(), 0, 255, "value")
    assert out.shape == (3, 8, 9) and out.dtype == np.float32
    np.testing.assert_allclose(out, img / 255.0, atol=1e-7, rtol=0)


def test_normalize_image_wrong_normalization_raises():
    with pytest.raises(Exception, match="Unknown normalization type"):
        bf.normalize_image("Hillshade", np.arange(4.0), 0, 1, "bad")


# ----------------------------------------------------------------------------------------------------------------------
# scale_within_0_and_1, scale_strict_0_to_1, scale_0_to_1
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scale_within_inside_range_returns_same_object(dtype):
    img = np.array([[0.0, 0.5, np.nan], [0.25, 1.0, 0.75]], dtype=dtype)
    out = bf.scale_within_0_and_1(img)
    assert out is img  # nothing to do: returned unchanged, NaN included


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scale_within_bounds_are_always_0_and_1(dtype):
    img = np.array([[-1.0, 0.5], [2.0, 3.0]], dtype=dtype)
    out = bf.scale_within_0_and_1(img.copy())
    # pins current behaviour, KNOWN_ISSUES candidate: np.array(0, actual_min) passes the minimum as the *dtype*
    # argument, so norm_min_value is always 0 and norm_max_value is always 1; (x - 0) / (1 - 0) leaves the values
    # unchanged instead of stretching them to 0..1.
    assert out.dtype == dtype
    assert_same(out, img)
    assert out is not img  # a new array, the argument is only touched by the NaN fill


def test_scale_within_snaps_small_negative_values_to_zero():
    img = np.array([[-0.005, 0.5], [1.5, 0.25]])
    out = bf.scale_within_0_and_1(img.copy())
    # values in (-0.01, 0) become 0 (the stretch itself is the identity, see above)
    assert_same(out, np.array([[0.0, 0.5], [1.5, 0.25]]))
    out = bf.scale_within_0_and_1(np.array([[-0.02, 0.5], [1.5, 0.25]]))
    assert out[0, 0] == -0.02  # below -0.01 the whole snapping step is skipped


def test_scale_within_divides_by_255_for_byte_like_data():
    img = np.array([[40.0, 100.0], [200.0, 255.0]])
    out = bf.scale_within_0_and_1(img.copy())
    assert_same(out, img / 255.0)  # 1 < max <= 255, max - min > 30, min > 0
    # float32 input: float32 / 255 stays float32
    out32 = bf.scale_within_0_and_1(img.astype(np.float32))
    assert out32.dtype == np.float32
    np.testing.assert_allclose(out32, img / 255.0, rtol=1e-6, atol=0)
    # not applied when min == 0 or when the span is at most 30: the identity stretch is used
    assert_same(bf.scale_within_0_and_1(np.array([[0.0, 100.0]])), np.array([[0.0, 100.0]]))
    assert_same(bf.scale_within_0_and_1(np.array([[10.0, 40.0]])), np.array([[10.0, 40.0]]))
    assert_same(bf.scale_within_0_and_1(np.array([[10.0, 41.0]])), np.array([[10.0, 41.0]]) / 255.0)
    # max above 255: not divided
    assert_same(bf.scale_within_0_and_1(np.array([[40.0, 300.0]])), np.array([[40.0, 300.0]]))


def test_scale_within_fills_nan_with_minimum_in_the_argument():
    img = np.array([[-1.0, np.nan], [2.0, 0.5]])
    out = bf.scale_within_0_and_1(img)
    # pins current behaviour, KNOWN_ISSUES candidate: NaN pixels are overwritten with the minimum *in the caller's
    # array* and the NaN mask of the output is lost.
    assert_same(img, np.array([[-1.0, -1.0], [2.0, 0.5]]))
    assert_same(out, np.array([[-1.0, -1.0], [2.0, 0.5]]))


def test_scale_strict_early_return_for_0_to_1():
    img = np.array([[0.0, 0.5], [np.nan, 1.0]])
    assert bf.scale_strict_0_to_1(img) is img


def test_scale_strict_raises_value_error_for_arrays():
    img = np.array([[1.0, 2.0], [3.0, np.nan]])
    # pins current behaviour, KNOWN_ISSUES candidate: `scaled[0 > scaled > -0.01] = 0` is a chained comparison; for
    # an array with more than one element it raises ValueError after the NaN fill, so the function never returns
    # for such input. NaN was already replaced by 0 in the caller's array.
    with pytest.raises(ValueError, match="truth value"):
        bf.scale_strict_0_to_1(img)
    assert_same(img, np.array([[1.0, 2.0], [3.0, 0.0]]))


def test_scale_strict_constant_input_gives_all_nan():
    # a constant array has max - min == 0: 0 / 0 = NaN everywhere; np.nanmin warns (all-NaN) and returns NaN, whose
    # comparison with -0.01 is False, so the chained comparison is not evaluated and the NaN array is returned
    # pins current behaviour, KNOWN_ISSUES candidate
    img = np.full((2, 2), 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = bf.scale_strict_0_to_1(img)
    assert np.isnan(out).all() and out.shape == (2, 2)


def test_scale_0_to_1_branch_unchanged_for_unit_range():
    # max in (0.9, 1] and min == 0: returned as is (same object), NaN kept
    img = np.array([[0.0, 0.95], [np.nan, 0.5]])
    assert bf.scale_0_to_1(img) is img
    img = np.array([[0.0, 1.0]])
    assert bf.scale_0_to_1(img) is img


def test_scale_0_to_1_wide_range_goes_to_scale_within():
    img = np.array([[40.0, 100.0], [200.0, 255.0]])
    assert_same(bf.scale_0_to_1(img.copy()), img / 255.0)
    # values already inside 0..1 but with a span above 0.3: unchanged
    img = np.array([[0.1, 0.9]])
    assert bf.scale_0_to_1(img) is img
    # out of range, span above 0.3: identity (the stretch bounds of scale_within_0_and_1 are always 0 and 1)
    # pins current behaviour, KNOWN_ISSUES candidate
    img = np.array([[-2.0, 3.0]])
    assert_same(bf.scale_0_to_1(img.copy()), img)


def test_scale_0_to_1_narrow_range_goes_to_scale_strict():
    # span <= 0.3 and not the 0..1 case: scale_strict_0_to_1 is used and raises for arrays
    # pins current behaviour, KNOWN_ISSUES candidate
    with pytest.raises(ValueError, match="truth value"):
        bf.scale_0_to_1(np.array([[0.2, 0.3], [0.25, 0.4]]))
    with pytest.raises(ValueError, match="truth value"):
        bf.scale_0_to_1(np.array([[1.2, 1.3], [1.25, 1.4]]))


# ----------------------------------------------------------------------------------------------------------------------
# apply_opacity
# ----------------------------------------------------------------------------------------------------------------------

def test_apply_opacity_percent_and_fraction():
    a = np.array([[0.0, 1.0], [0.5, 0.25]])
    b = np.array([[1.0, 0.0], [0.5, 0.75]])
    assert_same(bf.apply_opacity(a, b, 100), a)  # 100 % -> the active image
    assert_same(bf.apply_opacity(a, b, 0), b)  # 0 -> the background
    assert_same(bf.apply_opacity(a, b, 50), 0.5 * a + 0.5 * b)  # 0.5 is exact in binary
    assert_same(bf.apply_opacity(a, b, 25), 0.25 * a + 0.75 * b)
    # values up to 1 are fractions, values above 1 are percent
    assert_same(bf.apply_opacity(a, b, 0.5), bf.apply_opacity(a, b, 50))
    assert_same(bf.apply_opacity(a, b, 0.25), bf.apply_opacity(a, b, 25))


def test_apply_opacity_one_means_full_opacity():
    a = np.array([[0.2, 0.8]])
    b = np.array([[0.6, 0.1]])
    # pins current behaviour, KNOWN_ISSUES candidate: the limit is "opacity > 1", so opacity = 1 is the fraction 1.0
    # (100 %), not 1 %; there is no way to ask for 1 % opacity.
    assert_same(bf.apply_opacity(a, b, 1), a)
    assert_same(bf.apply_opacity(a, b, 1.5), a * 0.015 + b * (1 - 0.015))


def test_apply_opacity_extrapolates_above_100_percent():
    a = np.array([[1.0, 0.5]])
    b = np.array([[0.5, 0.5]])
    out = bf.apply_opacity(a, b, 150)  # a * 1.5 + b * (-0.5): outside 0..1 is not clipped here
    assert_same(out, np.array([[1.25, 0.5]]))


def test_apply_opacity_nan_and_dtype():
    a = np.array([[np.nan, 1.0]], dtype=np.float32)
    b = np.array([[0.5, np.nan]], dtype=np.float32)
    out = bf.apply_opacity(a, b, 50)
    assert np.isnan(out).all()
    # opacity 0 still propagates NaN of the active image (NaN * 0 = NaN)
    assert np.isnan(bf.apply_opacity(a, b, 0)[0, 0])
    # float32 inputs, Python scalar opacity: float32 or float64 depending on numpy promotion; the result is within 1e-7
    a2 = np.array([[0.2]], dtype=np.float32)
    b2 = np.array([[0.6]], dtype=np.float32)
    np.testing.assert_allclose(bf.apply_opacity(a2, b2, 50), [[0.4]], atol=1e-7, rtol=0)


# ----------------------------------------------------------------------------------------------------------------------
# cut_off_normalize
# ----------------------------------------------------------------------------------------------------------------------

def test_cut_off_normalize_value_mode():
    img = np.array([[-5.0, 0.0, 5.0], [10.0, 15.0, np.nan]])
    out = bf.cut_off_normalize(img.copy(), "value", 0, 10)
    assert_same(out, np.array([[0.0, 0.0, 0.5], [1.0, 1.0, np.nan]], dtype=np.float32))
    # bool_norm False: only the cut-off, in place, the same object is returned
    arg = img.copy()
    out = bf.cut_off_normalize(arg, "value", 0, 10, bool_norm=False)
    assert out is arg
    assert_same(arg, np.array([[0.0, 0.0, 5.0], [10.0, 10.0, np.nan]]))


def test_cut_off_normalize_value_defaults_to_data_range():
    img = np.array([[2.0, 4.0], [6.0, 10.0]])
    out = bf.cut_off_normalize(img.copy(), "value")  # min and max of the image: plain min-max stretch
    assert_same(out, np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32))
    # only the maximum given: the minimum is the data minimum
    out = bf.cut_off_normalize(img.copy(), "value", cutoff_max=6.0)
    assert_same(out, np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float32))
    # only the minimum given: the maximum is the data maximum
    out = bf.cut_off_normalize(img.copy(), "value", cutoff_min=6.0)
    assert_same(out, np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32))


@pytest.mark.parametrize("mode", ["perc", "percent", "Perc"])
def test_cut_off_normalize_percent_mode(mode):
    img = np.arange(101, dtype=np.float64)
    expected = ((np.clip(img, 2, 97) - 2) / 95).astype(np.float32)
    assert_same(bf.cut_off_normalize(img.copy(), mode, 2, 3), expected)
    # cut-off only: values are cut at the 2 % / 97 % values, not rescaled
    arg = img.copy()
    out = bf.cut_off_normalize(arg, mode, 2, 3, bool_norm=False)
    assert out is arg
    assert_same(arg, np.clip(img, 2, 97))
    # no limits given: 0 % and 0 % -> the data range
    assert_same(bf.cut_off_normalize(img.copy(), mode), (img / 100.0).astype(np.float32))
    assert_same(bf.cut_off_normalize(img.copy(), mode, bool_norm=False), img)


def test_cut_off_normalize_value_errors():
    img = np.arange(10.0)
    with pytest.raises(Exception, match="min and max cannot be the same"):
        bf.cut_off_normalize(img, "value", 3, 3)
    with pytest.raises(Exception, match="max can't be smaller than min"):
        bf.cut_off_normalize(img, "value", 5, 3)
    # pins current behaviour, KNOWN_ISSUES candidate: both checks compare mode with "value" case-sensitively; "Value"
    # passes both and then divides by zero (min == max) or reverses the limits.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.isnan(bf.cut_off_normalize(img.copy(), "Value", 3, 3)).all()
    # perc mode has no min/max check (the percentile function checks the sum instead)
    with pytest.raises(Exception, match="lin_cutoff_calc_from_perc"):
        bf.cut_off_normalize(img, "perc", 60, 50)


def test_cut_off_normalize_unknown_mode_returns_image():
    img = np.arange(6.0).reshape(2, 3)
    arg = img.copy()
    # pins current behaviour, KNOWN_ISSUES candidate: an unknown mode raises nothing and returns the argument as is
    assert bf.cut_off_normalize(arg, "bogus", 0, 1) is arg
    assert_same(arg, img)


# ----------------------------------------------------------------------------------------------------------------------
# lum, channel_min, channel_max, matrix_eq_*, clip_color
# ----------------------------------------------------------------------------------------------------------------------

def test_lum_weights_and_passthrough():
    rgb = np.zeros((3, 2, 2))
    rgb[0, 0, 0] = 1.0  # pure red
    rgb[1, 0, 1] = 1.0  # pure green
    rgb[2, 1, 0] = 1.0  # pure blue
    rgb[:, 1, 1] = 1.0  # white
    out = bf.lum(rgb)
    assert out.shape == (2, 2) and out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.3, 0.59], [0.11, 1.0]], atol=1e-7, rtol=0)  # float32 rounding of the weights
    # 2D arrays are returned as they are (same object)
    gray = np.array([[0.1, np.nan]])
    assert bf.lum(gray) is gray


def test_lum_random_matches_reference_and_keeps_nan():
    rng = np.random.default_rng(7)
    rgb = rng.uniform(-0.5, 1.5, size=(3, 6, 7))
    rgb[1, 2, 3] = np.nan
    out = bf.lum(rgb)
    ref = 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    np.testing.assert_allclose(out, ref, atol=2e-7, rtol=1e-6, equal_nan=True)  # float32 result of a float64 sum
    assert np.isnan(out[2, 3]) and np.isnan(out).sum() == 1
    out32 = bf.lum(rgb.astype(np.float32))
    assert out32.dtype == np.float32


def test_channel_min_max_match_numpy_without_nan():
    rng = np.random.default_rng(8)
    r, g, b = (rng.uniform(-1, 2, size=(9, 11)) for _ in range(3))
    g[0, 0] = r[0, 0]  # ties
    b[1, 1] = g[1, 1] = r[1, 1]
    lo = bf.channel_min(r, g, b)
    hi = bf.channel_max(r, g, b)
    assert_same(lo, np.minimum(np.minimum(r, g), b))
    assert_same(hi, np.maximum(np.maximum(r, g), b))
    # inputs are not modified
    r0, g0, b0 = r.copy(), g.copy(), b.copy()
    bf.channel_min(r, g, b)
    bf.channel_max(r, g, b)
    assert_same(r, r0)
    assert_same(g, g0)
    assert_same(b, b0)
    assert lo is not r and hi is not r


def test_channel_min_max_dtype_and_edge_values():
    r = np.array([[0.0, 1.0, 0.5]], dtype=np.float32)
    g = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    b = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    assert_same(bf.channel_min(r, g, b), np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert_same(bf.channel_max(r, g, b), np.array([[1.0, 1.0, 0.5]], dtype=np.float32))
    # integer channels: r * 1.0 makes the result float64
    ri = np.array([[1, 2]])
    out = bf.channel_min(ri, ri + 1, ri - 1)
    assert out.dtype == np.float64 and list(out[0]) == [0.0, 1.0]


def test_channel_min_max_nan_handling():
    # pins current behaviour, KNOWN_ISSUES candidate: the minimum starts at r and is replaced where g < min or b < min;
    # a comparison with NaN is False, so NaN in r propagates and NaN in g or b is ignored (np.minimum would give NaN).
    nan = np.nan
    r = np.array([[nan, 0.5, 0.5]])
    g = np.array([[0.1, nan, 0.9]])
    b = np.array([[0.2, 0.3, nan]])
    lo = bf.channel_min(r, g, b)
    hi = bf.channel_max(r, g, b)
    assert np.isnan(lo[0, 0]) and lo[0, 1] == 0.3 and lo[0, 2] == 0.5
    assert np.isnan(hi[0, 0]) and hi[0, 1] == 0.5 and hi[0, 2] == 0.9


def test_matrix_eq_min_lt_zero_formula():
    # r' = L + (r - L) * L / (L - min), applied only at idx; all values are dyadic, so the results are exact
    r = np.array([[-0.25, 0.75, 0.5]])
    lum_c = np.full((1, 3), 0.25)
    min_c = np.array([[-0.25, -0.25, 0.0]])
    idx = np.where(min_c < 0)
    out = bf.matrix_eq_min_lt_zero(r, idx, lum_c, min_c)
    assert out is r  # in place
    assert_same(r, np.array([[0.0, 0.5, 0.5]]))  # 0.25 + (-0.5 * 0.25) / 0.5 = 0 ; 0.25 + (0.5 * 0.25) / 0.5 = 0.5


def test_matrix_eq_max_gt_one_formula():
    # r' = L + (r - L) * (1 - L) / (max - L)
    r = np.array([[1.5, 0.25, 0.75]])
    lum_c = np.full((1, 3), 0.5)
    max_c = np.array([[1.5, 1.5, 1.0]])
    idx = np.where(max_c > 1)
    out = bf.matrix_eq_max_gt_one(r, idx, lum_c, max_c)
    assert out is r
    assert_same(r, np.array([[1.0, 0.375, 0.75]]))  # 0.5 + 1.0 * 0.5 / 1.0 ; 0.5 + (-0.25 * 0.5) / 1.0


def _clip_color_scalar(c_rgb, min_c=None, max_c=None):
    """Plain scalar reference of the clip_color formulas (W3C compositing spec, ClipColor), one pixel at a time,
    including the order used by rvt: the max correction uses the max channel computed *before* the min correction."""
    r, g, b = (float(v) for v in c_rgb)
    lum_c = 0.3 * r + 0.59 * g + 0.11 * b
    if min_c is None and max_c is None:
        min_c = min(r, g, b)
        max_c = max(r, g, b)
    if min_c < 0:
        r, g, b = (lum_c + (v - lum_c) * lum_c / (lum_c - min_c) for v in (r, g, b))
    if max_c > 1:
        r, g, b = (lum_c + (v - lum_c) * (1.0 - lum_c) / (max_c - lum_c) for v in (r, g, b))
    return r, g, b


def test_clip_color_inside_gamut_is_unchanged():
    c = np.zeros((3, 2, 3))
    c[0] = [[0.0, 0.125, 1.0], [0.5, 0.25, 0.75]]
    c[1] = [[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]]
    c[2] = [[0.0, 0.875, 1.0], [1.0, 0.0, 0.25]]  # 0, 1 and dyadic values are exact in float32
    out = bf.clip_color(c.copy())
    assert out.shape == c.shape and out.dtype == np.float64
    assert_same(out, c)


def test_clip_color_out_of_gamut_matches_scalar_reference():
    rng = np.random.default_rng(9)
    c = rng.uniform(-0.4, 1.4, size=(3, 5, 6))
    c[:, 0, 0] = [-0.2, 0.5, 1.4]  # both corrections at one pixel
    c[:, 0, 1] = [0.3, 0.4, 0.5]  # inside the gamut
    out = bf.clip_color(c.copy())
    ref = np.empty_like(c)
    for i in range(5):
        for j in range(6):
            ref[:, i, j] = _clip_color_scalar(c[:, i, j])
    # float32 channels and luminosity in the code, float64 here: 1e-4 covers the division by (L - min) for the
    # pixels drawn above (L - min >= 0.05 with probability one for this seed, checked by the assertion below)
    lum_c = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    assert np.min(lum_c - c.min(axis=0)) > 0.02
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=0)


def test_clip_color_corrections_preserve_luminosity_and_land_in_gamut():
    # one channel below 0 only: exact algebra gives lum' = lum and min' = 0
    c = np.zeros((3, 1, 3))
    c[:, 0, 0] = [-0.2, 0.6, 0.7]
    c[:, 0, 1] = [0.3, -0.1, 0.9]  # all three pixels have a luminosity above 0, the condition of the algebra
    c[:, 0, 2] = [0.4, 0.5, -0.1]
    lum_in = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    out = bf.clip_color(c.copy())
    lum_out = 0.3 * out[0] + 0.59 * out[1] + 0.11 * out[2]
    np.testing.assert_allclose(lum_out, lum_in, atol=1e-6, rtol=0)  # float32 channels
    assert out.min() > -1e-6
    np.testing.assert_allclose(out.min(axis=0), 0.0, atol=1e-6, rtol=0)
    # one channel above 1 only: max' = 1
    c = np.zeros((3, 1, 2))
    c[:, 0, 0] = [1.3, 0.2, 0.4]
    c[:, 0, 1] = [0.5, 0.4, 1.8]
    lum_in = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    out = bf.clip_color(c.copy())
    np.testing.assert_allclose(0.3 * out[0] + 0.59 * out[1] + 0.11 * out[2], lum_in, atol=1e-6, rtol=0)
    np.testing.assert_allclose(out.max(axis=0), 1.0, atol=1e-6, rtol=0)


def test_clip_color_explicit_limits_replace_the_channel_extremes():
    c = np.zeros((3, 1, 1))
    c[:, 0, 0] = [-0.25, 0.5, 0.5]
    # min_c = 0 given: no minimum correction even though a channel is negative
    out = bf.clip_color(c.copy(), np.zeros((1, 1)), np.zeros((1, 1)))
    assert_same(out, c)
    # max_c = 2 given (the real maximum is 0.5): the max correction runs with max_c = 2 and L = lum
    lum_c = float(np.float32(0.3 * -0.25 + 0.59 * 0.5 + 0.11 * 0.5))
    out = bf.clip_color(c.copy(), np.zeros((1, 1)), np.full((1, 1), 2.0))
    expected = [lum_c + (v - lum_c) * (1.0 - lum_c) / (2.0 - lum_c) for v in (-0.25, 0.5, 0.5)]
    np.testing.assert_allclose(out[:, 0, 0], expected, atol=1e-6, rtol=0)


def test_clip_color_only_one_limit_given_raises():
    c = np.full((3, 2, 2), 0.5)
    # pins current behaviour, KNOWN_ISSUES candidate: min_c and max_c are computed only when both are None; with one
    # of them given the other stays None and the comparison None < 0 raises TypeError.
    with pytest.raises(TypeError):
        bf.clip_color(c, min_c=np.zeros((2, 2)))
    with pytest.raises(TypeError):
        bf.clip_color(c, max_c=np.ones((2, 2)))


def test_clip_color_nan_and_gray_negative_pixel():
    c = np.zeros((3, 1, 3))
    c[:, 0, 0] = [np.nan, 0.5, 0.5]  # NaN pixel stays NaN in the bands that held it
    c[:, 0, 1] = [-0.125, -0.125, -0.125]  # neutral negative gray: L == min -> 0 / 0
    c[:, 0, 2] = [0.5, 0.5, 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = bf.clip_color(c.copy())
    assert np.isnan(out[0, 0, 0])
    # pins current behaviour, KNOWN_ISSUES candidate: for a neutral gray below 0, lum == min and the correction divides
    # by zero, so the pixel becomes NaN in all three bands instead of 0.
    assert np.isnan(out[:, 0, 1]).all()
    assert np.all(out[:, 0, 2] == 0.5)


# ----------------------------------------------------------------------------------------------------------------------
# blend modes on edge values (0, 1, NaN)
# ----------------------------------------------------------------------------------------------------------------------

EDGE = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
# all pairs (active, background) of the dyadic edge values, 25 pixels
ACT = np.repeat(EDGE, EDGE.size).reshape(EDGE.size, EDGE.size)
BAC = np.tile(EDGE, EDGE.size).reshape(EDGE.size, EDGE.size)


def _screen(a, b):
    return 1 - (1 - a) * (1 - b)


def _overlay(a, b):  # background decides the branch
    return np.where(b > 0.5, 1 - 2 * (1 - b) * (1 - a), 2 * a * b)


def _soft_light(a, b):  # active decides the branch
    return np.where(a < 0.5, 2 * b * a + b ** 2 * (1 - 2 * a), 2 * b * (1 - a) + np.sqrt(b) * (2 * a - 1))


def test_blend_screen_multiply_analytic():
    assert_same(bf.blend_screen(ACT, BAC), _screen(ACT, BAC))
    assert_same(bf.blend_multiply(ACT, BAC), ACT * BAC)
    ones, zeros = np.ones_like(EDGE), np.zeros_like(EDGE)
    assert_same(bf.blend_multiply(EDGE, ones), EDGE)  # 1 is neutral
    assert_same(bf.blend_multiply(EDGE, zeros), zeros)  # 0 is absorbing
    assert_same(bf.blend_screen(EDGE, zeros), EDGE)  # 0 is neutral
    assert_same(bf.blend_screen(EDGE, ones), ones)  # 1 is absorbing
    # symmetric in its arguments
    assert_same(bf.blend_screen(ACT, BAC), bf.blend_screen(BAC, ACT))
    assert_same(bf.blend_multiply(ACT, BAC), bf.blend_multiply(BAC, ACT))
    # NaN in either image gives NaN at that pixel only
    a = np.array([np.nan, 0.5, 0.5])
    b = np.array([0.5, np.nan, 0.5])
    for func in (bf.blend_screen, bf.blend_multiply):
        out = func(a, b)
        assert np.isnan(out[:2]).all() and not np.isnan(out[2])


def test_blend_normal_returns_active():
    a = np.array([[0.1, np.nan]])
    b = np.array([[0.9, 0.9]])
    assert bf.blend_normal(a, b) is a


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blend_overlay_analytic_and_in_place(dtype):
    a = ACT.astype(dtype)
    b = BAC.astype(dtype)
    expected = _overlay(ACT, BAC).astype(dtype)  # dyadic values: exact in both dtypes
    out = bf.blend_overlay(a, b)
    assert out is b  # the result is the background array, modified in place
    assert_same(out, expected)
    assert_same(a, ACT.astype(dtype))  # the active image is not touched
    # edge values: background 0 -> 0, background 1 -> 1, background 0.5 -> active
    assert np.all(expected[:, 0] == 0) and np.all(expected[:, -1] == 1)
    assert_same(expected[:, 2], ACT[:, 2].astype(dtype))


def test_blend_overlay_nan_handling():
    a = np.array([[np.nan, 0.5, 0.5], [np.nan, 0.5, 0.5]])
    b = np.array([[0.25, np.nan, 0.25], [0.75, np.nan, 0.75]])
    out = bf.blend_overlay(a, b.copy())
    # NaN in the active image gives NaN in both branches; NaN in the background matches neither mask: stays NaN
    assert np.isnan(out[:, 0]).all() and np.isnan(out[:, 1]).all()
    assert out[0, 2] == 0.25 and out[1, 2] == 0.75  # 2 * 0.25 * 0.5 ; 1 - 2 * 0.25 * 0.5


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blend_soft_light_analytic_and_in_place(dtype):
    a = ACT.astype(dtype)
    b = BAC.astype(dtype)
    # sqrt of the dyadic values is not exact: reference in float64, tolerance one float32 ulp at 1
    expected = _soft_light(ACT, BAC)
    out = bf.blend_soft_light(a, b)
    assert out is b
    assert out.dtype == dtype
    np.testing.assert_allclose(out, expected, atol=1e-7 if dtype == np.float32 else 1e-15, rtol=0)
    # edges: active 0 -> background squared; active 1 -> sqrt(background); active 0.5 -> background; background 0 -> 0
    np.testing.assert_allclose(expected[0], BAC[0] ** 2, atol=1e-15)
    np.testing.assert_allclose(expected[-1], np.sqrt(BAC[-1]), atol=1e-15)
    np.testing.assert_allclose(expected[2], BAC[2], atol=1e-15)
    assert np.all(expected[:, 0] == 0)


def test_blend_soft_light_nan_handling():
    a = np.array([[np.nan, 0.25, 0.75]])
    b = np.array([[0.5, np.nan, np.nan]])
    out = bf.blend_soft_light(a, b.copy())
    # pins current behaviour, KNOWN_ISSUES candidate: the branch is chosen by the active image; NaN there matches
    # neither mask, so the background value is returned unchanged (0.5) instead of NaN. NaN in the background is NaN.
    assert out[0, 0] == 0.5
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])


@pytest.mark.parametrize("mode", ["screen", "multiply", "overlay", "soft_light"])
def test_equation_blend_modes(mode):
    func = {"screen": _screen, "multiply": lambda a, b: a * b, "overlay": _overlay, "soft_light": _soft_light}[mode]
    expected = func(ACT, BAC)
    for name in (mode, mode.upper(), mode.title()):  # case-insensitive
        out = bf.equation_blend(name, ACT.copy(), BAC.copy())
        np.testing.assert_allclose(out, expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("mode", ["normal", "luminosity", "bogus", ""])
def test_equation_blend_other_modes_return_none(mode):
    # pins current behaviour, KNOWN_ISSUES candidate: "normal" and "luminosity" are handled by blend_images only;
    # equation_blend has no else branch and returns None for them (and for unknown names).
    assert bf.equation_blend(mode, ACT.copy(), BAC.copy()) is None


@pytest.mark.parametrize("mode", ["screen", "multiply"])
@pytest.mark.parametrize("dims", [(3, 3), (3, 2), (2, 3), (2, 2)])
def test_blend_multi_dim_images_shapes_and_analytic(mode, dims):
    rng = np.random.default_rng(10)
    h, w = 6, 7
    active = rng.uniform(0, 1, size=((3, h, w) if dims[0] == 3 else (h, w)))
    background = rng.uniform(0, 1, size=((3, h, w) if dims[1] == 3 else (h, w)))
    out = bf.blend_multi_dim_images(mode, active, background)
    func = _screen if mode == "screen" else (lambda a, b: a * b)
    # numpy broadcasting of a (h, w) image against (3, h, w) gives exactly the per-band pairing used by rvt
    expected = func(active, background)
    if dims == (2, 2):
        assert out.shape == (h, w)
    else:
        assert out.shape == (3, h, w) and out.dtype == np.float64
    assert_same(out, expected)


def test_blend_multi_dim_images_multiply_by_one_and_dtype():
    a = np.random.default_rng(11).uniform(0, 1, size=(3, 5, 6)).astype(np.float32)
    ones = np.ones((5, 6), dtype=np.float32)
    out = bf.blend_multi_dim_images("multiply", a, ones)
    assert out.dtype == np.float64  # np.zeros() target of the RGB cases
    assert_same(out, a.astype(np.float64))  # a * 1 = a, widened losslessly
    # 2D x 2D keeps the dtype of the operands
    out2 = bf.blend_multi_dim_images("multiply", a[0], ones)
    assert out2.dtype == np.float32
    assert_same(out2, a[0])


def test_blend_multi_dim_images_overlay_modifies_the_background():
    a = np.full((3, 4, 4), 0.25)
    b = np.full((3, 4, 4), 0.25)
    out = bf.blend_multi_dim_images("overlay", a, b)
    # pins current behaviour, KNOWN_ISSUES candidate: overlay and soft_light write into the background array; for an
    # RGB background the per-band views are written, so the caller's background changes (0.25 -> 2 * 0.25 * 0.25).
    assert_same(out, np.full((3, 4, 4), 0.125))
    assert_same(b, np.full((3, 4, 4), 0.125))
    assert_same(a, np.full((3, 4, 4), 0.25))


def test_blend_multi_dim_images_edge_values_and_nan():
    a = np.stack([ACT, ACT, ACT])
    a[1, 0, 0] = np.nan
    out = bf.blend_multi_dim_images("screen", a, BAC)
    assert np.isnan(out[1, 0, 0]) and np.isnan(out).sum() == 1
    assert_same(out[0], _screen(ACT, BAC))
    # unknown mode: equation_blend returns None; the RGB cases assign None into a float array -> NaN, the 2D case
    # returns None
    # pins current behaviour, KNOWN_ISSUES candidate
    assert bf.blend_multi_dim_images("bogus", ACT, BAC) is None
    assert np.isnan(bf.blend_multi_dim_images("bogus", a, BAC)).all()


def test_blend_images_dispatch():
    a = np.array([[0.2, 0.4]])
    b = np.array([[0.5, 0.6]])
    assert bf.blend_images("normal", a, b) is a
    assert bf.blend_images("Normal", a, b) is a
    assert bf.blend_images("something else", a, b) is a  # unknown names behave as "normal"
    assert_same(bf.blend_images("Multiply", a, b.copy()), a * b)
    assert_same(bf.blend_images("SCREEN", a, b.copy()), _screen(a, b))
    assert_same(bf.blend_images("overlay", a, b.copy()), _overlay(a, b))
    np.testing.assert_allclose(bf.blend_images("soft_light", a, b.copy()), _soft_light(a, b), atol=1e-15)
    # luminosity of two 2D images is the (luminosity of the) active image
    assert bf.blend_images("luminosity", a, b) is a


def test_blend_luminosity_known_colors():
    # background (0.2, 0.4, 0.6): L = 0.06 + 0.236 + 0.066 = 0.362; gray active 0.5 shifts every channel by 0.138;
    # the result (0.338, 0.538, 0.738) is inside the gamut: no clipping
    bg = np.zeros((3, 1, 1))
    bg[:, 0, 0] = [0.2, 0.4, 0.6]
    active = np.full((1, 1), 0.5)
    out = bf.blend_luminosity(active, bg)
    assert out.shape == (3, 1, 1) and out.dtype == np.float64
    np.testing.assert_allclose(out[:, 0, 0], [0.338, 0.538, 0.738], atol=1e-6, rtol=0)  # float32 luminosity
    # the luminosity of the result is the luminosity of the active image
    np.testing.assert_allclose(0.3 * out[0] + 0.59 * out[1] + 0.11 * out[2], 0.5, atol=1e-6, rtol=0)


def test_blend_luminosity_edges_and_nan():
    bg = np.zeros((3, 2, 3))
    bg[:, 0, 0] = [0.0, 0.0, 0.0]  # black background, white active -> white
    bg[:, 0, 1] = [1.0, 1.0, 1.0]  # white background, black active -> black
    bg[:, 0, 2] = [0.2, 0.4, 0.6]
    bg[:, 1, 0] = [0.2, 0.4, 0.6]
    bg[:, 1, 1] = [0.2, np.nan, 0.6]  # NaN in one background band: whole pixel NaN
    bg[:, 1, 2] = [0.2, 0.4, 0.6]
    active = np.array([[1.0, 0.0, 0.5], [np.nan, 0.5, 0.5]])  # NaN in the active image: whole pixel NaN
    out = bf.blend_luminosity(active, bg)
    np.testing.assert_allclose(out[:, 0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[:, 0, 1], 0.0, atol=1e-6)
    assert np.isnan(out[:, 1, 0]).all() and np.isnan(out[:, 1, 1]).all()
    assert not np.isnan(out[:, 0, :]).any() and not np.isnan(out[:, 1, 2]).any()


def test_blend_luminosity_2d_background_returns_luminosity_of_active():
    gray = np.array([[0.1, 0.9]])
    assert bf.blend_luminosity(gray, np.array([[0.5, 0.5]])) is gray
    rgb = np.zeros((3, 1, 2))
    rgb[:, 0, 0] = [1.0, 0.0, 0.0]
    rgb[:, 0, 1] = [0.0, 0.0, 1.0]
    out = bf.blend_luminosity(rgb, np.array([[0.5, 0.5]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.3, 0.11]], atol=1e-7, rtol=0)


def test_blend_luminosity_explicit_limits_use_the_clip_color_path():
    bg = np.zeros((3, 1, 1))
    bg[:, 0, 0] = [0.5, 0.5, 0.5]
    active = np.full((1, 1), 1.0)  # shift +0.5 -> channels 1.0, no channel above 1
    out = bf.blend_luminosity(active, bg, min_c=np.zeros((1, 1)), max_c=np.ones((1, 1)))
    np.testing.assert_allclose(out[:, 0, 0], 1.0, atol=1e-6)
    # with max_c = 2 the "above 1" correction is applied with that limit: L + (v - L) * (1 - L) / (2 - L) with v = L
    # (a gray pixel) is L itself, so the result is the luminosity 1.0 - unchanged
    out = bf.blend_luminosity(active, bg, min_c=np.zeros((1, 1)), max_c=np.full((1, 1), 2.0))
    np.testing.assert_allclose(out[:, 0, 0], 1.0, atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------------
# render_images / blend_images edge values (analytic complement of test_blend.py)
# ----------------------------------------------------------------------------------------------------------------------

def test_render_images_opacity_edges():
    a = np.array([[0.0, 1.0], [0.5, 0.25]])
    b = np.array([[1.0, 0.0], [0.5, 0.75]])
    assert_same(bf.render_images(a.copy(), b.copy(), 100), a)
    assert_same(bf.render_images(a.copy(), b.copy(), 0), b)
    assert_same(bf.render_images(a.copy(), b.copy(), 50), 0.5 * a + 0.5 * b)


def test_render_images_rgb_shapes_and_nan():
    a = np.full((3, 4, 5), 0.5)
    b = np.full((4, 5), 0.25)
    b[1, 1] = np.nan
    out = bf.render_images(a, b, 50)
    assert out.shape == (3, 4, 5) and out.dtype == np.float64
    assert np.isnan(out[:, 1, 1]).all() and np.isnan(out).sum() == 3
    assert np.all(out[:, 0, 0] == 0.375)
    out2 = bf.render_images(b, a, 50)  # gray active, RGB background
    assert out2.shape == (3, 4, 5)
    out3 = bf.render_images(a, a, 100)
    assert_same(out3, a)


def test_render_images_scales_inputs_outside_0_1():
    # an input with values outside 0..1 goes through scale_0_to_1 first (here the byte-like branch: / 255)
    a = np.array([[40.0, 255.0]])
    b = np.array([[0.0, 1.0]])
    out = bf.render_images(a, b, 100)
    np.testing.assert_allclose(out, a / 255.0, atol=1e-15)
