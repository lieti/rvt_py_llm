"""
Reference tests for the blending recipes of rvt.blend.e3mstp and rvt.blend.e4mstp.

tests/test_blend.py checks these two functions for determinism and accelerated == numpy only, and
tests/test_char_blend.py checks shape, dtype, range and NaN mask. Neither notices a change of one layer's opacity,
normalization bound or blend mode. This file rebuilds both recipes from the code at tag baseline-fixed
(rvt/blend.py: color_relief_image_map, e3mstp, e4mstp; BlenderCombination.render_all_images) with plain code:

- every layer image is computed with the public rvt.vis functions and the parameters that rvt.default.DefaultValues
  passes to them (get_slrm, get_mstp, get_sky_view_factor, get_neg_opns, get_local_dominance);
- every layer is normalized by the linear "value" stretch written out below (clip to [min, max], stretch to [0, 1],
  cast to float32, invert for the vis name "slope gradient"), colored with rvt.blend_func.gray_scale_to_color_ramp
  where the recipe has a colormap, and composed from the bottom layer up with rvt.blend_func.blend_images and
  rvt.blend_func.render_images, in the order of the render_all_images layer loop.

The reference repeats the operations of the code in the same order and dtype, so the comparison is exact
(assert_same: equal values, NaN at the same positions). A second test changes one recipe parameter at a time in the
reference and asserts the effect, so the first test is known to fail on such a change in rvt/blend.py.

Behaviour that looks wrong is pinned with the comment "# pins current behaviour, KNOWN_ISSUES candidate".
"""

import numpy as np
import pytest

import rvt.blend as rb
import rvt.blend_func as bf
import rvt.default
import rvt.vis
from conftest import assert_same, synthetic_dem


# ----------------------------------------------------------------------------------------------------------------------
# plain reference composition
# ----------------------------------------------------------------------------------------------------------------------

def _layer(image, lo, hi, mode, opacity, colormap=None, cut=(None, None), invert=False):
    """One layer: image, "value" normalization bounds, blend mode, opacity in percent, optional colormap."""
    return {"image": image, "lo": lo, "hi": hi, "mode": mode, "opacity": opacity, "colormap": colormap, "cut": cut,
            "invert": invert}


def _normalize_value(image, lo, hi, invert):
    """Linear "value" normalization as in rvt.blend_func.normalize_lin, on a copy (the library clips in place)."""
    img = np.array(image, copy=True)
    img[img > hi] = hi
    img[img < lo] = lo
    img = (img - lo) / (hi - lo)
    img[img > 1] = 1
    img[img < 0] = 0
    img = np.float32(img)
    if invert:  # normalize_image inverts the vis name "slope gradient" (high slopes dark)
        img = 1 - img
    return img


def _outside_0_1(arr):
    return np.nanmin(arr) < 0 or np.nanmax(arr) > 1


def _render(layers):
    """Compose the layers (top first, as created) from the bottom layer up. The bottom layer is the start image (its
    blend mode and opacity are not used); each layer above is blended onto the result and mixed with it by opacity."""
    rendered = None
    for lay in reversed(layers):
        norm = _normalize_value(lay["image"], lay["lo"], lay["hi"], lay["invert"])
        if lay["colormap"] is not None and np.ndim(lay["image"]) < 3:
            norm = bf.gray_scale_to_color_ramp(gray_scale=norm, colormap=lay["colormap"], output_8bit=False,
                                               min_colormap_cut=lay["cut"][0], max_colormap_cut=lay["cut"][1])
        if rendered is None:
            rendered = norm
            continue
        active, background = norm, rendered
        if _outside_0_1(active):
            active = bf.scale_0_to_1(active)
        if _outside_0_1(background):
            background = bf.scale_0_to_1(background)
        top = bf.blend_images(lay["mode"], active, background)
        # pins current behaviour, KNOWN_ISSUES candidate: blend_overlay and blend_soft_light write their result into
        # `background` (also into the band views of an RGB background), so for these two modes top and background
        # hold the same values here and the opacity has no visible effect (see test_recipe_sensitivity).
        rendered = bf.render_images(top, background, lay["opacity"])
    return rendered


def _svf(dem, resolution, d, compute_svf, compute_opns, r_max=None, noise=None):
    """rvt.vis.sky_view_factor with the parameters of DefaultValues.get_sky_view_factor (r_max / noise override)."""
    return rvt.vis.sky_view_factor(
        dem=dem, resolution=resolution, compute_svf=compute_svf, compute_opns=compute_opns, compute_asvf=False,
        svf_n_dir=d.svf_n_dir, svf_r_max=d.svf_r_max if r_max is None else r_max,
        svf_noise=d.svf_noise if noise is None else noise, asvf_dir=d.asvf_dir, asvf_level=d.asvf_level,
        ve_factor=d.ve_factor)


def _mstp(dem, d):
    # DefaultValues.get_mstp does not pass ve_factor (rvt.vis.mstp then uses ve_factor=1)
    return rvt.vis.mstp(dem=dem, local_scale=d.mstp_local_scale, meso_scale=d.mstp_meso_scale,
                        broad_scale=d.mstp_broad_scale, lightness=d.mstp_lightness)


def _e3_images(dem, resolution, d):
    """Layer images of e3mstp. color_relief_image_map sets svf_r_max = 10 on `default` before its openness."""
    opns_pos = _svf(dem, resolution, d, False, True, r_max=10)["opns"]
    opns_neg = _svf(-1 * dem, resolution, d, False, True, r_max=10)["opns"]
    return {
        "slrm": rvt.vis.slrm(dem=dem, radius_cell=d.slrm_rad_cell, ve_factor=d.ve_factor),
        "opns_pos_neg": opns_pos - opns_neg,
        "slope_rad": rvt.vis.slope_aspect(dem=dem, resolution_x=resolution, resolution_y=resolution,
                                          output_units="radian")["slope"],
        "mstp": _mstp(dem, d),
    }


def _crim_recipe(images, edit=None):
    """CRIM as called by e3mstp: openness pos - neg (-28..28 deg) overlay 50 %, the same luminosity 50 %, slope
    (0..0.8 rad) colored with OrRd (cut 0..1), normal 100 % (bottom)."""
    edit = edit or (lambda name, layers: layers)
    opd = images["opns_pos_neg"]
    return _render(edit("crim", [
        _layer(opd, -28, 28, "overlay", 50),
        _layer(opd, -28, 28, "luminosity", 50),
        _layer(images["slope_rad"], 0, 0.8, "normal", 100, colormap="OrRd", cut=(0, 1)),
    ]))


def _e3_recipe(images, edit=None):
    """e3MSTP: SLRM (-0.5..0.5) screen 25 %, CRIM (0..1) soft light 70 %, MSTP (0..1) normal 100 % (bottom).
    `edit(name, layers)` may change a recipe (sensitivity test).

    The docstring of e3mstp names the CRIM colormap "Reds_r(0.5-1)"; the code (and this reference) uses "OrRd" with
    cuts 0..1 (KNOWN_ISSUES candidate: docstring and code differ)."""
    edit = edit or (lambda name, layers: layers)
    crim = _crim_recipe(images, edit)
    return _render(edit("e3", [
        _layer(images["slrm"], -0.5, 0.5, "screen", 25),
        _layer(crim, 0, 1, "soft_light", 70),
        _layer(images["mstp"], 0, 1, "normal", 100),
    ]))


def _e4_images(dem, resolution, d):
    """Layer images of e4mstp. The flat-terrain sky-view factor uses a fresh DefaultValues with
    svf_r_max = int(10 / resolution) px (10 m) and svf_noise = 3, whatever `default` holds."""
    d2 = rvt.default.DefaultValues()
    svf_opns = _svf(dem, resolution, d, True, True)
    neg_opns = rvt.vis.sky_view_factor(dem=-1 * dem, resolution=resolution, svf_n_dir=d.svf_n_dir,
                                       svf_r_max=d.svf_r_max, svf_noise=d.svf_noise, compute_svf=False,
                                       compute_asvf=False, compute_opns=True, ve_factor=d.ve_factor)["opns"]
    return {
        "ld": rvt.vis.local_dominance(dem=dem, min_rad=d.ld_min_rad, max_rad=d.ld_max_rad, rad_inc=d.ld_rad_inc,
                                      angular_res=d.ld_anglr_res, observer_height=d.ld_observer_h,
                                      ve_factor=d.ve_factor).squeeze(),
        "svf": svf_opns["svf"].squeeze(),
        "opns_diff": svf_opns["opns"].squeeze() - neg_opns.squeeze(),
        "svf_flat": _svf(dem, resolution, d2, True, False, r_max=int(10 / resolution), noise=3)["svf"].squeeze(),
        "mstp": _mstp(dem, d),
        # pins current behaviour, KNOWN_ISSUES candidate: the slope is in radians (the slope_aspect default), but the
        # layer is normalized over 0..55 as the docstring's degrees, so the inverted layer stays within ~0.97..1.
        "slope": rvt.vis.slope_aspect(dem=dem, resolution_x=resolution, resolution_y=resolution)["slope"],
    }


def _e4_recipe(images, edit=None):
    """e4MSTP: MSTP (0..1) overlay 90 %, combined SVF (-0.5..0.5) multiply 25 %, combined openness / local dominance
    (0..1) multiply 100 %, slope (0..55, inverted) colored with Reds_r, normal 100 % (bottom).
    Combined SVF: SVF (0.7..1) normal 50 %, flat-terrain SVF (0.9..1) normal 100 % (bottom), cast to float32.
    Combined openness / LD: openness pos - neg (-15..15) normal 50 %, LD (0.5..1.8) normal 100 % (bottom), float32."""
    edit = edit or (lambda name, layers: layers)
    comb_svf = _render(edit("comb_svf", [
        _layer(images["svf"], 0.7, 1, "normal", 50),
        _layer(images["svf_flat"], 0.9, 1, "normal", 100),
    ])).astype("float32")
    comb_ol = _render(edit("comb_ol", [
        _layer(images["opns_diff"], -15, 15, "normal", 50),
        _layer(images["ld"], 0.5, 1.8, "normal", 100),
    ])).astype("float32")
    return _render(edit("e4", [
        _layer(images["mstp"], 0, 1, "overlay", 90),
        _layer(comb_svf, -0.5, 0.5, "multiply", 25),
        _layer(comb_ol, 0, 1, "multiply", 100),
        _layer(images["slope"], 0, 55, "normal", 100, colormap="Reds_r", cut=(0, 1), invert=True),
    ]))


# ----------------------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------------------

def _custom_default():
    """DefaultValues with other parameters than the defaults, to see that they are taken from `default`."""
    d = rvt.default.DefaultValues()
    d.ve_factor = 2
    d.slrm_rad_cell = 12  # rvt.vis.slrm accepts 10..50 px
    d.svf_n_dir = 8
    d.svf_r_max = 6
    d.svf_noise = 1
    d.ld_min_rad = 4
    d.ld_max_rad = 9
    d.ld_rad_inc = 2
    d.ld_anglr_res = 30
    d.ld_observer_h = 1.0
    d.mstp_local_scale = (1, 3, 1)
    d.mstp_meso_scale = (3, 12, 3)
    d.mstp_broad_scale = (12, 30, 6)
    d.mstp_lightness = 1.1
    return d


# (dem, resolution in m/px, DefaultValues factory)
CASES = {
    "plain": (lambda: synthetic_dem((40, 44), seed=61, relief=20.0), 1.0, rvt.default.DefaultValues),
    "nan_patch": (lambda: synthetic_dem((40, 44), seed=62, relief=20.0, nan_patch=True), 0.5,
                  rvt.default.DefaultValues),
    "offset_2500m": (lambda: synthetic_dem((40, 44), seed=63, relief=20.0, offset=2500.0, grid=True), 1.0,
                     rvt.default.DefaultValues),
    # pins current behaviour, KNOWN_ISSUES candidate: ve_factor of `default` is not applied to the MSTP layer
    # (DefaultValues.get_mstp) nor to the slope layers (slope_aspect is called without it)
    "custom_default": (lambda: synthetic_dem((40, 44), seed=64, relief=20.0), 1.0, _custom_default),
}


# ----------------------------------------------------------------------------------------------------------------------
# function == reference
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_color_relief_image_map_equals_reference_recipe(case):
    # The CRIM layer of e3mstp on its own: its float64 result is cast to float32 by the normalization in e3mstp,
    # which can hide float64 rounding differences (see test_recipe_sensitivity), so it is compared here directly.
    make_dem, resolution, make_default = CASES[case]
    dem = make_dem()
    expected = _crim_recipe(_e3_images(dem.copy(), resolution, make_default()))
    out = rb.color_relief_image_map(dem.copy(), resolution, default=make_default())
    assert out.shape == (3,) + dem.shape
    assert_same(out, expected)


@pytest.mark.parametrize("case", sorted(CASES))
def test_e3mstp_equals_reference_recipe(case):
    make_dem, resolution, make_default = CASES[case]
    dem = make_dem()
    expected = _e3_recipe(_e3_images(dem.copy(), resolution, make_default()))
    out = rb.e3mstp(dem.copy(), resolution, default=make_default())
    assert out.shape == (3,) + dem.shape
    assert_same(out, expected)
    assert np.array_equal(np.isnan(out), np.broadcast_to(np.isnan(dem), out.shape))


@pytest.mark.parametrize("case", sorted(CASES))
def test_e4mstp_equals_reference_recipe(case):
    make_dem, resolution, make_default = CASES[case]
    dem = make_dem()
    expected = _e4_recipe(_e4_images(dem.copy(), resolution, make_default()))
    out = rb.e4mstp(dem.copy(), resolution, default=make_default())
    assert out.shape == (3,) + dem.shape
    assert_same(out, expected)
    assert np.array_equal(np.isnan(out), np.broadcast_to(np.isnan(dem), out.shape))


# ----------------------------------------------------------------------------------------------------------------------
# sensitivity of the reference: which single-parameter changes the comparison above detects
# ----------------------------------------------------------------------------------------------------------------------

_OTHER_MODE = {"normal": "multiply", "multiply": "screen", "screen": "multiply", "overlay": "soft_light",
               "soft_light": "overlay", "luminosity": "normal"}

_RECIPES = {
    "e3mstp": (_e3_images, _e3_recipe, ("crim", "e3")),
    "e4mstp": (_e4_images, _e4_recipe, ("comb_svf", "comb_ol", "e4")),
}

_LAYER_COUNTS = {"crim": 3, "e3": 3, "comb_svf": 2, "comb_ol": 2, "e4": 4}


@pytest.fixture(scope="module")
def recipe_images():
    make_dem, resolution, make_default = CASES["nan_patch"]
    dem = make_dem()
    return {name: images(dem.copy(), resolution, make_default()) for name, (images, _, _) in _RECIPES.items()}


def _mutations():
    for func, (_, _, combs) in _RECIPES.items():
        for comb in combs:
            for i in range(_LAYER_COUNTS[comb]):
                for field in ("opacity", "lo", "hi", "mode"):
                    yield pytest.param(func, comb, i, field, id="{}-{}-{}-{}".format(func, comb, i, field))


def _mutate(lay, field):
    lay = dict(lay)
    if field == "opacity":
        lay["opacity"] = lay["opacity"] - 5  # 5 percentage points
    elif field in ("lo", "hi"):
        lay[field] = lay[field] + 0.05 * (lay["hi"] - lay["lo"])  # 5 % of the normalization range
    else:
        lay["mode"] = _OTHER_MODE[lay["mode"]]
    return lay


@pytest.mark.parametrize("func, comb, i, field", list(_mutations()))
def test_recipe_sensitivity(recipe_images, func, comb, i, field):
    _, recipe, _ = _RECIPES[func]
    images = recipe_images[func]
    base = recipe(images)
    seen = []

    def edit(name, layers):
        if name != comb:
            return layers
        seen.append(name)
        return [_mutate(lay, field) if j == i else lay for j, lay in enumerate(layers)]

    changed = recipe(images, edit)
    assert seen == [comb]
    if np.array_equal(np.isnan(changed), np.isnan(base)):
        max_diff = float(np.nanmax(np.abs(changed - base)))
    else:  # a different NaN mask (e.g. soft light <-> overlay at NaN pixels) counts as a visible change
        max_diff = np.inf
    bottom = i == _LAYER_COUNTS[comb] - 1
    if bottom and field in ("opacity", "mode"):
        # the bottom layer is the start image of the composition: its opacity and blend mode are not used
        assert max_diff == 0.0
    elif field == "opacity" and (comb, i) in (("crim", 0), ("e3", 1), ("e4", 0)):
        # pins current behaviour, KNOWN_ISSUES candidate: the opacity of an overlay / soft light layer has no visible
        # effect (see _render); the exact comparison above may still see float64 rounding differences
        assert max_diff < 1e-6
    else:
        # a visible change: well above the rounding level of the exact comparison
        assert max_diff > 1e-4
