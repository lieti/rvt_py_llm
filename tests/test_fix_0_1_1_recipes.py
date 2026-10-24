"""
0.1.1: CRIM, e3MSTP and e4MSTP rebuilt layer by layer through the public BlenderCombination API.

After the fixes X1-X5 (FIX_BRIEF_LLM.md section 1) each composite is its documented recipe and nothing else: the layer
images come from the public getters of a DefaultValues, each layer is added with its normalization, blend mode and
opacity, and BlenderCombination.render_all_images composes them. The composite function must give the same bits.

- CRIM (color_relief_image_map): openness positive - negative (value -28..28, overlay 50 %), the same image (luminosity
  50 %), slope in radians colored with the colormap (value 0..0.8, bottom); openness with svf_r_max = 10 px on a copy
  of ``default`` (X4).
- e3MSTP: SLRM (value -0.5..0.5, screen 25 %), CRIM with OrRd 0..1 (value 0..1, soft light 70 %), MSTP (value 0..1,
  bottom).
- e4MSTP: MSTP (value 0..1, overlay 90 %), combined SVF (value -0.5..0.5, multiply 25 %), combined openness / local
  dominance (value 0..1, multiply 100 %), slope in degrees (X2) inverted and colored with Reds_r (value 0..55, bottom).

tests/test_char_e3e4_recipe.py checks the same recipes against a reference written without render_all_images; this
file checks that the composite functions use the public API as documented. Synthetic data only.
"""

import copy

import numpy as np
import pytest

import rvt.blend as rb
import rvt.default
import rvt.vis
from conftest import assert_bitwise, synthetic_dem


def _crim(dem, resolution, default, colormap="OrRd", cut=(0, 1)):
    d = copy.deepcopy(default)
    d.svf_r_max = 10
    pos = d.get_sky_view_factor(dem_arr=dem, resolution=resolution, compute_svf=False, compute_asvf=False,
                                compute_opns=True)["opns"]
    neg = d.get_sky_view_factor(dem_arr=-1 * dem, resolution=resolution, compute_svf=False, compute_asvf=False,
                                compute_opns=True)["opns"]
    diff = pos - neg
    slope = rvt.vis.slope_aspect(dem=dem, resolution_x=resolution, resolution_y=resolution,
                                 output_units="radian")["slope"]
    comb = rb.BlenderCombination()
    comb.create_layer("Openness_Pos-Neg", "value", -28, 28, "overlay", 50, image=diff)
    comb.create_layer("Openness_Pos-Neg", "value", -28, 28, "luminosity", 50, image=diff.copy())
    comb.create_layer("slope gradient rad", "value", 0, 0.8, "normal", 100, colormap=colormap,
                      min_colormap_cut=cut[0], max_colormap_cut=cut[1], image=slope)
    return comb.render_all_images()


def _e3(dem, resolution, default):
    comb = rb.BlenderCombination()
    comb.create_layer("slrm", "value", -0.5, 0.5, "screen", 25, image=default.get_slrm(dem_arr=dem))
    comb.create_layer("crim_red", "value", 0, 1, "soft_light", 70, image=_crim(dem, resolution, default))
    comb.create_layer("mstp", "value", 0, 1, "normal", 100, image=default.get_mstp(dem_arr=dem))
    return comb.render_all_images()


def _e4(dem, resolution, default):
    ld = default.get_local_dominance(dem).squeeze()
    svf = default.get_sky_view_factor(dem, resolution, compute_svf=True, compute_opns=True)
    neg = default.get_neg_opns(dem, resolution).squeeze()
    flat_default = rvt.default.DefaultValues()
    flat_default.svf_r_max = int(10 / resolution)  # 10 m
    flat_default.svf_noise = 3
    flat = flat_default.get_sky_view_factor(dem, resolution, compute_svf=True, compute_opns=False)["svf"].squeeze()
    comb_svf = rb.BlenderCombination()
    comb_svf.create_layer("Sky-view factor", "Value", 0.7, 1, "normal", 50, image=svf["svf"].squeeze())
    comb_svf.create_layer("Sky-view factor", "Value", 0.9, 1, "normal", 100, image=flat)
    comb_ol = rb.BlenderCombination()
    comb_ol.create_layer("Openness difference", "Value", -15, 15, "normal", 50, image=svf["opns"].squeeze() - neg)
    comb_ol.create_layer("Local dominance", "Value", 0.5, 1.8, "normal", 100, image=ld)
    slope_deg = rvt.vis.slope_aspect(dem=dem, resolution_x=resolution, resolution_y=resolution,
                                     output_units="degree")["slope"]
    comb = rb.BlenderCombination()
    comb.create_layer("mstp", "value", 0, 1, "overlay", 90, image=default.get_mstp(dem_arr=dem))
    comb.create_layer("Comb svf", "value", -0.5, 0.5, "multiply", 25,
                      image=comb_svf.render_all_images(no_data=np.nan).astype("float32"))
    comb.create_layer("Comb openness LD", "value", 0, 1, "multiply", 100,
                      image=comb_ol.render_all_images(no_data=np.nan).astype("float32"))
    comb.create_layer("Slope gradient", "value", 0, 55, "normal", 100, colormap="Reds_r", min_colormap_cut=0,
                      max_colormap_cut=1, image=slope_deg)
    return comb.render_all_images(no_data=np.nan)


def _custom_default():
    d = rvt.default.DefaultValues()
    d.slrm_rad_cell = 12
    d.svf_n_dir = 8
    d.svf_r_max = 6
    d.svf_noise = 1
    d.ld_min_rad = 4
    d.ld_max_rad = 9
    d.mstp_local_scale = (1, 3, 1)
    d.mstp_meso_scale = (3, 12, 3)
    d.mstp_broad_scale = (12, 30, 6)
    return d


# (dem, resolution in m/px, DefaultValues factory)
CASES = {
    "plain": (lambda: synthetic_dem((40, 44), seed=111, relief=20.0), 1.0, rvt.default.DefaultValues),
    "nan_patch": (lambda: synthetic_dem((40, 44), seed=112, relief=20.0, nan_patch=True), 0.5,
                  rvt.default.DefaultValues),
    "custom_default": (lambda: synthetic_dem((40, 44), seed=113, relief=20.0, nan_patch=True), 1.0, _custom_default),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_color_relief_image_map_is_its_recipe(case):
    make_dem, resolution, make_default = CASES[case]
    out = rb.color_relief_image_map(make_dem(), resolution, default=make_default())
    assert_bitwise(out, _crim(make_dem(), resolution, make_default()))
    gray = rb.color_relief_image_map(make_dem(), resolution, default=make_default(), colormap="Reds_r",
                                     min_colormap_cut=0.5, max_colormap_cut=1)
    assert_bitwise(gray, _crim(make_dem(), resolution, make_default(), colormap="Reds_r", cut=(0.5, 1)))


@pytest.mark.parametrize("case", sorted(CASES))
def test_e3mstp_is_its_recipe(case):
    make_dem, resolution, make_default = CASES[case]
    out = rb.e3mstp(make_dem(), resolution, default=make_default())
    assert_bitwise(out, _e3(make_dem(), resolution, make_default()))


@pytest.mark.parametrize("case", sorted(CASES))
def test_e4mstp_is_its_recipe(case):
    make_dem, resolution, make_default = CASES[case]
    out = rb.e4mstp(make_dem(), resolution, default=make_default())
    assert_bitwise(out, _e4(make_dem(), resolution, make_default()))
