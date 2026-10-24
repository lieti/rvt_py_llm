"""
0.1.1 fix X2 (docs/KNOWN_ISSUES.md KI-47): the slope layer of e4mstp is in degrees.

At v0.1.0 e4mstp called rvt.vis.slope_aspect with its default unit, radians (about 0..1.5), and normalized the layer
over 0..55 as if in degrees, so the inverted slope layer stayed within about 0.97..1. The reading of FIX_BRIEF_LLM.md
section 1: ``rvt.vis.slope_aspect(..., output_units="degree")``, still normalized over 0..55.

Expected values: the slope in degrees computed here with numpy from the radian slope (``np.rad2deg``, the operation
slope_aspect applies for "degree"), and the other layers composed through the public BlenderCombination API
(tests/test_fix_0_1_1_recipes.py has the recipe tests of CRIM, e3MSTP and e4MSTP). A spy on rvt.vis.slope_aspect records
the unit e4mstp asks for. Synthetic data only.
"""

import inspect

import numpy as np
import pytest

import rvt.blend as rb
import rvt.default
import rvt.vis
from conftest import assert_same, synthetic_dem


def _steep_dem():
    # relief 40 m over 40 x 44 px at 1 m/px: slopes up to about 70 degrees
    return synthetic_dem((40, 44), seed=81, relief=40.0)


def test_e4mstp_asks_slope_aspect_for_degrees(monkeypatch):
    calls = []
    original = rvt.vis.slope_aspect
    signature = inspect.signature(original)

    def spy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(bound.arguments["output_units"])
        return original(*args, **kwargs)

    monkeypatch.setattr(rvt.vis, "slope_aspect", spy)
    rb.e4mstp(_steep_dem(), 1.0, default=rvt.default.DefaultValues())
    assert calls == ["degree"]


def test_slope_in_degrees_equals_rad2deg_of_the_radian_slope():
    # the numpy oracle of the slope layer used below: bitwise the degree output of slope_aspect
    dem = _steep_dem()
    rad = rvt.vis.slope_aspect(dem=dem, resolution_x=1.0, resolution_y=1.0, output_units="radian")["slope"]
    deg = rvt.vis.slope_aspect(dem=dem, resolution_x=1.0, resolution_y=1.0, output_units="degree")["slope"]
    assert_same(deg, np.rad2deg(rad))
    assert np.nanmax(deg) > 55  # the normalization 0..55 then clips the steepest pixels


def _e4_from_layers(dem, resolution, slope):
    """e4mstp rebuilt with BlenderCombination from the getters of a fresh DefaultValues and the given slope layer."""
    d = rvt.default.DefaultValues()
    ld = d.get_local_dominance(dem).squeeze()
    svf = d.get_sky_view_factor(dem, resolution, compute_svf=True, compute_opns=True)
    neg = d.get_neg_opns(dem, resolution).squeeze()
    d2 = rvt.default.DefaultValues()
    d2.svf_r_max = int(10 / resolution)
    d2.svf_noise = 3
    flat = d2.get_sky_view_factor(dem, resolution, compute_svf=True, compute_opns=False)["svf"].squeeze()
    comb_svf = rb.BlenderCombination()
    comb_svf.create_layer("Sky-view factor", "Value", 0.7, 1, "normal", 50, image=svf["svf"].squeeze())
    comb_svf.create_layer("Sky-view factor", "Value", 0.9, 1, "normal", 100, image=flat)
    comb_ol = rb.BlenderCombination()
    comb_ol.create_layer("Openness difference", "Value", -15, 15, "normal", 50, image=svf["opns"].squeeze() - neg)
    comb_ol.create_layer("Local dominance", "Value", 0.5, 1.8, "normal", 100, image=ld)
    comb = rb.BlenderCombination()
    comb.create_layer("mstp", "value", 0, 1, "overlay", 90, image=d.get_mstp(dem_arr=dem))
    comb.create_layer("Comb svf", "value", -0.5, 0.5, "multiply", 25,
                      image=comb_svf.render_all_images(no_data=np.nan).astype("float32"))
    comb.create_layer("Comb openness LD", "value", 0, 1, "multiply", 100,
                      image=comb_ol.render_all_images(no_data=np.nan).astype("float32"))
    comb.create_layer("Slope gradient", "value", 0, 55, "normal", 100, colormap="Reds_r", min_colormap_cut=0,
                      max_colormap_cut=1, image=slope)
    return comb.render_all_images(no_data=np.nan)


@pytest.mark.parametrize("resolution", [1.0, 0.5])
def test_e4mstp_slope_layer_is_the_slope_in_degrees(resolution):
    dem = _steep_dem()
    rad = rvt.vis.slope_aspect(dem=dem, resolution_x=resolution, resolution_y=resolution,
                               output_units="radian")["slope"]
    out = rb.e4mstp(dem.copy(), resolution, default=rvt.default.DefaultValues())
    assert_same(out, _e4_from_layers(dem.copy(), resolution, np.rad2deg(rad)))
    # the unit matters: the same recipe with the radian slope differs
    assert not np.array_equal(out, _e4_from_layers(dem.copy(), resolution, rad.copy()), equal_nan=True)
