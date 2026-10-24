"""
0.1.1 fix X4 (docs/KNOWN_ISSUES.md KI-45): color_relief_image_map does not modify the caller's DefaultValues.

At v0.1.0 color_relief_image_map set ``svf_r_max = 10`` on the DefaultValues object it received; e3mstp passes its own
``default`` on, and the default argument of both functions is one instance created at import time, so a later call
that reused the object computed sky-view factor and openness with r = 10 px. The reading of FIX_BRIEF_LLM.md section 1:
the function works on ``copy.deepcopy(default)``; e3mstp passes its ``default`` on unchanged.

Expected values: the attributes of the object before the call (``copy.deepcopy(vars(default))``), and the outputs of
the same functions called with a fresh DefaultValues (the output does not change in a first call). Synthetic data only.
"""

import copy
import inspect

import numpy as np
import pytest

import rvt.blend as rb
import rvt.default
from conftest import assert_bitwise, synthetic_dem


def _dem():
    return synthetic_dem((36, 40), seed=101, relief=8.0, nan_patch=True)


def _custom_default():
    d = rvt.default.DefaultValues()
    d.svf_r_max = 25
    d.svf_n_dir = 8
    d.slrm_rad_cell = 12
    return d


@pytest.mark.parametrize("func", ["color_relief_image_map", "e3mstp"])
def test_the_callers_default_is_not_modified(func):
    default = _custom_default()
    before = copy.deepcopy(vars(default))
    getattr(rb, func)(_dem(), 1.0, default=default)
    assert vars(default) == before
    assert default.svf_r_max == 25


@pytest.mark.parametrize("func", ["color_relief_image_map", "e3mstp"])
def test_the_output_does_not_depend_on_svf_r_max_and_a_reused_object_keeps_its_radius(func):
    # the openness of the CRIM is computed with r = 10 px whatever `default` holds, so the output equals that of a
    # DefaultValues with svf_r_max = 10; afterwards the reused object still computes with its own radius
    default = _custom_default()
    reference = _custom_default()
    reference.svf_r_max = 10
    assert_bitwise(getattr(rb, func)(_dem(), 1.0, default=default), getattr(rb, func)(_dem(), 1.0, default=reference))
    dem = _dem()
    svf = default.get_sky_view_factor(dem, 1.0, compute_svf=True, compute_opns=False)["svf"]
    expected = _custom_default().get_sky_view_factor(dem, 1.0, compute_svf=True, compute_opns=False)["svf"]
    assert_bitwise(svf, expected)


@pytest.mark.parametrize("func", ["color_relief_image_map", "e3mstp"])
def test_the_shared_default_instance_is_not_modified(func, monkeypatch):
    # the default argument is one instance created at import; give it another radius for this test (restored after)
    shared = inspect.signature(getattr(rb, func)).parameters["default"].default
    monkeypatch.setattr(shared, "svf_r_max", 25)
    before = copy.deepcopy(vars(shared))
    getattr(rb, func)(_dem(), 1.0)
    assert vars(shared) == before


def test_e3mstp_output_with_a_reused_default_equals_a_fresh_one():
    # "later calls that reuse the object": two calls with the same object give the same bits as a fresh object
    default = _custom_default()
    first = rb.e3mstp(_dem(), 1.0, default=default)
    crim = rb.color_relief_image_map(_dem(), 1.0, default=default)
    second = rb.e3mstp(_dem(), 1.0, default=default)
    assert_bitwise(first, second)
    assert_bitwise(crim, rb.color_relief_image_map(_dem(), 1.0, default=_custom_default()))
    assert np.isnan(first).any()  # the NaN patch of the DEM
