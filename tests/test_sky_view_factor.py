"""
Sky-view factor / anisotropic SVF / openness: rvt.vis.sky_view_factor_compute (horizon scan by slices over the output
window in row blocks, numba kernel `_svf_max_slope_kernel`) against the np.roll formulation of tag baseline-fixed.
"""

import numpy as np
import pytest

import rvt.vis
from conftest import assert_bitwise, assert_same, synthetic_dem


# ----------------------------------------------------------------------------------------------------------------------
# reference implementation (baseline-fixed, transcribed; horizon_shift_vector is unchanged and taken from rvt)
# ----------------------------------------------------------------------------------------------------------------------

def ref_sky_view_factor_compute(height_arr, radius_max=10, radius_min=1, num_directions=16, compute_svf=True,
                                compute_opns=False, compute_asvf=False, a_main_direction=315., a_poly_level=4,
                                a_min_weight=0.4):
    height = np.pad(height_arr, radius_max, mode='reflect')
    move = rvt.vis.horizon_shift_vector(num_directions=num_directions, radius_pixels=radius_max,
                                        min_radius=radius_min)
    svf_out = height * 0 if compute_svf else None
    if compute_asvf:
        asvf_out = height * 0
        w_m = a_min_weight
        w_a = np.deg2rad(a_main_direction)
        weight = np.arange(num_directions) * (2 * np.pi / num_directions)
        weight = (1 - w_m) * (np.cos((weight - w_a) / 2)) ** a_poly_level + w_m
    else:
        asvf_out = None
        weight = None
    opns_out = height * 0 if compute_opns else None

    for i_dir, direction in enumerate(move):
        max_slope = np.zeros(height.shape, dtype=np.float32) - 1000
        for i_rad, radius in enumerate(move[direction]["distance"]):
            shift_indx = move[direction]["shift"][i_rad]
            _ = (np.roll(height, shift_indx, axis=(0, 1)) - height) / radius
            max_slope = np.fmax(max_slope, _)
        max_slope = np.arctan(max_slope)
        if compute_svf:
            svf_out = svf_out + (1 - np.sin(np.fmax(max_slope, 0)))
        if compute_asvf:
            asvf_out = asvf_out + (1 - np.sin(np.fmax(max_slope, 0))) * weight[i_dir]
        if compute_opns:
            opns_out = opns_out + max_slope

    if compute_svf:
        svf_out = svf_out[radius_max:-radius_max, radius_max:-radius_max] / num_directions
    if compute_asvf:
        asvf_out = asvf_out[radius_max:-radius_max, radius_max:-radius_max] / np.sum(weight)
    if compute_opns:
        opns_out = np.rad2deg(0.5 * np.pi - (opns_out[radius_max:-radius_max, radius_max:-radius_max] /
                                             num_directions))
    out = {"svf": svf_out, "asvf": asvf_out, "opns": opns_out}
    return {k: v for k, v in out.items() if v is not None}


# ----------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------

OUTPUTS = [
    dict(compute_svf=True, compute_opns=False, compute_asvf=False),
    dict(compute_svf=False, compute_opns=True, compute_asvf=False),
    dict(compute_svf=False, compute_opns=False, compute_asvf=True),
    dict(compute_svf=True, compute_opns=True, compute_asvf=True),
]
SEARCH = [
    dict(radius_max=10, radius_min=1, num_directions=16),
    dict(radius_max=5, radius_min=2, num_directions=8),
    dict(radius_max=7, radius_min=1, num_directions=12, a_main_direction=90., a_poly_level=8, a_min_weight=0.1),
]


@pytest.mark.parametrize("outputs", OUTPUTS)
@pytest.mark.parametrize("search", SEARCH)
@pytest.mark.parametrize("nan_patch", [False, True])
def test_svf_compute_matches_reference(outputs, search, nan_patch):
    height = synthetic_dem((29, 38), seed=21, nan_patch=nan_patch)
    kwargs = dict(outputs, **search)
    assert_same(rvt.vis.sky_view_factor_compute(height.copy(), **kwargs),
                ref_sky_view_factor_compute(height.copy(), **kwargs))


def test_svf_compute_float64_and_multi_block():
    # float64 heights; a wide tile scanned in several row blocks (32768 elements per block)
    kwargs = dict(radius_max=4, radius_min=1, num_directions=8, compute_svf=True, compute_opns=True,
                  compute_asvf=True)
    for height in (synthetic_dem((23, 31), seed=22, nan_patch=True, dtype=np.float64),
                   synthetic_dem((30, 2500), seed=23, nan_patch=True)):
        assert_same(rvt.vis.sky_view_factor_compute(height.copy(), **kwargs),
                    ref_sky_view_factor_compute(height.copy(), **kwargs))


def test_sky_view_factor_wrapper():
    dem = synthetic_dem((27, 33), seed=24, nan_patch=True)
    out = rvt.vis.sky_view_factor(dem.copy(), resolution=0.5, compute_svf=True, compute_opns=True,
                                  compute_asvf=True, svf_n_dir=16, svf_r_max=8, svf_noise=2)
    dem_s = dem.astype(np.float32) * 1
    dem_s = dem_s / 0.5
    r_min = max(np.round(8 * 20. * 0.01, decimals=0), 1)  # as sky_view_factor computes it for svf_noise=2
    expected = ref_sky_view_factor_compute(dem_s, radius_max=8, radius_min=r_min, num_directions=16, compute_svf=True,
                                           compute_opns=True, compute_asvf=True, a_main_direction=315,
                                           a_poly_level=4, a_min_weight=0.4)
    for item in expected.values():
        item[np.isnan(dem)] = np.nan
    assert_same(out, expected)


def test_svf_elevation_offset_invariance():
    # height differences of the 1/256 m grid are exact in float32 at 0 m and at +2500 m, so the result is identical
    height0 = synthetic_dem((26, 30), seed=25, grid=True)
    height1 = height0 + np.float32(2500.0)
    kwargs = dict(radius_max=6, num_directions=16, compute_svf=True, compute_opns=True, compute_asvf=True)
    assert_same(rvt.vis.sky_view_factor_compute(height1, **kwargs), rvt.vis.sky_view_factor_compute(height0, **kwargs))


def test_svf_compute_deterministic():
    height = synthetic_dem((30, 31), seed=26, nan_patch=True)
    kwargs = dict(compute_svf=True, compute_opns=True, compute_asvf=True)
    assert_bitwise(rvt.vis.sky_view_factor_compute(height.copy(), **kwargs),
                   rvt.vis.sky_view_factor_compute(height.copy(), **kwargs))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_svf_compute_accel_equals_numpy(run_both, dtype):
    height = synthetic_dem((34, 37), seed=27, nan_patch=True, dtype=dtype)
    kwargs = dict(radius_max=8, compute_svf=True, compute_opns=True, compute_asvf=True)
    out_accel, out_numpy = run_both(lambda: rvt.vis.sky_view_factor_compute(height.copy(), **kwargs),
                                    kernels=[rvt.vis._svf_max_slope_kernel])
    assert_bitwise(out_accel, out_numpy)
    assert_same(out_numpy, ref_sky_view_factor_compute(height.copy(), **kwargs))


def test_svf_compute_numpy_path(numpy_path):
    height = synthetic_dem((25, 28), seed=28, nan_patch=True)
    kwargs = dict(radius_max=6, compute_svf=True, compute_opns=True, compute_asvf=True)
    assert_same(rvt.vis.sky_view_factor_compute(height.copy(), **kwargs),
                ref_sky_view_factor_compute(height.copy(), **kwargs))
