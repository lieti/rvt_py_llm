"""
Local dominance: rvt.vis.local_dominance (slices over the output window, masked float64 add, numba kernel
`_local_dominance_kernel`) against the np.roll / np.where formulation of tag baseline-fixed.
"""

import numpy as np
import pytest

import rvt.vis
from conftest import assert_bitwise, assert_same, synthetic_dem


# ----------------------------------------------------------------------------------------------------------------------
# reference implementation (baseline-fixed, transcribed)
# ----------------------------------------------------------------------------------------------------------------------

def ref_local_dominance(dem, min_rad=10, max_rad=20, rad_inc=1, angular_res=15, observer_height=1.7, ve_factor=1,
                        no_data=None):
    if no_data is not None:
        dem[dem == no_data] = np.nan
    dem = dem.astype(np.float32)
    pad_width = max_rad
    dem = np.pad(array=dem, pad_width=pad_width, mode="edge")
    dem = dem * ve_factor

    n_dist = int((max_rad - min_rad) / rad_inc + 1)
    distances = np.arange(n_dist * rad_inc, step=rad_inc) + min_rad
    n_ang = int(359 / angular_res + 1)
    angles = np.arange(n_ang * angular_res, step=angular_res)
    norma = np.sum((observer_height / distances) * (2 * distances + rad_inc)) * n_ang

    n_shifts = distances.size * angles.size
    x_t = (np.outer(np.cos(np.deg2rad(angles)), distances)).reshape(n_shifts)
    y_t = (np.outer(np.sin(np.deg2rad(angles)), distances)).reshape(n_shifts)
    distances = (np.outer(np.ones(n_ang), distances)).reshape(n_shifts)
    dist_factor = 2 * distances + rad_inc

    local_dom_out = dem * 0
    for i_s in range(n_shifts):
        dem_moved = np.roll(dem, int(round(y_t[i_s])), axis=0)
        dem_moved = np.roll(dem_moved, int(round(x_t[i_s])), axis=1)
        idx_lower = np.where((dem + observer_height) > dem_moved)
        if idx_lower[0].size > 0:
            local_dom_out[idx_lower[0], idx_lower[1]] = local_dom_out[idx_lower[0], idx_lower[1]] + \
                                                        (dem[idx_lower[0], idx_lower[1]] + observer_height -
                                                         dem_moved[idx_lower[0], idx_lower[1]]) / \
                                                        distances[i_s] * dist_factor[i_s]
    local_dom_out = local_dom_out / norma
    local_dom_out = local_dom_out[pad_width:-pad_width, pad_width:-pad_width]
    return local_dom_out


# ----------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------

PARAMS = [
    dict(),  # defaults: 10-20 px, 15 deg, 1.7 m
    dict(min_rad=2, max_rad=6, rad_inc=1, angular_res=30, observer_height=0.5),
    dict(min_rad=3, max_rad=9, rad_inc=2, angular_res=45, observer_height=1.7, ve_factor=2),
    dict(min_rad=1, max_rad=4, rad_inc=1, angular_res=20, observer_height=3, ve_factor=-1.5),
]


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("nan_patch", [False, True])
def test_local_dominance_matches_reference(params, nan_patch):
    dem = synthetic_dem((31, 43), seed=11, nan_patch=nan_patch)
    assert_same(rvt.vis.local_dominance(dem.copy(), **params), ref_local_dominance(dem.copy(), **params))


def test_local_dominance_float64_input_and_no_data():
    dem = synthetic_dem((25, 29), seed=12, dtype=np.float64)
    dem[3:6, 4:8] = -9999.0
    params = dict(min_rad=2, max_rad=5, angular_res=30, no_data=-9999.0)
    assert_same(rvt.vis.local_dominance(dem.copy(), **params), ref_local_dominance(dem.copy(), **params))


def test_local_dominance_multi_block():
    # several row blocks of the numpy path (32768 elements per block) and a tile narrower than the radius
    params = dict(min_rad=2, max_rad=5, angular_res=60)
    dem = synthetic_dem((40, 2000), seed=13, nan_patch=True)
    assert_same(rvt.vis.local_dominance(dem.copy(), **params), ref_local_dominance(dem.copy(), **params))
    dem = synthetic_dem((3, 7), seed=14)
    assert_same(rvt.vis.local_dominance(dem.copy(), **params), ref_local_dominance(dem.copy(), **params))


def test_local_dominance_deterministic():
    dem = synthetic_dem((30, 33), seed=15, nan_patch=True)
    assert_bitwise(rvt.vis.local_dominance(dem.copy()), rvt.vis.local_dominance(dem.copy()))


@pytest.mark.parametrize("nan_patch", [False, True])
def test_local_dominance_accel_equals_numpy(run_both, nan_patch):
    dem = synthetic_dem((36, 41), seed=16, nan_patch=nan_patch)
    out_accel, out_numpy = run_both(lambda: rvt.vis.local_dominance(dem.copy(), min_rad=4, max_rad=12),
                                    kernels=[rvt.vis._local_dominance_kernel])
    assert_bitwise(out_accel, out_numpy)
    assert_same(out_numpy, ref_local_dominance(dem.copy(), min_rad=4, max_rad=12))


def test_local_dominance_numpy_path(numpy_path):
    dem = synthetic_dem((28, 26), seed=17, nan_patch=True)
    assert_same(rvt.vis.local_dominance(dem.copy(), min_rad=3, max_rad=8),
                ref_local_dominance(dem.copy(), min_rad=3, max_rad=8))
