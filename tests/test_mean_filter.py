"""
Mean filter: rvt.vis.mean_filter (box sums by slicing the integral images over the output window, constant pixel
count without NaN) and its users slrm / msrm against the np.roll formulation of tag baseline-fixed.
"""

import numpy as np
import pytest

import rvt.vis
from conftest import assert_bitwise, assert_same, synthetic_dem


# ----------------------------------------------------------------------------------------------------------------------
# reference implementation (baseline-fixed, transcribed)
# ----------------------------------------------------------------------------------------------------------------------

def ref_integral_image(dem, data_type=np.float64):
    dem = dem.astype(data_type)
    return dem.cumsum(axis=0).cumsum(axis=1)


def ref_mean_filter(dem, kernel_radius):
    radius_cell = int(kernel_radius)
    if kernel_radius == 0:
        return dem
    idx_nan_dem = np.isnan(dem)
    dem_pad = np.pad(dem, (radius_cell + 1, radius_cell), mode="edge")
    idx_nan_dem_pad = np.isnan(dem_pad)
    dem_pad[idx_nan_dem_pad] = 0
    dem_i_nr_pixels = np.ones(dem_pad.shape)
    dem_i_nr_pixels[idx_nan_dem_pad] = 0
    dem_i_nr_pixels = ref_integral_image(dem_i_nr_pixels, np.int64)
    dem_i1 = ref_integral_image(dem_pad)
    kernel_nr_pix_arr = (np.roll(dem_i_nr_pixels, (radius_cell, radius_cell), axis=(0, 1)) +
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    mean_out = (np.roll(dem_i1, (radius_cell, radius_cell), axis=(0, 1)) +
                np.roll(dem_i1, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                np.roll(dem_i1, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                np.roll(dem_i1, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_out = mean_out / kernel_nr_pix_arr
    mean_out = mean_out.astype(np.float32)
    mean_out = mean_out[radius_cell:-(radius_cell + 1), radius_cell:-(radius_cell + 1)]
    mean_out[idx_nan_dem] = np.nan
    return mean_out


def ref_slrm(dem, radius_cell=20, ve_factor=1):
    dem = dem.astype(np.float32)
    dem = dem * ve_factor
    return dem - ref_mean_filter(dem, radius_cell)


def ref_msrm(dem, resolution, feature_min, feature_max, scaling_factor, ve_factor=1):
    dem = dem.astype(np.float32)
    dem = dem * ve_factor
    if feature_min < resolution:
        feature_min = resolution
    scaling_factor = int(scaling_factor)
    i = int(np.floor(((feature_min - resolution) / (2 * resolution)) ** (1 / scaling_factor)))
    n = int(np.ceil(((feature_max - resolution) / (2 * resolution)) ** (1 / scaling_factor)))
    relief_models_sum = np.zeros(dem.shape)
    nr_relief_models = 0
    last_lpf_surface = 0
    for ndx in range(i, n + 1, 1):
        lpf_surface = ref_mean_filter(dem, ndx ** scaling_factor)
        if not ndx == i:
            relief_models_sum += (last_lpf_surface - lpf_surface)
            nr_relief_models += 1
        last_lpf_surface = lpf_surface
    return relief_models_sum / nr_relief_models


# ----------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [1, 2, 5, 20, 40])
@pytest.mark.parametrize("nan_patch", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mean_filter_matches_reference(radius, nan_patch, dtype):
    # radius 40 exceeds the tile size (edge padding reaches the far side)
    dem = synthetic_dem((33, 47), seed=41, nan_patch=nan_patch, dtype=dtype)
    assert_same(rvt.vis.mean_filter(dem.copy(), radius), ref_mean_filter(dem.copy(), radius))


def test_mean_filter_radius_zero_and_all_nan_window():
    dem = synthetic_dem((20, 21), seed=42)
    assert rvt.vis.mean_filter(dem, 0) is dem
    dem[5:12, 6:14] = np.nan  # windows of radius 2 fully inside NaN: 0 / 0
    assert_same(rvt.vis.mean_filter(dem.copy(), 2), ref_mean_filter(dem.copy(), 2))


def test_slrm_and_msrm_match_reference():
    dem = synthetic_dem((60, 70), seed=43, nan_patch=True)
    assert_same(rvt.vis.slrm(dem.copy(), radius_cell=10), ref_slrm(dem.copy(), radius_cell=10))
    assert_same(rvt.vis.slrm(dem.copy(), radius_cell=25, ve_factor=2), ref_slrm(dem.copy(), 25, 2))
    assert_same(rvt.vis.msrm(dem.copy(), resolution=0.5, feature_min=1, feature_max=8, scaling_factor=2),
                ref_msrm(dem.copy(), 0.5, 1, 8, 2))


def test_mean_filter_offset_and_determinism():
    dem = synthetic_dem((30, 40), seed=44, nan_patch=True)
    assert_bitwise(rvt.vis.mean_filter(dem.copy(), 7), rvt.vis.mean_filter(dem.copy(), 7))
    dem1 = dem + np.float32(2500.0)
    assert_same(rvt.vis.mean_filter(dem1.copy(), 7), ref_mean_filter(dem1.copy(), 7))
