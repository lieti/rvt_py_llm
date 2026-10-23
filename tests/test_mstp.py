"""
MSTP deviation: rvt.vis.max_elevation_deviation (sliced box sums `_mstp_box_sum`, numba kernel
`_mstp_dev_max_kernel`) and rvt.vis.mstp against the np.roll / np.where formulation of tag baseline-fixed.
"""

import numpy as np
import pytest

import rvt.vis
from conftest import assert_bitwise, assert_same, kernel_ran, synthetic_dem

# NaN cells give 0 / 0 in the deviation (outside an errstate), in the reference as in rvt
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")


# ----------------------------------------------------------------------------------------------------------------------
# reference implementation (baseline-fixed, transcribed)
# ----------------------------------------------------------------------------------------------------------------------

def ref_integral_image(dem, data_type=np.float64):
    dem = dem.astype(data_type)
    return dem.cumsum(axis=0).cumsum(axis=1)


def ref_topographic_dev(dem, dem_i_nr_pixels, dem_i1, dem_i2, kernel_radius):
    radius_cell = int(kernel_radius)
    if radius_cell <= 0:
        return dem

    kernel_nr_pix_arr = (np.roll(dem_i_nr_pixels, (radius_cell, radius_cell), axis=(0, 1)) +
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    dem_mean = (np.roll(dem_i1, (radius_cell, radius_cell), axis=(0, 1)) +
                np.roll(dem_i1, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                np.roll(dem_i1, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                np.roll(dem_i1, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    with np.errstate(divide='ignore', invalid='ignore'):
        dem_mean = dem_mean / kernel_nr_pix_arr
    dem_std = (np.roll(dem_i2, (radius_cell, radius_cell), axis=(0, 1)) +
               np.roll(dem_i2, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
               np.roll(dem_i2, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
               np.roll(dem_i2, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    with np.errstate(divide='ignore', invalid='ignore'):
        dem_std = np.sqrt(np.abs(dem_std / kernel_nr_pix_arr - dem_mean ** 2))
    dev_out = (np.roll(dem, (-1, -1), axis=(0, 1)) - dem_mean) / (dem_std + 1e-6)
    return dev_out


def ref_max_elevation_deviation(dem, minimum_radius, maximum_radius, step):
    minimum_radius = int(minimum_radius)
    maximum_radius = int(maximum_radius)
    step = int(step)
    idx_nan_dem = np.isnan(dem)
    dem_pad = np.pad(dem, (maximum_radius + 1, maximum_radius), mode="symmetric")
    idx_nan_dem_pad = np.isnan(dem_pad)
    dem_pad[idx_nan_dem_pad] = 0
    dem_i_nr_pixels = np.ones(dem_pad.shape)
    dem_i_nr_pixels[idx_nan_dem_pad] = 0
    dem_i_nr_pixels = ref_integral_image(dem_i_nr_pixels, np.int64)
    dem_i1 = ref_integral_image(dem_pad)
    dem_i2 = ref_integral_image(dem_pad.astype(np.float64) ** 2)
    for kernel_radius in range(minimum_radius, maximum_radius + 1, step):
        dev = ref_topographic_dev(dem_pad, dem_i_nr_pixels, dem_i1, dem_i2, kernel_radius)[
              maximum_radius:-(maximum_radius + 1),
              maximum_radius:-(maximum_radius + 1)]
        if kernel_radius == minimum_radius:
            dev_max_out = dev
        else:
            dev_max_out = np.where(np.abs(dev_max_out) >= np.abs(dev), dev_max_out, dev)
    dev_max_out[idx_nan_dem] = np.nan
    return dev_max_out.astype(np.float32)


def ref_mstp(dem, local_scale, meso_scale, broad_scale, lightness=1.2, ve_factor=1):
    dem = dem.astype(np.float32)
    dem = dem * ve_factor
    local_dev = ref_max_elevation_deviation(dem, local_scale[0], local_scale[1], local_scale[2])
    meso_dev = ref_max_elevation_deviation(dem, meso_scale[0], meso_scale[1], meso_scale[2])
    broad_dev = ref_max_elevation_deviation(dem, broad_scale[0], broad_scale[1], broad_scale[2])
    red = 1 - np.exp(-lightness * np.abs(broad_dev))
    green = 1 - np.exp(-lightness * np.abs(meso_dev))
    blue = 1 - np.exp(-lightness * np.abs(local_dev))
    for band in (red, green, blue):
        band[band < 0] = 0
        band[band > 1] = 1
    return np.asarray([red, green, blue])


# ----------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------

RADII = [(1, 5, 1), (5, 15, 5), (2, 9, 3), (3, 3, 1), (0, 3, 1), (4, 12, 4)]


@pytest.mark.parametrize("radii", RADII)
@pytest.mark.parametrize("nan_patch", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_max_elevation_deviation_matches_reference(radii, nan_patch, dtype):
    dem = synthetic_dem((37, 52), seed=1, nan_patch=nan_patch, dtype=dtype)
    expected = ref_max_elevation_deviation(dem.copy(), *radii)
    actual = rvt.vis.max_elevation_deviation(dem.copy(), *radii)
    assert_same(actual, expected)


def test_max_elevation_deviation_wide_tile():
    # radius larger than the tile height (symmetric padding beyond the edge) and a wide tile
    dem = synthetic_dem((9, 300), seed=2, nan_patch=True)
    for radii in [(5, 20, 5), (1, 4, 1)]:
        assert_same(rvt.vis.max_elevation_deviation(dem.copy(), *radii),
                    ref_max_elevation_deviation(dem.copy(), *radii))


def test_max_elevation_deviation_constant_dem():
    # zero variance everywhere: dev = 0 / 1e-6
    dem = np.full((20, 23), 312.5, dtype=np.float32)
    assert_same(rvt.vis.max_elevation_deviation(dem.copy(), 1, 4, 1),
                ref_max_elevation_deviation(dem.copy(), 1, 4, 1))


@pytest.mark.parametrize("nan_patch", [False, True])
def test_mstp_matches_reference(nan_patch):
    dem = synthetic_dem((41, 47), seed=3, nan_patch=nan_patch)
    scales = dict(local_scale=(1, 5, 1), meso_scale=(5, 15, 5), broad_scale=(15, 30, 15))
    actual = rvt.vis.mstp(dem.copy(), **scales)
    expected = ref_mstp(dem.copy(), scales["local_scale"], scales["meso_scale"], scales["broad_scale"])
    assert_same(actual, expected)


def test_mstp_elevation_offset_invariance():
    # The same terrain at 0 m and at +2500 m (exact in float32): the deviation is translation invariant up to the
    # float64 rounding of the E[x^2] - E[x]^2 variance. The optimized code reproduces the reference at both offsets.
    dem0 = synthetic_dem((45, 50), seed=4, relief=8.0, grid=True)
    dem1 = dem0 + np.float32(2500.0)
    assert np.array_equal(dem1 - np.float32(2500.0), dem0)  # the offset is exact
    for radii in [(1, 5, 1), (5, 20, 5)]:
        dev0 = rvt.vis.max_elevation_deviation(dem0.copy(), *radii)
        dev1 = rvt.vis.max_elevation_deviation(dem1.copy(), *radii)
        assert_same(dev1, ref_max_elevation_deviation(dem1.copy(), *radii))
        # measured: about 1 float32 ulp of the deviation (|dev| < 3); float32 squares would fail this by far
        assert np.max(np.abs(dev1.astype(np.float64) - dev0)) < 1e-4
    scales = dict(local_scale=(1, 5, 1), meso_scale=(5, 15, 5), broad_scale=(15, 30, 15))
    m0 = rvt.vis.mstp(dem0.copy(), **scales)
    m1 = rvt.vis.mstp(dem1.copy(), **scales)
    # 8-bit output (as e3MSTP / e4MSTP use it) differs by at most 1 DN
    assert np.max(np.abs(np.round(m1 * 255.0) - np.round(m0 * 255.0))) <= 1


def test_mstp_deterministic():
    dem = synthetic_dem((40, 44), seed=5, nan_patch=True)
    scales = dict(local_scale=(1, 5, 1), meso_scale=(5, 15, 5), broad_scale=(15, 30, 15))
    assert_bitwise(rvt.vis.mstp(dem.copy(), **scales), rvt.vis.mstp(dem.copy(), **scales))


@pytest.mark.parametrize("nan_patch", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_max_elevation_deviation_accel_equals_numpy(run_both, nan_patch, dtype):
    dem = synthetic_dem((33, 61), seed=6, nan_patch=nan_patch, dtype=dtype)
    out_accel, out_numpy = run_both(lambda: rvt.vis.max_elevation_deviation(dem.copy(), 2, 10, 2),
                                    kernels=[rvt.vis._mstp_dev_max_kernel])
    assert_bitwise(out_accel, out_numpy)
    assert_same(out_numpy, ref_max_elevation_deviation(dem.copy(), 2, 10, 2))


def test_max_elevation_deviation_numpy_path(numpy_path):
    dem = synthetic_dem((30, 35), seed=7, nan_patch=True)
    assert_same(rvt.vis.max_elevation_deviation(dem.copy(), 1, 6, 1),
                ref_max_elevation_deviation(dem.copy(), 1, 6, 1))


def test_max_elevation_deviation_kernel_threads(run_both, monkeypatch):
    dem = synthetic_dem((30, 40), seed=8)
    outs = []
    for n in ("1", "2"):
        monkeypatch.setenv("RVT_NUM_THREADS", n)
        outs.append(run_both(lambda: rvt.vis.max_elevation_deviation(dem.copy(), 1, 5, 1),
                             kernels=[rvt.vis._mstp_dev_max_kernel])[0])
    assert kernel_ran(rvt.vis._mstp_dev_max_kernel)
    assert_bitwise(outs[0], outs[1])
