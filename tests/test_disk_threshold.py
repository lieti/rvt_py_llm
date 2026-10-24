"""
0.2.0 disk windows: below DISK_MIN_RADIUS_PX = 8 px every window stays square, bitwise.

Windows smaller than 8 px are discretisation-dominated as disks (review C-15: disk DEV ratio 1.51 at r = 2 px), so
``kernel_shape="disk"`` and ``"disk_equal_area"`` leave them on the square code: the result must be bitwise the
``v0.1.1`` (default) result of the same call. Covered: ``mean_filter`` at r = 0..7 px, ``max_elevation_deviation``
whose largest radius is < 8 px, the blue (local, 1..5 px) band of ``mstp``, ``msrm`` whose largest radius k^s is
< 8 px. Also: ``kernel_shape="square"`` passed explicitly gives the default result bitwise for all five functions,
also at radii >= 8 px.

Expected values are the default calls of the same functions (the square path, frozen by the 0.1.1 tests); no value
is pasted from a run.
"""

import numpy as np
import pytest

import rvt.vis
from conftest import assert_bitwise, synthetic_dem

SHAPES = ("disk", "disk_equal_area")
ALL_SHAPES = ("square",) + SHAPES


def _dem(nan):
    return synthetic_dem((70, 83), seed=301, relief=20.0, nan_patch=nan)


NAN_IDS = ["plain", "nan"]


@pytest.mark.parametrize("nan", [False, True], ids=NAN_IDS)
@pytest.mark.parametrize("shape", ALL_SHAPES)
@pytest.mark.parametrize("r", range(0, 8))
def test_mean_filter_below_8_px_is_the_square_result(r, shape, nan):
    dem = _dem(nan)
    expected = rvt.vis.mean_filter(dem.copy(), r)
    out = rvt.vis.mean_filter(dem, r, kernel_shape=shape)
    if r == 0:
        assert out is dem  # the r = 0 shortcut returns the input itself, as on the square path
    assert_bitwise(out, expected)


@pytest.mark.parametrize("nan", [False, True], ids=NAN_IDS)
@pytest.mark.parametrize("shape", ALL_SHAPES)
@pytest.mark.parametrize("band", [(1, 5, 1), (1, 7, 1), (2, 6, 2), (7, 7, 1), (3, 7, 4)])
def test_max_elevation_deviation_below_8_px_is_the_square_result(band, shape, nan):
    dem = _dem(nan)
    expected = rvt.vis.max_elevation_deviation(dem.copy(), *band)
    out = rvt.vis.max_elevation_deviation(dem.copy(), *band, kernel_shape=shape)
    assert_bitwise(out, expected)


SCALES = dict(local_scale=(1, 5, 1), meso_scale=(5, 20, 5), broad_scale=(10, 40, 10))


@pytest.mark.parametrize("nan", [False, True], ids=NAN_IDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_mstp_local_band_stays_square(shape, nan):
    dem = _dem(nan)
    expected = rvt.vis.mstp(dem.copy(), lightness=1.1, **SCALES)
    out = rvt.vis.mstp(dem.copy(), lightness=1.1, kernel_shape=shape, **SCALES)
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert_bitwise(out[2], expected[2])  # blue = local scale, radii 1..5 px
    # the broad band (10..40 px, all disks) does change: the shape is applied at all
    assert not np.array_equal(out[0], expected[0], equal_nan=True)


# (resolution m/px, feature_min m, feature_max m, scaling_factor): largest radius k^s < 8 px
MSRM_SMALL = {
    "radii_0_1_4": (1.0, 1.0, 9.0, 2),  # i = 0, n = 2: radii 0, 1, 4 px
    "radii_1_to_7": (1.0, 3.0, 15.0, 1),  # i = 1, n = 7: radii 1..7 px
}


@pytest.mark.parametrize("nan", [False, True], ids=NAN_IDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("case", sorted(MSRM_SMALL))
def test_msrm_with_largest_radius_below_8_px_is_the_square_result(case, shape, nan):
    dem = _dem(nan)
    expected = rvt.vis.msrm(dem.copy(), *MSRM_SMALL[case])
    out = rvt.vis.msrm(dem.copy(), *MSRM_SMALL[case], kernel_shape=shape)
    assert_bitwise(out, expected)


@pytest.mark.parametrize("nan", [False, True], ids=NAN_IDS)
def test_explicit_square_is_the_default_for_every_function(nan):
    dem = _dem(nan)
    assert_bitwise(rvt.vis.mean_filter(dem.copy(), 13, kernel_shape="square"), rvt.vis.mean_filter(dem.copy(), 13))
    assert_bitwise(rvt.vis.max_elevation_deviation(dem.copy(), 5, 20, 5, kernel_shape="square"),
                   rvt.vis.max_elevation_deviation(dem.copy(), 5, 20, 5))
    assert_bitwise(rvt.vis.mstp(dem.copy(), kernel_shape="square", **SCALES), rvt.vis.mstp(dem.copy(), **SCALES))
    assert_bitwise(rvt.vis.slrm(dem.copy(), 20, 2, kernel_shape="square"), rvt.vis.slrm(dem.copy(), 20, 2))
    assert_bitwise(rvt.vis.msrm(dem.copy(), 1.0, 1.0, 33.0, 2, kernel_shape="square"),
                   rvt.vis.msrm(dem.copy(), 1.0, 1.0, 33.0, 2))


@pytest.mark.parametrize("shape", SHAPES)
def test_8_px_is_the_first_disk_radius(shape):
    # r = 8 px is a disk (it differs from the square), r = 7 px is not (bitwise square, tested above)
    dem = _dem(False)
    assert not np.array_equal(rvt.vis.mean_filter(dem.copy(), 8, kernel_shape=shape),
                              rvt.vis.mean_filter(dem.copy(), 8))
    assert not np.array_equal(rvt.vis.max_elevation_deviation(dem.copy(), 8, 8, 1, kernel_shape=shape),
                              rvt.vis.max_elevation_deviation(dem.copy(), 8, 8, 1))
