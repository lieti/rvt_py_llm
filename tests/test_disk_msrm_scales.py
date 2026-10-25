"""
0.2.0 disk windows: the number of scales of ``rvt.vis.msrm`` with ``kernel_shape="disk"`` and ``"disk_equal_area"``.

The radii of ``msrm`` are ``k ** s`` px for k = i .. n with (docstring of ``msrm``, feature sizes in m)
``i = floor(((feature_min - resolution) / (2 resolution)) ** (1 / s))`` and
``n = ceil(((feature_max - resolution) / (2 resolution)) ** (1 / s))``. The cases of test_disk_oracle.py and
test_disk_threshold.py use feature sizes for which these powers are integers, so floor and ceil agree there and a
wrong rounding of i or n on the disk path would not be seen. Here the powers are not integers (checked in the test,
at least 0.1 from the nearest integer): ceil and floor give different n, and in the i > 0 cases different i.

Radii (``_radii``) are derived in the test from the documented formula and checked against the values written out by
hand. Reference: the brute-force float64 disk mean of test_disk_oracle.py (copied here, mean only): the tile is padded
by floor(r_w) px in edge mode, every offset ``dx^2 + dy^2 <= r_w^2`` of the disk (``r_w = r`` for "disk",
``(2r + 1) / sqrt(pi)`` for "disk_equal_area") is added in float64, NaN pixels are left out and the pixel count of
every window comes from the same mask. The largest radius of every case is >= 8 px (DISK_MIN_RADIUS_PX), so the disk
is used at every radius, also those below 8 px; the mean of the differences of consecutive surfaces telescopes to
``(L_first - L_last) / (number of differences)``. Tolerance: 2 float32 spacings of the largest |z|, as
``_msrm_tol`` of test_disk_oracle.py (each surface and each difference may be rounded to float32).

Each case also checks that its reference differs by more than 100 tolerances from the reference built with the other
rounding of n (floor instead of ceil; square windows if the largest radius then falls below 8 px), so the test tells
the two numbers of scales apart.

Inputs: seeded synthetic tiles <= 128 px on a 1/256 m grid, with and without NaN.
"""

import math

import numpy as np
import pytest

import rvt.vis
from conftest import synthetic_dem

SHAPES = ("disk", "disk_equal_area")
MIN_DISK_RADIUS = 8  # px; DISK_MIN_RADIUS_PX of the 0.2.0 definition

# (resolution m/px, feature_min m, feature_max m, scaling_factor, radii k^s px written out by hand)
CASES = {
    # (9 / 2) ** (1/2) = 2.12: n = 3 (floor: 2, radii 0, 1, 4 px)
    "res1_0_to_10m_s2": (1.0, 1.0, 10.0, 2, (0, 1, 4, 9)),
    # i: (5 / 2) ** (1/2) = 1.58 -> 1; n: (24 / 2) ** (1/2) = 3.46 -> 4 (floor: 3)
    "res1_6_to_25m_s2": (1.0, 6.0, 25.0, 2, (1, 4, 9, 16)),
    # 0.5 m/px: i: (1.5 / 1) ** (1/3) = 1.14 -> 1; n: (19.5 / 1) ** (1/3) = 2.69 -> 3 (floor: 2, radii 1, 8 px)
    "res05_2_to_20m_s3": (0.5, 2.0, 20.0, 3, (1, 8, 27)),
}


# ----------------------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------------------

def _tile(name):
    """Seeded tiles: "plain" 128 x 96 px, "nan" 101 x 117 px with NaN; relief 20 m on a 1/256 m grid."""
    if name == "plain":
        return synthetic_dem((128, 96), seed=211, relief=20.0, grid=True)
    z = synthetic_dem((101, 117), seed=212, relief=20.0, grid=True, nan_patch=True)
    z[40:62, 50:75] = np.nan  # a block larger than an r = 8 px window
    z[85:101, 0:6] = np.nan  # a block on the border (bottom left corner)
    z[0, 30:34] = np.nan  # part of the first row
    z[17, 100] = np.nan
    return z


TILES = ("plain", "nan")


# ----------------------------------------------------------------------------------------------------------------------
# radii from the documented formula
# ----------------------------------------------------------------------------------------------------------------------

def _powers(resolution, feature_min, feature_max, scaling):
    feature_min = max(feature_min, resolution)
    p_min = ((feature_min - resolution) / (2 * resolution)) ** (1 / scaling)
    p_max = ((feature_max - resolution) / (2 * resolution)) ** (1 / scaling)
    return p_min, p_max


def _radii(resolution, feature_min, feature_max, scaling, round_n=math.ceil):
    """Radii k^s px for k = floor(p_min) .. round_n(p_max) (``round_n`` = ceil as documented)."""
    p_min, p_max = _powers(resolution, feature_min, feature_max, scaling)
    return tuple(k ** scaling for k in range(math.floor(p_min), round_n(p_max) + 1))


# ----------------------------------------------------------------------------------------------------------------------
# brute-force reference (mean only, from test_disk_oracle.py)
# ----------------------------------------------------------------------------------------------------------------------

def _window(r, shape, always_disk=False):
    """(r_w, disk) of the window of radius r px: square below 8 px (unless always_disk), else the disk of the shape."""
    if shape == "square" or (r < MIN_DISK_RADIUS and not always_disk):
        return float(r), False
    if shape == "disk":
        return float(r), True
    return (2 * r + 1) / math.sqrt(math.pi), True


def _offsets(r_w, disk):
    """Integer offsets (dy, dx) of the window: |dy|, |dx| <= floor(r_w), and dx^2 + dy^2 <= r_w^2 for a disk."""
    big = int(math.floor(r_w))
    dy, dx = np.mgrid[-big:big + 1, -big:big + 1]
    keep = (dx * dx + dy * dy <= r_w * r_w) if disk else np.ones(dy.shape, dtype=bool)
    if disk:
        # no integer d^2 lies within 1e-6 of r_w^2 for the radii used here, so the float64 test is unambiguous
        d2 = (dx * dx + dy * dy).astype(np.float64)
        assert np.min(np.abs(d2 - r_w * r_w)) > 1e-6 or float(r_w).is_integer()
    return list(zip(dy[keep].tolist(), dx[keep].tolist()))


def _ref_mean_filter(tile, r, shape, always_disk=False):
    """Window mean in float64 with edge padding, NaN left out (count from the mask); NaN where the tile is NaN."""
    z = np.asarray(tile, dtype=np.float64)
    if r == 0:
        return z.copy()
    r_w, disk = _window(r, shape, always_disk)
    pad = int(math.floor(r_w))
    zp = np.pad(z, pad, mode="edge")
    valid = ~np.isnan(zp)
    zp0 = np.where(valid, zp, 0.0)
    n_rows, n_cols = z.shape
    count = np.zeros(z.shape)
    s1 = np.zeros(z.shape)
    for dy, dx in _offsets(r_w, disk):
        sl = (slice(pad + dy, pad + dy + n_rows), slice(pad + dx, pad + dx + n_cols))
        count += valid[sl]
        s1 += zp0[sl]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / count
    mean[np.isnan(z)] = np.nan
    return mean


def _ref_msrm(dem, radii, shape):
    """Telescoped mean of the differences of consecutive surfaces; disk at every radius when the largest is >= 8 px,
    square at every radius otherwise."""
    always_disk = max(radii) >= MIN_DISK_RADIUS
    win_shape = shape if always_disk else "square"
    first = _ref_mean_filter(dem, radii[0], win_shape, always_disk)
    last = _ref_mean_filter(dem, radii[-1], win_shape, always_disk)
    return (first - last) / (len(radii) - 1)


def _msrm_tol(dem):
    # each surface may be rounded to float32 (half a spacing each) and the difference of two rounded in float32
    # (half a spacing): <= 1.5 spacings of the largest |z| per difference, and so for their mean
    return 2.0 * float(np.spacing(np.float32(np.nanmax(np.abs(dem)))))


# ----------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_case_powers_are_not_integers_and_radii_follow_the_formula(case):
    resolution, f_min, f_max, scaling, radii = CASES[case]
    p_min, p_max = _powers(resolution, f_min, f_max, scaling)
    assert abs(p_max - round(p_max)) > 0.1  # ceil(p_max) != floor(p_max): n depends on the rounding
    if radii[0] > 0:
        assert abs(p_min - round(p_min)) > 0.1  # i > 0 and floor(p_min) != ceil(p_min)
    assert _radii(resolution, f_min, f_max, scaling) == radii
    assert max(radii) >= MIN_DISK_RADIUS  # the disk applies at every radius
    assert _radii(resolution, f_min, f_max, scaling, round_n=math.floor) == radii[:-1]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("tile", TILES)
def test_msrm_disk_number_of_scales_with_non_integer_powers(tile, case, shape):
    resolution, f_min, f_max, scaling, _ = CASES[case]
    radii = _radii(resolution, f_min, f_max, scaling)
    dem = _tile(tile)
    before = dem.copy()
    out = rvt.vis.msrm(dem.copy(), resolution, f_min, f_max, scaling, kernel_shape=shape)
    assert np.array_equal(dem, before, equal_nan=True)
    assert out.dtype == np.float64
    assert out.shape == dem.shape
    assert np.array_equal(np.isnan(out), np.isnan(dem))
    ref = _ref_msrm(dem, radii, shape)
    tol = _msrm_tol(dem)
    diff = np.abs(out - ref)
    assert np.nanmax(diff) <= tol, "max |out - ref| = {} > {}".format(np.nanmax(diff), tol)
    # the other rounding of n (one scale fewer) gives a reference far outside the tolerance
    ref_floor = _ref_msrm(dem, _radii(resolution, f_min, f_max, scaling, round_n=math.floor), shape)
    assert np.nanmax(np.abs(ref_floor - ref)) > 100 * tol
