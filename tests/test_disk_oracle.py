"""
0.2.0 disk windows: ``rvt.vis.mean_filter``, ``max_elevation_deviation`` and ``msrm`` with
``kernel_shape="disk"`` and ``"disk_equal_area"`` against a brute-force float64 reference kept in this file.

The window definition being tested: a window of radius r px is the exact discrete disk ``dx^2 + dy^2 <= r_w^2``
(dx, dy integer offsets from the centre pixel), with ``r_w = r`` for "disk" and ``r_w = (2r + 1) / sqrt(pi)`` for
"disk_equal_area" (the disk with the area of the (2r + 1)^2 square). Windows with r >= 8 px (DISK_MIN_RADIUS_PX) are
disks, smaller ones stay square; ``msrm`` uses one shape for all its radii, the disk at every radius when its
largest radius is >= 8 px.

Reference (``_window_stats``): the tile is padded by floor(r_w) px with the padding mode of the function
(``mean_filter`` and ``msrm``: edge; ``max_elevation_deviation``: symmetric; the padding values at a given distance
from the edge do not depend on the pad width); for every offset (dy, dx) of the footprint the shifted padded tile is
added in float64. NaN pixels are left out of the sums and the pixel count of every window is counted from the same
mask. The variance is computed in two passes (squared deviations from the window mean), so the reference has no
cancellation at any elevation. DEV = (z - mean) / (std + 1e-6) as in the square path, and DEV_max keeps the running
maximum where ``|max| >= |dev|`` (radii in increasing order), the tie rule of the square path.

Tolerances (CIRC_BRIEF_LLM P1): the float64 result of the function may differ from the reference by 1e-9 m (window
mean) and 1e-7 (DEV) before the final rounding to float32. Rounding to nearest is monotone, so the float32 output
must lie in [float32(ref - tol), float32(ref + tol)] (``assert_f32_within``); this is the tolerance itself, not a
widening of it. The window std is checked through DEV (it is not returned by any function). MSRM returns float64
but adds differences of mean-filtered surfaces that the square path rounds to float32; its tolerance is
2 float32 spacings of the largest |z| (``_msrm_tol``).

Inputs: seeded synthetic tiles <= 128 px on a 1/256 m grid (exact in float32, also after +2 500 m), with and
without NaN (a large block, a block on the border, single pixels); borders are compared like the interior.
"""

import math

import numpy as np
import pytest

import rvt.vis
from conftest import synthetic_dem

SHAPES = ("disk", "disk_equal_area")
RADII = (8, 9, 13, 20)
MIN_DISK_RADIUS = 8  # px; DISK_MIN_RADIUS_PX of the 0.2.0 definition


# ----------------------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------------------

def _tile(name):
    """Seeded tiles: "plain" 128 x 96 px, "nan" 101 x 117 px with NaN; relief 20 m on a 1/256 m grid."""
    if name == "plain":
        return synthetic_dem((128, 96), seed=201, relief=20.0, grid=True)
    z = synthetic_dem((101, 117), seed=202, relief=20.0, grid=True, nan_patch=True)
    z[40:62, 50:75] = np.nan  # a block larger than an r = 8 px window
    z[85:101, 0:6] = np.nan  # a block on the border (bottom left corner)
    z[0, 30:34] = np.nan  # part of the first row
    z[17, 100] = np.nan
    return z


TILES = ("plain", "nan")


ISLANDS = [(r, c) for r in (12, 37, 62) for c in (55, 77)]  # isolated valid pixels, >= 22 px apart


def _islands_tile():
    """96 x 100 px: terrain (2 m relief) in the first 30 columns, NaN elsewhere except the ISLANDS pixels, each more
    than r_w = 15.2 px (disk_equal_area at r = 13 px) from every other valid pixel: their windows hold one pixel."""
    z = synthetic_dem((96, 100), seed=205, relief=2.0, grid=True)
    out = np.full(z.shape, np.nan, dtype=np.float32)
    out[:, :30] = z[:, :30]
    for r, c in ISLANDS:
        out[r, c] = z[r, c]
    return out


def _low_relief_tile(nan):
    """Relief of a few cm (window std of a few cm): DEV there is sensitive to cancellation at +2 500 m."""
    z = synthetic_dem((96, 88), seed=203, relief=0.5, grid=True, nan_patch=nan)
    if nan:
        z[30:44, 40:52] = np.nan
    return z


# ----------------------------------------------------------------------------------------------------------------------
# brute-force reference
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


def _window_stats(tile, mode, r_w, disk):
    """Window mean, variance (two passes) and valid-pixel count of every pixel, float64, NaN left out."""
    z = np.asarray(tile, dtype=np.float64)
    pad = int(math.floor(r_w))
    zp = np.pad(z, pad, mode=mode)
    valid = ~np.isnan(zp)
    zp0 = np.where(valid, zp, 0.0)
    n_rows, n_cols = z.shape
    offsets = _offsets(r_w, disk)
    count = np.zeros(z.shape)
    s1 = np.zeros(z.shape)
    for dy, dx in offsets:
        sl = (slice(pad + dy, pad + dy + n_rows), slice(pad + dx, pad + dx + n_cols))
        count += valid[sl]
        s1 += zp0[sl]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / count
    ss = np.zeros(z.shape)
    for dy, dx in offsets:
        sl = (slice(pad + dy, pad + dy + n_rows), slice(pad + dx, pad + dx + n_cols))
        d = np.where(valid[sl], zp0[sl] - mean, 0.0)
        ss += d * d
    with np.errstate(divide="ignore", invalid="ignore"):
        var = ss / count
    return mean, var, count


def _ref_mean_filter(tile, r, shape, always_disk=False):
    mean, _, _ = _window_stats(tile, "edge", *_window(r, shape, always_disk))
    if r == 0:
        mean = np.asarray(tile, dtype=np.float64).copy()
    mean[np.isnan(tile)] = np.nan
    return mean


def _ref_dev(tile, r, shape):
    """DEV of one radius, as topographic_dev defines it: (z - mean) / (sqrt(var) + 1e-6)."""
    mean, var, _ = _window_stats(tile, "symmetric", *_window(r, shape))
    z = np.asarray(tile, dtype=np.float64)
    dev = (z - mean) / (np.sqrt(var) + 1e-6)
    dev[np.isnan(tile)] = np.nan
    return dev


def _ref_dev_max(tile, radii, shape):
    """DEV_max over the radii in order, tie rule of the square path; also returns the DEV of every radius."""
    devs = [_ref_dev(tile, r, shape) for r in radii]
    dev_max = devs[0].copy()
    for dev in devs[1:]:
        dev_max = np.where(np.abs(dev_max) >= np.abs(dev), dev_max, dev)
    return dev_max, devs


# ----------------------------------------------------------------------------------------------------------------------
# comparisons
# ----------------------------------------------------------------------------------------------------------------------

def _within(out, ref, tol):
    """out (float32) is the float32 rounding of some value within tol of ref (float64); NaN where ref is NaN."""
    lo = (ref - tol).astype(np.float32)
    hi = (ref + tol).astype(np.float32)
    return ((out >= lo) & (out <= hi)) | (np.isnan(ref) & np.isnan(out))


def assert_f32_within(out, ref, tol):
    out = np.asarray(out)
    assert out.dtype == np.float32
    assert out.shape == ref.shape
    assert np.array_equal(np.isnan(out), np.isnan(ref)), "NaN mask differs from the reference"
    ok = _within(out, ref, tol)
    if not ok.all():
        diff = np.abs(out.astype(np.float64) - ref)
        raise AssertionError("{} of {} pixels outside float32([ref - {}, ref + {}]); max |out - ref| = {}".format(
            int(np.count_nonzero(~ok)), ok.size, tol, tol, float(np.nanmax(np.where(ok, 0.0, diff)))))


def assert_dev_max_within(out, ref_max, devs, tol):
    """As assert_f32_within for DEV_max; where two radii have |DEV| within 2 tol of the maximum (a near tie that a
    perturbation of tol may decide either way) the DEV of any of these radii is accepted. Such pixels must be rare."""
    out = np.asarray(out)
    assert out.dtype == np.float32
    assert np.array_equal(np.isnan(out), np.isnan(ref_max)), "NaN mask differs from the reference"
    ok = _within(out, ref_max, tol)
    near = np.zeros(ok.shape, dtype=bool)
    top = np.abs(ref_max)
    for dev in devs:
        cand = (np.abs(dev) >= top - 2 * tol) & ~np.isnan(dev)
        accepted = cand & _within(out, dev, tol) & ~ok
        near |= accepted
    assert np.count_nonzero(near) <= 0.001 * ok.size
    bad = ~(ok | near)
    if bad.any():
        diff = np.abs(out.astype(np.float64) - ref_max)
        raise AssertionError("{} of {} pixels outside float32([ref - {}, ref + {}]); max |out - ref| = {}".format(
            int(np.count_nonzero(bad)), bad.size, tol, tol, float(np.nanmax(np.where(bad, diff, 0.0)))))


# ----------------------------------------------------------------------------------------------------------------------
# (a) mean_filter and max_elevation_deviation against the reference
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("r", RADII)
@pytest.mark.parametrize("tile", TILES)
def test_mean_filter_disk_equals_brute_force(tile, r, shape):
    dem = _tile(tile)
    before = dem.copy()
    out = rvt.vis.mean_filter(dem, r, kernel_shape=shape)
    assert np.array_equal(dem, before, equal_nan=True)  # the input is not modified
    assert_f32_within(out, _ref_mean_filter(dem, r, shape), 1e-9)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("r", RADII)
@pytest.mark.parametrize("tile", TILES)
def test_max_elevation_deviation_one_disk_radius_equals_brute_force(tile, r, shape):
    dem = _tile(tile)
    before = dem.copy()
    out = rvt.vis.max_elevation_deviation(dem, r, r, 1, kernel_shape=shape)
    assert np.array_equal(dem, before, equal_nan=True)  # the input is not modified
    ref_max, devs = _ref_dev_max(dem, [r], shape)
    assert_dev_max_within(out, ref_max, devs, 1e-7)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tile", TILES)
def test_max_elevation_deviation_mixed_band_square_below_8_px(tile, shape):
    # radii 5, 10, 15, 20 px: the largest is >= 8 px, so 5 px is a square window and 10-20 px are disks (STATUS J1)
    dem = _tile(tile)
    out = rvt.vis.max_elevation_deviation(dem, 5, 20, 5, kernel_shape=shape)
    ref_max, devs = _ref_dev_max(dem, [5, 10, 15, 20], shape)
    assert_dev_max_within(out, ref_max, devs, 1e-7)


@pytest.mark.parametrize("shape", SHAPES)
def test_max_elevation_deviation_meso_band_5_to_50_px(shape):
    # the default meso band of MSTP (5..50 px step 5, i.e. 1.25-12.5 m at 0.25 m/px) on a 64 x 72 px tile with NaN;
    # the windows reach far beyond the tile (equal-area r_w = 57 px at r = 50 px): padding everywhere
    dem = synthetic_dem((64, 72), seed=204, relief=20.0, grid=True, nan_patch=True)
    out = rvt.vis.max_elevation_deviation(dem, 5, 50, 5, kernel_shape=shape)
    ref_max, devs = _ref_dev_max(dem, list(range(5, 51, 5)), shape)
    assert_dev_max_within(out, ref_max, devs, 1e-7)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("r", [8, 13])
def test_windows_holding_a_single_valid_pixel(r, shape):
    # the pixel count comes from the NaN mask: a window that holds only its centre has count 1, mean = z, var = 0
    # and DEV = 0 / (0 + 1e-6) = 0
    dem = _islands_tile()
    ref_mean = _ref_mean_filter(dem, r, shape)
    ref_max, devs = _ref_dev_max(dem, [r], shape)
    for p in ISLANDS:
        assert ref_mean[p] == dem[p] and ref_max[p] == 0.0  # the reference sees one pixel there
    assert_f32_within(rvt.vis.mean_filter(dem, r, kernel_shape=shape), ref_mean, 1e-9)
    assert_dev_max_within(rvt.vis.max_elevation_deviation(dem, r, r, 1, kernel_shape=shape), ref_max, devs, 1e-7)


# ----------------------------------------------------------------------------------------------------------------------
# (c) elevation offset +2 500 m
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("nan", [False, True], ids=["plain", "nan"])
def test_offset_2500_m_dev_and_mean(shape, nan):
    z0 = _low_relief_tile(nan)
    z1 = z0 + np.float32(2500.0)
    assert np.array_equal(z1 - np.float32(2500.0), z0, equal_nan=True)  # the offset is exact on the 1/256 m grid
    # DEV: the reference at 0 m; the function at 0 m and at +2 500 m within 1e-7 of it
    ref_max, devs = _ref_dev_max(z0, [8, 12, 16, 20], shape)
    assert_dev_max_within(rvt.vis.max_elevation_deviation(z0, 8, 20, 4, kernel_shape=shape), ref_max, devs, 1e-7)
    assert_dev_max_within(rvt.vis.max_elevation_deviation(z1, 8, 20, 4, kernel_shape=shape), ref_max, devs, 1e-7)
    # mean filter: the reference at 0 m shifted by 2 500 m, within 1e-9 m before the rounding to float32
    ref_mean = _ref_mean_filter(z0, 13, shape)
    assert_f32_within(rvt.vis.mean_filter(z1, 13, kernel_shape=shape), ref_mean + 2500.0, 1e-9)
    assert_f32_within(rvt.vis.mean_filter(z0, 13, kernel_shape=shape), ref_mean, 1e-9)


# ----------------------------------------------------------------------------------------------------------------------
# (e) msrm: one shape for all radii, the disk at every radius when the largest radius is >= 8 px
# ----------------------------------------------------------------------------------------------------------------------

# (resolution m/px, feature_min m, feature_max m, scaling_factor, radii k^s px written out by hand):
# i = floor(((feature_min - res) / (2 res)) ** (1/s)), n = ceil(((feature_max - res) / (2 res)) ** (1/s))
MSRM_CASES = {
    "radii_0_1_4_9_16": (1.0, 1.0, 33.0, 2, (0, 1, 4, 9, 16)),  # i = 0, n = 4
    "radii_1_4_9_16": (1.0, 3.0, 33.0, 2, (1, 4, 9, 16)),  # i = 1, n = 4
    "radii_4_9_16": (1.0, 9.0, 33.0, 2, (4, 9, 16)),  # i = 2, n = 4
}


def _msrm_tol(dem):
    # each surface may be rounded to float32 (half a spacing each) and the difference of two rounded in float32
    # (half a spacing): <= 1.5 spacings of the largest |z| per difference, and so for their mean
    return 2.0 * float(np.spacing(np.float32(np.nanmax(np.abs(dem)))))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("case", sorted(MSRM_CASES))
@pytest.mark.parametrize("tile", TILES)
def test_msrm_uses_the_disk_at_every_radius(tile, case, shape):
    resolution, f_min, f_max, scaling, radii = MSRM_CASES[case]
    dem = _tile(tile)
    out = rvt.vis.msrm(dem.copy(), resolution, f_min, f_max, scaling, kernel_shape=shape)
    assert out.dtype == np.float64
    assert out.shape == dem.shape
    # the disk also at the radii below 8 px (0 px is the pixel itself); the mean of the differences of consecutive
    # surfaces telescopes to (L_first - L_last) / (number of differences)
    surfaces = [_ref_mean_filter(dem, r, shape, always_disk=True) for r in radii]
    ref = (surfaces[0] - surfaces[-1]) / (len(radii) - 1)
    assert np.array_equal(np.isnan(out), np.isnan(dem))
    diff = np.abs(out - ref)
    tol = _msrm_tol(dem)
    assert np.nanmax(diff) <= tol, "max |out - ref| = {} > {}".format(np.nanmax(diff), tol)
    disk_first = _offsets(*_window(radii[0], shape, always_disk=True))
    if len(disk_first) != len(_offsets(radii[0], False)):
        # the intermediate surfaces cancel in the sum, the first does not: a square first window would differ from
        # the disk by far more than the tolerance, so this case tells "disk at every radius" apart (not at 0 px, and
        # not for disk_equal_area at 1 px, whose disk r_w = 1.69 px is the full 3 x 3 square)
        ref_square_first = (_ref_mean_filter(dem, radii[0], "square") - surfaces[-1]) / (len(radii) - 1)
        assert np.nanmax(np.abs(ref_square_first - ref)) > 100 * tol
