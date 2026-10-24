"""
0.2.0 disk windows: the anisotropy regression of review C-15, the reason for the feature.

Input (C-15): a straight Gaussian bank 1 m high, 3 m = 12 px wide (exp(-d^2 / (2 * (1.5 m)^2)), d = distance from
the bank axis), 0.25 m/px, offset 0 m, once along the x axis (0 deg) and once on the diagonal (45 deg), through
the centre pixel of an odd-sized tile. The crest response ratio is DEV(45 deg) / DEV(0 deg) at the centre pixel
(1 = isotropic). Review numbers (crest pixel, 513 px tile): square 0.855 at r = 16 px = 4 m and 0.847 at r = 48 px
= 12 m; disk 0.993 and 0.999. Here: with ``kernel_shape="disk"`` or ``"disk_equal_area"`` the ratio is within 2 %
of 1; with the square (default) it stays below 0.9. The mean-filter crest response (z - mean, what SLRM shows) of
the disk is within 2 % of isotropic too.

The windows at the centre pixel lie inside the tile (161 px, centre 80, largest r_w = 55 px), so padding plays no
part.
"""

import numpy as np
import pytest

import rvt.vis

N = 161
C = N // 2
RES = 0.25  # m/px


def _bank(angle_deg):
    c = (np.arange(N, dtype=np.float64) - (N - 1) / 2.0) * RES
    x, y = np.meshgrid(c, c)
    a = np.deg2rad(angle_deg)
    d = -np.sin(a) * x + np.cos(a) * y  # signed distance from the bank axis, m
    return np.exp(-(d ** 2) / (2 * 1.5 ** 2)).astype(np.float32)


def _crest_dev(angle_deg, r, **kw):
    return float(rvt.vis.max_elevation_deviation(_bank(angle_deg), r, r, 1, **kw)[C, C])


def _crest_relief(angle_deg, r, **kw):
    z = _bank(angle_deg)
    return float(z[C, C]) - float(rvt.vis.mean_filter(z, r, **kw)[C, C])


@pytest.mark.parametrize("shape", ["disk", "disk_equal_area"])
@pytest.mark.parametrize("r", [16, 48])
def test_disk_dev_crest_ratio_is_isotropic(r, shape):
    ratio = _crest_dev(45, r, kernel_shape=shape) / _crest_dev(0, r, kernel_shape=shape)
    assert abs(ratio - 1) < 0.02, ratio


@pytest.mark.parametrize("shape", ["disk", "disk_equal_area"])
@pytest.mark.parametrize("r", [16, 48])
def test_disk_mean_filter_crest_ratio_is_isotropic(r, shape):
    ratio = _crest_relief(45, r, kernel_shape=shape) / _crest_relief(0, r, kernel_shape=shape)
    assert abs(ratio - 1) < 0.02, ratio


@pytest.mark.parametrize("r", [16, 48])
def test_square_dev_crest_ratio_stays_anisotropic(r):
    # the default path is unchanged: the diagonal bank stays > 10 % weaker (review C-15: 0.855, 0.847)
    ratio = _crest_dev(45, r) / _crest_dev(0, r)
    assert ratio < 0.9, ratio
