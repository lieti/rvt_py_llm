"""
Shared setup of the equivalence tests for the optimized code paths of rvt.

- The repository root (the parent of this directory) is put first on sys.path, and the imported ``rvt`` package
  must come from this repository (not from an installed copy).
- Helpers: synthetic DEMs (seeded, platform-independent generator), comparison asserts, and switches between the
  accelerated (numba) path and the numpy fallback path of ``rvt._accel``.

The tests run with the accelerators as found in the environment; set ``RVT_NO_ACCEL=1`` to run the whole suite on
the numpy fallback path. Tests that compare the two paths switch ``RVT_NO_ACCEL`` themselves (restored afterwards)
and are skipped when numba is not available.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[:1] != [REPO_ROOT]:
    sys.path.insert(0, REPO_ROOT)

import rvt  # noqa: E402

_RVT_FILE = os.path.normcase(os.path.realpath(rvt.__file__))
assert os.path.commonpath([_RVT_FILE, os.path.normcase(os.path.realpath(REPO_ROOT))]) == \
    os.path.normcase(os.path.realpath(REPO_ROOT)), \
    "rvt imported from {}, not from the repository {}".format(rvt.__file__, REPO_ROOT)

import rvt.vis  # noqa: E402
from rvt import _accel  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
# synthetic inputs
# ----------------------------------------------------------------------------------------------------------------------

def synthetic_dem(shape, seed=0, relief=20.0, offset=0.0, nan_patch=False, dtype=np.float32, grid=False):
    """Seeded synthetic terrain: smooth hills, a ridge, a pit, a terrace step and noise. With grid=True the values lie
    on a 1/256 m grid (|z| < 1024 m): differences are then exact in float32 and a dyadic offset such as +2500 m is
    exact; by default they are generic float32 values (differences rounded). Optional NaN patches (a block and a
    few single pixels, one on the border)."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = shape
    y, x = np.mgrid[0:n_rows, 0:n_cols].astype(np.float64)
    z = relief * 0.3 * np.sin(x / 7.0 + 0.3) * np.cos(y / 5.0)
    z += relief * 0.2 * np.exp(-((x - 0.6 * n_cols) ** 2 + (y - 0.4 * n_rows) ** 2) / (0.05 * n_rows * n_cols + 1.0))
    z -= relief * 0.15 * np.exp(-((x - 0.3 * n_cols) ** 2 + (y - 0.7 * n_rows) ** 2) / 20.0)
    z += np.where(x > 0.75 * n_cols, 0.1 * relief, 0.0)
    z += np.where(np.abs(x - y) < 1.5, 0.05 * relief, 0.0)
    z += rng.normal(0.0, 0.02 * relief, size=shape)
    if grid:
        z = np.round(z * 256.0) / 256.0
    z = z + offset
    z = z.astype(dtype)
    if nan_patch:
        r0, c0 = n_rows // 3, n_cols // 4
        z[r0:r0 + 3, c0:c0 + 4] = np.nan
        z[n_rows // 2, (2 * n_cols) // 3] = np.nan
        z[0, n_cols - 2] = np.nan
    return z


# ----------------------------------------------------------------------------------------------------------------------
# comparisons
# ----------------------------------------------------------------------------------------------------------------------

def assert_same(actual, expected):
    """Same type, shape, dtype and values (NaN at the same positions, +0 == -0): the exactness used by the harness."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert sorted(actual) == sorted(expected)
        for key in expected:
            assert_same(actual[key], expected[key])
        return
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if not np.array_equal(actual, expected, equal_nan=actual.dtype.kind in "fc"):
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        raise AssertionError("values differ at {} of {} positions, max |diff| = {}".format(
            int(np.count_nonzero(~(diff == 0) & ~(np.isnan(actual) & np.isnan(expected)))), actual.size,
            np.nanmax(diff)))


def assert_bitwise(actual, expected):
    """Bitwise identical (also the sign of zero and the NaN payload)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert sorted(actual) == sorted(expected)
        for key in expected:
            assert_bitwise(actual[key], expected[key])
        return
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes()


# ----------------------------------------------------------------------------------------------------------------------
# accelerated / numpy paths
# ----------------------------------------------------------------------------------------------------------------------

def numba_available():
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


def kernel_ran(kernel):
    """True if the numba kernel of rvt._accel has been compiled and has not failed."""
    return kernel["dispatcher"] is not None and not kernel["failed"]


@pytest.fixture
def numpy_path(monkeypatch):
    """Switches the accelerators off (RVT_NO_ACCEL=1) for the test."""
    monkeypatch.setenv("RVT_NO_ACCEL", "1")
    assert not _accel.enabled()


@pytest.fixture
def run_both(monkeypatch):
    """Returns run(func): func() evaluated on the accelerated path and on the numpy path, as (accel, numpy).
    Skips the test when numba is not available. Asserts that the accelerated run did not fall back."""
    if not numba_available():
        pytest.skip("numba not available")

    def run(func, kernels=()):
        monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
        assert _accel.enabled() and _accel.get_numba() is not None
        out_accel = func()
        for k in kernels:
            assert kernel_ran(k), "the numba kernel did not run (fallback taken)"
        monkeypatch.setenv("RVT_NO_ACCEL", "1")
        out_numpy = func()
        monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
        return out_accel, out_numpy

    return run
