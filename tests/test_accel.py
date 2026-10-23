"""
rvt._accel: lazy numba import, RVT_NO_ACCEL switch, RVT_NUM_THREADS / NUMBA_NUM_THREADS, automatic fallback to the
numpy path (numba missing, compile or runtime failure).
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import rvt.vis
from rvt import _accel
from conftest import REPO_ROOT, assert_bitwise, assert_same, numba_available, synthetic_dem


def test_import_does_not_load_numba():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import rvt, rvt.vis, rvt.blend_func, rvt.blend, "
            "rvt.default, rvt._accel; assert rvt.__file__.startswith(sys.argv[1]), rvt.__file__; "
            "print('numba' in sys.modules)")
    env = dict(os.environ)
    env.pop("RVT_NO_ACCEL", None)
    res = subprocess.run([sys.executable, "-c", code, REPO_ROOT], capture_output=True, text=True, env=env,
                         timeout=120)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.parametrize("value,enabled", [(None, True), ("", True), ("0", True), ("false", True), ("No", True),
                                           ("off", True), (" 0 ", True), ("1", False), ("true", False),
                                           ("yes", False), ("anything", False)])
def test_enabled(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    else:
        monkeypatch.setenv("RVT_NO_ACCEL", value)
    assert _accel.enabled() is enabled
    if not enabled:
        assert _accel.get_numba() is None


@pytest.mark.parametrize("rvt_n,numba_n,expected", [(None, None, None), ("3", None, 3), (None, "2", 2),
                                                    ("1", "4", 1), ("x", "4", 4), ("0", None, None),
                                                    ("-2", "5", 5), (" 2 ", None, 2)])
def test_num_threads(monkeypatch, rvt_n, numba_n, expected):
    for name, val in (("RVT_NUM_THREADS", rvt_n), ("NUMBA_NUM_THREADS", numba_n)):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, val)
    assert _accel.num_threads() == expected


def _add_one(a, out):
    for i in range(a.shape[0]):
        out[i] = a[i] + 1.0


def test_call_switched_off(monkeypatch):
    kernel = _accel.jit_kernel(_add_one)
    monkeypatch.setenv("RVT_NO_ACCEL", "1")
    out = np.zeros(3)
    assert _accel.call(kernel, np.arange(3.0), out) is False
    assert kernel["dispatcher"] is None and not kernel["failed"]
    assert np.all(out == 0)


def test_call_numba_missing(monkeypatch):
    # numba not importable: every kernel reports False and the numpy path runs, with the same results
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    monkeypatch.setitem(_accel._NUMBA_STATE, "tried", True)
    monkeypatch.setitem(_accel._NUMBA_STATE, "module", None)
    kernel = _accel.jit_kernel(_add_one)
    assert _accel.call(kernel, np.arange(3.0), np.zeros(3)) is False
    from test_local_dominance import ref_local_dominance
    from test_mstp import ref_max_elevation_deviation
    from test_sky_view_factor import ref_sky_view_factor_compute
    dem = synthetic_dem((24, 27), seed=51, nan_patch=True)
    assert_same(rvt.vis.local_dominance(dem.copy(), min_rad=2, max_rad=6),
                ref_local_dominance(dem.copy(), min_rad=2, max_rad=6))
    assert_same(rvt.vis.max_elevation_deviation(dem.copy(), 1, 4, 1), ref_max_elevation_deviation(dem.copy(), 1, 4, 1))
    assert_same(rvt.vis.sky_view_factor_compute(dem.copy(), radius_max=5, compute_opns=True),
                ref_sky_view_factor_compute(dem.copy(), radius_max=5, compute_opns=True))


@pytest.mark.skipif(not numba_available(), reason="numba not available")
def test_call_compiles_and_runs(monkeypatch):
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    kernel = _accel.jit_kernel(_add_one)
    out = np.zeros(4)
    assert _accel.call(kernel, np.arange(4.0), out) is True
    assert kernel["dispatcher"] is not None and not kernel["failed"]
    assert np.array_equal(out, np.arange(4.0) + 1.0)


@pytest.mark.skipif(not numba_available(), reason="numba not available")
def test_call_failure_falls_back(monkeypatch):
    # a typing / compile error at the first call: False, the kernel is marked failed and never retried
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    kernel = _accel.jit_kernel(_add_one)
    assert _accel.call(kernel, object(), np.zeros(3)) is False
    assert kernel["failed"]
    assert _accel.call(kernel, np.arange(3.0), np.zeros(3)) is False
    assert _accel.dispatcher(kernel) is None


def _fill_parallel(out):
    for i in range(out.shape[0]):
        out[i] = i * 0.5


@pytest.mark.skipif(not numba_available(), reason="numba not available")
def test_parallel_kernel_honours_thread_count(monkeypatch):
    import numba
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    kernel = _accel.jit_kernel(_fill_parallel, parallel=True)
    before = numba.get_num_threads()
    try:
        for n in (1, 2):
            monkeypatch.setenv("RVT_NUM_THREADS", str(n))
            out = np.zeros(10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numba.NumbaPerformanceWarning)  # nothing to parallelize here
                assert _accel.call(kernel, out) is True
            assert numba.get_num_threads() == min(n, numba.config.NUMBA_NUM_THREADS)
            assert np.array_equal(out, np.arange(10) * 0.5)
    finally:
        numba.set_num_threads(before)


def test_numpy_path_deterministic(numpy_path):
    dem = synthetic_dem((26, 30), seed=52, nan_patch=True)
    for func in (lambda d: rvt.vis.local_dominance(d, min_rad=2, max_rad=6),
                 lambda d: rvt.vis.max_elevation_deviation(d, 1, 6, 1),
                 lambda d: rvt.vis.sky_view_factor_compute(d, radius_max=6, compute_asvf=True),
                 lambda d: rvt.vis.sky_illumination(d, 0.5, num_directions=8, max_fine_radius=30)):
        assert_bitwise(func(dem.copy()), func(dem.copy()))
