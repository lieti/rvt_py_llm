"""
Characterization tests for the public helpers of rvt._accel: enabled, num_threads, call, jit_kernel (and the
accessors get_numba and dispatcher that they rely on).

test_accel.py covers the on/off switch values, the thread-count variables, a kernel that compiles and runs, a
compile failure and the fallback of vis functions. This file adds the kernel registry structure, the exact return
types, the switch applied after a kernel has been compiled, runtime failures, and the numba accessor.
The kernels used here are tiny, so compilation takes well under a second.
"""

import numpy as np
import pytest

from rvt import _accel
from conftest import numba_available

needs_numba = pytest.mark.skipif(not numba_available(), reason="numba not available")


def _scale(a, factor, out):
    for i in range(a.shape[0]):
        out[i] = a[i] * factor


def _divide_by_zero(out):
    # an integer division by zero raises ZeroDivisionError in a numba kernel (default error model)
    zero = out.shape[0] - out.shape[0]
    out[0] = 1 // zero


# ----------------------------------------------------------------------------------------------------------------------
# jit_kernel
# ----------------------------------------------------------------------------------------------------------------------

def test_jit_kernel_returns_registry_entry_without_compiling():
    kernel = _accel.jit_kernel(_scale)
    assert isinstance(kernel, dict)
    assert sorted(kernel) == ["dispatcher", "failed", "parallel", "py_func"]
    assert kernel["py_func"] is _scale
    assert kernel["parallel"] is False and kernel["failed"] is False
    assert kernel["dispatcher"] is None  # compiled on the first call, not at registration


def test_jit_kernel_decorator_forms():
    @_accel.jit_kernel
    def plain(a):
        return a

    @_accel.jit_kernel(parallel=True)
    def parallel(a):
        return a

    assert plain["py_func"].__name__ == "plain" and plain["parallel"] is False
    assert parallel["py_func"].__name__ == "parallel" and parallel["parallel"] is True
    assert _accel.jit_kernel(_scale, parallel=True)["parallel"] is True
    assert _accel.jit_kernel(_scale, parallel=1)["parallel"] is True  # coerced to bool
    assert _accel.jit_kernel(_scale, parallel=0)["parallel"] is False


def test_jit_kernels_are_independent_entries():
    first, second = _accel.jit_kernel(_scale), _accel.jit_kernel(_scale)
    assert first is not second
    first["failed"] = True
    assert second["failed"] is False


# ----------------------------------------------------------------------------------------------------------------------
# enabled, num_threads
# ----------------------------------------------------------------------------------------------------------------------

def test_enabled_returns_bool(monkeypatch):
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    assert _accel.enabled() is True
    monkeypatch.setenv("RVT_NO_ACCEL", "1")
    assert _accel.enabled() is False


@pytest.mark.parametrize("rvt_n,numba_n,expected", [("2.5", "3", 3), ("  ", "4", 4), ("abc", None, None),
                                                    ("1", None, 1), ("007", None, 7), (None, "0", None),
                                                    (None, "-1", None), ("0", "0", None)])
def test_num_threads_unusable_values_fall_through(monkeypatch, rvt_n, numba_n, expected):
    for name, value in (("RVT_NUM_THREADS", rvt_n), ("NUMBA_NUM_THREADS", numba_n)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    result = _accel.num_threads()
    assert result == expected
    assert result is None or type(result) is int


# ----------------------------------------------------------------------------------------------------------------------
# get_numba, dispatcher, call
# ----------------------------------------------------------------------------------------------------------------------

def test_get_numba_none_when_switched_off(monkeypatch):
    monkeypatch.setenv("RVT_NO_ACCEL", "1")
    assert _accel.get_numba() is None
    kernel = _accel.jit_kernel(_scale)
    assert _accel.dispatcher(kernel) is None
    assert kernel["dispatcher"] is None and kernel["failed"] is False  # switching off does not mark a failure


@needs_numba
def test_get_numba_returns_the_numba_module(monkeypatch):
    import numba
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    assert _accel.get_numba() is numba
    assert _accel.get_numba() is _accel.get_numba()


@needs_numba
def test_call_passes_scalars_and_arrays_and_returns_true(monkeypatch):
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    kernel = _accel.jit_kernel(_scale)
    out = np.zeros(4)
    result = _accel.call(kernel, np.arange(4.0), 2.5, out)
    assert result is True
    assert np.array_equal(out, [0.0, 2.5, 5.0, 7.5])  # results are written into the argument, nothing is returned


@needs_numba
def test_call_compiles_once_and_switch_applies_after_compilation(monkeypatch):
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    kernel = _accel.jit_kernel(_scale)
    assert _accel.call(kernel, np.arange(3.0), 1.0, np.zeros(3)) is True
    compiled = kernel["dispatcher"]
    assert compiled is not None
    monkeypatch.setenv("RVT_NO_ACCEL", "1")  # an already compiled kernel is not used while accelerators are off
    out = np.zeros(3)
    assert _accel.call(kernel, np.arange(3.0), 1.0, out) is False
    assert np.all(out == 0)
    assert kernel["dispatcher"] is compiled and kernel["failed"] is False
    monkeypatch.delenv("RVT_NO_ACCEL")
    assert _accel.call(kernel, np.arange(3.0), 3.0, out) is True
    assert kernel["dispatcher"] is compiled  # same compiled function, not recompiled
    assert np.array_equal(out, [0.0, 3.0, 6.0])


@needs_numba
def test_call_runtime_failure_marks_kernel_failed(monkeypatch):
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    kernel = _accel.jit_kernel(_divide_by_zero)
    assert _accel.call(kernel, np.zeros(2, dtype=np.int64)) is False
    assert kernel["failed"] is True
    assert _accel.dispatcher(kernel) is None  # a failed kernel is never used again, the caller runs its numpy path
    assert _accel.call(kernel, np.zeros(2, dtype=np.int64)) is False


@needs_numba
def test_serial_kernel_leaves_the_numba_thread_count_alone(monkeypatch):
    import numba
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    monkeypatch.setenv("RVT_NUM_THREADS", "1")
    before = numba.get_num_threads()
    kernel = _accel.jit_kernel(_scale)  # parallel=False: the thread variables are applied to parallel kernels only
    assert _accel.call(kernel, np.arange(3.0), 1.0, np.zeros(3)) is True
    assert numba.get_num_threads() == before
