"""
Private accelerator support for rvt (not part of the public API).

Optional numba kernels sit behind an automatic fallback to the numpy path:

- numba is imported lazily, on the first call of an accelerated function, never at ``import rvt``;
- if numba is missing, fails to import, or a kernel fails to compile or run, ``call`` returns False and the
  caller runs its numpy path;
- ``RVT_NO_ACCEL=1`` switches every accelerated path off;
- kernels are compiled in memory only (no on-disk cache), without fastmath and with the default error
  model, so every kernel performs exactly the IEEE operations written in it;
- thread count: ``num_threads`` honours ``RVT_NUM_THREADS`` and then ``NUMBA_NUM_THREADS``. Kernels
  declared with ``jit_kernel(parallel=True)`` apply it before each call; serial kernels (the default) use
  one thread.

Usage::

    @_accel.jit_kernel
    def _my_kernel(a, out):
        ...

    if not _accel.call(_my_kernel, a, out):
        ...  # numpy path
"""

import threading as _threading

_LOCK = _threading.RLock()
_NUMBA_STATE = {"tried": False, "module": None}


def _env(name):
    import os
    return os.environ.get(name, "").strip()


def enabled():
    """False when accelerators are switched off with RVT_NO_ACCEL (any value except '', '0', 'false', 'no', 'off')."""
    return _env("RVT_NO_ACCEL").lower() in ("", "0", "false", "no", "off")


def get_numba():
    """The numba module, imported on first use; None when switched off, missing or failing to import."""
    if not enabled():
        return None
    if not _NUMBA_STATE["tried"]:
        with _LOCK:
            if not _NUMBA_STATE["tried"]:
                try:
                    import numba as numba_mod
                except Exception:
                    numba_mod = None
                _NUMBA_STATE["module"] = numba_mod
                _NUMBA_STATE["tried"] = True
    return _NUMBA_STATE["module"]


def num_threads():
    """Requested thread count for parallel kernels: RVT_NUM_THREADS, else NUMBA_NUM_THREADS, else None
    (library default). Values that are not positive integers are ignored."""
    for name in ("RVT_NUM_THREADS", "NUMBA_NUM_THREADS"):
        val = _env(name)
        if val:
            try:
                n = int(val)
            except ValueError:
                continue
            if n >= 1:
                return n
    return None


def _apply_threads(numba_mod):
    n = num_threads()
    if n is not None:
        numba_mod.set_num_threads(max(1, min(n, int(numba_mod.config.NUMBA_NUM_THREADS))))


def jit_kernel(py_func=None, parallel=False):
    """Decorator: register a numba-compatible function as a kernel, compiled with numba.njit on its first call
    (in memory, no fastmath, no cache). Returns the kernel state used by ``call``."""
    if py_func is None:
        return lambda func_: jit_kernel(func_, parallel=parallel)
    return {"py_func": py_func, "parallel": bool(parallel), "dispatcher": None, "failed": False}


def dispatcher(k):
    """The numba dispatcher of kernel ``k``, or None when accelerators are off, numba is unavailable or the
    kernel failed before."""
    if k["failed"]:
        return None
    numba_mod = get_numba()
    if numba_mod is None:
        return None
    if k["dispatcher"] is None:
        with _LOCK:
            if k["dispatcher"] is None and not k["failed"]:
                try:
                    k["dispatcher"] = numba_mod.njit(nogil=True, parallel=k["parallel"])(k["py_func"])
                except Exception:
                    k["failed"] = True
    return k["dispatcher"]


def call(k, *args):
    """Run kernel ``k`` on ``args`` (outputs written in place). True on success, False when the caller has to
    run its numpy path instead (accelerators off, numba missing, compile or runtime failure). After False,
    any output array passed in may be partially written and has to be re-initialised by the caller."""
    disp = dispatcher(k)
    if disp is None:
        return False
    try:
        if k["parallel"]:
            _apply_threads(get_numba())
        disp(*args)
    except Exception:
        k["failed"] = True
        return False
    return True
