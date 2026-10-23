"""
Optional numba accelerators for rvt (private module, not part of the public API).

This module lets ``rvt.vis`` run a few inner loops as numba kernels while keeping the numpy code as the
reference and the fallback. The rules:

- numba is imported lazily, on the first call of an accelerated function, never at ``import rvt``;
- if numba is missing, fails to import, or a kernel fails to compile or run, ``call`` returns False and the
  caller runs its numpy path;
- ``RVT_NO_ACCEL=1`` switches every accelerated path off;
- kernels are compiled in memory only (no on-disk cache), without fastmath and with the default error
  model, so every kernel performs exactly the IEEE operations written in it. This is what lets each kernel
  reproduce its numpy path bitwise; the contract of each kernel is stated in its docstring in ``rvt.vis``;
- thread count: ``num_threads`` honours ``RVT_NUM_THREADS`` and then ``NUMBA_NUM_THREADS``. Kernels
  declared with ``jit_kernel(parallel=True)`` apply it before each call; serial kernels (the default) use
  one thread. All kernels in rvt are serial, so results do not depend on the thread count.

Kernels in use (all in ``rvt.vis``): ``_svf_max_slope_kernel``, ``_local_dominance_kernel``,
``_sky_illum_search_kernel``, ``_sky_illum_search_literal_kernel``, ``_mstp_dev_max_kernel``.

Usage::

    @_accel.jit_kernel
    def _my_kernel(a, out):
        ...

    if not _accel.call(_my_kernel, a, out):
        ...  # numpy path

The state of a kernel is a plain dict (``py_func``, ``parallel``, ``dispatcher``, ``failed``), shared by
all threads; compilation is guarded by a re-entrant lock. A kernel that failed once stays disabled for the
life of the process.
"""

import threading as _threading

# Guards the one-time numba import and the one-time compilation of each kernel across threads.
_LOCK = _threading.RLock()
# Result of the lazy numba import: "tried" becomes True after the first attempt, "module" is numba or None.
_NUMBA_STATE = {"tried": False, "module": None}


def _env(name):
    """Return the environment variable ``name`` stripped of white space, or '' when it is not set."""
    import os
    return os.environ.get(name, "").strip()


def enabled():
    """
    Return whether accelerators are switched on.

    Returns
    -------
    bool
        False when ``RVT_NO_ACCEL`` is set to any value other than '', '0', 'false', 'no', 'off'
        (case-insensitive); True otherwise. Read on every call, so the variable can be changed at run time.
    """
    return _env("RVT_NO_ACCEL").lower() in ("", "0", "false", "no", "off")


def get_numba():
    """
    Return the numba module, importing it on first use.

    Returns
    -------
    module or None
        ``numba``; None when accelerators are switched off (``enabled()`` is False) or numba is missing or
        fails to import. The import is attempted once per process; a failure is remembered.
    """
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
    """
    Return the requested thread count for parallel kernels.

    Returns
    -------
    int or None
        ``RVT_NUM_THREADS`` if it is a positive integer, else ``NUMBA_NUM_THREADS`` if it is a positive
        integer, else None (library default). Values that are not positive integers are ignored.
    """
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
    """Set numba's thread count to ``num_threads()``, clipped to [1, numba.config.NUMBA_NUM_THREADS]
    (numba raises for a value above the size of its thread pool); no change when ``num_threads()`` is None."""
    n = num_threads()
    if n is not None:
        numba_mod.set_num_threads(max(1, min(n, int(numba_mod.config.NUMBA_NUM_THREADS))))


def jit_kernel(py_func=None, parallel=False):
    """
    Register a numba-compatible function as a kernel (decorator).

    The function is not compiled here; ``dispatcher`` compiles it with ``numba.njit(nogil=True,
    parallel=parallel)`` on its first call, in memory, without fastmath and without cache. The decorated
    name is bound to the kernel state, not to a callable; run it with ``call``.

    Parameters
    ----------
    py_func : callable or None
        The kernel as a plain Python function. None when the decorator is used with arguments
        (``@jit_kernel(parallel=True)``).
    parallel : bool
        Compile with ``parallel=True`` and apply ``num_threads()`` before each call. Default False (serial).

    Returns
    -------
    dict or callable
        The kernel state ``{"py_func", "parallel", "dispatcher": None, "failed": False}``; when ``py_func``
        is None, a decorator that returns that state.
    """
    if py_func is None:
        return lambda func_: jit_kernel(func_, parallel=parallel)
    return {"py_func": py_func, "parallel": bool(parallel), "dispatcher": None, "failed": False}


def dispatcher(k):
    """
    Return the compiled numba dispatcher of a kernel, compiling it on first use.

    Parameters
    ----------
    k : dict
        Kernel state returned by ``jit_kernel``. ``k["dispatcher"]`` and ``k["failed"]`` are updated in place.

    Returns
    -------
    numba dispatcher or None
        None when accelerators are off, numba is unavailable, or the kernel failed before (a failure in
        ``numba.njit`` marks it failed for the rest of the process). numba compiles the machine code for an
        argument signature lazily, at the first ``call`` with that signature.
    """
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
    """
    Run a kernel on its arguments; outputs are written in place into the arrays passed in.

    Parameters
    ----------
    k : dict
        Kernel state returned by ``jit_kernel``.
    *args
        Positional arguments of the kernel (numpy arrays and scalars).

    Returns
    -------
    bool
        True on success. False when the caller has to run its numpy path instead: accelerators off, numba
        missing, compilation failure (typing errors included) or an exception while running. After any
        exception the kernel is marked failed and is not tried again in this process.

    Notes
    -----
    After False, an output array passed in may be partially written (a run-time exception can occur
    after some writes) and has to be re-initialised by the caller. The callers in ``rvt.vis`` either
    re-initialise their outputs or guard their inputs so that only typing/compilation, which happen
    before any write, can fail.
    """
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
