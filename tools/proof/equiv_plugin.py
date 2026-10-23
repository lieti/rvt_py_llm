"""pytest plugin: record a hash of every value returned by a public rvt function during a test run.

Loaded by ``equiv.py`` with ``-p equiv_plugin``; not part of the library. For every test it records, in
call order, each return from a public function or method of ``rvt.*`` (name without a leading
underscore, defined in the package):

- the qualified name;
- a canonical hash of the returned value (arrays: dtype, shape and SHA-256 of the raw bytes, so a
  single-bit difference, -0.0 vs 0.0 or a different NaN payload counts);
- a canonical hash of the array arguments as they are after the call (in-place effects);
- after the test, a hash of every file the test wrote under ``tmp_path`` (text files with timestamps:
  line count only).

Exceptions leaving a public function are recorded by the test outcome itself (the test asserts them);
the profiler sees them as a return of ``None``.

Paths are normalised (the tree root and the pytest base temp directory are replaced by placeholders),
so two trees in different directories give identical records when they behave identically.
The JSON log is written to the path in the environment variable ``EQUIV_OUT``.
"""

import hashlib
import inspect
import json
import os
import re
import sys
from enum import Enum
from pathlib import Path, PurePath

import numpy as np

_ROOT = os.environ["EQUIV_ROOT"]
_BASETEMP = os.environ["EQUIV_BASETEMP"]
_OUT = os.environ["EQUIV_OUT"]
_TEXT_SUFFIXES = {".txt", ".log"}

_codes = {}            # code object -> qualified name
_records = {}          # nodeid -> list of records
_current = None        # list of records of the running test
_stack = []            # per active public frame: (frame, [array args])


def _norm_str(s):
    for prefix, tag in ((_BASETEMP, "<BT>"), (_ROOT, "<ROOT>")):
        for p in {prefix, prefix.replace("\\", "/"), prefix.replace("/", "\\")}:
            s = s.replace(p, tag)
    s = s.replace("\\", "/")
    s = re.sub(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", "<DATETIME>", s)  # log file names
    return re.sub(r"pytest-\d+", "pytest-N", s)


def _canon(obj, depth=0):
    """Return a JSON-able canonical description of ``obj``."""
    if depth > 6:
        return "<deep>"
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        if a.dtype == object:
            return {"nd": "object", "shape": list(a.shape), "items": [_canon(x, depth + 1) for x in a.ravel()]}
        return {"nd": a.dtype.str, "shape": list(a.shape), "sha": hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(obj, np.generic):
        return {"np": obj.dtype.str, "v": _canon(obj.item(), depth + 1)}
    if isinstance(obj, bool) or obj is None or isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return {"f": obj.hex()}
    if isinstance(obj, str):
        return _norm_str(obj)
    if isinstance(obj, PurePath):
        return {"path": _norm_str(str(obj))}
    if isinstance(obj, Enum):
        return {"enum": f"{type(obj).__name__}.{obj.name}"}
    if isinstance(obj, dict):
        return {"dict": [[_canon(k, depth + 1), _canon(v, depth + 1)] for k, v in
                         sorted(obj.items(), key=lambda kv: repr(kv[0]))]}
    if isinstance(obj, (list, tuple)):
        return {type(obj).__name__: [_canon(x, depth + 1) for x in obj]}
    if isinstance(obj, (set, frozenset)):
        return {"set": sorted(json.dumps(_canon(x, depth + 1), sort_keys=True) for x in obj)}
    mod = type(obj).__module__ or ""
    if mod.startswith("rvt") and hasattr(obj, "__dict__"):
        return {"obj": type(obj).__qualname__,
                "vars": _canon({k: v for k, v in vars(obj).items() if not callable(v)}, depth + 1)}
    if mod.startswith("matplotlib") and callable(obj):
        return {"mpl": type(obj).__name__, "lut": _canon(np.asarray(obj(np.linspace(0, 1, 64))), depth + 1)}
    return {"type": f"{mod}.{type(obj).__qualname__}"}


def _digest(obj):
    return hashlib.sha256(json.dumps(_canon(obj), sort_keys=True).encode()).hexdigest()[:24]


def _register(module):
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            _codes[obj.__code__] = f"{module.__name__}.{name}"
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for mname, mobj in vars(obj).items():
                if mname.startswith("_") and mname != "__init__":
                    continue
                f = mobj.__func__ if isinstance(mobj, (staticmethod, classmethod)) else mobj
                if inspect.isfunction(f):
                    _codes[f.__code__] = f"{module.__name__}.{name}.{mname}"


def _profile(frame, event, arg):
    if _current is None:
        return
    code = frame.f_code
    if code not in _codes:
        return
    if event == "call":
        arrays = [v for v in frame.f_locals.values() if isinstance(v, np.ndarray)]
        _stack.append((frame, arrays))
    elif event == "return" and _stack and _stack[-1][0] is frame:
        _, arrays = _stack.pop()
        _current.append([_codes[code], _digest(arg), _digest(arrays)])


def pytest_sessionstart(session):
    sys.path.insert(0, _ROOT)
    import rvt  # noqa: F401
    import rvt._accel  # noqa: F401
    import rvt.blend
    import rvt.blend_func
    import rvt.default
    import rvt.tile
    import rvt.vis
    if not Path(rvt.__file__).resolve().is_relative_to(Path(_ROOT).resolve()):
        raise RuntimeError(f"rvt imported from {rvt.__file__}, not from {_ROOT}")
    for m in (rvt.vis, rvt.blend_func, rvt.blend, rvt.default, rvt.tile, rvt._accel):
        _register(m)


def pytest_runtest_setup(item):
    global _current
    _current = _records.setdefault(item.nodeid, [])
    _stack.clear()
    sys.setprofile(_profile)


def pytest_runtest_teardown(item, nextitem):
    global _current
    sys.setprofile(None)
    _current = None
    tmp = getattr(item, "funcargs", {}).get("tmp_path")
    if tmp is not None and Path(tmp).exists():
        files = []
        for p in sorted(Path(tmp).rglob("*")):
            if not p.is_file():
                continue
            rel = _norm_str(p.relative_to(tmp).as_posix())
            if p.suffix.lower() in _TEXT_SUFFIXES:
                n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                files.append([rel, f"lines={n}"])
            else:
                files.append([rel, hashlib.sha256(p.read_bytes()).hexdigest()[:24]])
        _records.setdefault(item.nodeid, []).append(["<files>", files, ""])


def pytest_sessionfinish(session, exitstatus):
    Path(_OUT).write_text(json.dumps({"exitstatus": int(exitstatus), "records": _records}), encoding="utf-8")
