"""Line coverage of the rvt package by the test suite, with the standard library only.

The frozen environment has no coverage.py, so this uses ``trace.Trace`` (count mode) around
``pytest.main``. Report only, not a gate. With numba on, kernel bodies run as compiled code and are
not traced; ``--no-accel`` shows the numpy-only picture (kernels then count as not executed).

Usage (from the repository root)::

    python tools/proof/linecov.py [--ref main] [--no-accel] [--out .proof_scratch/coverage.md]

Builds the tree like equiv.py (package from ``--ref``, tests from the same ref) and writes a markdown
table: per module executable lines, executed lines, percent, plus the public functions whose body was
never executed.
"""

import argparse
import ast
import os
import sys
import trace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equiv  # noqa: E402

MODULES = ("vis.py", "blend.py", "blend_func.py", "default.py", "tile.py", "_accel.py", "_blocking.py",
           "__init__.py")


def public_bodies(path):
    """Yield (qualified name, set of body line numbers) of public functions and methods."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        items = [(node.name, node)] if isinstance(node, ast.FunctionDef) else []
        if isinstance(node, ast.ClassDef):
            items += [(f"{node.name}.{m.name}", m) for m in node.body if isinstance(m, ast.FunctionDef)]
        for name, fn in items:
            if name.split(".")[-1].startswith("_") and not name.endswith("__init__"):
                continue
            lines = set()
            for stmt in fn.body:
                lines.update(range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1))
            yield name, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="main")
    ap.add_argument("--no-accel", action="store_true")
    ap.add_argument("--out", default=str(equiv.SCRATCH / "coverage.md"))
    a = ap.parse_args()

    equiv.SCRATCH.mkdir(parents=True, exist_ok=True)
    root = equiv.assemble("COV", ref=a.ref, tests_ref=a.ref)
    if a.no_accel:
        os.environ["RVT_NO_ACCEL"] = "1"
    os.chdir(root)
    sys.path.insert(0, str(root))
    import pytest

    tracer = trace.Trace(count=1, trace=0, ignoredirs=[sys.prefix, sys.exec_prefix])
    rc = tracer.runfunc(pytest.main, ["tests", "-q", "-p", "no:cacheprovider", "--rootdir", str(root)])
    counts = tracer.results().counts

    pkg = (root / "rvt").resolve()
    hit = {}
    for (fname, lineno), n in counts.items():
        p = Path(fname).resolve()
        if p.parent == pkg and n:
            hit.setdefault(p.name, set()).add(lineno)

    rows, never = [], []
    tot_exe = tot_hit = 0
    for mod in MODULES:
        path = pkg / mod
        exe = set(trace._find_executable_linenos(str(path)))  # stdlib helper used by trace itself
        h = hit.get(mod, set()) & exe
        tot_exe += len(exe)
        tot_hit += len(h)
        rows.append(f"| {mod} | {len(exe)} | {len(h)} | {100 * len(h) / max(len(exe), 1):.1f} % |")
        for name, body in public_bodies(path):
            if body and not (body & hit.get(mod, set())):
                never.append(f"{mod}:{name}")
    mode = "RVT_NO_ACCEL=1" if a.no_accel else "accelerators on"
    text = [f"# Line coverage of rvt by the test suite ({a.ref}, {mode})", "",
            f"pytest exit code: {int(rc)}. Tool: tools/proof/linecov.py (stdlib trace).", "",
            "| module | executable lines | executed | coverage |", "|---|---|---|---|", *rows,
            f"| **total** | {tot_exe} | {tot_hit} | {100 * tot_hit / max(tot_exe, 1):.1f} % |", "",
            "Public functions/methods whose body never executed: " + (", ".join(never) or "none") + "."]
    Path(a.out).write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text[-10:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
