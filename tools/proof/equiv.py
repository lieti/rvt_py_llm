"""Prove that two rvt trees behave bitwise identically on everything the test suite exercises.

Golden-free: nothing is stored between runs. Both trees run the *same* test files (taken from one git
ref, normally the tag ``tests-frozen``) under the recording plugin ``equiv_plugin.py``, which logs a
hash of every value returned by a public rvt function, of the array arguments after the call, and of
every file a test writes. The two logs must be identical, record by record. A test that passes in both
trees but returns a different bit pattern (inside a test tolerance) is therefore still a FAIL.

Variants (each run for both trees, in fresh processes):
- ``accel``: default environment (numba kernels on, default threading);
- ``t1``: ``RVT_NUM_THREADS=NUMBA_NUM_THREADS=OMP_NUM_THREADS=...=1``;
- ``noaccel``: ``RVT_NO_ACCEL=1`` (numpy fallback path).

Usage (from the repository root)::

    python tools/proof/equiv.py                                  # start vs main, frozen tests
    python tools/proof/equiv.py --b-dir <checkout>               # start vs a checkout as it is on disk
    python tools/proof/equiv.py --a start --b HEAD --tests-ref tests-frozen --variants accel

Trees are assembled under ``<repo>/.proof_scratch/`` (package ``rvt/`` and ``settings/`` from the tree,
``tests/`` from ``--tests-ref``). The last line is ``EQUIV PASS`` or ``EQUIV FAIL``. The report also
lists public functions that no test called (execution coverage of the public API).
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]  # repository root
SCRATCH = REPO / ".proof_scratch"
THREAD_VARS = ("RVT_NUM_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
VARIANTS = ("accel", "t1", "noaccel")


def git_bytes(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def extract(ref, paths, dest):
    """Extract ``paths`` of git ``ref`` (of this repository) into ``dest``."""
    data = git_bytes("archive", "--format=tar", ref, *paths)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest, filter="data")


def assemble(label, ref=None, src_dir=None, tests_ref="tests-frozen"):
    root = SCRATCH / label
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if src_dir is not None:
        for sub in ("rvt", "settings"):
            shutil.copytree(Path(src_dir) / sub, root / sub,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        extract(ref, ["rvt", "settings"], root)
    extract(tests_ref, ["tests"], root)
    return root


def run(root, label, variant):
    out = SCRATCH / f"log_{label}_{variant}.json"
    bt = SCRATCH / f"bt_{label}_{variant}"
    if bt.exists():
        shutil.rmtree(bt)
    env = dict(os.environ)
    env.update({"EQUIV_ROOT": str(root), "EQUIV_BASETEMP": str(bt), "EQUIV_OUT": str(out),
                "PYTHONPATH": str(HERE), "PYTHONHASHSEED": "0"})
    env.pop("RVT_NO_ACCEL", None)
    if variant == "t1":
        env.update({k: "1" for k in THREAD_VARS})
    elif variant == "noaccel":
        env["RVT_NO_ACCEL"] = "1"
    # --rootdir: the tree root, so that node ids are "tests/..." wherever the scratch directory lies
    cmd = [sys.executable, "-m", "pytest", str(root / "tests"), "-q", "-p", "no:cacheprovider",
           "-p", "equiv_plugin", "--basetemp", str(bt), "--rootdir", str(root)]
    r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
    log = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"exitstatus": -1, "records": {}}
    return log, tail


def compare(la, lb, limit=15):
    """Return a list of mismatch descriptions between two logs."""
    ra, rb = la["records"], lb["records"]
    bad = []
    for nid in sorted(set(ra) ^ set(rb)):
        bad.append(f"test only in {'A' if nid in ra else 'B'}: {nid}")
    n_records = 0
    for nid in sorted(set(ra) & set(rb)):
        a, b = ra[nid], rb[nid]
        n_records += len(a)
        if a == b:
            continue
        if len(a) != len(b):
            bad.append(f"{nid}: {len(a)} vs {len(b)} records")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                bad.append(f"{nid}: record {i} {x[0]}: A={x[1:]} B={y[1:]}")
                break
    return bad, n_records


def public_api(root):
    """Qualified names of all public functions/methods of the tree (same rule as the plugin)."""
    code = ("import sys, json, inspect; sys.path.insert(0, sys.argv[1]);"
            "import rvt, rvt.vis, rvt.blend, rvt.blend_func, rvt.default, rvt.tile, rvt._accel;"
            "out=[]\n"
            "for m in (rvt.vis, rvt.blend_func, rvt.blend, rvt.default, rvt.tile, rvt._accel):\n"
            "  for n,o in vars(m).items():\n"
            "    if n.startswith('_') or getattr(o,'__module__',None)!=m.__name__: continue\n"
            "    if inspect.isfunction(o): out.append(m.__name__+'.'+n)\n"
            "    elif inspect.isclass(o):\n"
            "      out += [m.__name__+'.'+n+'.'+k for k,v in vars(o).items()"
            " if (not k.startswith('_') or k=='__init__') and inspect.isfunction(v)]\n"
            "print(json.dumps(out))")
    r = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", default="start", help="git ref of rvt_py_llm for tree A (default: start)")
    ap.add_argument("--b", default="main", help="git ref of rvt_py_llm for tree B (default: main)")
    ap.add_argument("--b-dir", help="use this directory (its rvt/ and settings/) as tree B instead of --b")
    ap.add_argument("--tests-ref", default="tests-frozen", help="git ref providing tests/ for both trees")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tag", help="scratch name prefix, so that parallel runs do not collide "
                                  "(default: name of --b-dir, else 'ref')")
    a = ap.parse_args()
    tag = a.tag or (Path(a.b_dir).name if a.b_dir else "ref")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    root_a = assemble(f"{tag}_A", ref=a.a, tests_ref=a.tests_ref)
    root_b = assemble(f"{tag}_B", ref=a.b, src_dir=a.b_dir, tests_ref=a.tests_ref)
    print(f"A = rvt_py_llm@{a.a}; B = {a.b_dir or 'rvt_py_llm@' + a.b}; tests from {a.tests_ref}")

    ok = True
    called = set()
    for v in [x.strip() for x in a.variants.split(",") if x.strip()]:
        la, ta = run(root_a, f"{tag}_A", v)
        lb, tb = run(root_b, f"{tag}_B", v)
        bad, n = compare(la, lb)
        # a skipped test weakens the proof silently (e.g. numba not importable), so it counts as a failure
        skipped = [f"{side}: {t}" for side, t in (("A", ta), ("B", tb)) if "skipped" in t]
        bad += [f"tests skipped, proof incomplete: {s}" for s in skipped]
        status = "ok" if not bad and la["exitstatus"] == 0 and lb["exitstatus"] == 0 else "FAIL"
        ok &= status == "ok"
        print(f"[{v}] {status}: {len(la['records'])} tests, {n} records compared; "
              f"A: {ta} | B: {tb}")
        for line in bad[:15]:
            print(f"    {line}")
        if len(bad) > 15:
            print(f"    ... {len(bad) - 15} more")
        for recs in lb["records"].values():
            called.update(r[0] for r in recs)
    api = public_api(root_b)
    missing = sorted(set(api) - called)
    print(f"public API executed by the tests: {len(api) - len(missing)}/{len(api)}")
    if missing:
        print("  never called: " + ", ".join(missing))
    print("EQUIV PASS" if ok else "EQUIV FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
