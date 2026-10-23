"""Prove that the frozen tests are untouched and still pass, without skips, on a given commit.

The tests present at the tag ``tests-frozen`` of rvt_py_llm are the contract of this work. This
script checks, for a commit (default ``main``):

1. ``git diff --name-status tests-frozen..<commit> -- tests`` lists only added files, and every added
   file is a new ``test_*.py`` module (no added ``conftest.py`` that could skip or deselect tests);
   no pytest configuration file (pytest.ini, tox.ini, pyproject.toml, setup.cfg, conftest.py at the
   repository root) differs from ``tests-frozen`` in a line that mentions pytest;
2. the SHA-256 of every frozen test file equals the value in ``tools/proof/frozen_manifest.json``
   (written once with ``--freeze`` when the tag is set);
3. the commit's package, run against its own ``tests/`` directory, reports **PASSED** for every frozen
   test node (no skip, xfail, deselect or error), with accelerators on and with ``RVT_NO_ACCEL=1``.

Usage (from the repository root)::

    python tools/proof/frozen_check.py --freeze          # once, right after tagging tests-frozen
    python tools/proof/frozen_check.py                   # check main
    python tools/proof/frozen_check.py --ref <commit>

The last line is ``FROZEN PASS`` or ``FROZEN FAIL``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equiv  # noqa: E402  (reuses the tree assembly)

REPO = equiv.REPO
TAG = "tests-frozen"
MANIFEST = HERE / "frozen_manifest.json"
CONFIG_FILES = ("pytest.ini", "tox.ini", "pyproject.toml", "setup.cfg", "conftest.py")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def file_hashes(ref):
    names = [n for n in git("ls-tree", "-r", "--name-only", ref, "--", "tests").splitlines() if n]
    return {n: hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{n}"], check=True,
                                             capture_output=True).stdout).hexdigest() for n in names}


def node_outcomes(root, no_accel):
    """Run pytest on ``root/tests`` and return {nodeid: outcome} from the -rA summary."""
    env = dict(os.environ)
    env.pop("RVT_NO_ACCEL", None)
    if no_accel:
        env["RVT_NO_ACCEL"] = "1"
    # --rootdir: the tree root, so that node ids are "tests/..." wherever the scratch directory lies
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-rA", "-p", "no:cacheprovider",
                        "--rootdir", str(root)],
                       cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split(" ", 1)
        # only summary lines that name a node id; captured stderr such as "ERROR 4: ..." from GDAL is not one
        if len(parts) == 2 and parts[0] in ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS") \
                and parts[1].startswith("tests/"):
            # PASSED lines carry no message; node ids may themselves contain " - "
            nid = parts[1].strip() if parts[0] == "PASSED" else parts[1].split(" - ")[0].strip()
            out[nid] = parts[0]
    return out, (r.stdout.strip().splitlines() or ["<no output>"])[-1]


def freeze():
    nodes, tail = node_outcomes(equiv.assemble("FROZEN", ref=TAG, tests_ref=TAG), no_accel=False)
    bad = {n: o for n, o in nodes.items() if o != "PASSED"}
    if bad or not nodes:
        print(f"cannot freeze: {len(bad)} nodes not PASSED ({tail})")
        return False
    data = {"tag": TAG, "commit": git("rev-parse", TAG + "^{commit}").strip(),
            "files": file_hashes(TAG), "nodes": sorted(nodes)}
    MANIFEST.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"manifest written: {len(data['files'])} files, {len(data['nodes'])} nodes, commit {data['commit'][:8]}")
    return True


def check(ref):
    problems = []
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if git("rev-parse", TAG + "^{commit}").strip() != man["commit"]:
        problems.append(f"tag {TAG} was moved (manifest has {man['commit'][:8]})")
    for line in git("diff", "--no-renames", "--name-status", f"{TAG}..{ref}", "--", "tests").splitlines():
        status, path = line.split("\t", 1)
        name = path.rsplit("/", 1)[-1]
        if status != "A":
            problems.append(f"frozen test file changed: {status} {path}")
        elif not (name.startswith("test_") and name.endswith(".py")):
            problems.append(f"added file under tests/ is not a test module: {path}")
    for cfg in CONFIG_FILES:
        diff = git("diff", f"{TAG}..{ref}", "--", cfg)
        if "pytest" in diff.lower():
            problems.append(f"pytest configuration changed in {cfg}")
    now = file_hashes(ref)
    for path, digest in man["files"].items():
        if now.get(path) != digest:
            problems.append(f"hash mismatch: {path}")
    root = equiv.assemble("FROZENCHK", ref=ref, tests_ref=ref)
    for no_accel in (False, True):
        nodes, tail = node_outcomes(root, no_accel)
        label = "RVT_NO_ACCEL=1" if no_accel else "accel"
        not_passed = [n for n in man["nodes"] if nodes.get(n) != "PASSED"]
        print(f"[{label}] {tail}; frozen nodes PASSED: {len(man['nodes']) - len(not_passed)}/{len(man['nodes'])}")
        problems += [f"[{label}] frozen node {nodes.get(n, 'MISSING')}: {n}" for n in not_passed[:20]]
    print(f"checked {ref} ({git('rev-parse', '--short', ref).strip()}) against {TAG}")
    for p in problems:
        print(f"  {p}")
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="main")
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    ok = freeze() if a.freeze else check(a.ref)
    print("FROZEN PASS" if ok else "FROZEN FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
