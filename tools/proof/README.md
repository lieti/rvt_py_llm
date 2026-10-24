# Proof tools

These scripts prove that a commit of this repository keeps the contract of rvt_py_llm 0.1.0: every output of
every public function, method and class is bitwise identical to tag `start`, the frozen tests are untouched and
pass, and every commit changes only what its class allows. [docs/TESTING.md](../../docs/TESTING.md) explains the
method and records the results; this file says how to run each tool.

The tools use the standard library, numpy and pytest only, and find everything relative to their own location:
the repository root is `Path(__file__).resolve().parents[2]`, git refs (`start`, `main`, `tests-frozen`) are read
from that repository, and scratch trees and logs go to `<repo>/.proof_scratch/` (ignored by git, recreated on every
run, never read back between runs). Run them from the repository root. In the development workspace prefix every
command with `pixi run` (e.g. `pixi run python tools/proof/doclint.py`); outside it, use a Python with the
packages of the test suite (`pip install ".[test,accel]"`).

## What each tool proves

| tool | proves | default | last line |
|---|---|---|---|
| `astcheck.py` | every commit of a range changes only what its subject class allows: `docs:` the AST of `rvt/**/*.py` without docstrings is identical; `msg:` as `docs:` except string constants in `raise` / `warn(...)`; `meta:` nothing under `rvt/` changes; `test:` only new files under `tests/`; `refactor:` is listed as `needs-equiv`. Classes of the 0.1.1 fix phase: `fix:` only existing `rvt/**/*.py` files modified, with a code change, docstrings and comments changed only in functions whose code changes, module and class docstrings unchanged (listed as `needs-tests+equiv_diff`); `retire:` only files of the frozen test set (`frozen_manifest.json` at the parent) modified, no file added or deleted, no pytest configuration (listed as `needs-retire_check`); `tool:` only `tools/proof/**` (listed as `needs-selftest`). Self-test: `astcheck_selftest.py` | `--repo` = this repository; `--range` or `--base/--head/--class` required | `ASTCHECK PASS` / `ASTCHECK FAIL` |
| `frozen_check.py` | the files under `tests/` at tag `tests-frozen` are unchanged (SHA-256 against `frozen_manifest.json`), no file there was modified, renamed or deleted, no pytest configuration changed, and all 1089 frozen test nodes report PASSED with accelerators and with `RVT_NO_ACCEL=1` | `--ref main` | `FROZEN PASS` / `FROZEN FAIL` |
| `equiv.py` (with `equiv_plugin.py`) | two trees (A and B) give bitwise identical records for every public call made by the frozen tests: return value hash, array arguments after the call, files written; variants `accel`, `t1` (all thread variables 1), `noaccel` (`RVT_NO_ACCEL=1`); a skipped test fails the run | A = `start`, B = `main`, tests from `tests-frozen`; `--b-dir <checkout>` uses a directory as it is on disk | `EQUIV PASS` / `EQUIV FAIL` |
| `mutants.py` | the frozen tests and `equiv.py` detect 21 hand-written wrong changes (tests kill >= 90 %, equiv kills 100 %) and do not report 2 behaviour-neutral control edits | `--ref main`, `--tests-ref tests-frozen` | `MUTANTS PASS` / `MUTANTS FAIL` |
| `linecov.py` | report only: line coverage of `rvt` by the test suite, with `trace.Trace` (no coverage.py needed) | `--ref main`, `--out .proof_scratch/coverage.md` | the coverage table |
| `doclint.py` | every public module, class, function and method has a numpy-style docstring whose `Parameters` name every parameter in signature order, and a `Returns` section when it returns a value | `--dir` = this repository (the working tree, not a git ref) | `DOCLINT PASS` / `DOCLINT FAIL (<n> problems)` |

`frozen_manifest.json` holds the commit of `tests-frozen`, the SHA-256 of the 16 frozen files and the 1089 node
ids. It was written once with `frozen_check.py --freeze` when the tag was set and is never regenerated. The manifest
hashes are of the git blobs (LF line endings, read with `git show`), not of the files on disk: on a checkout with
`core.autocrlf=true` a hash of a working-tree file (CRLF) does not match, and `frozen_check.py` is still correct.

## Commands

The commands are the same in Windows PowerShell and in bash; only setting an environment variable differs (not
needed here).

| step | command | expected last line | typical runtime |
|---|---|---|---|
| commit classes since `start` | `python tools/proof/astcheck.py --repo . --range start..HEAD` | `ASTCHECK PASS` | 5 s |
| one commit against a class | `python tools/proof/astcheck.py --base HEAD~1 --head HEAD --class docs` | `ASTCHECK PASS` | 1 s |
| astcheck self-test | `python tools/proof/astcheck_selftest.py` | `ASTCHECK SELFTEST PASS` | 5 s |
| frozen tests | `python tools/proof/frozen_check.py` (or `--ref <branch or commit>`) | `FROZEN PASS` | 22 s |
| bitwise equivalence, `start` against `main` | `python tools/proof/equiv.py` | `EQUIV PASS` | 100 s |
| bitwise equivalence of a checkout as it is on disk | `python tools/proof/equiv.py --b-dir <checkout>` | `EQUIV PASS` | 100 s |
| mutation check | `python tools/proof/mutants.py` | `MUTANTS PASS` | 24 runs of the frozen tests (not timed for this file) |
| line coverage | `python tools/proof/linecov.py --out .proof_scratch/coverage.md` | the total row of the table | 35 s |
| docstrings | `python tools/proof/doclint.py` | `DOCLINT PASS` | < 1 s |

Runtimes were measured on the development machine (Windows 11, Intel i9-14900K, one run at a time); `equiv.py`
runs the 1089 frozen tests six times (two trees, three variants), `frozen_check.py` twice.

Notes:

- The tools compare git refs of the repository they live in. `main`, `start` and `tests-frozen` must exist locally
  (`git fetch --tags` in a clone).
- `equiv.py --tag <name>` prefixes the scratch directories, so that two runs can go in parallel.
- `equiv.py` puts `tools/proof` on `PYTHONPATH` so that pytest can load `equiv_plugin.py`. Do not add a module
  to this directory whose name shadows a package (docs/TESTING.md, section 5, describes what happened with a
  file named `coverage.py`).
- The pytest root directory of each scratch tree is pinned with `--rootdir` to the tree itself, so that node ids
  are `tests/...` and do not depend on where the scratch directory lies.
