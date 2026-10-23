# Testing and proof of equivalence

The contract of rvt_py_llm 0.1.0 is: every output of every public function, method and class is bitwise identical
to tag `start` (the optimized code, itself bitwise identical to the fixed upstream baseline on the evaluation sets),
with and without numba, for any thread count; the public API is unchanged. This file describes the tests that
encode the contract, the tools that prove each commit keeps it, and how to run them.

## 1. Test suites

All tests are in `tests/` (1089 tests, about 10 s in one process on the development machine). `tests/conftest.py`
puts the repository root first on `sys.path` and asserts that the imported `rvt` comes from this repository, so the
tests never run against an installed copy.

| suite | files | tests | origin | expected values |
|---|---|---|---|---|
| equivalence tests of the optimized paths | `test_accel.py`, `test_mean_filter.py`, `test_sky_view_factor.py`, `test_local_dominance.py`, `test_sky_illumination.py`, `test_mstp.py`, `test_blend.py` (+ `conftest.py`) | 223 | commit `8413d3e` (tag `start`) | a transcription of the upstream `np.roll` / `np.where` code of tag `baseline-fixed`, kept in each test module; most comparisons are bitwise |
| characterization tests | `test_char_vis.py`, `test_char_blend_func.py`, `test_char_blend.py`, `test_char_e3e4_recipe.py`, `test_char_default.py`, `test_char_tile.py`, `test_char_accel.py` | 859 | 0.1.0, commits `48f2386`, `dcd3d17`, `77c245d`, `c341ec8` | analytic results, properties, plain reference implementations in the test file, hand-written tables |
| API snapshot | `test_api_snapshot.py` | 7 | 0.1.0, commit `77c245d` | the public API at `start`, generated once by `collect_api()` |

What the equivalence tests cover: each optimized function against the upstream formulation on small seeded
synthetic DEMs (multi-block sizes, NaN patches, float32 and float64, edge radii); MSTP offset invariance at
+2,500 m; SVF exact invariance on a 1/256 m grid; two calls bitwise equal; `RVT_NUM_THREADS=1` against `2`; the
accelerated path against the numpy path (skipped when numba is missing); numba missing or failing to compile or run.

What the characterization tests cover: every public function and method of `rvt.vis`, `rvt.blend_func`,
`rvt.blend`, `rvt.default`, `rvt.tile` and the helpers of `rvt._accel`: output dtype and shape, NaN mask, edge
behaviour, parameter checks (exception type), in-place effects on arguments, settings files, output file names,
GeoTIFF outputs (written to `tmp_path`), and the e3MSTP / e4MSTP recipes rebuilt layer by layer with plain code.
Behaviour that looks wrong is pinned with the comment `# pins current behaviour` and listed in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### How expected values are derived

No test reads a stored result file: there are no golden files, binary or text. Expected values are:

- analytic: a plane has slope `atan(|grad|)`; a flat DEM has hillshade `sin(elevation)` and sky-view factor 1; the
  integral image of ones is `(i + 1)(j + 1)`; a wall of height h at distance d has horizon `atan(h / d)`; local
  dominance of a 10 m peak is `11.7 / 1.7`;
- properties: shape, dtype, NaN mask, transpose and flip symmetry, invariance to an elevation offset, the identity
  of a returned object, whether an argument changed;
- a plain reference implementation kept in the test file (the upstream code for the equivalence tests, a plain
  byte scale for `float_to_8bit`, the e3/e4 recipes written out layer by layer), never produced by calling the
  function under test;
- hand-written tables (default values, file names from the formatting rules).

The single exception is the API snapshot: `SNAPSHOT` in `tests/test_api_snapshot.py` was generated once from the
code at `start` and pasted. It fails when a public name, a signature, a default argument or a `DefaultValues` default
changes; an intended change edits `SNAPSHOT` in the same commit (not allowed in 0.1.0, see section 2).

Tolerances: comparisons with the upstream transcription are bitwise unless the test states a tolerance and why
(e.g. `1e-6` for a float32 sum, `1e-4` for MSTP at +2,500 m against 0 m).

### Platform note

`tests/test_char_default.py::test_json_default_path_is_settings_default_settings_json_in_cwd` pins the Windows
behaviour of a path with a backslash (KNOWN_ISSUES KI-52) and is expected to fail on Linux and macOS;
`tests/test_char_blend.py:94` is a Windows-only test for the same pattern (KI-35). The suite was run on Windows 11
only.

## 2. The frozen-test rule

The tests are the contract, so they may not move while the code moves.

- Tag `tests-frozen` (annotated tag on commit `c341ec8`) marks the test set. After it, files that existed under
  `tests/` at the tag may not be modified, renamed or deleted, and no `conftest.py`, `pytest.ini`, `tox.ini`,
  `pyproject.toml` or `setup.cfg` may skip, deselect or xfail them. New tests go into new files.
- A manifest holds the SHA-256 of every frozen file (16 files, 1089 test nodes).
- `frozen_check.py` proves for a commit that (1) `git diff tests-frozen..<commit> -- tests` lists only added
  `test_*.py` files and no pytest configuration changed in a line that mentions pytest; (2) every frozen file
  matches its hash; (3) every frozen test node reports PASSED (no skip, xfail, deselect or error), with accelerators
  on and with `RVT_NO_ACCEL=1`. Last line `FROZEN PASS` or `FROZEN FAIL`.

A failing frozen test means the code change is wrong, not the test.

## 3. The proof chain

The proof tools are in [tools/proof/](../tools/proof/README.md) of this repository and run from the repository
root (`python tools/proof/<tool>.py`, in the development workspace `pixi run python tools/proof/<tool>.py`). They
find the repository from their own location, read the git refs `start`, `main` and `tests-frozen` of this
repository, and build their scratch trees in `.proof_scratch/` (ignored by git). They use the standard library,
numpy and pytest only. The 0.1.0 work ran the same scripts from the parent workspace (`work/llm/` there) before
they were copied into the repository; the copies differ only in paths (repository root, scratch directory) and
in pinning the pytest root directory of a scratch tree, so that node ids do not depend on its location.

### 3.1 astcheck: what a commit may change, proven without running code

`astcheck.py` classifies every commit of a range by its subject prefix and checks it against the parent:

| class | rule |
|---|---|
| `docs:` | the AST of every `rvt/**/*.py` file with docstrings removed is identical (comments and blank lines are not in the AST) |
| `msg:` | as `docs:`, but string constants inside `raise` statements and `warn(...)` calls may differ |
| `meta:` | no file under `rvt/` changes |
| `test:` | only files under `tests/` are added; no existing file under `tests/` changes |
| `refactor:` | any change; such a commit must pass the tests and `equiv.py` |

`--range start..main` checks every commit; merge commits are skipped. Last line `ASTCHECK PASS` or `ASTCHECK FAIL`.
The tool was tested on commits that hide a code change inside a `docs:` or `msg:` commit, and catches them.

### 3.2 equiv: bitwise record comparison of two trees

`equiv.py` runs the same test files (normally from tag `tests-frozen`) against two trees, A (normally `start`) and B
(the candidate), under the pytest plugin `equiv_plugin.py`. The plugin records, for every test and in call order,
every return from a public function or method of `rvt.*`: the qualified name, a hash of the returned value (arrays:
dtype, shape and SHA-256 of the raw bytes, so one flipped bit, -0.0 against 0.0 or a different NaN payload counts),
a hash of the array arguments after the call (in-place effects), and after the test a hash of every file the test
wrote. The two record logs must be identical record by record. A test that passes in both trees but returns a
different bit pattern inside its tolerance is therefore still a failure. Nothing is stored between runs.

Variants, each in fresh processes for both trees:

- `accel`: default environment (numba kernels on, default threading);
- `t1`: `RVT_NUM_THREADS`, `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS` and the other thread variables set to 1;
- `noaccel`: `RVT_NO_ACCEL=1` (numpy fallback path).

The report also lists public functions that no test called. Any skipped test fails the run. Last line `EQUIV PASS`
or `EQUIV FAIL`.

Results:

| run | tests | records accel / t1 / noaccel | public API executed | result |
|---|---|---|---|---|
| `start` against the then `main`, tests from `main` before the freeze | 1021 | 47,024 / 47,024 / 21,320 | 153 / 153 | EQUIV PASS |
| `start` against `main` at `tests-frozen`, tests from `tests-frozen` | 1089 | 58,182 / 58,182 / 28,958 | 153 / 153 | EQUIV PASS |
| each `refactor:` commit (69525a0, 2e1a306, 8839afd) | 1089 | accel, t1, noaccel | 153 / 153 | EQUIV PASS |

The record count depends on what the tests call: the frozen test `test_class_docstring_attributes_all_exist` creates
one `DefaultValues` per attribute entry in its docstring, so a docstring edit that merges two entries changes the
count. This is why the documentation of the local dominance radii was kept at 75 entries.

What equiv covers: every public function, method and class reached by the 1089 tests, on the inputs of those tests
(small synthetic arrays and GeoTIFFs), on one machine. It does not cover inputs the tests do not use (large
rasters, other CPUs); the evaluation harness (section 3.6) covers real DEM windows.

### 3.3 mutants: do the tests and equiv notice wrong changes?

No mutation library is in the environment, so `mutants.py` has 21 hand-written mutants, each one plausible mistake
in one place: wrong padding mode, float32 instead of float64 accumulation, a changed constant, swapped operands, a
changed NaN rule in a numba kernel, an off-by-one in the tile overlap, a changed default, an `RVT_NO_ACCEL` value no
longer accepted, a changed layer opacity in the e3/e4 recipes. For each mutant a scratch tree is built, the frozen
tests run once under the recording plugin, and the mutant counts as killed by tests if the run fails and as killed
by equiv if its records differ from the unmutated tree. Two control edits (a local rename, a docstring edit) must
not be reported.

Thresholds, fixed before the first run and never lowered: tests kill >= 90 %, equiv kills 100 %, both controls
equivalent.

| run | killed by tests | killed by equiv | controls | result |
|---|---|---|---|---|
| before the freeze (tests from `main`, 1021 tests) | 20 / 21 = 95 % (m21, the e3/e4 SLRM layer opacity, survived) | 21 / 21 | equivalent | MUTANTS PASS |
| after the freeze (tests from `tests-frozen`, 1089 tests) | 21 / 21 = 100 % | 21 / 21 | equivalent | MUTANTS PASS |

The surviving mutant m21 led to `tests/test_char_e3e4_recipe.py` (commit `c341ec8`), added before the freeze.

### 3.4 linecov: line coverage

`linecov.py` measures line coverage of `rvt` by the test suite with `trace.Trace` (the environment has no
coverage.py). Report only, not a gate. With numba on, kernel bodies run compiled and count as not executed.

| module | executable lines | executed | coverage |
|---|---|---|---|
| vis.py | 1259 | 1017 | 80.8 % |
| blend.py | 1113 | 1043 | 93.7 % |
| blend_func.py | 423 | 414 | 97.9 % |
| default.py | 1908 | 1444 | 75.7 % |
| tile.py | 334 | 318 | 95.2 % |
| _accel.py | 67 | 46 | 68.7 % |
| total | 5105 | 4282 | 83.9 % |

Measured on `main` before the freeze, accelerators on. No public function or method has a body that never executed.

### 3.5 doclint: docstring completeness

`doclint.py` checks that every public module, class, function and method has a numpy-style docstring whose
`Parameters` section names every parameter in signature order and that functions returning a value have a
`Returns` section. Problems at `start`: 175; at `main` after the documentation pass: 0.

### 3.6 Evaluation harness of the optimization run

The optimization run was judged by a frozen harness (`eval/` of the workspace): correctness of 11 visualizations
on real DEM windows (`dev`, `holdout`, `bench_final`), also with numba made unimportable; determinism with default,
1 and 2 threads; benchmarks with a fresh process per call. On `start` it reported `GATE FINAL PASS` with every output
bitwise identical to `baseline-fixed` (`out/gate_final/` of the workspace). The 0.1.0 release reruns these checks on
the final `main`; their results are recorded in section 6.

## 4. Commands

Run from the repository root unless stated. Do not run plain `pytest` at the root: the upstream scripts
`test_tile.py` and `test_rvt_tile.py` there are not part of the suite; always pass the `tests` path.

| step | Windows PowerShell | Linux bash |
|---|---|---|
| tests, default path | `python -m pytest tests -q` | `python -m pytest tests -q` |
| tests, numpy fallback | `$env:RVT_NO_ACCEL="1"; python -m pytest tests -q; Remove-Item Env:RVT_NO_ACCEL` | `RVT_NO_ACCEL=1 python -m pytest tests -q` |
| tests, one thread | `$env:RVT_NUM_THREADS="1"; python -m pytest tests -q; Remove-Item Env:RVT_NUM_THREADS` | `RVT_NUM_THREADS=1 python -m pytest tests -q` |

Expected last line: `1089 passed` (3 warnings) in all three runs; with `RVT_NO_ACCEL=1` the tests that compare
the numba path with the numpy path run too (they switch the accelerators on themselves), so nothing is skipped.
Without numba installed, the tests that need it are skipped. In the development workspace, prefix with `pixi run`
and pass the path of the checkout, e.g. `pixi run pytest rvt_py_llm/tests -q`.

Proof tools, from the repository root (PowerShell and bash are the same command; in the development workspace
prefix `pixi run`). Expected last lines and runtimes: [tools/proof/README.md](../tools/proof/README.md).

```
python tools/proof/astcheck.py --repo . --range start..main      # ASTCHECK PASS
python tools/proof/frozen_check.py --ref main                    # FROZEN PASS
python tools/proof/equiv.py                                      # start vs main, frozen tests; EQUIV PASS
python tools/proof/equiv.py --b-dir <checkout>                   # start vs a checkout as it is on disk
python tools/proof/mutants.py --ref main --tests-ref tests-frozen   # MUTANTS PASS
python tools/proof/linecov.py --ref main --out <file.md>         # coverage table
python tools/proof/doclint.py                                    # DOCLINT PASS
```

## 5. Lesson: a helper script that shadowed a package

The first line-coverage helper was saved as `work/llm/coverage.py` in the workspace. The proof tools put their own
directory (then `work/llm`, now `tools/proof`) on `PYTHONPATH` so that pytest can load `equiv_plugin.py`; numba
imports the `coverage` package when it is present, and found the helper script instead, so numba failed to import
in those runs. The test suite then skipped the 20 tests
that need numba, the accelerated paths fell back to numpy, and two mutants in numba kernels (m06, m07) survived
because no kernel ran. Nothing was wrong with the code under test; the measuring setup had changed it. The helper was
renamed to `linecov.py`, and `equiv.py` and `mutants.py` now fail on any skipped test, so that a silent loss of the
accelerated path cannot pass again. The general point: a proof run has to check that it exercised what it claims to
exercise (no skips, kernels actually ran), not only that nothing failed.

## 6. Release evidence

The evaluation harness of the optimization run (`eval/` of the parent workspace, frozen, see section 3.6) is
rerun on the release candidate of 0.1.0 (2026-10-23, Windows 11, Intel i9-14900K, 32 logical CPUs, Python 3.11.17,
numpy 2.4.6). Copies of the reports are in [release_evidence/0.1.0/](release_evidence/0.1.0/).

Measured commit: `3df56b7` (package content hash `5d053dca2dcf`, as computed by `eval/common.py` `impl_hash`).
Later commits up to the tag `v0.1.0` change no file under `rvt/` (`git diff --stat 3df56b7 v0.1.0 -- rvt` is empty),
so the package that was measured is the released one.

| check | what it compares | result | report |
|---|---|---|---|
| eval check `dev` | 11 visualizations on the 8 `dev` DEM windows against `baseline-fixed` (8-bit tolerance, exactness reported) | PASS, exact | `check_dev_20261023-144312_19692.json` |
| eval check `dev --no-accel` | as `dev`, with `RVT_NO_ACCEL=1` and numba made unimportable | PASS, exact | `check_dev_noaccel_20261023-144549_19088.json` |
| eval determinism | default threading twice, 1 thread, 2 threads: bitwise equal | PASS | `check_determinism_20261023-145142_29184.json` |
| eval check `holdout` | 11 visualizations on the `holdout` windows (never used during development) | PASS, exact | `check_holdout_20261023-144724_30768.json` |
| eval check `bench_final` | outputs on the three 2048 x 2048 px `bench_final` windows | PASS, exact | `check_bench_final_20261023-144937_33688.json` |
| bench quick against `opt/perf` | speed of `3df56b7` against `rvt` `opt/perf` (= tag `start`), 1 thread, `bench_quick` | pipeline x1.00 [0.995-1.012], no target regressed | `bench_quick_1t_20261023-144020_5544.json` |
| final gate, speed, 1 thread | 11 targets against `baseline-fixed` on `bench_final` | pipeline x11.41 [11.32-11.43] (opt/perf: x11.44 [11.35-11.46]) | `bench_final_1t_20261023-170737_32096.json` |
| final gate, speed, default threads (informative) | as above, default threading | pipeline x11.37 [11.31-11.57] | `bench_final_mt_20261023-174445_18544.json` |
| final gate | all of the above | `GATE FINAL PASS` | `gate_final.log` |

"exact" means that every output differs from `baseline-fixed` by at most 1e-5 in float with an identical NaN mask
(`eval/common.py`); 0 DN difference in every 8-bit comparison.

Incident during these measurements: a stray `python -` process left by a crashed Git Bash command (not related to
rvt) occupied one CPU core from 12:03 to 17:38, so the bench quick run and most of the final gate ran with one core
busy. Baseline and candidate are timed alternately in the same rounds, and the 1-thread pipeline factor agrees
with the opt/perf gate within 0.3 % (intervals overlap), so the comparison is not biased; absolute times may be
slightly longer than on an idle machine.
