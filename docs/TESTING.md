# Testing and proof of equivalence

The contract of rvt_py_llm 0.1.0 is: every output of every public function, method and class is bitwise identical
to tag `start` (the optimized code, itself bitwise identical to the fixed upstream baseline on the evaluation sets),
with and without numba, for any thread count; the public API is unchanged. This file describes the tests that
encode the contract, the tools that prove each commit keeps it, and how to run them.

0.1.1 changes the outputs of 15 public functions on purpose (the blend path of CRIM, e3MSTP and e4MSTP, see
[FIXES_0_1_1.md](FIXES_0_1_1.md)); every other public function stays bitwise identical to `v0.1.0`. Section 7
describes how this was proven and how the frozen test set moved.

0.2.0 adds the opt-in keyword `kernel_shape` (disk windows, [DISK_WINDOWS.md](DISK_WINDOWS.md)); with the default
`"square"` every public function stays bitwise identical to `v0.1.1`. Section 8 describes the proof.

## 1. Test suites

All tests are in `tests/` (1428 tests in 28 files, all frozen at tag `tests-frozen` since 0.2.0: the 1089 tests of
the 0.1.0 freeze, `test_proof_box_sum.py` (4), the 0.1.1 fix tests `test_fix_0_1_1_*.py` (101) and the 0.2.0
disk-window tests `test_disk_*.py` (234); about 14 s in one process on the development machine). `tests/conftest.py`
puts the repository root first on `sys.path` and asserts that the imported `rvt` comes from this repository, so the
tests never run against an installed copy.

| suite | files | tests | origin | expected values |
|---|---|---|---|---|
| equivalence tests of the optimized paths | `test_accel.py`, `test_mean_filter.py`, `test_sky_view_factor.py`, `test_local_dominance.py`, `test_sky_illumination.py`, `test_mstp.py`, `test_blend.py` (+ `conftest.py`) | 223 | commit `8413d3e` (tag `start`) | a transcription of the upstream `np.roll` / `np.where` code of tag `baseline-fixed`, kept in each test module; most comparisons are bitwise |
| characterization tests | `test_char_vis.py`, `test_char_blend_func.py`, `test_char_blend.py`, `test_char_e3e4_recipe.py`, `test_char_default.py`, `test_char_tile.py`, `test_char_accel.py` | 859 | 0.1.0, commits `48f2386`, `dcd3d17`, `77c245d`, `c341ec8` | analytic results, properties, plain reference implementations in the test file, hand-written tables |
| API snapshot | `test_api_snapshot.py` | 7 | 0.1.0, commit `77c245d` | the public API at `start`, generated once by `collect_api()` |
| order-sensitive `_box_sum` tests | `test_proof_box_sum.py` | 4 | after the 0.1.0 audit, commit `422ad9f` | inputs with full 53-bit mantissas and the upstream order of the subtractions |
| 0.1.1 fix tests | `test_fix_0_1_1_x1_blend_new_array.py`, `..._x2_e4mstp_slope_degrees.py`, `..._x3_clip_and_scale.py`, `..._x4_default_not_modified.py`, `..._x5_opacity_check.py`, `test_fix_0_1_1_recipes.py` | 101 | 0.1.1, commit `7aa8b03`, written before the fixes | the opacity formula `o * B(a, b) + (1 - o) * b` in the test, band independence, the slope layer from degrees computed with numpy, clipping with NaN kept, `DefaultValues` attributes before and after, the raised exception, e3MSTP / e4MSTP rebuilt through `BlenderCombination`; at opacity 100 a copy of the old in-place overlay / soft light gives the same bits |
| 0.2.0 disk-window tests | `test_disk_oracle.py`, `test_disk_threshold.py`, `test_disk_anisotropy.py`, `test_disk_composition.py`, `test_disk_settings.py`, `test_disk_msrm_scales.py` | 234 | 0.2.0, commit `450d49e` (219, written before the feature) and `685c920` (15, after the mutant run of section 3.3) | a brute-force float64 disk window in the test file (two-pass variance, NaN excluded, counts from the mask, the padding mode of each function), the square result for windows below 8 px (bitwise), the review C-15 Gaussian bank, the composition of the lower-level functions, the 0.1.1 settings keys written out by hand, the e3MSTP / e4MSTP recipes rebuilt in the test |

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

- Tag `tests-frozen` marks the test set: an annotated tag on commit `c341ec8` for 0.1.0 (now also tag
  `tests-frozen-0.1.0`), moved to `5b5cbad` in 0.1.1 (section 7; now also tag `tests-frozen-0.1.1`) and to
  `b691587` in 0.2.0 (section 8). After it, files that existed under
  `tests/` at the tag may not be modified, renamed or deleted, and no `conftest.py`, `pytest.ini`, `tox.ini`,
  `pyproject.toml` or `setup.cfg` may skip, deselect or xfail them. New tests go into new files.
- A manifest holds the SHA-256 of every frozen file (0.1.0: 16 files, 1089 test nodes; 0.1.1: 23 files, 1194 test
  nodes; 0.2.0: 29 files, 1428 test nodes).
- `frozen_check.py` proves for a commit that (1) `git diff tests-frozen..<commit> -- tests` lists only added
  `test_*.py` files and no pytest configuration changed in a line that mentions pytest; (2) every frozen file
  matches its hash; (3) every frozen test node reports PASSED (no skip, xfail, deselect or error), with accelerators
  on and with `RVT_NO_ACCEL=1`. Last line `FROZEN PASS` or `FROZEN FAIL`.

A failing frozen test means the code change is wrong, not the test.

## 3. The proof chain

The proof tools are in [tools/proof/](../tools/proof/README.md) of this repository and run from the repository
root (`python tools/proof/<tool>.py`, with the pinned pixi environment `pixi run python tools/proof/<tool>.py`). They
find the repository from their own location, read the git refs `start`, `main` and `tests-frozen` of this
repository, and build their scratch trees in `.proof_scratch/` (ignored by git). They use the standard library,
numpy and pytest only. The 0.1.0 work ran the same scripts from a directory outside this repository before
they were copied into it; the copies differ only in paths (repository root, scratch directory) and
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
| `fix:` (0.1.1) | only existing `rvt/**/*.py` files change and the AST without docstrings of at least one differs; a function whose code is unchanged keeps its docstring and source text, module and class docstrings are unchanged; listed as needing the full tests and the record comparison of section 7 |
| `retire:` (0.1.1) | only files of the frozen test set change, none is added or deleted, no pytest configuration changes; listed as needing the retire check of section 7 |
| `tool:` (0.1.1) | only files under `tools/proof/` change; listed as needing the tool's self-test |
| `feat:` (0.2.0) | only existing files under `rvt/` (Python) and `settings/` change, none added or deleted, with a code change; docstrings and comments change only in functions whose code changes, a class docstring only with a change of that class, module docstrings not at all; listed as needing the full tests, the record comparison and the determinism check of section 8 |

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

What equiv does not compare: raised exceptions, their types and messages (covered by the frozen tests and by
astcheck); warnings; the contents of `.txt` and `.log` files a test writes (they hold timestamps, so only their
line count is compared); in-place changes of dict or object members (only
array arguments are hashed; the known in-place effects are asserted by the frozen tests); private functions (only
through the public calls that reach them); inputs outside the frozen tests.

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
| final `main` (`de0f665`, after the refactors; tests from `tests-frozen`, 1089 tests) | 21 / 21 = 100 % | 21 / 21 | equivalent | MUTANTS PASS |
| final `main` (`02f0b11`) with m22-m23 on the refactored helpers; tests from `tests-frozen`, 1089 tests | 22 / 23 = 96 % (m22 survived) | 22 / 23 = 96 % (m22 survived) | equivalent | MUTANTS FAIL (equiv below 100 %) |
| final `main` (`02f0b11`), tests from `main` (1093 tests, incl. `test_proof_box_sum.py`) | 23 / 23 = 100 % | 23 / 23 | equivalent | MUTANTS PASS |
| 0.1.1 `main` (`81e50b1`) with m24-m28 on the fixed code; tests from the new `tests-frozen` (1194 tests) | 28 / 28 = 100 % | 28 / 28 | equivalent | MUTANTS PASS |
| 0.2.0 `main` (`1668231`) with m29-m33 on the disk path; tests from the new `tests-frozen` (1413 tests) | 31 / 33 = 94 % (m13 and m31 survived) | 33 / 33 | equivalent | MUTANTS PASS |

m31 (FFT pixel counts not rounded to integers) changes the disk-path results only below the test tolerances
(1e-9 m on the window mean, 1e-7 on DEV); equiv, which compares bits, kills it. m13 (the number of MSRM scales)
survived because its first match had moved: 0.2.0 repeats the expression in `vis._msrm_radii` (disk path), and every
disk MSRM test used feature sizes for which `ceil` and `floor` agree. The gap was closed with the new file
`tests/test_disk_msrm_scales.py` (`685c920`) and the mutants m13 (scoped to `msrm`) and m34 (the same edit in
`_msrm_radii`, `b691587`), before the second freeze:

| run | killed by tests | killed by equiv | controls | result |
|---|---|---|---|---|
| 0.2.0 `main` (`0b218ad`) with m29-m34; tests from `tests-frozen` (1428 tests) | 33 / 34 = 97 % (m31 survived) | 34 / 34 | equivalent | MUTANTS PASS |

The surviving mutant m21 led to `tests/test_char_e3e4_recipe.py` (commit `c341ec8`), added before the freeze.
The surviving mutant m22 (the two subtractions of `_box_sum` in the other order) shows a limit of the frozen
corpus, not of the code: the frozen tests use small DEMs whose float64 integral images are exact, so the order of
the subtractions cannot change a bit there. The refactor 8839afd keeps the original order (checked by reading the
diff, by the release audit, and by the audit's out-of-corpus bitwise comparison of `start` and `main`). The gap was
closed with the new, additive test file `tests/test_proof_box_sum.py` (commit `422ad9f`; inputs with full 53-bit
mantissas, where the two orders round differently); no frozen file was changed, and the threshold was not lowered:
the run against the frozen tests is reported as FAIL above.
The logs of the two final runs are in [release_evidence/0.1.0/](release_evidence/0.1.0/)
(`mutants_final_frozen.log`, `mutants_final_all_main_tests.log`, with `frozen_final.log`). Mutants m01-m21 do not touch the code that the three refactors changed (`_accumulate`, `_box_sum`,
`BLOCK_ELEMS`); m22-m23 are written for those helpers.

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

The optimization run was judged by a frozen acceptance harness (not shipped, section 9): correctness of 11
visualizations on real DEM windows (`dev`, `holdout`, `bench_final`), also with numba made unimportable;
determinism with default, 1 and 2 threads; benchmarks with a fresh process per call. On `start` it reported
`GATE FINAL PASS` with every output bitwise identical to `baseline-fixed` (speed report
`release_evidence/0.1.0/bench/bench_final_1t_20261023-082005_12380.json`). The 0.1.0 release reruns these checks on
the final `main`; their results are recorded in section 6.

## 4. Commands

Run from the repository root unless stated. Do not run plain `pytest` at the root: the upstream scripts
`test_tile.py` and `test_rvt_tile.py` there are not part of the suite; always pass the `tests` path.

| step | Windows PowerShell | Linux bash |
|---|---|---|
| tests, default path | `python -m pytest tests -q` | `python -m pytest tests -q` |
| tests, numpy fallback | `$env:RVT_NO_ACCEL="1"; python -m pytest tests -q; Remove-Item Env:RVT_NO_ACCEL` | `RVT_NO_ACCEL=1 python -m pytest tests -q` |
| tests, one thread | `$env:RVT_NUM_THREADS="1"; python -m pytest tests -q; Remove-Item Env:RVT_NUM_THREADS` | `RVT_NUM_THREADS=1 python -m pytest tests -q` |

Expected last line: `1428 passed` (3 warnings) in all three runs; with `RVT_NO_ACCEL=1` the tests that compare
the numba path with the numpy path run too (they switch the accelerators on themselves), so nothing is skipped.
Without numba installed, the tests that need it are skipped. With the pinned pixi environment (README, Install,
route A), prefix with `pixi run`: `pixi run python -m pytest tests -q`.

Proof tools, from the repository root (PowerShell and bash are the same command; with the pixi environment
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

The first line-coverage helper was saved as `coverage.py` next to the other proof scripts (then outside this
repository, see section 3). The proof tools put their own directory (now `tools/proof`) on `PYTHONPATH` so that
pytest can load `equiv_plugin.py`; numba
imports the `coverage` package when it is present, and found the helper script instead, so numba failed to import
in those runs. The test suite then skipped the 20 tests
that need numba, the accelerated paths fell back to numpy, and two mutants in numba kernels (m06, m07) survived
because no kernel ran. Nothing was wrong with the code under test; the measuring setup had changed it. The helper was
renamed to `linecov.py`, and `equiv.py` and `mutants.py` now fail on any skipped test, so that a silent loss of the
accelerated path cannot pass again. The general point: a proof run has to check that it exercised what it claims to
exercise (no skips, kernels actually ran), not only that nothing failed.

## 6. Release evidence

The acceptance harness of the optimization run (not shipped, see sections 3.6 and 9) is
rerun on the release candidate of 0.1.0 (2026-10-23, Windows 11, Intel i9-14900K, 32 logical CPUs, Python 3.11.17,
numpy 2.4.6). Copies of the reports are in [release_evidence/0.1.0/](release_evidence/0.1.0/).

Measured commit: `3df56b7` (package content hash `5d053dca2dcf`: the first 12 hex digits of the SHA-256 over the
relative path and the bytes of every file under `rvt/`, bytecode excluded, in path order, as the harness computes it).
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
(the comparison of the harness, section 9); 0 DN difference in every 8-bit comparison.

Incident during these measurements: a stray `python -` process left by a crashed Git Bash command (not related to
rvt) occupied one CPU core from 12:03 to 17:38, so the bench quick run and most of the final gate ran with one core
busy. Baseline and candidate are timed alternately in the same rounds, and the 1-thread pipeline factor agrees
with the opt/perf gate within 0.3 % (x11.41 against x11.44, intervals overlap), so the comparison is not biased.

The absolute times are not comparable with the opt/perf gate. In the release run (report
`bench_final_1t_20261023-170737_32096.json`) baseline and candidate medians are both 2.30-2.45 times longer than in
the final gate of the optimization run on an idle machine (2026-10-23 08:20, report
`release_evidence/0.1.0/bench/bench_final_1t_20261023-082005_12380.json`), for every one of
the 33 rows (11 targets x 3 windows); per-row speed-ups agree within 2.2 %, except slope on the synthetic window
(x1.00 against x0.95, 4.6 %). Medians, `medium` window, opt/perf gate -> release run:

| target | baseline | candidate | speed-up |
|---|---|---|---|
| local dominance | 33.97 s -> 80.51 s (x2.37) | 0.429 s -> 1.014 s (x2.36) | x79.16 -> x79.36 |
| sky illumination | 38.17 s -> 87.67 s (x2.30) | 5.629 s -> 12.991 s (x2.31) | x6.78 -> x6.75 |
| e4MSTP | 67.27 s -> 158.52 s (x2.36) | 4.136 s -> 9.780 s (x2.36) | x16.26 -> x16.21 |

A uniform factor of about 2.4 on both sides is not what one busy core of 32 does to processes free to run
elsewhere. The probable cause (not verified) is that the harness pins each timed 1-thread process to the first CPU
of its affinity list (the runner of the harness, not shipped: `cpu_affinity(cpus[: max(1, a.threads)])`), and the stray
process ran on that same CPU, so every timed process shared its CPU with it. Absolute times of this release are
therefore quoted from the opt/perf gate, with that report named next to them.

## 7. The 0.1.1 fixes: changed outputs, proven scope

0.1.1 fixes five defects of the blend path ([FIXES_0_1_1.md](FIXES_0_1_1.md)). The work ran under a written plan
with fixed readings; the lists and scripts named below are not in this repository (section 9). The steps:

1. **Lists written before any code changed.** An allow-list of the 15 public functions whose records may change
   (`blend_func.blend_overlay`, `blend_soft_light`, `equation_blend`, `blend_multi_dim_images`, `blend_images`,
   `render_images`, `scale_within_0_and_1`, `scale_strict_0_to_1`, `scale_0_to_1`; `blend.BlenderLayer.check_data`,
   `BlenderCombination.check_data`, `BlenderCombination.render_all_images`, `color_relief_image_map`, `e3mstp`,
   `e4mstp`) and a list of the frozen 0.1.0 nodes that pinned a fixed behaviour (66 nodes in 22 test functions,
   plus two helpers of the e3/e4 reference recipe).
2. **Tests first** (`7aa8b03`, `test:`): 101 new tests in six files; on `v0.1.0` 75 of them fail (the fixed
   behaviour) and 26 pass (invariants such as the opacity-100 bits).
3. **Retire** (`8f569ca`, `retire:`): the listed nodes get the fixed expectation, marked `# 0.1.1: fixed, KI-nn`; no
   node was removed. A retire check parses every frozen file at the 0.1.0 freeze and at the commit: every function
   outside the list is AST-identical and byte-identical, no function is added, every non-listed node is still
   collected, no pytest configuration changes. Result `RETIRE OK` (66 retired, 1023 kept).
4. **Tools** (`65ef5e8`, `abe40a5`, `tool:`): the astcheck classes `fix:`, `retire:`, `tool:` (section 3.1) and
   mutants m24-m28 on the fixed code (section 3.3).
5. **One `fix:` commit per defect** (`a37ab44` X5, `c72646c` X4, `33a2bd5` X2, `c2456b1` X1, `d09e45d` X3), then
   `docs:` and `meta:` commits. Every commit was checked in its own worktree: astcheck; the full tests with and
   without `RVT_NO_ACCEL=1` (a fix commit fails only the tests of the later fixes; from `d09e45d` on all pass);
   a record comparison with `v0.1.0` (below); the harness check on `dev` with and without
   accelerators and its determinism check; the preview of the harness against the previous commit as per-fix
   attribution.
6. **Freeze** (`81e50b1`): tag `tests-frozen` moved to `5b5cbad` (23 files, 1194 nodes, all PASSED); the 0.1.0
   freeze stays as tag `tests-frozen-0.1.0`.

The record comparison is `equiv.py` in a scoped form: both trees run the 0.1.0 frozen tests under the recording
plugin in the variants accel, t1 and noaccel; every differing record is classified as *allowed* (function on the
allow-list), *propagated* (another function, in a test where an allowed function also differs: the plugin
records outputs and arguments after the call, not the inputs, so a function fed a changed array cannot be told
apart from a changed function) or *unexplained* (fails). The same tool compares the accel, t1 and noaccel logs of
the new tree with each other (records of `rvt._accel`, which report the switch state, left out). Result on
`main`: allowed 14, propagated 12 (ten functions of `rvt.blend` and `rvt.blend_func` that receive changed arrays,
such as `apply_opacity` and `normalize_image`; `vis.slope_aspect` called
with `output_units="degree"` by X2, `DefaultValues.__init__` in a test where 0.1.1 raises before rendering),
unexplained 0 in every variant; accel = t1 = noaccel. A tree with one unrelated function changed by a factor
1.0000001 (`blend_func.lum`) was reported as unexplained in the tool's self-test.

Results on the release candidate (`5b5cbad`; later commits change no file under `rvt/`), Windows 11, the machine of
section 6; copies in [release_evidence/0.1.1/](release_evidence/0.1.1/):

| check | result | report |
|---|---|---|
| tests, accel / `RVT_NO_ACCEL=1` | 1194 passed / 1194 passed | `frozen_check_main.log` |
| `frozen_check.py` on `main` | FROZEN PASS (1194 / 1194 frozen nodes PASSED, both variants) | `frozen_check_main.log` |
| `astcheck.py --range v0.1.0..main` | ASTCHECK PASS | `astcheck_main.log` |
| retire check, 0.1.0 freeze against `main` | RETIRE OK | `retire_check_main.log` |
| record comparison against `v0.1.0`, and accel / t1 / noaccel | allowed 14, propagated 12, unexplained 0; EQUIV_DIFF OK | `equiv_diff_main.log` |
| `mutants.py` | MUTANTS PASS, 28 / 28 | `mutants_main.log` |
| `doclint.py` | DOCLINT PASS | `doclint_main.log` |
| eval check `dev` / `dev --no-accel` (88 rows) | e3mstp and e4mstp rows differ (the fixes); every other row PASS, exact | `check_dev_20261024-120634_19552.json`, `check_dev_noaccel_20261024-120916_3168.json` |
| eval determinism | PASS | `check_determinism_20261024-121122_16772.json` |
| e3mstp / e4mstp time, 0.1.1 against 0.1.0, medium 2048 x 2048 px, 1 thread, median of 3 | x0.872 / x1.030 (limit x1.05) | `timing.log` |
| per commit: tests, astcheck, record comparison, eval checks, preview | see the summaries | `acceptance_summary.txt` |

## 8. The 0.2.0 disk windows: default path unchanged, new path tested

0.2.0 adds the keyword `kernel_shape` ([DISK_WINDOWS.md](DISK_WINDOWS.md)). The work ran under a written plan; the
lists and scripts named below are not in this repository (section 9). The steps:

1. **Lists written before any code changed.** An allow-list of functions whose records may differ from `v0.1.1`:
   empty. A list of the frozen 0.1.1 nodes that pin the API the feature extends, found by a dry run of the frozen
   tests against a stub of the new API (keyword, attributes, settings keys; no behaviour): 4 nodes in
   `test_api_snapshot.py` and `test_char_default.py`, plus the helper `_changed_copy`. The same dry run showed the
   only record differences: the hash of the settings json that two settings tests write (it gains the three keys;
   a new test checks its content) and the records of the 4 retired nodes. These are listed by node and left out of
   the comparison.
2. **Tests first** (`450d49e`, `test:`): 219 tests in five files; on `v0.1.1` 213 fail (missing keyword or
   attribute) and 6 pass (the default path). Their reference is a brute-force float64 disk window written in the
   test file.
3. **Retire** (`9a313f1`): the 4 nodes add the 0.2.0 entries to their expectation (`# 0.2.0: kernel_shape`); module
   level literals unchanged. `RETIRE OK` (1194 base nodes, 4 retired, 1190 kept).
4. **Tools** (`bcebe7e`): astcheck class `feat:` (section 3.1), mutants m29-m33.
5. **Four `feat:` commits** (`202aa15`, `7468298`, `982d615`, `2ee3ecf`), then `docs:` and `meta:`. Every commit was
   checked in its own worktree: the full tests with and without `RVT_NO_ACCEL=1` (a feat commit fails only tests of
   the later steps); astcheck; the retire check; the record comparison with `v0.1.1` and `--self`; eval check
   `dev` with and without numba (every row equal to the row of `v0.1.1`) and `--determinism`; from the second
   commit on a determinism check of the disk path in fresh processes (default threading, every thread variable 1
   and 2, `RVT_NO_ACCEL=1`; mstp, slrm, msrm, e3mstp, e4mstp in both disk shapes on the 8 `dev` tiles).
6. **Freeze** (`1668231`), a mutant run that exposed the MSRM scale-count gap (section 3.3), the test
   `test_disk_msrm_scales.py` (`685c920`), mutants m13 / m34 (`b691587`), and the second freeze (`0b218ad`): tag
   `tests-frozen` on `b691587`, 29 files, 1428 nodes; the 0.1.1 freeze stays as tag `tests-frozen-0.1.1`.

Results on the release candidate (`main` before the tag `v0.2.0`; the last commit under `rvt/` is `c073b10`),
Windows 11; copies in
[release_evidence/0.2.0/](release_evidence/0.2.0/):

| check | result | report |
|---|---|---|
| tests, accel / `RVT_NO_ACCEL=1` | 1428 passed / 1428 passed | `frozen_check_main.log` |
| `frozen_check.py` on `main` | FROZEN PASS | `frozen_check_main.log` |
| `astcheck.py --range v0.1.1..main` | ASTCHECK PASS | `astcheck_main.log` |
| retire check, 0.1.1 freeze against `main` | RETIRE OK | `retire_check_main.log` |
| record comparison against `v0.1.1` (frozen 0.1.1 tests; accel, t1, noaccel) and `--self` | allowed 0, propagated 0, unexplained 0; 0 differing nodes between variants; EQUIV_DIFF OK | `equiv_diff_main.log` |
| determinism of the disk path, fresh processes, 4 variants | DET OK (all 5 products, both shapes, different from square) | `det_check_main.log` |
| `mutants.py` | MUTANTS PASS, tests 33 / 34, equiv 34 / 34 | `mutants_main.log` |
| `doclint.py` | DOCLINT PASS | `doclint_main.log` |
| eval check `dev` / `dev --no-accel` (88 rows) | every row equal to `v0.1.1` (e3mstp and e4mstp rows differ from the 0.1.0 reference since 0.1.1) | `acceptance_summary.txt` |
| eval determinism | PASS | `acceptance_summary.txt` |
| e3mstp / e4mstp time, 0.2.0 against 0.1.1, medium 2048 x 2048 px, 1 thread, median of 5 | x1.002 / x0.997 (limit x1.02) | `timing.log` |
| comparison runs, square of 0.2.0 against `v0.1.1` (3 tiles of 1024 x 1024 px, 3 full DEMs, 5 products) | 30 / 30 bitwise equal | `compare_tiles.md`, `compare_full.md` |

## 9. External evidence: the acceptance harness is not shipped

Everything in this repository can be rerun from a clone: the tests, the proof tools of `tools/proof/` and the
commands of section 4. Two things cannot, because they live in a private workspace outside this repository and are
not shipped:

- **The acceptance harness.** A set of scripts that judged every performance commit and every release against the
  upstream code. `check.py` computes the 11 visualizations of a target list with the candidate and with the cached
  baseline on a set of DEM windows and compares them; the tolerance is stated in DN of rvt's own 8-bit display
  stretch of each output (p99.9 of the absolute difference at most 5 DN, maximum 13 DN, mean 0.5 DN; "exact" in the
  tables above means a float difference of at most 1e-5 with an identical NaN mask, which every release met). Its
  exit code is 0 for pass, 1 for a failing candidate and 2 for an error of the harness itself. With
  `--determinism` it runs the candidate twice with default threading, once with 1 and once with 2 threads and
  requires bitwise identical outputs. `bench.py` times baseline and candidate in a fresh process per call, alternating
  the two in every round (so a change of machine load hits both), and reports medians and bootstrap intervals of
  the ratio per target and for the pipeline. `gate.py` runs the checks and the benchmark in sequence and prints
  `GATE ... PASS` or `FAIL`. The JSON reports and logs that these scripts wrote are copied into
  `release_evidence/<version>/` (and `release_evidence/0.1.0/bench/` for the per-commit benchmarks of
  [CHANGES_VS_UPSTREAM.md](CHANGES_VS_UPSTREAM.md)); the scripts are not.
- **The DEM sets.** Real lidar DEMs at 0.25 m/px, cut into the windows `dev` (8 windows used during development),
  `holdout` (never used during development), `bench_quick` (one 1024 x 1024 px window) and `bench_final` (three
  2048 x 2048 px windows: two real, one synthetic at +300 m), plus three full DEMs of about 10 000 x 8 000 px.
  They are not part of the repository.

A reader without the harness can still check the two claims that matter for using the package: the test suite
(section 4) proves the behaviour of every public function, and `tools/proof/equiv.py --a v0.2.0 --b-dir .` proves
that a change keeps every public call bitwise identical to the tag `v0.2.0` on the inputs of the frozen tests.
The absolute speed-ups are reproducible in order of magnitude on any DEM with the functions named in
[CHANGES_VS_UPSTREAM.md](CHANGES_VS_UPSTREAM.md); the exact ratios depend on the DEM and the machine.
