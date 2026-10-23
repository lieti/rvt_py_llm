# Changelog

All notable changes of rvt_py_llm. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Details and evidence: [docs/CHANGES_VS_UPSTREAM.md](docs/CHANGES_VS_UPSTREAM.md).

## [0.1.0] - 2026-10-23

First release of rvt_py_llm, a drop-in of rvt_py (import name `rvt`). Outputs are bitwise identical to tag `start`,
which is bitwise identical to the upstream code with the precision fix (`baseline-fixed`) on the evaluation sets.

### Fixed
- MSTP computes the local variance from float64 squares (`03702f4`). MSTP, e3MSTP and e4MSTP outputs therefore
  differ from upstream: upstream changed the 8-bit e3MSTP/e4MSTP by up to 207-210 DN (99.9th percentile 150 DN)
  when the same terrain was shifted by +17.3 m (512 x 512 px synthetic DEM at 300 m); after the fix by at most 1 DN
  (measurement in the message of `03702f4`).
- 40 exception messages in `rvt.vis` and `rvt.blend_func` name the right module (`rvt.vis` instead of
  `rvt.visualization`), function and bound; exception types are unchanged (`51a882a`, `66a4709`).

### Changed
- Performance (`21284eb` .. `bd173ea`): the eleven visualizations of a tiling pipeline run x11.44 [11.35-11.46]
  faster on `bench_final` (2048 x 2048 px, 0.25 m/px, 1 thread) with bitwise identical outputs; local dominance
  x79.2, MSTP x13.6, e4MSTP x16.3, e3MSTP x8.0, SVF/openness x7.1-x8.2, sky illumination x6.8, MSRM x4.8, SLRM x4.4
  (medium window).
- Three behaviour-neutral readability refactors, proven bitwise identical: the block size in `rvt/_blocking.py`
  (`69525a0`), one accumulation helper `rvt.vis._accumulate` (`2e1a306`), one box-sum helper `rvt.vis._box_sum`
  (`8839afd`).
- Packaging: name `rvt_py_llm`, version 0.1.0, `packages=["rvt"]`; `install_requires` lists the runtime imports
  (numpy, scipy, matplotlib, gdal) instead of upstream's list that also had pandas, geopandas, jupyter and rasterio;
  `python_requires=">=3.9"` (upstream `>=3.6, <3.12`; tested on 3.11 only); extras `accel` (numba) and `test`
  (pytest, rasterio); `setup.cfg` removed (it repeated the metadata with other values and pointed to a missing
  package directory).

### Added
- Optional numba acceleration of four inner loops (five kernels), with automatic fallback to numpy and
  `RVT_NO_ACCEL=1` (`rvt/_accel.py`, `0ddd8e0` .. `bd173ea`).
- Proof tools in `tools/proof/` (`astcheck.py`, `frozen_check.py`, `equiv.py` with `equiv_plugin.py`,
  `mutants.py`, `linecov.py`, `doclint.py`, `frozen_manifest.json`), run from the repository root.
- Tests: 223 equivalence tests (`8413d3e`) and 866 characterization and API-snapshot tests (frozen at tag
  `tests-frozen`).
- Docstrings for every public module, class, function and method; equivalence contracts for private helpers and
  kernels.
- Documents: `docs/ARCHITECTURE.md`, `docs/CHANGES_VS_UPSTREAM.md`, `docs/KNOWN_ISSUES.md` (76 issues, documented,
  not changed), `docs/TESTING.md`, `docs/CONVENTIONS.md`, `AGENTS.md`. The upstream README moved to
  `docs/UPSTREAM_README.md`.

### Verification
- Tests: 1089 passed, with accelerators and with `RVT_NO_ACCEL=1` (no test skipped).
- At the freeze: `equiv.py` (accel, t1, noaccel; 153/153 public functions executed) and `mutants.py` (21/21 mutants
  killed by the tests and by equiv) PASS. For each refactor commit: tests (accel and `RVT_NO_ACCEL=1`) and
  `equiv.py` PASS. Every commit of `start..main` passes `astcheck.py` for its class (see docs/TESTING.md).
- Evaluation harness (frozen `eval/`) on the measured commit `3df56b7`: GATE FINAL PASS; dev, dev without numba, holdout
  and bench_final exact against `baseline-fixed`; determinism PASS; pipeline x11.41 [11.32-11.43] on `bench_final`, 1
  thread; x1.00 [0.995-1.012] against `opt/perf`. Details: docs/TESTING.md, section 6 (Release evidence).

## Earlier history

Based on rvt_py 2.2.3 (upstream, https://github.com/EarthObservation/RVT_py): master commit `bab109c` of 2026-08-05,
117 commits after the upstream tag `v2.2.3`. The upstream release notes are in `docs/releases_rvtpy.rst`.
