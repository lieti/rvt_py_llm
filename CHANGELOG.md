# Changelog

All notable changes of rvt_py_llm. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Details and evidence: [docs/CHANGES_VS_UPSTREAM.md](docs/CHANGES_VS_UPSTREAM.md).

## [0.2.0] - 2026-10-25

Opt-in disk windows for MSTP, SLRM and MSRM. With the default `kernel_shape="square"` every public function returns
the same bits as 0.1.1, with and without numba, for any thread count. Details, measurements and open questions:
[docs/DISK_WINDOWS.md](docs/DISK_WINDOWS.md).

### Added
- Keyword `kernel_shape` (`"square"`, `"disk"`, `"disk_equal_area"`), last parameter of `rvt.vis.mean_filter`,
  `max_elevation_deviation`, `mstp`, `slrm` and `msrm` (`202aa15`, `7468298`, `982d615`). Windows with
  r >= 8 px = 2 m at 0.25 m/px become exact discrete disks (dx² + dy² <= r_w², r_w = r or (2r + 1) / √π); smaller
  windows stay square; MSRM uses one shape for all its radii. Window sums by FFT (`scipy.fft`, float64,
  `workers=1`) of the mean-removed elevations; no numba, no threads on this path.
- `DefaultValues.slrm_kernel_shape`, `msrm_kernel_shape`, `mstp_kernel_shape` = `"square"`, json keys in the three
  groups (files without them load as `"square"`), txt names equal to the attribute names; output file names get
  `_disk` / `_diskea`; the tile overlap of SLRM and MSTP grows to floor(r_w) for `disk_equal_area` (`2ee3ecf`).
- 234 tests of the disk windows (`tests/test_disk_*.py`, brute-force float64 disk reference): 219 written before
  the feature, 15 added after a mutant run exposed an untested MSRM scale count (`685c920`); 4 frozen
  0.1.1 nodes that pinned the API (signatures, attribute set, settings keys) list the 0.2.0 additions. Tag
  `tests-frozen` moved to the 0.2.0 test set (1428 tests); the 0.1.1 freeze is tag `tests-frozen-0.1.1`.
- `astcheck.py` commit class `feat:`; mutants m29-m34 on the disk path; m13 scoped to `msrm`.
- `docs/DISK_WINDOWS.md`, KNOWN_ISSUES KI-77 (MSTP with square windows depends on the absolute elevation in flat
  terrain on large tiles; documented, not changed), `docs/release_evidence/0.2.0/`.

### Changed
- Nothing for default calls. Disk-path cost on the 10 000 x 8 000 px test DEMs, 1 thread: `get_mstp` 110-114 s
  against 16-19 s, peak 9.4 GB against 3.8 GB; `e3mstp` 150-153 s against 54-57 s; `e4mstp` 173-176 s against
  80-83 s. 2112 x 2112 px tile (0.25 m/px): MSTP 7.6-7.8 s against 0.96 s, peak 0.9 GB against 0.5 GB
  (`docs/release_evidence/0.2.0/cost2048.log`).

## [0.1.1] - 2026-10-24

Five fixes of the blend path. The outputs of CRIM, e3MSTP and e4MSTP change on purpose; every other public function
returns the same bits as 0.1.0, with and without numba. Details, evidence and open questions:
[docs/FIXES_0_1_1.md](docs/FIXES_0_1_1.md).

### Fixed
- X1 (KI-33, KI-34): `blend_overlay` and `blend_soft_light` return a new array instead of writing into the
  background, so the opacity of overlay and soft light layers applies (CRIM 50 %, e3MSTP 70 %, e4MSTP 90 %) and an
  RGB layer over a gray background no longer chains its bands (`c2456b1`). e3MSTP on the three 10 000 x 8 000 px
  test DEMs at 0.25 m/px: p99.9 16-31 DN, max 42 DN against 0.1.0.
- X2 (KI-47): e4MSTP computes its slope layer in degrees, as the 0..55 normalization expects (`33a2bd5`). e4MSTP
  (X1 and X2) on the same DEMs: p99.9 66-206 DN.
- X3 (KI-19, KI-20, KI-21): blend operands outside [0, 1] are clipped, NaN kept, instead of passing through
  `scale_0_to_1`; `scale_within_0_and_1`, `scale_strict_0_to_1` and `scale_0_to_1` use the actual minimum and
  maximum, keep NaN, leave the argument unchanged and no longer raise (`d09e45d`).
- X4 (KI-45): `color_relief_image_map` (and through it `e3mstp`) no longer sets `svf_r_max = 10` on the caller's
  `DefaultValues` (`c72646c`).
- X5 (KI-36): `BlenderLayer.check_data` raises for an opacity outside [0, 100] (`a37ab44`).

### Changed
- e3MSTP 0.87 times and e4MSTP 1.03 times the 0.1.0 time (medium 2048 x 2048 px window, 1 thread, median of 3).

### Added
- 101 tests written before the fixes (`tests/test_fix_0_1_1_*.py`); 66 frozen 0.1.0 tests that pinned the old
  behaviour now assert the fixed one, each marked `# 0.1.1: fixed, KI-nn`. Tag `tests-frozen` moved to the 0.1.1
  test set (1194 tests); the 0.1.0 freeze is tag `tests-frozen-0.1.0`.
- `astcheck.py` commit classes `fix:`, `retire:`, `tool:`; mutants m24-m28 on the fixed code.
- `docs/FIXES_0_1_1.md`, `docs/repro/fixes_0_1_1.py`, `docs/release_evidence/0.1.1/`.

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
- Tests: 1093 passed (1089 frozen + 4 added after the release audit), with accelerators and with `RVT_NO_ACCEL=1` (no test skipped).
- At the freeze: `equiv.py` (accel, t1, noaccel; 153/153 public functions executed) and `mutants.py` (21/21 mutants
  killed by the tests and by equiv) PASS. For each refactor commit: tests (accel and `RVT_NO_ACCEL=1`) and
  `equiv.py` PASS. Every commit of `start..main` passes `astcheck.py` for its class (see docs/TESTING.md).
- Evaluation harness (frozen `eval/`) on the measured commit `3df56b7`: GATE FINAL PASS; dev, dev without numba, holdout
  and bench_final exact against `baseline-fixed`; determinism PASS; pipeline x11.41 [11.32-11.43] on `bench_final`, 1
  thread; x1.00 [0.995-1.012] against `opt/perf`. Details: docs/TESTING.md, section 6 (Release evidence).

## Earlier history

Based on rvt_py 2.2.3 (upstream, https://github.com/EarthObservation/RVT_py): master commit `bab109c` of 2026-08-05,
117 commits after the upstream tag `v2.2.3`. The upstream release notes are in `docs/releases_rvtpy.rst`.
