# Changes compared to upstream rvt_py

rvt_py_llm 0.1.0 is the upstream Relief Visualization Toolbox Python library (rvt_py,
https://github.com/EarthObservation/RVT_py, by Žiga Kokalj et al.) at master commit `bab109c` (tag `baseline`,
`setup.py` version 2.2.3), plus:

1. one precision fix, commit `03702f4` (tag `baseline-fixed`);
2. eleven performance commits and one test commit (`21284eb` .. `8413d3e`, tag `start`; the branch `opt/perf` of
   the optimization run);
3. the 0.1.0 pass: documentation, characterization tests, exception message corrections, three behaviour-neutral
   readability refactors proven bitwise identical, and packaging (`start..main`).

The import name stays `rvt`; the public API (names, signatures, defaults, `DefaultValues` attributes, settings files,
return shapes and dtypes, exception types) is unchanged.

## What a user notices

- **Faster.** The eleven visualizations that a tiling pipeline computes run together 11.44 times faster (95 %
  interval 11.35-11.46) than the fixed baseline, measured on `bench_final` (three 2048 x 2048 px windows at
  0.25 m/px: two real DEMs, `medium` and `high`, and one synthetic window at +300 m; 1 thread per process, 5 rounds;
  Windows 11, Intel i9-14900K). Per visualization (medium window): local dominance 33.97 s -> 0.429 s (x79.2), MSTP
  12.11 s -> 0.892 s (x13.6), e4MSTP 67.27 s -> 4.136 s (x16.3), e3MSTP 23.17 s -> 2.891 s (x8.0), SVF + anisotropic
  SVF + openness 5.53 s -> 0.775 s (x7.1), negative openness x8.2, sky illumination x6.8, MSRM x4.8, SLRM x4.4; slope
  and hillshade are unchanged (x1.0). Peak memory added by a call is lower or equal (ratio 0.29-1.00).
  Source of these absolute times and ratios: the final gate of the optimization run on `opt/perf` (= tag `start`),
  2026-10-23 08:20, idle machine, report `bench_final_1t_20261023-082005_12380.json` in `out/gate_final/` of the
  optimization workspace (not in this repository). The release evidence of 0.1.0
  ([bench_final_1t_20261023-170737_32096.json](release_evidence/0.1.0/bench_final_1t_20261023-170737_32096.json))
  has the same ratios with absolute times about 2.4 times longer, e.g. local dominance 80.51 s -> 1.014 s (x79.4),
  pipeline x11.41 [11.32-11.43]; [TESTING.md](TESTING.md) section 6 explains the difference.
- **Same pictures.** Every output is bitwise identical to the fixed baseline (`baseline-fixed`) on all evaluation
  sets, with and without numba, for 1, 2 and the default number of threads.
- **MSTP, e3MSTP and e4MSTP differ from raw upstream**, because of the precision fix. Upstream squares the padded DEM
  in float32 before the integral images of the local variance E[x²] - E[x]²; at real elevations this loses most of
  the precision, so MSTP depended on the absolute elevation: the same terrain shifted by +17.3 m changed the 8-bit
  e3MSTP/e4MSTP output by up to 207-210 DN (99.9th percentile 150 DN) on a 512 x 512 px synthetic DEM at 300 m. With
  float64 squares the difference is at most 1 DN. All other visualizations are bitwise identical to raw upstream.
- **numba is optional.** With numba installed, four inner loops run compiled (five kernels: sky illumination has
  two); without it, or with
  `RVT_NO_ACCEL=1`, the numpy code runs and gives the same bits. numba costs about 0.2 s to import and 0.24-0.40 s to
  compile per kernel, once per process.
- **Documentation and tests.** Every public function has a numpy-style docstring with units, dtypes, edges and NaN
  handling; the repository has architecture, testing and known-issue documents; 1093 tests.
- **Exception texts.** 40 messages were corrected (module name `rvt.vis` instead of `rvt.visualization`, wrong
  function or bound named, typos); exception types are unchanged. The list is at the end of this file.

## Precision fix (tag `baseline-fixed`)

### 03702f4 mstp: compute the local variance from float64 squares
- What changed: `max_elevation_deviation` squares the padded DEM in float64 instead of its input dtype (float32)
  before building the integral image of squares.
- Why: see "What a user notices" above.
- Equivalence: not equivalent on purpose; changes MSTP, e3MSTP and e4MSTP outputs relative to `baseline`. Whether
  upstream adopts it is the upstream authors' decision.
- Measured gain: none intended.

## Performance commits (`baseline-fixed..start`)

Each commit was accepted on its own against the branch as it was then: exact on the evaluation sets `dev`,
`bench_quick`, `dev` without accelerators, and determinism; affected-target pipeline speed-up >= 1.10 with its 95 %
interval above 1 on `bench_quick` (1024 x 1024 px `medium` window, 5 rounds, 1 thread). "Incremental" is that
speed-up over the affected targets, against the previous commit. Commit messages hold the full equivalence argument.
Report paths are in the optimization workspace (`work/eval/...`), not in this repository; the final reports are
copied to `out/gate_final/` there.

### 21284eb K-01: MSTP box sums by slicing the integral images over the output window
- What: `max_elevation_deviation` evaluates the 12 rolled tables per radius as slices of the output window; pixel
  count (2r+1)² without NaN, int64 table otherwise; in-place cumsum and running maximum; the unused local
  `rad_max_out` was removed.
- Why equivalent: on the output window `np.roll` never wraps, so the rolled tables equal slices; the float64
  expression order ((b + a) - c - d, / n, - mean², abs, sqrt, + 1e-6, (z - mean) / std) is unchanged.
- Exact. numpy, no threads. Affected: mstp, e3mstp, e4mstp.
- Gain: incremental x1.63 [1.61-1.63] (mstp x8.02, e3mstp x2.17, e4mstp x1.26),
  `work/eval/bench_quick_1t_20261023-002717_28020.json`.

### 0ddb966 K-02: local_dominance by slices on the output window, masked float64 add into float32
- What: the shifted DEM is a slice of the edge-padded DEM; only the output window is computed, in row blocks;
  masked float64 add into the float32 accumulator instead of index gather/scatter; the original roll loop runs if a
  shift exceeds the padding.
- Why equivalent: every shift is at most `max_rad` = the padding width (checked at run time); per shift and pixel
  the arithmetic is unchanged (float32 comparison and difference, float64 division and weight, float64 add rounded
  to float32); the accumulator starts as `dem * 0` (keeps NaN and -0.0).
- Exact. numpy. Affected: local_dominance, e4mstp.
- Gain: x3.06 [3.01-3.08] (local_dominance x12.10, e4mstp x2.14); terrain `high` x2.66;
  `work/eval/bench_quick_1t_20261023-004625_24872.json`.

### b816b26 K-05: sky_illumination horizon search by slicing, trig on the unpadded extent
- What: the level-0 search runs on the unpadded extent (every level-0 shift is at most 20 px = 5 m at 0.25 m/px,
  the padding width); the 1 -> 0 spline is evaluated at unpadded points only; trigonometric terms once per direction
  on the unpadded extent.
- Why equivalent: slices equal the rolled arrays there; coarse levels keep `np.roll` semantics on their full extent
  because the spline reads all of them; same ufuncs, operand order and dtypes; nothing tile-wide is taken before
  the crop.
- Exact. numpy/scipy. Affected: sky_illumination.
- Gain: x3.03 [2.97-3.04]; `high` x3.09; `work/eval/bench_quick_1t_20261023-010342_25868.json`.

### 47f22ab K-06: mean_filter by slicing integral images
- What: the four `np.roll` shifts of the integral images are slices of the output window; constant pixel count
  without NaN.
- Why equivalent: with padding (r + 1, r) the shifts never wrap inside the kept window; same float64 order
  ((T1 + T2) - T3) - T4, same division, float32 cast and NaN restore.
- Exact. numpy. Affected: slrm, msrm, e3mstp.
- Gain: x1.13 [1.12-1.14] (msrm x4.67, slrm x4.42, e3mstp x1.01);
  `work/eval/bench_quick_1t_20261023-020331_33172.json`.

### 530ced9 K-03: sky_view_factor_compute horizon scan by slicing over the output window, in row blocks
- What: the reflect-padded DEM is read by slices on the output window, in contiguous row blocks of about 32768
  elements; the full-extent `np.roll` loop stays as the reference when the guard fails.
- Why equivalent: every shift from `horizon_shift_vector` is at most `radius_max` (checked at run time); per pixel
  the same float32 difference, float64 division, `np.fmax` from -1000, `np.arctan`, and the same accumulation per
  direction in the same order.
- Exact (bitwise equality of the blocked transcendentals verified on an AVX2 CPU). numpy. Affected: svf_asvf_opns,
  neg_opns, e3mstp, e4mstp.
- Gain: x2.61 [2.60-2.63] (svf x4.42, neg_opns x4.34, e3mstp x2.02, e4mstp x2.60); `high` x2.58;
  `work/eval/bench_quick_1t_20261023-021655_31124.json`.

### 379177c K-05b: sky_illumination level search and directional terms in cached row blocks, fit-free spline
- What: row blocks on every level (rows a multiple of 16); the floor at 0 applied once; for kx = ky = 1 on the
  integer knot grid the spline is evaluated with the same FITPACK `bisplev` without the fit.
- Why equivalent: pixels are independent; `max(ms, max(q1, 0), ...)` equals `max(max(ms, 0), q1, ...)` including
  NaN (only a zero's sign may differ, which no output sees); the fitted coefficients of that linear spline equal the
  data exactly; non-finite data or other knots take the original `RectBivariateSpline` path.
- Exact. numpy/scipy. Affected: sky_illumination.
- Gain: x1.58 [1.55-1.63]; `high` x1.59; `work/eval/bench_quick_1t_20261023-022943_3752.json`.

### 8e81581 K-11: blend/render path of e3mstp/e4mstp in cache-sized blocks, no fancy indexing, reused range tests
- What: `normalize_lin`, `blend_multi_dim_images`, `render_images`, `blend_overlay`, `blend_soft_light`,
  `blend_luminosity` + `clip_color` evaluate the same expressions on blocks of 32768 elements; overlay and soft
  light by masked copy, still in place; `render_all_images` reuses range tests on unchanged arrays.
- Why equivalent: elementwise expressions with the same operand dtypes and order; masks taken before any write;
  floating-point warnings reproduced; any other input runs the original code.
- Exact. numpy. Affected: e3mstp, e4mstp.
- Gain: x1.13 [1.10-1.14] (e3mstp x1.26, e4mstp x1.06); `work/eval/bench_quick_1t_20261023-035510_13668.json`.

### 0ddd8e0 K-08a: accel infrastructure (rvt/_accel.py) and numba local_dominance kernel
- What: `rvt/_accel.py`: lazy numba import, automatic fallback, `RVT_NO_ACCEL=1`, `RVT_NUM_THREADS` /
  `NUMBA_NUM_THREADS`; serial kernel for local dominance; no fastmath, no cache.
- Why equivalent: per pixel and shift the same operations in the same order and dtypes as the numpy path; NaN
  compares False and is skipped as `where=mask` does; no FMA contraction without fastmath.
- Exact (numba == numpy bitwise). numba, 1 thread. Affected: local_dominance, e4mstp.
- Gain: x1.58 [1.55-1.64] (local_dominance x6.14, e4mstp x1.31); `high` x1.60;
  `work/eval/bench_quick_1t_20261023-041009_28376.json`. One-off: import numba 0.22 s, compile 0.24 s.

### b8c9a9f K-08b: numba MSTP deviation kernel (max_elevation_deviation)
- What: serial kernel for the per-radius loop of the K-01 path.
- Why equivalent: same float64 box sums, mean, variance, abs, sqrt, + 1e-6 and (z - mean) / std in the same order;
  running maximum with the same NaN rule; inputs outside 0 < r_min <= r_max, step > 0, float32/float64 run numpy.
- Exact. Affected: mstp, e3mstp, e4mstp.
- Gain: x1.30 [1.29-1.31] (mstp x2.12, e3mstp x1.29, e4mstp x1.18);
  `work/eval/bench_quick_1t_20261023-042037_16132.json`. Compile 0.35 s.

### 87b49f6 K-08c: numba kernel for the SVF horizon search (max_slope per direction)
- What: the radius loop of `_svf_compute_window` runs in a serial kernel; `np.arctan`, `np.sin` and the
  accumulation stay in numpy.
- Why equivalent: same float32 difference, float64 division, `fmax` rule (NaN skipped, maximum never NaN); only
  the sign of a zero maximum on ties may differ, which every later operation maps to identical outputs.
- Exact. Affected: svf_asvf_opns, neg_opns, e3mstp, e4mstp.
- Gain: x1.43 [1.42-1.44] (svf x1.58, neg_opns x1.83, e3mstp x1.29, e4mstp x1.46); `high` x1.42;
  `work/eval/bench_quick_1t_20261023-043041_2132.json`. Compile 0.27 s.

### bd173ea K-08d: numba kernel for the sky_illumination level search
- What: two serial kernels replace the numpy row-block loop of `_sky_illum_search_blocked` (floor-once sequence,
  and the literal sequence of the shadow direction at level 0).
- Why equivalent: same operations per pixel and shift; `np.maximum` written out with numpy's NaN rule and its
  operand choice on +0/-0 ties (numpy 2.4); guards send other inputs to numpy before any write.
- Exact. Affected: sky_illumination.
- Gain: x1.38 [1.37-1.40]; `high` x1.39; `work/eval/bench_quick_1t_20261023-044042_26480.json`. Compile 0.40 s.

### 8413d3e tests: equivalence tests for the optimized code paths
- What: `tests/` with 223 tests (about 5 s): each optimized function against a transcription of the
  `baseline-fixed` code, MSTP offset invariance at +2,500 m, determinism, `RVT_NUM_THREADS` 1 against 2,
  accelerator fallbacks; pass with and without `RVT_NO_ACCEL=1`. No change under `rvt/`.

### Result of the series

Final gate of the optimization run on `start` (`out/gate_final/` of the optimization workspace): `GATE FINAL PASS`;
outputs bitwise identical to `baseline-fixed` on `dev`, `dev` without accelerators, `holdout` and `bench_final`;
determinism PASS; pipeline x11.44 [11.35-11.46] with 1 thread, x10.86 [9.86-11.27] with default threading
(informative).

Tried and rejected: slope/hillshade without pad/roll (x1.02 over the affected targets), the remaining sky
illumination passes (x1.07), one horizon scan for both openness signs (estimated x1.07), one integral-image set for
the three MSTP scales (approximate, estimated <= x1.08), float32 slope arithmetic in horizon scans (approximate).

## The 0.1.0 pass (`start..main`)

Commit classes (subject prefix) and what each class may change; each commit was checked by its class rule
(see [TESTING.md](TESTING.md)). Counts of `git log --format=%s start..de0f665` (`de0f665` is the release commit of
0.1.0); later commits (documentation rounds, proof-tool updates) are not counted here, `git log --format=%s
start..main` lists them:

| class | count | allowed content | check |
|---|---|---|---|
| `test:` | 4 | new test files only | frozen check |
| `docs:` | 22 (21 in the list below, plus 3001dc3) | docstrings, comments, blank lines | AST without docstrings identical |
| `msg:` | 2 | text of exception messages only | AST differs only in string constants inside `raise` |
| `refactor:` | 3 | behaviour-neutral code edits | full test suite + bitwise record comparison (`equiv.py`) PASS |
| `meta:` | 12 (e6b518f, plus the packaging and repository-document commits after the list below) | packaging, repository documents | no change under `rvt/` |

Commits up to the merge of the refactors. The packaging and document commits of this release (`meta:`, including the
one that adds this file) come after them, and so does one `docs:` commit, 3001dc3 (`docs: shadow_horizon and
get_sky_illumination state the measured NaN effect`, docstrings of `rvt/vis.py` and `rvt/default.py` only):

| commit | subject |
|---|---|
| e6b518f | `meta: add docs/CONVENTIONS.md (docstring and units standard)` |
| 48f2386 | `test: characterization tests for rvt.vis` |
| dcd3d17 | `test: characterization tests for rvt.blend and rvt.blend_func` |
| 77c245d | `test: characterization tests for rvt.default, rvt.tile and the public API snapshot` |
| c341ec8 | `test: reference tests for the e3MSTP and e4MSTP recipes` |
| 2956265 | `docs: module docstrings of rvt, rvt.vis and rvt._accel` |
| 9818ba9 | `docs: vis docstrings for byte_scale, slope/hillshade and mean filters` |
| 7b7067d | `docs: vis docstrings for the horizon search, SVF and local dominance` |
| 4639961 | `docs: vis docstrings for sky illumination and its pyramid helpers` |
| 05f4592 | `docs: vis docstrings for MSRM, MSTP, integral images and fill_where_nan` |
| eb22b37 | `docs: vis private helpers and numba kernels: equivalence contracts` |
| b0aa300 | `docs: vis: remove commented-out byte_scale code, correct comments` |
| 51a882a | `msg: vis exception texts name rvt.vis and state the actual checks` |
| dc2a8c2 | `docs: module docstrings of rvt.blend and rvt.blend_func` |
| 624cea1 | `docs: numpy docstrings for every public function of rvt.blend_func` |
| ec44a2d | `docs: docstrings of BlenderLayer, BlenderCombination, compare_2_combinations` |
| c2f8269 | `docs: docstrings of BlenderCombinations and the terrain preset classes` |
| a9efeb7 | `docs: CRIM, e3MSTP and e4MSTP recipes as the code implements them` |
| f1ef4ce | `docs: private helpers of rvt.blend_func and their equivalence contracts` |
| 932172f | `docs: remove commented-out code in rvt.blend and rvt.blend_func` |
| 66a4709 | `msg: gray_scale_to_color_ramp cut-order message states the check` |
| 96b8d15 | `docs: tile.py module and function docstrings, overlap table` |
| 9d9c63e | `docs: default.py module, RVTVisualization and DefaultValues docstrings` |
| c220535 | `docs: default.py settings file and output file-name docstrings` |
| b2333d2 | `docs: default.py get_*, save_*, float_to_8bit, dispatch, log, raster I/O` |
| 98450e1 | `docs: Structure block in save_sky_view_factor, why-comments in tile loops` |
| 6453efd | `docs: correct the byte no-data note, name the in-place dict filling` |
| 4a2144c | `docs: merge the local dominance radii into one DefaultValues entry` |
| 69525a0 | `refactor: define the 32768-element block size once in rvt/_blocking.py` |
| 2e1a306 | `refactor: merge _svf_accumulate and _sky_illum_iadd into _accumulate` |
| 8839afd | `refactor: one module-level _box_sum for mean_filter and MSTP` |

### Refactors

The three `refactor:` commits change private code only; each was checked with the full test suite (1089 tests, with
accelerators and with `RVT_NO_ACCEL=1`) and with `equiv.py` (`EQUIV PASS` in the variants accel, t1 and noaccel;
153 of 153 public functions executed). Timing at 1024 x 1024 px, 1 thread, best of 5, stayed within about 2 % of
`start` (noise), e.g. SVF 0.240 s -> 0.238 s, sky illumination 0.862 s -> 0.866 s; these are interim numbers of the
refactor worker, not a benchmark of the evaluation harness.

- 69525a0: the 32768-element block size is defined once, as `BLOCK_ELEMS` in the new private module
  `rvt/_blocking.py`, and imported as `_BLOCK_ELEMS` by `rvt/vis.py` and `rvt/blend_func.py`. It is a module of its
  own because `rvt.blend_func` needs the constant but does not use the numba module `rvt._accel`.
- 2e1a306: `_svf_accumulate` and `_sky_illum_iadd` are merged into `rvt.vis._accumulate`.
- 8839afd: one module-level `rvt.vis._box_sum` is used by `mean_filter` and `max_elevation_deviation`; it replaces
  the nested `mean_filter._box_sum` and `_mstp_box_sum`.

Considered and declined: one shift-array builder for all horizon scans (`local_dominance` builds two arrays, the
others three); a simpler `normalize_lin` (merging its loops changes the in-place clipping when a
FloatingPointError is raised); removal of dead private code (every private function has a caller).

### Exception messages changed (`msg:` commits 51a882a and 66a4709)

Verbatim old and new text of every changed message, with the line of the `raise` at tag `start`. Exception types
are unchanged; no warning text changed. 39 of the 41 messages raised in `rvt/vis.py` and 1 of the 9 in
`rvt/blend_func.py` changed; no message in `rvt/blend.py`, `rvt/default.py`, `rvt/tile.py` or `rvt/_accel.py` changed.

| raise at `start` | function | old text | new text |
|---|---|---|---|
| `rvt/vis.py:159` | slope_aspect | `rvt.visualization.slope_aspect: dem has to be 2D np.array!` | `rvt.vis.slope_aspect: dem has to be 2D np.array!` |
| `rvt/vis.py:161` | slope_aspect | `rvt.visualization.slope_aspect: ve_factor must be between -10000 and 10000!` | `rvt.vis.slope_aspect: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:163` | slope_aspect | `rvt.visualization.slope_aspect: resolution must be a positive number!` | `rvt.vis.slope_aspect: resolution must be a positive number!` |
| `rvt/vis.py:194` | slope_aspect | `rvt.visualization.calculate_slope: Wrong function input 'output_units'!` | `rvt.vis.slope_aspect: Wrong function input 'output_units'!` |
| `rvt/vis.py:272` | hillshade | `rvt.visualization.hillshade: dem has to be 2D np.array!` | `rvt.vis.hillshade: dem has to be 2D np.array!` |
| `rvt/vis.py:274` | hillshade | `rvt.visualization.hillshade: ve_factor must be between -10000 and 10000!` | `rvt.vis.hillshade: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:276` | hillshade | `rvt.visualization.hillshade: sun_azimuth must be [0-360] and sun_elevation [0-90]!` | `rvt.vis.hillshade: sun_azimuth must be [0-360] and sun_elevation [0-90]!` |
| `rvt/vis.py:278` | hillshade | `rvt.visualization.hillshade: resolution must be a positive number!` | `rvt.vis.hillshade: resolution must be a positive number!` |
| `rvt/vis.py:356` | multi_hillshade | `rvt.visualization.multi_hillshade: dem has to be 2D np.array!` | `rvt.vis.multi_hillshade: dem has to be 2D np.array!` |
| `rvt/vis.py:358` | multi_hillshade | `rvt.visualization.multi_hillshade: sun_elevation must be [0-90]!` | `rvt.vis.multi_hillshade: sun_elevation must be [0-90]!` |
| `rvt/vis.py:360` | multi_hillshade | `rvt.visualization.multi_hillshade: resolution must be a positive number!` | `rvt.vis.multi_hillshade: resolution must be a positive number!` |
| `rvt/vis.py:362` | multi_hillshade | `rvt.visualization.multi_hillshade: nr_directions must be a positive number!` | `rvt.vis.multi_hillshade: nr_directions must be a positive number!` |
| `rvt/vis.py:364` | multi_hillshade | `rvt.visualization.multi_hillshade: ve_factor must be between -10000 and 10000!` | `rvt.vis.multi_hillshade: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:476` | slrm | `rvt.visualization.slrm: dem has to be 2D np.array!` | `rvt.vis.slrm: dem has to be 2D np.array!` |
| `rvt/vis.py:478` | slrm | `rvt.visualization.slrm: Radius for trend assessment needs to be in interval 10-50 pixels!` | `rvt.vis.slrm: Radius for trend assessment needs to be in interval 10-50 pixels!` |
| `rvt/vis.py:480` | slrm | `rvt.visualization.slrm: ve_factor must be between -10000 and 10000!` | `rvt.vis.slrm: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:942` | sky_view_factor | `rvt.visualization.sky_view_factor: dem has to be 2D np.array!` | `rvt.vis.sky_view_factor: dem has to be 2D np.array!` |
| `rvt/vis.py:944` | sky_view_factor | `rvt.visualization.sky_view_factor: ve_factor must be between -10000 and 10000!` | `rvt.vis.sky_view_factor: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:946` | sky_view_factor | `rvt.visualization.sky_view_factor: svf_noise must be one of the followingvalues (0-don't remove, 1-low, 2-med, 3-high)!` | `rvt.vis.sky_view_factor: svf_noise must be one of the following values (0-don't remove, 1-low, 2-med, 3-high)!` |
| `rvt/vis.py:949` | sky_view_factor | `rvt.visualization.sky_view_factor: asvf_leve must be one of the followingvalues (1-low, 2-high)!` | `rvt.vis.sky_view_factor: asvf_level must be one of the following values (1-low, 2-high)!` |
| `rvt/vis.py:952` | sky_view_factor | `rvt.visualization.sky_view_factor: All computes are false!` | `rvt.vis.sky_view_factor: All computes are false!` |
| `rvt/vis.py:954` | sky_view_factor | `rvt.visualization.sky_view_factor: resolution must be a positive number!` | `rvt.vis.sky_view_factor: resolution must be a positive number!` |
| `rvt/vis.py:1068` | local_dominance | `rvt.visualization.local_dominance: dem has to be 2D np.array!` | `rvt.vis.local_dominance: dem has to be 2D np.array!` |
| `rvt/vis.py:1070` | local_dominance | `rvt.visualization.local_dominance: ve_factor must be between -10000 and 10000!` | `rvt.vis.local_dominance: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:1540` | sky_illumination | `rvt.visualization.sky_illumination: ve_factor must be between -10000 and 10000!` | `rvt.vis.sky_illumination: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:1542` | sky_illumination | `rvt.visualization.sky_illumination: shadow_az must be between 0 and 360!` | `rvt.vis.sky_illumination: shadow_az must be between 0 and 360!` |
| `rvt/vis.py:1544` | sky_illumination | `rvt.visualization.sky_illumination: shadow_el must be between 0 and 90!` | `rvt.vis.sky_illumination: shadow_el must be between 0 and 90!` |
| `rvt/vis.py:1546` | sky_illumination | `rvt.visualization.sky_illumination: resolution must be a positive number!` | `rvt.vis.sky_illumination: resolution must be a positive number!` |
| `rvt/vis.py:1562` | sky_illumination | `rvt.visualization.sky_illumination: sky_model must be overcast or uniform!` | `rvt.vis.sky_illumination: sky_model must be overcast or uniform!` |
| `rvt/vis.py:1877` | shadow_horizon | `rvt.visualization.shadow_horizon: ve_factor must be between -10000 and 10000!` | `rvt.vis.shadow_horizon: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:1879` | shadow_horizon | `rvt.visualization.shadow_horizon: shadow_az must be between 0 and 360!` | `rvt.vis.shadow_horizon: shadow_az must be between 0 and 360!` |
| `rvt/vis.py:1881` | shadow_horizon | `rvt.visualization.shadow_horizon: shadow_el must be between 0 and 90!` | `rvt.vis.shadow_horizon: shadow_el must be between 0 and 90!` |
| `rvt/vis.py:1883` | shadow_horizon | `rvt.visualization.shadow_horizon: resolution must be a positive number!` | `rvt.vis.shadow_horizon: resolution must be a positive number!` |
| `rvt/vis.py:1925` | msrm | `rvt.visualization.msrm: ve_factor must be between -10000 and 10000!` | `rvt.vis.msrm: ve_factor must be between -10000 and 10000!` |
| `rvt/vis.py:1927` | msrm | `rvt.visualization.msrm: resolution must be a positive number!` | `rvt.vis.msrm: resolution must be a positive number!` |
| `rvt/vis.py:2258` | mstp | `rvt.visualization.mstp: local_scale, meso_scale, broad_scale min has to be smaller than max!` | `rvt.vis.mstp: local_scale, meso_scale, broad_scale min has to be smaller than max!` |
| `rvt/vis.py:2261` | mstp | `rvt.visualization.mstp: local_scale, meso_scale, broad_scale step has to be within min and max!` | `rvt.vis.mstp: local_scale, meso_scale, broad_scale step has to be within min and max!` |
| `rvt/vis.py:2264` | mstp | `rvt.visualization.mstp: ve_factor must be between -10000 and 10000!` | `rvt.vis.mstp: ve_factor must be between -1000 and 10000!` |
| `rvt/vis.py:2383` | fill_where_nan | `rvt.visualization.fill_where_nan: Wrong method!` | `rvt.vis.fill_where_nan: Wrong method!` |
| `rvt/blend_func.py:107` | gray_scale_to_color_ramp | `rvt.blend_func.gray_scale_to_color_ramp: min_colormap_cut can't be smaller than max_colormap_cut!` | `rvt.blend_func.gray_scale_to_color_ramp: min_colormap_cut has to be smaller than max_colormap_cut!` |
