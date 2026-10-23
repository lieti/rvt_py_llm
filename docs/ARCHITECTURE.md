# Architecture of rvt (rvt_py_llm 0.1.0)

This document describes how the package `rvt` is organised, which functions do the work behind each public entry
point, how data flows through a visualization, and how the optional accelerated paths are built. It is written for
a developer who knows relief visualization but not this code, and for an LLM session that has to change the code
without breaking it. Units and conventions are in [CONVENTIONS.md](CONVENTIONS.md); latent bugs in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md); the tests and proofs in [TESTING.md](TESTING.md).

`file:line` references are at tag `start` (`8413d3e`), the optimized code before the documentation pass. In 0.1.0
the same functions carry longer docstrings, so their line numbers are larger; the function names are the stable
reference. Three behaviour-neutral refactor commits of 0.1.0 renamed or merged private helpers; where a name
differs between `start` and 0.1.0, both are given ("`start` name -> 0.1.0 name"):

- 69525a0: the block size 32768 is defined once, as `BLOCK_ELEMS` in the new private module `rvt/_blocking.py`, and
  imported as `_BLOCK_ELEMS` by `rvt/vis.py` and `rvt/blend_func.py`. It is a module of its own because
  `rvt.blend_func` needs the constant but does not use the numba module `rvt._accel`;
- 2e1a306: `_svf_accumulate` and `_sky_illum_iadd` (`vis.py:583`, `:1297`) are merged into `rvt.vis._accumulate`;
- 8839afd: one module-level `rvt.vis._box_sum` is used by `mean_filter` and `max_elevation_deviation`; it replaces
  the nested `mean_filter._box_sum` and `_mstp_box_sum` (`vis.py:2216`).

## 1. Module map

| module | responsibility | imports | public entry points |
|---|---|---|---|
| `rvt/__init__.py` | package docstring only; importing `rvt` imports no submodule | none | none |
| `rvt/vis.py` | the numerical algorithms on 2D numpy arrays; no file I/O | numpy, scipy, `rvt._accel`, `rvt._blocking` | `slope_aspect`, `hillshade`, `multi_hillshade`, `mean_filter`, `slrm`, `msrm`, `sky_view_factor`, `sky_view_factor_compute`, `local_dominance`, `sky_illumination`, `shadow_horizon`, `mstp`, `max_elevation_deviation`, `integral_image`, `topographic_dev`, `byte_scale`, `fill_where_nan`, pyramid helpers |
| `rvt/default.py` | `DefaultValues`: the default parameters of every visualization, settings files (json, legacy txt), `get_<vis>` (compute from an array), `save_<vis>` (read a GeoTIFF, compute, write GeoTIFFs), 8-bit stretch (`float_to_8bit`), log file; GDAL raster helpers; `RVTVisualization` enum | `rvt.vis`, `rvt.blend_func`, `rvt.tile`, osgeo.gdal, numpy | `DefaultValues`, `RVTVisualization`, `get_raster_arr`, `get_raster_size`, `save_raster` |
| `rvt/blend_func.py` | per-array blending operations: normalization, blend modes, opacity, color ramps | numpy, matplotlib, `rvt._blocking` | `normalize_image`, `blend_images`, `render_images`, `gray_scale_to_color_ramp`, ... |
| `rvt/blend.py` | the layer model (`BlenderLayer`, `BlenderCombination`, `BlenderCombinations`), JSON combination and terrain files, the composite recipes `color_relief_image_map`, `e3mstp`, `e4mstp` | `rvt.default`, `rvt.vis`, `rvt.blend_func` | `BlenderCombination.render_all_images`, `e3mstp`, `e4mstp`, `color_relief_image_map`, `TerrainSettings`, ... |
| `rvt/tile.py` | tile-by-tile processing of rasters larger than memory, with an overlap per visualization | `rvt.default`, osgeo.gdal, numpy | `save_visualization_tile_by_tile`, `save_rvt_visualization_tile_by_tile` |
| `rvt/_accel.py` | private: optional numba kernels with a numpy fallback, the `RVT_NO_ACCEL` switch and thread-count control | `threading`, `os`; numba lazily | none (private) |
| `rvt/_blocking.py` | private, new in 0.1.0 (69525a0): `BLOCK_ELEMS = 32768`, the block size of every row-blocked loop, and why | none | none (private) |

Dependency direction: `vis` depends only on numpy, scipy, `_accel` and `_blocking`; `blend_func` only on numpy,
matplotlib and `_blocking`;
`default` on `vis`, `blend_func`, `tile`; `tile` on `default` (imported at module level, `rvt/tile.py`); `blend` on
`default`, `vis`, `blend_func`. Nothing imports `blend`.

The settings files used by `rvt.default` and `rvt.blend` live in `settings/` at the repository root, not inside the
package.

## 2. Call graph of the evaluated entry points

The optimization run measured eleven entry points, each called as a tile pipeline calls it: a `DefaultValues()`
instance with its defaults, or the composite function (`eval/targets.py` of the optimization workspace). The
functions that hold the time are named; `[numba]` marks a kernel run through `rvt._accel.call`, with the numpy
code next to it as the fallback.

```
DefaultValues.get_slope                 default.py:1206
  vis.slope_aspect                      vis.py:118     edge pad 1 px, central differences

DefaultValues.get_hillshade             default.py:1290
  vis.hillshade                         vis.py:232
    vis.slope_aspect                    vis.py:299 (call)

DefaultValues.get_slrm                  default.py:1487
  vis.slrm                              vis.py:451
    vis.mean_filter                     vis.py:391     integral images, box sums by slices (K-06);
                                                       0.1.0: box sums via vis._box_sum (8839afd)

DefaultValues.get_msrm                  default.py:1979
  vis.msrm                              vis.py:1890
    vis.mean_filter (per scale)         vis.py:1954 (call)

DefaultValues.get_sky_view_factor       default.py:1563   (target svf_asvf_opns: svf, asvf, opns)
DefaultValues.get_neg_opns              default.py:1727   (same path on -1 * dem)
  vis.sky_view_factor                   vis.py:887     dem / resolution, ve_factor
    vis.sky_view_factor_compute         vis.py:750     reflect pad r_max
      vis._svf_compute_window           vis.py:636     output window, row blocks (K-03)
        vis._svf_max_slope_kernel       vis.py:591     [numba] radius loop (K-08c); arctan/sin stay numpy
      (reference: full-extent np.roll loop in sky_view_factor_compute, used when _svf_window_ok is False)

DefaultValues.get_local_dominance       default.py:1898
  vis.local_dominance                   vis.py:1030    edge pad max_rad, slices, row blocks (K-02)
    vis._local_dominance_kernel         vis.py:1007    [numba] (K-08a)

DefaultValues.get_sky_illumination      default.py:1811
  vis.sky_illumination                  vis.py:1490
    vis.slope_aspect                    vis.py:1565 (call)
    vis.horizon_generate_pyramids       vis.py:1218 -> horizon_generate_coarse_dem vis.py:1156
    vis._sky_illum_search_blocked       vis.py:1416    level search in row blocks (K-05, K-05b)
      vis._sky_illum_search_accel       vis.py:1395
        _sky_illum_search_kernel        vis.py:1356    [numba] (K-08d)
        _sky_illum_search_literal_kernel vis.py:1377   [numba] shadow direction, level 0
    vis._sky_illum_level_search         vis.py:1304    numpy level search (non-sliced cases)
    vis._sky_illum_resample             vis.py:1460    linear spline without refit (K-05b)

DefaultValues.get_mstp                  default.py:2060
  vis.mstp                              vis.py:2224    three scales
    vis.max_elevation_deviation         vis.py:2097    symmetric pad, float64 integral images (x3, vis.py:2273-2277)
      vis.integral_image                vis.py:1965
      vis._mstp_dev_max_kernel          vis.py:2057    [numba] (K-08b)
      vis._mstp_box_sum                 vis.py:2216    numpy path (K-01); 0.1.0: vis._box_sum (8839afd)

blend.e3mstp                            blend.py:1339
  DefaultValues.get_slrm                blend.py:1365
  blend.color_relief_image_map          blend.py:1366 -> blend.py:1266
    DefaultValues.get_sky_view_factor   blend.py:1308 (opns of dem), blend.py:1311 (opns of -dem)
    vis.slope_aspect                    blend.py:1316
    BlenderCombination.render_all_images blend.py:1334
  DefaultValues.get_mstp                blend.py:1368
  BlenderCombination.render_all_images  blend.py:1383

blend.e4mstp                            blend.py:1387
  DefaultValues.get_local_dominance     blend.py:1429
  DefaultValues.get_sky_view_factor     blend.py:1430 (svf + opns, r = 10 px)
  DefaultValues.get_neg_opns            blend.py:1433
  DefaultValues.get_sky_view_factor     blend.py:1439 (flat-terrain svf, r = int(10 / resolution) px = 10 m)
  DefaultValues.get_mstp                blend.py:1442
  vis.slope_aspect                      blend.py:1487
  BlenderCombination.render_all_images  blend.py:1458, 1479, 1519 (three combinations)

BlenderCombination.render_all_images    blend.py:340
  blend_func.normalize_image            per layer (blend.py:411-588)
  blend_func.gray_scale_to_color_ramp   blend.py:595 (layers with a colormap)
  blend_func._outside_0_1, scale_0_to_1 blend.py:611-617 (range tests reused, K-11)
  blend_func.blend_images               blend.py:621
  blend_func._render_images             blend.py:626   opacity, row blocks of _BLOCK_ELEMS (K-11)
```

Where the time goes after the optimization (1024² medium window, 1 thread, sum about 3.9 s over the eleven entry
points; optimization report §4): sky illumination 1.33 s (numpy arctan/cos/sin/power per direction 57 %, numba search
24 %, `bisplev` 13 %); e4MSTP 1.01 s; e3MSTP 0.71 s; `_svf_compute_window` 1.05 s summed over four entry points.

## 3. Data flow through a visualization

The common path of a `rvt.vis` function, with the variations that matter:

1. **Input**: a 2D `numpy.ndarray` of elevations in m, row 0 north. `DefaultValues.get_*` passes it on unchanged;
   `save_*` reads it with `get_raster_arr` (`default.py:2540`), which returns the file's data type.
2. **No-data**: pixels equal to `no_data` become NaN. Most functions do this in the caller's array *before* the
   dtype conversion (`vis.py:282`, `:368`, `:484`, `:1074`, `:1550`, `:1931`, `:2268`), so the caller's array changes
   and an integer array raises (KNOWN_ISSUES KI-17). `slope_aspect` (`vis.py:166-170`) and `sky_view_factor`
   (`vis.py:957-969`) convert first.
3. **dtype**: `dem.astype(np.float32)` (e.g. `vis.py:166`, `:284`, `:486`, `:957`, `:1076`, `:1552`, `:1933`,
   `:2270`), then `* ve_factor`. `sky_view_factor` also divides by `resolution` (`vis.py:976`); `sky_illumination`
   does not (KI-06).
4. **Padding**: each function pads with its own mode and width, so that the neighbourhood of an edge pixel exists:
   edge 1 px (`slope_aspect`, `hillshade`, `vis.py:176`, `:286`); edge `(r + 1, r)` (`mean_filter`, `vis.py:405`);
   reflect `r_max` (`sky_view_factor_compute`, `vis.py:807`); edge `max_rad` (`local_dominance`, `vis.py:1079`);
   symmetric, constant (`dem.min()`) and edge in the sky illumination pyramid (`vis.py:1188-1269`); symmetric
   `(r_max + 1, r_max)` (`max_elevation_deviation`, `vis.py:2125`).
5. **Algorithm** on the padded array; the optimized paths compute only the output window (section 4).
6. **Output**: the padding is cropped; NaN is restored at input NaN pixels where the function does so. Output dtypes
   differ per function, as asserted by the frozen tests: `slope_aspect` float32 (`tests/test_char_vis.py:220`),
   `hillshade` float64 (`:388`), `mean_filter` and `slrm` float32 (`:513`, `:549`), `msrm` float64 (`:598`),
   `sky_view_factor_compute` float64 (`:695`), `local_dominance` float64 (`:806`), `sky_illumination` float64
   (`:1043`), `max_elevation_deviation` float32 (`:1230`), `mstp` float32 `(3, H, W)` (`:1255`). The composites
   `color_relief_image_map`, `e3mstp`, `e4mstp` return float64 `(3, H, W)` in [0, 1].
7. **8-bit**: `DefaultValues.float_to_8bit` (`default.py:1086`) stretches with the `<vis>_bytscl` setting through
   `rvt.blend_func.normalize_image` and `rvt.vis.byte_scale`; it clips its argument in place (KI-60).

## 4. Accelerated paths

The optimization (commits K-01 to K-11, see [CHANGES_VS_UPSTREAM.md](CHANGES_VS_UPSTREAM.md)) changed how five
functions of `rvt.vis` and the blend kernels of `rvt.blend_func` compute, never what they compute. The design
rules, which every future change has to keep:

- **Output window by slicing.** The upstream code shifted the whole padded array with `np.roll` once per direction
  and radius. Every shift is at most the padding width, so on the output window `np.roll(a, (dy, dx))[i, j]` equals
  `a_padded[p + i - dy, p + j - dx]` and never a wrapped value. The optimized code reads these slices instead of
  building rolled copies, and computes only the output window. Where this is not guaranteed by construction, a guard
  checks it at run time and otherwise runs the original full-extent loop, which is kept in the code as the reference:
  `_svf_window_ok` (`vis.py:565`), the shift range check of `local_dominance`, the level-0 check of
  `sky_illumination`, the radius and dtype guards of `max_elevation_deviation`.
- **Same IEEE operations.** Per element, the optimized code performs the same operations in the same order with the
  same operand dtypes as the upstream code: float32 differences, float64 division by the `np.float64` radius,
  float64 integral images and squares, the same accumulation order over directions. This is why the outputs are
  bitwise identical, not only close. Each private helper and kernel states its equivalence contract in its
  docstring.
- **Blocking for the cache.** SVF (`_SVF_BLOCK_ELEMS = 32768`, `vis.py:562`), local dominance (`vis.py:1116`),
  sky illumination (`_sky_illum_block_rows`, `vis.py:1343`) and the blend kernels (`_BLOCK_ELEMS = 32768`,
  `blend_func.py:33`) work in row blocks of about 32768 elements, i.e. 256 KiB of float64 per buffer, so the
  temporaries of one block stay in the CPU cache. In 0.1.0 the four literals are one constant,
  `rvt._blocking.BLOCK_ELEMS` (69525a0). The block size affects speed only: every blocked operation is elementwise
  or per pixel. Sky illumination uses a multiple of 16 rows per block so that the
  positions of elements in SIMD bodies and tails match the whole-array operation (`vis.py:1343-1346`).
- **numba kernels via `_accel`.** Four inner loops have a numba kernel: the SVF radius loop
  (`_svf_max_slope_kernel`), local dominance (`_local_dominance_kernel`), the sky illumination level search
  (`_sky_illum_search_kernel`, `_sky_illum_search_literal_kernel`) and the MSTP deviation loop
  (`_mstp_dev_max_kernel`). A kernel is declared with `@_accel.jit_kernel` and run with `_accel.call(kernel, ...)`,
  which returns False when the caller must run its numpy path (`_accel.py:105` at `start`).
- **Fallback contract.** numba is imported lazily on the first accelerated call, never at `import rvt`
  (`_accel.py:42`). `call` returns False when `RVT_NO_ACCEL` is set (any value except '', '0', 'false', 'no', 'off',
  `_accel.py:39`), when numba is missing or fails to import, or when a kernel fails to compile or run; a kernel that
  failed once stays disabled for the process. The callers either re-initialise their outputs after False or guard
  their inputs so that only typing and compilation, which happen before any write, can fail.
- **Bitwise contract of the kernels.** Kernels are compiled with `numba.njit(nogil=True)` (`_accel.py:99`): no
  fastmath, no FMA contraction, no reassociation, default error model, no on-disk cache. NaN rules are written out
  explicitly where numpy's `fmax`/`maximum` define them. Accelerated and numpy paths are bitwise equal; the tests
  compare them (`tests/test_accel.py`, the per-function test files).
- **No threads in kernels.** All kernels are serial (`parallel=False`), so results do not depend on the thread
  count. `_accel.num_threads` (`_accel.py:58`) reads `RVT_NUM_THREADS`, then `NUMBA_NUM_THREADS`, and is applied only
  to kernels declared `parallel=True` (none). The numpy work uses no BLAS. A pipeline that runs one tile per process
  can set all thread variables to 1 without losing speed.
- **Transcendentals stay in numpy.** arctan, sin, cos, power run in numpy in every path, because a different
  implementation would not be bitwise identical. Bitwise equality of the blocked numpy transcendentals was verified on
  one AVX2 CPU only.

One-off cost of numba per process (fresh process, 1 thread, Windows 11, i9-14900K): `import numba` 0.14-0.22 s;
compilation per kernel: local dominance 0.24 s, SVF 0.25-0.27 s, MSTP 0.35-0.37 s, sky illumination 0.39-0.40 s.
For a process that computes one cheap visualization on a small tile, `RVT_NO_ACCEL=1` can be faster overall.

## 5. Tests

| file | kind | covers |
|---|---|---|
| `tests/conftest.py` | shared setup | repository root first on `sys.path` (asserts that `rvt` comes from this repository), seeded synthetic DEMs, bitwise asserts, accel/numpy switches |
| `tests/test_accel.py` | equivalence (from `start`) | `rvt._accel`: lazy import, `RVT_NO_ACCEL`, thread variables, fallback on missing numba and on compile/run failure |
| `tests/test_mean_filter.py` | equivalence | `mean_filter`, `slrm`, `msrm` against a transcription of the `baseline-fixed` np.roll code |
| `tests/test_sky_view_factor.py` | equivalence | `sky_view_factor_compute`, `sky_view_factor`, SVF kernel |
| `tests/test_local_dominance.py` | equivalence | `local_dominance`, its kernel |
| `tests/test_sky_illumination.py` | equivalence | `sky_illumination`, `shadow_horizon`, both search kernels, `_sky_illum_resample` against `RectBivariateSpline` |
| `tests/test_mstp.py` | equivalence | `max_elevation_deviation`, `mstp`, MSTP kernel, offset invariance at +2,500 m |
| `tests/test_blend.py` | equivalence | blocked blend kernels and the `render_all_images` layer loop against `baseline-fixed` |
| `tests/test_char_vis.py` | characterization (0.1.0) | every public function of `rvt.vis`: dtype, shape, NaN mask, edges, parameter checks, in-place effects, analytic values |
| `tests/test_char_blend_func.py` | characterization | every public function of `rvt.blend_func` not covered by `test_blend.py` |
| `tests/test_char_blend.py` | characterization | classes and functions of `rvt.blend`, JSON files, CRIM/e3/e4 shape and NaN mask |
| `tests/test_char_e3e4_recipe.py` | characterization | the e3MSTP and e4MSTP recipes rebuilt with plain code (layer by layer) |
| `tests/test_char_default.py` | characterization | `DefaultValues`: defaults, settings files, file names, `get_*` wiring, `float_to_8bit`, `save_*` on small GeoTIFFs |
| `tests/test_char_tile.py` | characterization | both tiling functions on a 120 x 130 px GeoTIFF in 50 x 50 px tiles |
| `tests/test_char_accel.py` | characterization | `rvt._accel` registry structure, return types, runtime failures |
| `tests/test_api_snapshot.py` | API snapshot | signatures, defaults, `DefaultValues` attributes, `RVTVisualization` members |

The suite has 1089 tests and runs in about 10 s on one process; every file listed above is frozen at tag
`tests-frozen` (see [TESTING.md](TESTING.md)).

## 6. If you need to change X, look at Y

| you need to change | look at | keep in mind |
|---|---|---|
| a default parameter | `DefaultValues.__init__` (`default.py:55` ff.), `save_default_to_file`, `read_default_from_file`, `settings/default_settings.json` | the API snapshot test fails on purpose; file names encode parameters (`get_*_file_name`) |
| a visualization algorithm | the function in `rvt/vis.py`, its private helpers and kernel | if it has an accelerated path, the numpy reference path, the sliced path and the kernel must stay bitwise equal; the per-function equivalence test compares them |
| a numba kernel | the kernel docstring (equivalence contract), `rvt/_accel.py` | no fastmath, no parallel, no on-disk cache; keep the guard that sends unsupported inputs to numpy; run with and without `RVT_NO_ACCEL=1` |
| the numba switch or thread count | `rvt/_accel.py` (`enabled`, `num_threads`, `call`) | `RVT_NO_ACCEL` values and `RVT_NUM_THREADS` order are tested in `tests/test_accel.py` and `tests/test_char_accel.py` |
| the block size of the blocked loops | `rvt/_blocking.py` (`BLOCK_ELEMS`) | speed only; sky illumination also rounds its block to a multiple of 16 rows (`_sky_illum_block_rows`) |
| a blend mode or normalization | `rvt/blend_func.py` (blocked path and its `_ref` function) | blocked and reference paths must give the same bits and the same in-place effects and warnings |
| a composite recipe (CRIM, e3MSTP, e4MSTP) | `rvt/blend.py` functions of that name | `tests/test_char_e3e4_recipe.py` rebuilds the recipes layer by layer |
| layer rendering | `BlenderCombination.render_all_images` (`blend.py:340`) | range tests are reused between layers (K-11); overlay and soft light write into the background (KI-33) |
| reading or writing rasters | `get_raster_arr`, `save_raster` (`default.py:2540`, `:2602`), `rvt/tile.py` | GDAL only inside `rvt.default` and `rvt.tile` |
| tiling overlaps | `_get_rvt_visualization_overlap` (`tile.py:263`) | measured minimum buffers are in KNOWN_ISSUES, section "Tiling facts" |
| an exception message | the `raise` line | only in a `msg:` commit; exception types must not change |
| behaviour that looks wrong | KNOWN_ISSUES.md and the docstring Notes | do not fix it in this repository; document it and pin it with a test |

## 7. Why some numerics look the way they do

- float64 squares and integral images in MSTP: the variance E[x²] - E[x]² loses its precision in float32 at real
  elevations; the precision fix (commit 03702f4) squares in float64. Upstream squared in float32, so MSTP depended on
  the absolute elevation (up to 207-210 DN in 8-bit e3MSTP/e4MSTP for a +17.3 m shift on a 512 x 512 synthetic DEM at
  300 m; at most 1 DN after the fix).
- `dem * 0` as the start of accumulators keeps NaN and -0.0 where upstream had them.
- Ties in `max` and the sign of zero: kernels reproduce numpy's NaN rule of `fmax`/`maximum` explicitly; the
  sky illumination kernel also reproduces numpy 2.4's choice of operand on +0/-0 ties. No output depends on the sign
  of a zero maximum.
- Linear spline without refit (`_sky_illum_resample`): for `kx = ky = 1` on the integer knot grid the FITPACK
  coefficients equal the data, so the same `bisplev` evaluation is called without the fit; non-finite data take the
  original `RectBivariateSpline` path.
