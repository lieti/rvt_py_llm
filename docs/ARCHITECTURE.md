# Architecture of rvt (rvt_py_llm 0.1.1)

This document describes how the package `rvt` is organised, which functions do the work behind each public entry
point, how data flows through a visualization, and how the optional accelerated paths are built. It is written for
a developer who knows relief visualization but not this code, and for an LLM session that has to change the code
without breaking it. Units and conventions are in [CONVENTIONS.md](CONVENTIONS.md); latent bugs in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md); the tests and proofs in [TESTING.md](TESTING.md).

**Line numbers refer to commit `234b156`; the function names are the stable references.** Code is named by its
qualified name (`vis.sky_illumination` is `rvt.vis.sky_illumination`, `DefaultValues.get_mstp` is
`rvt.default.DefaultValues.get_mstp`); `vis.py:2048` means `rvt/vis.py` line 2048 at that commit. Lines given for a
function are the line of its `def`; lines given for a call are the line of the call. Every name below exists at
that commit.

History of the private names: three behaviour-neutral refactor commits of 0.1.0 renamed or merged private helpers
of tag `start` (`8413d3e`). Older documents, commit messages and the `start` line numbers in KNOWN_ISSUES.md use
the old names:

| at `start` | at 0.1.0 | commit |
|---|---|---|
| `_SVF_BLOCK_ELEMS` (vis.py), `_BLOCK_ELEMS` (blend_func.py) and two literals 32768 | `BLOCK_ELEMS` in `rvt/_blocking.py:19`, imported as `_BLOCK_ELEMS` (`vis.py:76`, `blend_func.py:77`) | 69525a0 |
| `_svf_accumulate`, `_sky_illum_iadd` | `vis._accumulate` (`vis.py:838`) | 2e1a306 |
| `_mstp_box_sum`, the nested `mean_filter._box_sum` | `vis._box_sum` (`vis.py:557`) | 8839afd |

`rvt/_blocking.py` is a module of its own because `rvt.blend_func` needs the constant but does not use the numba
module `rvt._accel`.

## 1. Module map

| module | responsibility | imports | public entry points |
|---|---|---|---|
| `rvt/__init__.py` | package docstring only; importing `rvt` imports no submodule | none | none |
| `rvt/vis.py` | the numerical algorithms on 2D numpy arrays; no file I/O | numpy, scipy, `rvt._accel`, `rvt._blocking` | `slope_aspect`, `hillshade`, `multi_hillshade`, `mean_filter`, `slrm`, `msrm`, `sky_view_factor`, `sky_view_factor_compute`, `local_dominance`, `sky_illumination`, `shadow_horizon`, `mstp`, `max_elevation_deviation`, `integral_image`, `topographic_dev`, `byte_scale`, `fill_where_nan`, pyramid helpers |
| `rvt/default.py` | `DefaultValues`: the default parameters of every visualization, settings files (json, legacy txt), `get_<vis>` (compute from an array), `save_<vis>` (read a GeoTIFF, compute, write GeoTIFFs), 8-bit stretch (`float_to_8bit`), log file; GDAL raster helpers; `RVTVisualization` enum | `rvt.vis`, `rvt.blend_func`, `rvt.tile`, osgeo.gdal, numpy | `DefaultValues`, `RVTVisualization`, `get_raster_arr`, `get_raster_size`, `save_raster` |
| `rvt/blend_func.py` | per-array blending operations: normalization, the blend modes, opacity, color ramps | numpy, matplotlib, `rvt._blocking` | `normalize_image`, `blend_images`, `render_images`, `gray_scale_to_color_ramp`, ... |
| `rvt/blend.py` | the layer model (`BlenderLayer`, `BlenderCombination`, `BlenderCombinations`), JSON combination and terrain files, the composite recipes `color_relief_image_map`, `e3mstp`, `e4mstp` | `rvt.default`, `rvt.vis`, `rvt.blend_func` | `BlenderCombination.render_all_images`, `e3mstp`, `e4mstp`, `color_relief_image_map`, `TerrainSettings`, ... |
| `rvt/tile.py` | tile-by-tile processing of rasters larger than memory, with an overlap per visualization | `rvt.default`, osgeo.gdal, numpy | `save_visualization_tile_by_tile`, `save_rvt_visualization_tile_by_tile` |
| `rvt/_accel.py` | private: optional numba kernels with a numpy fallback, the `RVT_NO_ACCEL` switch and thread-count control | `threading`, `os`; numba lazily | none (private) |
| `rvt/_blocking.py` | private, new in 0.1.0 (69525a0): `BLOCK_ELEMS = 32768`, the block size of every row-blocked loop, and why | none | none (private) |

Dependency direction: `vis` depends only on numpy, scipy, `_accel` and `_blocking`; `blend_func` only on numpy,
matplotlib and `_blocking`; `default` on `vis`, `blend_func`, `tile`; `tile` on `default` (imported at module
level, `rvt/tile.py`); `blend` on `default`, `vis`, `blend_func`. Nothing imports `blend`.

The settings files used by `rvt.default` and `rvt.blend` live in `settings/` at the repository root, not inside the
package.

## 2. Call graph of the evaluated entry points

The optimization run measured eleven entry points, each called as a tile pipeline calls it: a `DefaultValues()`
instance with its defaults, or the composite function (`eval/targets.py` of the optimization workspace). The
functions that hold the time are named; `[numba]` marks a kernel run through `rvt._accel.call`, with the numpy
code next to it as the fallback. File names are relative to `rvt/`; a line is a `def` unless marked "(call)".

```
DefaultValues.get_slope                 default.py:1889
  vis.slope_aspect                      vis.py:195     edge pad 1 px, central differences

DefaultValues.get_hillshade             default.py:2077
  vis.hillshade                         vis.py:350
    vis.slope_aspect                    vis.py:442 (call)

DefaultValues.get_slrm                  default.py:2442
  vis.slrm                              vis.py:667
    vis.mean_filter                     vis.py:576     integral image, box sums by slices (K-06)
      vis._box_sum                      vis.py:557     box sum of a summed-area table on the output window

DefaultValues.get_msrm                  default.py:3318
  vis.msrm                              vis.py:2542
    vis.mean_filter (per scale)         vis.py:2631 (call)

DefaultValues.get_sky_view_factor       default.py:2589   (target svf_asvf_opns: svf, asvf, opns)
DefaultValues.get_neg_opns              default.py:2844   (same path on -1 * dem)
  vis.sky_view_factor                   vis.py:1213    dem / resolution, ve_factor
    vis.sky_view_factor_compute         vis.py:1048    reflect pad r_max (call at vis.py:1334)
      vis._svf_window_ok                vis.py:810     guard: window path or reference loop (vis.py:1139)
      vis._svf_compute_window           vis.py:922     output window, row blocks (K-03)
        vis._svf_kernel_ok              vis.py:896     guard of the kernel (decision at vis.py:962)
        vis._svf_max_slope_kernel       vis.py:857     [numba] radius loop (K-08c), call at vis.py:982;
                                                       arctan/sin stay numpy
        vis._accumulate                 vis.py:838     sums over directions (vis.py:1014-1018)
      (reference: full-extent np.roll loop in sky_view_factor_compute, used when _svf_window_ok is False)

DefaultValues.get_local_dominance       default.py:3165
  vis.local_dominance                   vis.py:1394    edge pad max_rad, slices, row blocks (K-02)
    vis._local_dominance_kernel         vis.py:1355    [numba] (K-08a), call at vis.py:1514

DefaultValues.get_sky_illumination      default.py:3001
  vis.sky_illumination                  vis.py:2048
    vis.slope_aspect                    vis.py:2201 (call)
    vis.horizon_generate_pyramids       vis.py:1665    (call at vis.py:2206)
      vis.horizon_generate_coarse_dem   vis.py:1560
    vis._sky_illum_search_blocked       vis.py:1949    level search in row blocks (K-05, K-05b),
                                                       calls at vis.py:2325 (level 0), :2330 (coarse levels)
      vis._sky_illum_block_rows         vis.py:1835    rows per block, a multiple of 16
      vis._sky_illum_search_accel       vis.py:1918    guards, then one of the two kernels (call at vis.py:1981)
        vis._sky_illum_search_kernel    vis.py:1864    [numba] (K-08d)
        vis._sky_illum_search_literal_kernel vis.py:1893 [numba] shadow direction, level 0
    vis._sky_illum_level_search         vis.py:1786    numpy whole-array level search (call at vis.py:2332)
    vis._sky_illum_resample             vis.py:2008    linear spline without refit (K-05b), call at vis.py:2347
    vis._accumulate                     vis.py:838     directional sums (vis.py:2413-2434)

DefaultValues.get_mstp                  default.py:3473
  vis.mstp                              vis.py:2967    three scales: max_elevation_deviation called three times
                                                       (vis.py:3044, :3046, :3048)
    vis.max_elevation_deviation         vis.py:2816    symmetric pad; float64 summed-area tables of the DEM and
                                                       of its squares, accumulated in place (vis.py:2892-2898)
      vis.integral_image                vis.py:2642    only for the pixel-count table when the DEM has NaN
                                                       (call at vis.py:2884)
      vis._mstp_dev_max_kernel          vis.py:2758    [numba] (K-08b), call at vis.py:2915
      vis._box_sum                      vis.py:557     numpy path (K-01), calls at vis.py:2933-2937
  (vis.topographic_dev                  vis.py:2682    public, kept, not called inside rvt: the reference
                                                       formula that the two paths above evaluate on the
                                                       output window; KNOWN_ISSUES KI-04)

blend.e3mstp                            blend.py:1968
  DefaultValues.get_slrm                blend.py:2018 (call)
  blend.color_relief_image_map          blend.py:2019 (call) -> def blend.py:1876
    DefaultValues.get_sky_view_factor   blend.py:1937 (call; opns of dem), blend.py:1940 (call; opns of -dem)
    vis.slope_aspect                    blend.py:1945 (call)
    BlenderCombination.render_all_images blend.py:1963 (call)
  DefaultValues.get_mstp                blend.py:2021 (call)
  BlenderCombination.render_all_images  blend.py:2036 (call)

blend.e4mstp                            blend.py:2040
  DefaultValues.get_local_dominance     blend.py:2102 (call)
  DefaultValues.get_sky_view_factor     blend.py:2103 (call; svf + opns, r = 10 px)
  DefaultValues.get_neg_opns            blend.py:2106 (call)
  DefaultValues.get_sky_view_factor     blend.py:2112 (call; flat-terrain svf, r = int(10 / resolution) px = 10 m)
  DefaultValues.get_mstp                blend.py:2115 (call)
  vis.slope_aspect                      blend.py:2160 (call; degrees since 0.1.1, radians before)
  BlenderCombination.render_all_images  blend.py:2131, 2152, 2192 (calls; three combinations)

BlenderCombination.render_all_images    blend.py:616
  layer loop, bottom layer first        blend.py:740
  blend_func.normalize_image            per layer (calls in blend.py:769-946)
  blend_func.gray_scale_to_color_ramp   blend.py:953 (call; layers with a colormap)
  blend_func._outside_0_1, np.clip      blend.py:969-975 (range tests reused, K-11; the clip replaced
                                                       scale_0_to_1 in 0.1.1)
  blend_func.blend_images               blend.py:979 (call) -> def blend_func.py:1171
  blend_func._render_images             blend.py:984 (call) -> def blend_func.py:1262; opacity, row blocks of
                                                       _BLOCK_ELEMS (K-11)
```

`vis.topographic_dev` (`vis.py:2682`) is public and kept with its upstream behaviour, but no function in `rvt`
calls it since K-01: `max_elevation_deviation` evaluates the same float64 expressions only on the output window,
with `_box_sum` (numpy) or `_mstp_dev_max_kernel` (numba). A change to `topographic_dev` therefore does not change
MSTP; a change to the MSTP numerics has to be made in `max_elevation_deviation` and its kernel.

Where the time goes after the optimization (1024² medium window, 1 thread, sum about 3.9 s over the eleven entry
points; optimization report §4): sky illumination 1.33 s (numpy arctan/cos/sin/power per direction 57 %, numba search
24 %, `bisplev` 13 %); e4MSTP 1.01 s; e3MSTP 0.71 s; `_svf_compute_window` 1.05 s summed over four entry points.

### Blend modes

`blend_func.blend_images` (`blend_func.py:1171`) dispatches on the mode name, case-insensitive:

| mode | function | path | in-place effect |
|---|---|---|---|
| `normal` (and any unknown name) | `blend_normal` (`blend_func.py:761`) | returns `active` | none |
| `multiply` | `blend_multiply` (`blend_func.py:799`) via `blend_multi_dim_images` (`:1113`) and `equation_blend` (`:1049`) | one numpy expression; when a layer is RGB, `blend_multi_dim_images` assigns each band per row block through `_assign_elementwise` (`:1082`) | none |
| `screen` | `blend_screen` (`blend_func.py:780`), same route as multiply | as multiply | none |
| `overlay` | `blend_overlay` (`blend_func.py:895`), same route | `_blend_in_place_blocked` (`:818`) with `_overlay_masks` / `_overlay_branches` (`:871`, `:876`); reference `_blend_overlay_ref` (`:881`) | none: both paths write into a copy of `background` (0.1.1; up to 0.1.0 into `background` itself, KI-33) |
| `soft_light` | `blend_soft_light` (`blend_func.py:940`), same route | `_blend_in_place_blocked` with `_soft_light_masks` / `_soft_light_branches` (`:929`, `:934`); reference `_blend_soft_light_ref` (`:973`) | none: as overlay (0.1.1; KI-33 up to 0.1.0) |
| `luminosity` | `blend_luminosity` (`blend_func.py:988`) | `_luminosity_clip_blocked` (`:698`) for float `(3, H, W)` backgrounds without `min_c`/`max_c`; otherwise the original whole-array code in the same function | none |

Only overlay and soft light keep the original code as a separate `_ref` function; multiply and screen are
unchanged one-line expressions (the blocking is in `_assign_elementwise`), and for luminosity the original
whole-array code is the non-blocked branch of `blend_luminosity` itself.

## 3. Data flow through a visualization

The common path of a `rvt.vis` function, with the variations that matter:

1. **Input**: a 2D `numpy.ndarray` of elevations in m, row 0 north. `DefaultValues.get_*` passes it on unchanged;
   `save_*` reads it with `get_raster_arr` (`default.py:4125`), which returns the file's data type.
2. **No-data**: pixels equal to `no_data` become NaN. Most functions do this in the caller's array *before* the
   dtype conversion (`vis.py:425` `hillshade`, `:534` `multi_hillshade`, `:718` `slrm`, `:1476`
   `local_dominance`, `:2186` `sky_illumination`, `:2608` `msrm`, `:3039` `mstp`), so the caller's array changes and
   an integer array raises (KNOWN_ISSUES KI-17). `slope_aspect` (`vis.py:259-263`) and `sky_view_factor`
   (`vis.py:1305-1317`) convert first. The table in [CONVENTIONS.md](CONVENTIONS.md) section 3.1 lists every case.
3. **dtype**: `dem.astype(np.float32)` (e.g. `vis.py:259`, `:427`, `:720`, `:1305`, `:1478`, `:2188`, `:2610`,
   `:3041`), then `* ve_factor`. `sky_view_factor` also divides by `resolution` (`vis.py:1324`); `sky_illumination`
   does not (KI-06).
4. **Padding**: each function pads with its own mode and width, so that the neighbourhood of an edge pixel exists:
   edge 1 px (`slope_aspect`, `hillshade`, `vis.py:269`, `:429`); edge `(r + 1, r)` (`mean_filter`, `vis.py:627`);
   reflect `r_max` (`sky_view_factor_compute`, `vis.py:1131`); edge `max_rad` (`local_dominance`, `vis.py:1481`);
   symmetric, constant (`dem.min()`) and edge in the sky illumination pyramid (`horizon_generate_coarse_dem`
   `vis.py:1635-1659`, `horizon_generate_pyramids` `vis.py:1758`); symmetric `(r_max + 1, r_max)`
   (`max_elevation_deviation`, `vis.py:2874`).
5. **Algorithm** on the padded array; the optimized paths compute only the output window (section 4).
6. **Output**: the padding is cropped; NaN is restored at input NaN pixels where the function does so. Output dtypes
   differ per function, as asserted by the frozen tests: `slope_aspect` float32 (`tests/test_char_vis.py:220`),
   `hillshade` float64 (`:388`), `mean_filter` and `slrm` float32 (`:513`, `:549`), `msrm` float64 (`:598`),
   `sky_view_factor_compute` float64 (`:695`), `local_dominance` float64 (`:806`), `sky_illumination` float64
   (`:1043`), `max_elevation_deviation` float32 (`:1230`), `mstp` float32 `(3, H, W)` (`:1255`). The composites
   `color_relief_image_map`, `e3mstp`, `e4mstp` return float64 `(3, H, W)` in [0, 1].
7. **8-bit**: `DefaultValues.float_to_8bit` (`default.py:1705`) stretches with the `<vis>_bytscl` setting through
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
  `vis._svf_window_ok` (`vis.py:810`, used at `vis.py:1139`), the shift range check of `local_dominance`
  (`vis.py:1501`), the level-0 check of `sky_illumination` (`inner0`, `vis.py:2228`, and `blocked`, `vis.py:2276`),
  the radius and dtype guards of `max_elevation_deviation` (`vis.py:2907-2909`).
- **Same IEEE operations.** Per element, the optimized code performs the same operations in the same order with the
  same operand dtypes as the upstream code: float32 differences, float64 division by the `np.float64` radius,
  float64 summed-area tables and squares, the same accumulation order over directions. This is why the outputs are
  bitwise identical, not only close. Each private helper and kernel states its equivalence contract in its
  docstring.
- **Blocking for the cache.** SVF (`_svf_compute_window`, `vis.py:968`), local dominance (`vis.py:1518`), sky
  illumination (`_sky_illum_block_rows`, `vis.py:1835`) and the blend kernels (`blend_func._row_slices`,
  `blend_func.py:90`) work in row blocks of about `BLOCK_ELEMS = 32768` elements (`rvt/_blocking.py:19`), i.e.
  256 KiB of float64 per buffer, so the temporaries of one block stay in the CPU cache. The block size affects speed
  only: every blocked operation is elementwise or per pixel. Sky illumination uses a multiple of 16 rows per block
  so that the positions of elements in SIMD bodies and tails match the whole-array operation (`vis.py:1846`).
- **numba kernels via `_accel`.** Four algorithms have numba kernels, five kernels in total (every
  `@_accel.jit_kernel` in `rvt/vis.py`): the SVF radius loop (`vis._svf_max_slope_kernel`, `vis.py:857`), local
  dominance (`vis._local_dominance_kernel`, `vis.py:1355`), the sky illumination level search, which has two kernels
  (`vis._sky_illum_search_kernel`, `vis.py:1864`, and `vis._sky_illum_search_literal_kernel`, `vis.py:1893`, which
  keeps the literal operation order needed for the zero sign in the shadow direction), and the MSTP deviation loop
  (`vis._mstp_dev_max_kernel`, `vis.py:2758`). A kernel is declared with `@_accel.jit_kernel` (`_accel.py:116`)
  and run with `_accel.call(kernel, ...)` (`_accel.py:174`), which returns False when the caller must run its numpy
  path. `rvt.blend_func` has no numba kernel; its blocked paths are numpy.
- **Fallback contract.** numba is imported lazily on the first accelerated call, never at `import rvt`
  (`_accel.get_numba`, `_accel.py:62`, import at `:78`). `call` returns False when `RVT_NO_ACCEL` is set (any value
  except '', '0', 'false', 'no', 'off'; `_accel.enabled`, `_accel.py:49`), when numba is missing or fails to import,
  or when a kernel fails to compile or run; a kernel that failed once stays disabled for the process. The callers
  either re-initialise their outputs after False or guard their inputs so that only typing and compilation, which
  happen before any write, can fail.
- **Bitwise contract of the kernels.** Kernels are compiled with `numba.njit(nogil=True)` (`_accel.dispatcher`,
  `_accel.py:143`, the `njit` call at `:168`): no fastmath, no FMA contraction, no reassociation, default error
  model, no on-disk cache. NaN rules are written out explicitly where numpy's `fmax`/`maximum` define them.
  Accelerated and numpy paths are bitwise equal; the tests compare them (`tests/test_accel.py`, the per-function test files).
- **No threads in kernels.** All kernels are serial (`parallel=False`), so results do not depend on the thread
  count. `_accel.num_threads` (`_accel.py:86`) reads `RVT_NUM_THREADS`, then `NUMBA_NUM_THREADS`, and is applied only
  to kernels declared `parallel=True` (none). The numpy work uses no BLAS. A pipeline that runs one tile per process
  can set all thread variables to 1 without losing speed.
- **Transcendentals stay in numpy.** arctan, sin, cos, power run in numpy in every path, because a different
  implementation would not be bitwise identical. Bitwise equality of the blocked numpy transcendentals was verified on
  one AVX2 CPU only.

### Where the numba / numpy decision is taken

The global switch is `_accel.enabled()` (`_accel.py:49`), read on every call; `_accel.call` (`_accel.py:174`)
returns False whenever the kernel cannot run (switch off, numba missing, compile or run failure). Each call site
adds its own guard, which sends inputs the kernel does not support to numpy before the kernel is tried:

| visualization | function and guard | kernel call | after False |
|---|---|---|---|
| SVF, anisotropic SVF, openness | `vis._svf_compute_window`: `use_kernel = _accel.enabled() and _svf_kernel_ok(height, scan)` (`vis.py:962`; `_svf_kernel_ok`, `vis.py:896`) | `vis.py:982` | numpy scan from that direction on (`vis.py:986`) |
| local dominance | `vis.local_dominance`: shifts within the padding (`vis.py:1501`) and float dtypes (`vis.py:1510-1511`) | `vis.py:1514` | output re-initialised, numpy row blocks (`vis.py:1517`) |
| sky illumination, shadow | `vis.sky_illumination` chooses the blocked path (`blocked`, `vis.py:2276`); `vis._sky_illum_search_blocked` tries the kernel for float64 quotients (`vis.py:1981`); `vis._sky_illum_search_accel` checks `_accel.enabled()` and the input guards (`vis.py:1933-1941`) and picks the kernel by `floor_once` (`vis.py:1945`) | `vis.py:1946` | numpy row blocks (`vis.py:1983` ff.); the guards leave `max_slope` untouched |
| MSTP, e3MSTP, e4MSTP | `vis.max_elevation_deviation`: radii > 0, step > 0, float dtypes, float64 tables (`vis.py:2907-2909`) | `vis.py:2915` | numpy path with `_box_sum` (`vis.py:2921` ff.) |

SVF and sky illumination test `_accel.enabled()` themselves so that they do not build the kernel's argument arrays
when accelerators are off; local dominance and MSTP rely on `_accel.call` returning False.

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
| `tests/test_proof_box_sum.py` | proof (added after the release audit) | `rvt.vis._box_sum` against the reference operation order on inputs where the order changes the rounding |
| `tests/test_fix_0_1_1_*.py` (6 files) | 0.1.1 fix tests | X1 overlay / soft light into a new array and the opacity mix, X2 e4MSTP slope in degrees, X3 clip and `scale_*`, X4 `default` unchanged, X5 opacity range, e3/e4 rebuilt through `BlenderCombination` |

The suite has 1194 tests and runs in about 12 s on one process; every file listed above is frozen at tag
`tests-frozen` (0.1.1 freeze, 1194 tests; the 0.1.0 freeze of 1089 tests is tag `tests-frozen-0.1.0`, see
[TESTING.md](TESTING.md)).

## 6. If you need to change X, look at Y

| you need to change | look at | keep in mind |
|---|---|---|
| a default parameter | `DefaultValues.__init__` (`default.py:317`), `DefaultValues.save_default_to_file` (`default.py:413`), `DefaultValues.read_default_from_file` (`default.py:697`), `settings/default_settings.json` | the API snapshot test fails on purpose; file names encode parameters (`get_*_file_name`) |
| a visualization algorithm | the function in `rvt/vis.py`, its private helpers and kernel (section 2) | if it has an accelerated path, the numpy reference path, the sliced path and the kernel must stay bitwise equal; the per-function equivalence test compares them |
| a numba kernel | the kernel docstring (equivalence contract), its call site and guard (section 4, "Where the numba / numpy decision is taken"), `rvt/_accel.py` | no fastmath, no parallel, no on-disk cache; keep the guard that sends unsupported inputs to numpy; run with and without `RVT_NO_ACCEL=1` |
| the numba switch or thread count | `_accel.enabled` (`_accel.py:49`), `_accel.num_threads` (`_accel.py:86`), `_accel.call` (`_accel.py:174`) | `RVT_NO_ACCEL` values and `RVT_NUM_THREADS` order are tested in `tests/test_accel.py` and `tests/test_char_accel.py` |
| the block size of the blocked loops | `BLOCK_ELEMS` (`_blocking.py:19`) | speed only; sky illumination also rounds its block to a multiple of 16 rows (`vis._sky_illum_block_rows`, `vis.py:1835`) |
| a blend mode | the mode's function in `rvt/blend_func.py` (table "Blend modes" in section 2) and, for overlay and soft light, its `_blend_*_ref` function | blocked and reference paths must give the same bits and the same in-place effects and warnings |
| a normalization | `blend_func.normalize_image` (`blend_func.py:1504`), `advanced_normalization` (`:420`), `normalize_lin` (`:278`), `normalize_perc` (`:387`) | `normalize_lin` has a blocked path (`_row_slices`); the percent stretch is per image |
| a composite recipe (CRIM, e3MSTP, e4MSTP) | `blend.color_relief_image_map` (`blend.py:1876`), `blend.e3mstp` (`blend.py:1968`), `blend.e4mstp` (`blend.py:2040`) | `tests/test_char_e3e4_recipe.py` rebuilds the recipes layer by layer; CRIM works on a deep copy of `default` (0.1.1) |
| layer rendering | `BlenderCombination.render_all_images` (`blend.py:616`) | range tests are reused between layers (K-11); operands outside [0, 1] are clipped (0.1.1); the background range test is recomputed after overlay and soft light (conservative since 0.1.1, when they stopped writing into the background) |
| reading or writing rasters | `default.get_raster_arr`, `default.save_raster` (`default.py:4125`, `:4194`), `rvt/tile.py` | GDAL only inside `rvt.default` and `rvt.tile` |
| tiling overlaps | `tile._get_rvt_visualization_overlap` (`tile.py:372`) | measured minimum buffers are in KNOWN_ISSUES, section "Tiling facts" |
| an exception message | the `raise` line | only in a `msg:` commit; exception types must not change |
| behaviour that looks wrong | KNOWN_ISSUES.md and the docstring Notes | do not fix it in this repository; document it and pin it with a test |

## 7. Why some numerics look the way they do

- float64 squares and summed-area tables in MSTP: the variance E[x²] - E[x]² loses its precision in float32 at real
  elevations; the precision fix (commit 03702f4) squares in float64. Upstream squared in float32, so MSTP depended on
  the absolute elevation: the same terrain shifted by +17.3 m changed the 8-bit e3MSTP/e4MSTP output by up to
  207-210 DN (99.9th percentile 150 DN) on a 512 x 512 px synthetic DEM at 300 m, and by at most 1 DN after the fix
  (measurement stated in the message of commit 03702f4).
- `dem * 0` as the start of accumulators keeps NaN and -0.0 where upstream had them.
- Ties in `max` and the sign of zero: kernels reproduce numpy's NaN rule of `fmax`/`maximum` explicitly; the
  sky illumination kernel also reproduces numpy 2.4's choice of operand on +0/-0 ties. No output depends on the sign
  of a zero maximum.
- Linear spline without refit (`vis._sky_illum_resample`, `vis.py:2008`): for `kx = ky = 1` on the integer knot
  grid the FITPACK coefficients equal the data, so the same `bisplev` evaluation is called without the fit;
  non-finite data take the original `RectBivariateSpline` path.
