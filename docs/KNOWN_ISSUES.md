# Known issues of rvt_py_llm 0.1.0

This file lists behaviour of `rvt` that looks unintended. None of it is changed in 0.1.0: the rule of this
release is that every output stays bitwise identical to tag `start` (see [TESTING.md](TESTING.md)). Each entry
is described in the docstring of the function concerned ("Known issue ... see docs/KNOWN_ISSUES.md"), and most
are pinned by a frozen characterization test, so a fix shows up as a failing test that has to be replaced on
purpose. Deciding which of these to fix, and how, is left to the upstream authors. Release 0.1.1 fixes the entries
KI-19, KI-20, KI-21, KI-33, KI-34, KI-36, KI-45 and KI-47 (their Status line names the commit; the fixes, readings
and open questions are described in [FIXES_0_1_1.md](FIXES_0_1_1.md)); the entries keep their 0.1.0 text as
evidence.

Conventions of this list:

- **Where** gives `file:line` at tag `start` (`8413d3e`, the code of `opt/perf`), where the issue was found; these
  lines are kept as evidence. In 0.1.0 the same code has docstrings and comments added, so its line numbers are
  200-1600 lines larger. **Now** gives the same places in 0.1.0: the enclosing function (the stable reference) and
  the current line at commit `234b156`, found by matching the unchanged source line between the two commits.
- **Repro** uses the preamble below; the result after `->` was printed on Windows 11, Python 3.11.17, numpy 2.4.6,
  numba 0.68.0, by code that is behaviour-identical to `start`.
- **Pinned** names the frozen test that asserts the current behaviour (`tests/<file>:<line>` or
  `tests/<file>::<test>`); "not pinned" means no test asserts it.
- **Status** is "documented, not changed in 0.1.0" for every entry not fixed in 0.1.1, and "fixed in 0.1.1
  (<commit>), see docs/FIXES_0_1_1.md" for the fixed ones.
- Upstream review numbers (`AUDIT #n`) refer to the internal review that preceded 0.1.0; they are given only to
  connect the two lists.

Entries are grouped by the module that holds the cause: `rvt.vis` (KI-01 to KI-18), `rvt.blend_func` (KI-19 to
KI-34), `rvt.blend` (KI-35 to KI-50), `rvt.default` (KI-51 to KI-69), `rvt.tile` (KI-70 to KI-76); KI-77
(`rvt.vis`, found in 0.2.0) follows at the end. A last section
lists tiling facts that are not defects but matter for tiled processing.

## Preamble used by the repros

```python
import sys, os, tempfile, warnings
sys.path.insert(0, "<repository root>"); warnings.simplefilter("ignore")
import numpy as np
from pathlib import Path
from osgeo import gdal
import rvt.vis as v, rvt.blend as rb, rvt.blend_func as bf, rvt.default, rvt.tile
from rvt.default import DefaultValues, RVTVisualization as V
os.chdir(tempfile.mkdtemp()); TMP = os.getcwd()          # repros that read settings/ run from the repository root

def write_dem(name, arr, gt=(0, 1, 0, 60, 0, -1), nodata=None, etype=gdal.GDT_Float32):
    ds = gdal.GetDriverByName("GTiff").Create(name, arr.shape[1], arr.shape[0], 1, etype)
    ds.SetGeoTransform(gt); ds.GetRasterBand(1).WriteArray(arr)
    if nodata is not None: ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None; return os.path.abspath(name)

y, x = np.mgrid[0:60, 0:70]
DEM = (np.sin(x / 7.0) * 5 + np.cos(y / 5.0) * 3 + 0.05 * x * y).astype(np.float32)   # 60 x 70 px, m
```

## rvt.vis

### KI-01 fill_where_nan "idw": the window end column is clipped to the number of rows
- Where: `rvt/vis.py:2350` at `start` (`fill_where_nan`): `i_column_end = dem.shape[0]` (`shape[1]` is meant).
- Description: when the IDW window passes the right edge, its end column becomes H (rows) instead of W (columns).
- Impact: arrays wider than tall: valid pixels in columns >= H are ignored; a NaN pixel at a column >= H gets a
  window without its own column (IndexError) or an empty window (stays NaN). Arrays taller than wide are not
  affected (numpy slicing stops at the edge).
- Repro:
  ```python
  d = np.zeros((8, 14)); d[:, 8:] = 1000.0; d[3, 7] = np.nan
  v.fill_where_nan(d, "idw_8_2")[3, 7]          # -> 0.0 (the 1000 m columns 8..13 are ignored)
  d = np.ones((8, 14)); d[3, 12] = np.nan
  v.fill_where_nan(d, "idw_5_2")                # -> IndexError
  ```
- Pinned: `tests/test_char_vis.py` (fill_where_nan tests). AUDIT #24.
- Status: documented, not changed in 0.1.0.

### KI-02 multi_hillshade returns (n, H - 2, W - 2); the saved and the tiled rasters inherit it
- Where: `rvt/vis.py:311` at `start` (`hillshade` always crops 1 px: `hillshade_out[1:-1, 1:-1]`), called by
  `multi_hillshade` at `rvt/vis.py:383` with a slope and aspect of shape (H, W). Consequences:
  `rvt/default.py:1467-1471` (`save_multi_hillshade` writes the cropped array with the DEM's geotransform) and
  `rvt/tile.py:238-239`, `:419-440` (`save_rvt_visualization_tile_by_tile` assumes a result of tile size).
- Now: `hillshade` (`rvt/vis.py:454`); `multi_hillshade` (`rvt/vis.py:549`); `DefaultValues.save_multi_hillshade`
  (`rvt/default.py:2422-2426`); `_create_rvt_visualization_blank_raster` (`rvt/tile.py:347-348`);
  `save_rvt_visualization_tile_by_tile` (`rvt/tile.py:619-640`) (lines at `234b156`).
- Description: `hillshade` crops 1 px on each side of whatever slope it gets; this fits only its own
  (H + 2, W + 2) slope. `multi_hillshade` passes an (H, W) slope, so its result is (n, H - 2, W - 2); pixel
  [i, j] belongs to DEM pixel [i + 1, j + 1].
- Impact: multi-direction hillshade loses a 1 px border and has a different shape from every other
  visualization. The float GeoTIFF of `save_multi_hillshade` is 2 px smaller than the DEM and shifted by 1 px
  to the north-west. Tiled float output: inside every tile pixel [i, j] holds DEM pixel [i + 1, j + 1] and the
  last 2 rows and columns of every tile stay NaN.
- Repro:
  ```python
  v.multi_hillshade(np.zeros((20, 30), np.float32), 1, 1).shape                     # -> (16, 18, 28)
  p = write_dem("dem.tif", DEM); dv = DefaultValues(); dv.save_multi_hillshade(p, save_float=True, save_8bit=False)
  ds = gdal.Open(dv.get_multi_hillshade_path(p)); print(ds.RasterXSize, ds.RasterYSize)   # -> 68 58 (DEM 70 x 60)
  dv.tile_size = (50, 50); rvt.tile.save_rvt_visualization_tile_by_tile(V.MULTI_HILLSHADE, dv, Path(p), Path("."), True, False)
  t = rvt.default.get_raster_arr(dv.get_multi_hillshade_path(p))["array"]; [i for i in range(60) if np.isnan(t[:, i, :]).all()]  # -> [48, 49, 58, 59]
  ```
- Pinned: `tests/test_char_vis.py` (multi_hillshade shape), `tests/test_char_tile.py::test_tiled_multi_hillshade_geometry`.
  AUDIT #27; note 7 of the notes for the upstream authors.
- Status: documented, not changed in 0.1.0.

### KI-03 sky_illumination: a tile matches the full raster exactly only when its origin is a multiple of 8 px
- Where: `rvt/vis.py:1570` at `start` (`horizon_generate_pyramids(dem, ...)` inside `sky_illumination`),
  `rvt/vis.py:1218-1294` (`horizon_generate_pyramids`), `rvt/vis.py:1156` (`horizon_generate_coarse_dem`).
- Now: `sky_illumination` (`rvt/vis.py:2206`); `horizon_generate_pyramids` (`rvt/vis.py:1665-1783`);
  `horizon_generate_coarse_dem` (`rvt/vis.py:1560`) (lines at `234b156`).
- Description: the coarse pyramid levels (scale 2, 3 levels with the defaults) are anchored at the top-left
  corner of the array passed in, so the coarse grid of a tile depends on the tile origin modulo 2^3 = 8 px.
- Impact: tiled processing (`rvt.tile`, or any external tiling) differs from the full-raster result. Measured on a
  synthetic 1024 px window with 512 px tiles and a 128 px buffer: origins 248 and 256 give an identical interior;
  origins 252, 254 and 255 differ by up to 1.9e-4, 3.1e-4 and 6.1e-4 (1 level on a 0-1 -> 0-255 scale).
- Repro: [repro/sim_tile_origin.py](repro/sim_tile_origin.py) (self-contained, imports `rvt` from the repository
  root, about 5 s with numba): `python docs/repro/sim_tile_origin.py` ->
  ```
  tile origin 256 (mod 8 = 0): max |tile - full| = 0, max difference on a 0..1 -> 0..255 scale 0 DN
  tile origin 248 (mod 8 = 0): max |tile - full| = 0, max difference on a 0..1 -> 0..255 scale 0 DN
  tile origin 252 (mod 8 = 4): max |tile - full| = 0.000192, max difference on a 0..1 -> 0..255 scale 1 DN
  tile origin 254 (mod 8 = 6): max |tile - full| = 0.000307, max difference on a 0..1 -> 0..255 scale 1 DN
  tile origin 255 (mod 8 = 7): max |tile - full| = 0.000612, max difference on a 0..1 -> 0..255 scale 1 DN
  ```
  (first output line, the import path, omitted; printed at `de0f665`). See also the tiling section below.
- Pinned: not pinned. AUDIT #28; note 6 of the notes for the upstream authors.
- Status: documented, not changed in 0.1.0.

### KI-04 topographic_dev is public but no rvt function calls it
- Where: `rvt/vis.py:1999` at `start` (`topographic_dev`).
- Now: `topographic_dev` (`rvt/vis.py:2682`) (lines at `234b156`).
- Description: `max_elevation_deviation` evaluates the same expressions on the output window (numpy path with
  `_box_sum`, called `_mstp_box_sum` at `start`; numba path `_mstp_dev_max_kernel`); `topographic_dev` is kept as
  public API only.
- Impact: none on results; a change to `topographic_dev` does not change MSTP.
- Repro:
  ```python
  import ast, pathlib
  [(p.name, n.lineno) for p in pathlib.Path("rvt").glob("*.py") for n in ast.walk(ast.parse(p.read_text("utf-8")))
   if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "topographic_dev"]   # -> []
  ```
- Pinned: not applicable (documentation). AUDIT #23.
- Status: documented, not changed in 0.1.0.

### KI-05 mstp accepts ve_factor >= -1000 only
- Where: `rvt/vis.py:2263` at `start` (`mstp`): `if not (10000 >= ve_factor >= -1000)`.
- Now: `mstp` (`rvt/vis.py:3034`) (lines at `234b156`).
- Description: every other function accepts [-10000, 10000]. Until 0.1.0 the message said "-10000"; the `msg:`
  commit 51a882a made the message state the actual check, which is unchanged.
- Impact: `ve_factor` in [-10000, -1001] raises for MSTP only. MSTP is nearly invariant to `ve_factor` (only the
  1e-6 added to the standard deviation depends on it).
- Repro:
  ```python
  v.mstp(np.random.default_rng(1).random((20, 20)).astype(np.float32), (1, 3, 1), (3, 6, 1), (6, 9, 1), ve_factor=-1001)
  # -> Exception: rvt.vis.mstp: ve_factor must be between -1000 and 10000!
  ```
- Pinned: `tests/test_char_vis.py` (mstp parameter checks). AUDIT #17.
- Status: documented, message corrected in 0.1.0, check not changed.

### KI-06 sky_illumination and shadow_horizon: horizon angles ignore the pixel size
- Where: `rvt/vis.py:1552-1553` at `start` (DEM converted and multiplied by `ve_factor`, never divided by
  `resolution`); compare `sky_view_factor` at `rvt/vis.py:976` (`dem = dem / resolution`). In `sky_illumination`
  `resolution` is used only for slope and aspect (`rvt/vis.py:1565`).
- Now: `sky_illumination` (`rvt/vis.py:2188-2189`, `:2201`); `sky_view_factor` (`rvt/vis.py:1324`) (lines at
  `234b156`).
- Description: the horizon search divides height differences in m by distances in px, so horizon angles are
  correct only at 1 m/px.
- Impact: shadow, horizon and the horizon terms of sky illumination are wrong at any other pixel size; at
  0.25 m/px the tangent of every horizon angle is 4 times too small.
- Repro:
  ```python
  dem = np.cumsum(np.random.default_rng(0).normal(size=(40, 40)), 0).astype(np.float32)
  np.array_equal(v.shadow_horizon(dem.copy(), 1)["horizon"], v.shadow_horizon(dem.copy(), 10)["horizon"])  # -> True
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-07 sky_illumination, shadow_horizon, get_shadow: one NaN pixel spoils the whole result
- Where: `rvt/vis.py:1269` at `start` (pyramid padding with `constant_values=dem.min()`), `rvt/vis.py:1213`
  (coarse padding with `dem_coarse.min()`), the spline fit on non-finite data (`_sky_illum_resample` falls back to
  `RectBivariateSpline`), `rvt/vis.py:1813` (`overcast_out.max()`); `shadow_horizon` at `rvt/vis.py:1885`;
  through `DefaultValues.get_shadow` (`rvt/default.py:1284`) and `get_sky_illumination` (`rvt/default.py:1811`).
- Now: `horizon_generate_pyramids` (`rvt/vis.py:1758`); `horizon_generate_coarse_dem` (`rvt/vis.py:1660`);
  `sky_illumination` (`rvt/vis.py:2456`); `shadow_horizon` (`rvt/vis.py:2537`); `DefaultValues.get_shadow`
  (`rvt/default.py:2037`); `DefaultValues.get_sky_illumination` (`rvt/default.py:3001`) (lines at `234b156`).
- Description: `dem.min()` is NaN when the DEM holds a NaN, so the pyramid padding and every coarse level become
  NaN, the spline spreads NaN to every pixel, and the horizon of every pixel is NaN. The shadow is computed by
  comparisons with that horizon, which are False for NaN, so the shadow is 0 everywhere.
- Impact: a DEM with any no-data pixel (or a `no_data` value present) gives an all-NaN sky illumination and
  horizon and an all-zero shadow. Measured: 40 x 40 px, one NaN: 1600 of 1600 sky illumination pixels NaN
  (overcast and uniform), horizon 1600 of 1600 NaN, shadow 1600 of 1600 zero; `DefaultValues` defaults on the
  60 x 70 px DEM with one NaN: 4200 of 4200 pixels NaN.
- Repro:
  ```python
  d = np.cumsum(np.random.default_rng(0).normal(size=(40, 40)), 0).astype(np.float32); d[5, 5] = np.nan
  [int(np.isnan(v.sky_illumination(d.copy(), 1, sky_model=m)).sum()) for m in ("overcast", "uniform")]  # -> [1600, 1600]
  s = v.shadow_horizon(d.copy(), 1); int(np.isnan(s["horizon"]).sum()), int((s["shadow"] == 0).sum())   # -> (1600, 1600)
  ```
- Pinned: `tests/test_char_default.py::test_shadow_of_a_dem_with_nan_is_all_zero`,
  `tests/test_char_default.py::test_sky_illumination_spreads_a_single_nan_over_most_of_the_output` (asserts more than
  half); `tests/test_char_vis.py::test_sky_illumination_char_nan_pixels` asserts only "NaN at least at the input NaN".
- Status: documented, not changed in 0.1.0.

### KI-08 sky_illumination: shadow_az is matched without wrap-around (0 and 360 differ)
- Where: `rvt/vis.py:1616` at `start`: `i = np.argmin(np.abs(_ - (360 - shadow_az)))`.
- Now: `sky_illumination` (`rvt/vis.py:2252`) (lines at `234b156`).
- Description: the direction keys run from 0 to 348.75 (32 directions); `shadow_az = 0` looks for 360 and picks
  348.75 (the key of `shadow_az = 11.25`) instead of key 0, which `shadow_az = 360` picks.
- Impact: a sun exactly in the north casts its shadow 11.25° off; 0 and 360 give different results.
- Repro:
  ```python
  dem = np.cumsum(np.random.default_rng(0).normal(size=(40, 40)), 0).astype(np.float32)
  h = lambda az: v.shadow_horizon(dem.copy(), 1, shadow_az=az)["horizon"]
  np.array_equal(h(0), h(11.25)), np.array_equal(h(0), h(360))    # -> (True, False)
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-09 sky_illumination: a plane receives different light by facing direction, and more than flat terrain
- Where: `rvt/vis.py:1767` at `start`: `_d_aspect = -2 * np.sin(da) * np.cos(dir_rad - aspect)`, combined with the
  direction keys of `horizon_shift_vector` (`rvt/vis.py:528-529`, counterclockwise from north) and the aspect of
  `slope_aspect` (clockwise from north). The mixed conventions are a plausible cause; this is not proven.
- Now: `sky_illumination` (`rvt/vis.py:2408`); `horizon_shift_vector` (`rvt/vis.py:777-778`) (lines at `234b156`).
- Description: under a uniform sky a plane of a given slope should receive the same light whatever its aspect,
  and less than flat terrain (1).
- Impact: on a 26.6° plane east- and west-facing slopes get about 11 % more light than north- and south-facing
  ones, and values exceed 1.
- Repro:
  ```python
  yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) * 0.5          # slope 26.6 deg
  [round(float(v.sky_illumination(p, 1, sky_model="uniform", max_fine_radius=20)[32, 32]), 4)
   for p in (-xx, xx, yy, -yy)]                                    # E, W, N, S -> [1.0597, 1.0597, 0.9516, 0.9516]
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-10 sky_illumination: the shaded uniform model is not divided by pi
- Where: `rvt/vis.py:1815` at `start` (`uniform_sh_out = (0.8 * uniform_out + 0.2 * shadow_out)`), computed before
  `rvt/vis.py:1820` (`uniform_out = uniform_out / np.pi`).
- Now: `sky_illumination` (`rvt/vis.py:2458`, `:2463`) (lines at `234b156`).
- Impact: `sky_model="uniform", compute_shadow=True` returns values around 0.8 pi + 0.2 = 2.71 on flat terrain
  instead of values in [0, 1].
- Repro:
  ```python
  v.sky_illumination(np.zeros((30, 30), np.float32), 1, sky_model="uniform", compute_shadow=True)[15, 15]  # -> 2.7133
  ```
- Pinned: `tests/test_char_vis.py` (sky_illumination uniform with shadow).
- Status: documented, not changed in 0.1.0.

### KI-11 sky_view_factor: the anisotropic main direction is measured counterclockwise from north
- Where: `rvt/vis.py:831` and `rvt/vis.py:654` at `start`
  (`weight = (1 - w_m) * (np.cos((weight - w_a) / 2)) ** a_poly_level + w_m`, `weight` = the direction keys), keys from
  `rvt/vis.py:528-529` (key 0 north, 90 west).
- Now: `sky_view_factor_compute` (`rvt/vis.py:1157`); `_svf_compute_window` (`rvt/vis.py:952`); `horizon_shift_vector`
  (`rvt/vis.py:777-778`) (lines at `234b156`).
- Description: `asvf_dir` is compared with keys measured counterclockwise from north, while `sun_azimuth` and
  `shadow_az` are clockwise. The default `asvf_dir = 315` weights the north-east sky most, not the north-west.
- Impact: the anisotropic SVF is mirrored east-west relative to the azimuth convention of the other functions,
  unless this is intended.
- Repro (one 50 m pixel 4 px north-west or north-east of the centre; a lower value means an obstacle in the
  weighted sky):
  ```python
  for dy, dx in ((-4, -4), (-4, 4)):                               # NW, NE
      d = np.zeros((41, 41), np.float32); d[20 + dy, 20 + dx] = 50
      print(v.sky_view_factor(d, 1, compute_svf=False, compute_asvf=True, asvf_level=2)["asvf"][20, 20])  # -> 0.972, 0.8206
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-12 local_dominance with max_rad = 0 returns an empty array
- Where: `rvt/vis.py:1151` at `start`: `local_dom_out[pad_width:-pad_width, pad_width:-pad_width]` with
  `pad_width = 0` is `[0:-0]`.
- Now: `local_dominance` (`rvt/vis.py:1555`) (lines at `234b156`).
- Impact: edge case only (no shift exists then).
- Repro: `v.local_dominance(np.zeros((10, 10), np.float32), 0, 0).shape   # -> (0, 0)`
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-13 max_elevation_deviation and mstp: a radius <= 0 contributes the elevation, shifted by one pixel
- Where: `rvt/vis.py:2175` at `start` (`dev = dem_pad[maximum_radius:-(maximum_radius + 1), ...]`), which reproduces
  `topographic_dev` (`rvt/vis.py:2024`, returns the padded DEM for radius <= 0) and the reference crop.
- Now: `max_elevation_deviation` (`rvt/vis.py:2926`); `topographic_dev` (`rvt/vis.py:2725`) (lines at `234b156`).
- Description: for D <= 0 the "deviation" is the elevation itself, taken at (i - 1, j - 1), because the window
  centre is at +1 in the padded array.
- Impact: `minimum_radius = 0`, which `mstp`'s checks allow, mixes elevations in metres into DEV_max.
- Repro: `v.max_elevation_deviation(np.arange(30, dtype=np.float32).reshape(5, 6), 0, 0, 1)[2, 2]   # -> 7.0 = z[1, 1]`
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-14 msrm divides by zero when only one filtered surface exists
- Where: `rvt/vis.py:1960` at `start`: `msrm_out = relief_models_sum / nr_relief_models` with `nr_relief_models = 0`
  when i == n.
- Now: `msrm` (`rvt/vis.py:2637`) (lines at `234b156`).
- Impact: all-NaN output and a RuntimeWarning, e.g. for `feature_min == feature_max == resolution`.
- Repro: `np.isnan(v.msrm(np.random.default_rng(2).random((20, 20)).astype(np.float32), 1, 1, 1, 1)).all()   # -> True`
- Pinned: `tests/test_char_vis.py` (msrm with one scale).
- Status: documented, not changed in 0.1.0.

### KI-15 byte_scale: offset wraps, constant band becomes 255, uint8 branch, 3D input modified
- Where: `rvt/vis.py:109` at `start` (`+ low` after clipping, in uint8); `rvt/vis.py:91` with `rvt/vis.py:105`
  (`c_scale` set to 1, but the division uses `c_max - c_min`); `rvt/vis.py:94` (uint8 branch: `high + 1 = 256`
  overflows uint8 under numpy 2, and `return` inside the band loop); `rvt/vis.py:81` (no-data replaced in the
  bands, which are views of a 3D input).
- Now: `byte_scale` (`rvt/vis.py:186`, `:169`, `:182`, `:174`, `:159`) (lines at `234b156`).
- Impact: `low > 0` wraps the top values and the NaN code (255 + 10 = 9); a constant band becomes all 255; uint8
  input raises OverflowError with the default `high`, and with `high < 255` a 3D uint8 input returns only its first
  band; a 3D input with `no_data` is modified.
- Repro:
  ```python
  v.byte_scale(np.array([[0., 1., np.nan]]), low=10).tolist()                     # -> [[10, 9, 9]]
  v.byte_scale(np.full((2, 2), 3.0)).tolist()                                     # -> [[255, 255], [255, 255]]
  v.byte_scale(np.zeros((3, 4, 4), np.uint8) + 5, high=200).shape                # -> (4, 4)
  v.byte_scale(np.array([[0, 10], [20, 255]], dtype=np.uint8))                    # -> OverflowError
  ```
- Pinned: `tests/test_char_vis.py` (byte_scale tests), except the 3D uint8 return.
- Status: documented, not changed in 0.1.0.

### KI-16 slope_aspect: aspect in (-pi, pi], and "percent" leaves the aspect in radians
- Where: `rvt/vis.py:202` at `start` (`np.arctan2`, no shift to [0, 2 pi)); `rvt/vis.py:203` (only "degree"
  converts the aspect).
- Now: `slope_aspect` (`rvt/vis.py:295`, `:296`) (lines at `234b156`).
- Description: the upstream docstring said "0 or 2pi for north" (a clockwise azimuth in [0, 2 pi)); the code
  returns north 0, east pi/2, south pi, west -pi/2. The 0.1.0 docstring describes the code.
- Repro:
  ```python
  yy, xx = np.mgrid[0:9, 0:9].astype(np.float32)
  v.slope_aspect(xx * 0.5)["aspect"][4, 4]                                        # -> -1.5708 (down-slope west)
  ```
- Pinned: `tests/test_char_vis.py` (slope_aspect aspect range).
- Status: documented, not changed in 0.1.0.

### KI-17 no_data is replaced in the caller's array; integer DEMs with no_data then raise
- Where: `dem[dem == no_data] = np.nan` before the float32 conversion at `start` in `hillshade`
  (`rvt/vis.py:282`), `multi_hillshade` (`:368`), `slrm` (`:484`), `local_dominance` (`:1074`), `sky_illumination`
  and `shadow_horizon` (`:1550`), `msrm` (`:1931`), `mstp` (`:2268`). `slope_aspect` (`:170`) and `sky_view_factor`
  (`:969`) convert first and leave the input unchanged. `rvt/default.py:2564` (`get_raster_arr`) returns the data
  type of the file.
- Now: `hillshade` (`rvt/vis.py:425`); `multi_hillshade` (`rvt/vis.py:534`); `slrm` (`rvt/vis.py:718`);
  `local_dominance` (`rvt/vis.py:1476`); `sky_illumination` (`rvt/vis.py:2186`); `msrm` (`rvt/vis.py:2608`); `mstp`
  (`rvt/vis.py:3039`); `slope_aspect` (`rvt/vis.py:263`); `sky_view_factor` (`rvt/vis.py:1317`); `get_raster_arr`
  (`rvt/default.py:4154`) (lines at `234b156`).
- Impact: the caller's DEM is modified. An integer DEM with `no_data` raises ValueError, so `save_hillshade`,
  `save_slrm`, `save_local_dominance`, `save_sky_illumination`, `save_msrm`, `save_mstp`, `save_multi_hillshade` (and
  the matching `get_*` with an integer array) fail on Int16/Int32 GeoTIFFs that carry a no-data value.
- Repro:
  ```python
  d = np.full((12, 12), 5.0, np.float32); d[3, 3] = -9999; v.slrm(d, 10, no_data=-9999); np.isnan(d[3, 3])  # -> True
  d = DEM.astype(np.int16); d[0, 0] = -9999; p = write_dem("dem_int.tif", d, nodata=-9999, etype=gdal.GDT_Int16)
  DefaultValues().save_hillshade(p, save_float=True, save_8bit=False)      # -> ValueError: cannot convert float NaN to integer
  ```
- Pinned: `tests/test_char_vis.py` (no_data in place, float input).
- Status: documented, not changed in 0.1.0.

### KI-18 horizon_generate_coarse_dem: the maximum starts at 0, and even scales wrap to the last row
- Where: `rvt/vis.py:1195` at `start` (`dem_convolve = np.zeros(dem_fine.shape)`), `rvt/vis.py:1198` (`np.roll` over
  offsets 0..1 for scale 2 reaches one row/column before the first, which the padding does not supply).
- Now: `horizon_generate_coarse_dem` (`rvt/vis.py:1642`, `:1645`) (lines at `234b156`).
- Impact: coarse levels of a DEM below 0 m are 0 (the horizon search of below-sea-level terrain sees a flat 0 m far
  field); the first coarse row/column takes the maximum with the opposite edge of the level.
- Repro:
  ```python
  v.horizon_generate_coarse_dem(np.full((12, 12), -5.0), 2, 0, 1, 3).max()      # -> 0.0
  d = np.zeros((12, 12)); d[-4, 3:-3] = 8.0                                       # last data row is 8 m
  v.horizon_generate_coarse_dem(d, 2, 0, 1, 3)[3, 3:6]                            # -> [4., 4., 4.] (first coarse row)
  ```
- Pinned: `tests/test_char_vis.py` (horizon_generate_coarse_dem tests).
- Status: documented, not changed in 0.1.0.

## rvt.blend_func

### KI-19 scale_within_0_and_1: the stretch limits are always 0 and 1, so scale_0_to_1 does not rescale
- Where: `rvt/blend_func.py:564` and `:567` at `start` (`np.nanmax(np.array(0, actual_min))`); caller
  `scale_0_to_1` at `rvt/blend_func.py:603-609`.
- Now: `scale_within_0_and_1` (`rvt/blend_func.py:1369`, `:1372`); `scale_0_to_1` (`rvt/blend_func.py:1437-1471`)
  (lines at `234b156`).
- Description: `np.array(0, actual_min)` passes the minimum as the *dtype* argument (numpy accepts a numpy scalar
  as dtype), so `norm_min_value` is always 0 and `norm_max_value` always 1, and `(x - 0) / (1 - 0)` returns the
  values unchanged.
- Impact: arrays outside [0, 1] with a range above 0.3 are not rescaled before blending (except the 8-bit branch,
  divided by 255); `render_images` then mixes values outside [0, 1] and `render_all_images` warns "Rendered image
  scale distorted".
- Repro:
  ```python
  bf.scale_within_0_and_1(np.array([-1.0, 0.5, 3.0]))                 # -> [-1.   0.5  3. ]
  bf.scale_0_to_1(np.array([-2.0, 3.0]))                              # -> [-2.  3.]
  ```
- Pinned: `tests/test_char_blend_func.py:369` (`test_scale_within_bounds_are_always_0_and_1`), `:452`. AUDIT #26.
- Status: fixed in 0.1.1 (d09e45d), see docs/FIXES_0_1_1.md

### KI-20 scale_within_0_and_1 overwrites NaN in the caller's array and loses the NaN mask
- Where: `rvt/blend_func.py:561` at `start` (the restoring lines were commented out upstream).
- Now: `scale_within_0_and_1` (`rvt/blend_func.py:1365`) (lines at `234b156`).
- Impact: no-data pixels become valid values (the minimum) in the result and in the caller's array.
- Repro:
  ```python
  img = np.array([-1.0, np.nan, 2.0]); out = bf.scale_within_0_and_1(img); img, out   # -> [-1. -1.  2.] [-1. -1.  2.]
  ```
- Pinned: `tests/test_char_blend_func.py:405`.
- Status: fixed in 0.1.1 (d09e45d), see docs/FIXES_0_1_1.md

### KI-21 scale_strict_0_to_1 raises ValueError for every non-constant array (chained comparison)
- Where: `rvt/blend_func.py:598` at `start` (`scaled[0 > scaled > -0.01] = 0`); NaN set to 0 in place at `:590`;
  caller `scale_0_to_1` at `:603-609`.
- Now: `scale_strict_0_to_1` (`rvt/blend_func.py:1432`, `:1424`); `scale_0_to_1` (`rvt/blend_func.py:1437-1471`)
  (lines at `234b156`).
- Description: the chained comparison is `(0 > scaled) and (scaled > -0.01)`; `and` asks for the truth value of an
  array. After the min-max stretch the minimum is 0, so the line is reached for every non-constant input and
  raises; NaN has already been replaced by 0 in the argument. A constant array gives all NaN.
- Impact: `scale_0_to_1` raises for a non-constant array with a range up to 0.3 that is not inside [0, 1]; a layer
  with a narrow range outside [0, 1] makes rendering fail.
- Repro:
  ```python
  img = np.array([1.0, 2.0, np.nan]); bf.scale_strict_0_to_1(img)    # -> ValueError: The truth value of an array ...
  img                                                                 # -> [1. 2. 0.]
  bf.scale_0_to_1(np.array([0.2, 0.3]))                               # -> ValueError
  ```
- Pinned: `tests/test_char_blend_func.py:418`, `:429`, `:459`.
- Status: fixed in 0.1.1 (d09e45d), see docs/FIXES_0_1_1.md

### KI-22 apply_opacity: opacity 1 means 100 %, values above 100 extrapolate
- Where: `rvt/blend_func.py:613` at `start` (`if opacity > 1: opacity = opacity / 100`).
- Now: `apply_opacity` (`rvt/blend_func.py:1499`) (lines at `234b156`).
- Description: values <= 1 are used as fractions, values > 1 as percent; 1 % cannot be requested and 1.5 means
  1.5 %. There is no range check; 150 gives `1.5 * a - 0.5 * b`.
- Repro:
  ```python
  a = np.array([0.9]); b = np.array([0.2])
  bf.apply_opacity(a, b, 1), bf.apply_opacity(a, b, 1.5), bf.apply_opacity(a, b, 150)   # -> [0.9] [0.2105] [1.25]
  ```
- Pinned: `tests/test_char_blend_func.py:485`, `:491`.
- Status: documented, not changed in 0.1.0.

### KI-23 advanced_normalization(normalization=None) raises AttributeError
- Where: `rvt/blend_func.py:206-210` at `start` (the `elif normalization is None` branch comes after `.lower()`).
- Now: `advanced_normalization` (`rvt/blend_func.py:466-470`) (lines at `234b156`).
- Impact: None cannot be used to skip normalization; the branch meant for it is unreachable.
- Repro: `bf.advanced_normalization(np.arange(4.0), 0, 1, None)   # -> AttributeError: 'NoneType' object has no attribute 'lower'`
- Pinned: `tests/test_char_blend_func.py:267`.
- Status: documented, not changed in 0.1.0.

### KI-24 advanced_normalization: the limit checks are case-sensitive
- Where: `rvt/blend_func.py:197`, `:201` at `start` (`normalization == "value"`), while the dispatch uses `.lower()`.
- Now: `advanced_normalization` (`rvt/blend_func.py:457`, `:461`) (lines at `234b156`).
- Description: "Value", the spelling of the shipped JSON files, skips the `min == max` and `min > max` checks.
- Impact: `min == max` gives an all-NaN layer, `min > max` an all -0.0 layer, without an error.
- Repro:
  ```python
  bf.advanced_normalization(np.arange(4.0), 1, 1, "Value")           # -> [nan nan nan nan]
  bf.advanced_normalization(np.arange(4.0), 2, 1, "Value")           # -> [-0. -0. -0. -0.]
  ```
- Pinned: `tests/test_char_blend_func.py:275`.
- Status: documented, not changed in 0.1.0.

### KI-25 normalize_image: abbreviations and "percent" are case-sensitive
- Where: `rvt/blend_func.py:625` at `start` (`normalization == "percent"`), `:632` ("mhs"), `:643-644` ("slp",
  "neg_opns").
- Now: `normalize_image` (`rvt/blend_func.py:1549`, `:1556`, `:1567-1568`) (lines at `234b156`).
- Description: the long visualization names are compared with `.lower()`, the abbreviations and "percent" exactly.
- Impact: "SLP" is not inverted; "Percent" raises "Unknown normalization type".
- Repro:
  ```python
  img = np.linspace(0.0, 10.0, 3)
  bf.normalize_image("slp", img.copy(), 0, 10, "value"), bf.normalize_image("SLP", img.copy(), 0, 10, "value")  # -> [1. 0.5 0.] [0. 0.5 1.]
  bf.normalize_image("Hillshade", img.copy(), 2, 3, "Percent")       # -> Exception: ... Unknown normalization type: Percent
  ```
- Pinned: `tests/test_char_blend_func.py:310`.
- Status: documented, not changed in 0.1.0.

### KI-26 cut_off_normalize: the limit checks are case-sensitive
- Where: `rvt/blend_func.py:658`, `:661` at `start` (`mode == "value"`).
- Now: `cut_off_normalize` (`rvt/blend_func.py:1617`, `:1620`) (lines at `234b156`).
- Impact: "Value" with equal or reversed limits gives all NaN or all -0.0 without an error.
- Repro:
  ```python
  bf.cut_off_normalize(np.arange(5.0), "Value", 3, 3)                # -> [nan nan nan nan nan]
  bf.cut_off_normalize(np.arange(5.0), "Value", 3, 1)                # -> [-0. -0. -0. -0. -0.]
  ```
- Pinned: `tests/test_char_blend_func.py:559`.
- Status: documented, not changed in 0.1.0.

### KI-27 cut_off_normalize accepts an unknown mode without an error
- Where: `rvt/blend_func.py:666-689` at `start` (no `else` branch).
- Now: `cut_off_normalize` (`rvt/blend_func.py:1625-1648`) (lines at `234b156`).
- Impact: a typo in `mode` returns the argument unchanged.
- Repro: `arg = np.arange(4.0); bf.cut_off_normalize(arg, "bogus", 0, 1) is arg   # -> True`
- Pinned: `tests/test_char_blend_func.py:572`.
- Status: documented, not changed in 0.1.0.

### KI-28 channel_min and channel_max: NaN in r propagates, NaN in g or b is ignored
- Where: `rvt/blend_func.py:244-259` at `start`.
- Now: `channel_min` and `channel_max` (`rvt/blend_func.py:572-632`) (lines at `234b156`).
- Description: the minimum starts as `r` and is replaced where `g < min` and then where `b < min`; a comparison with
  NaN is False.
- Impact: the luminosity blend treats NaN differently per band (only reachable with NaN in some bands).
- Repro:
  ```python
  r = np.array([np.nan, 0.5]); g = np.array([0.1, np.nan]); b = np.array([0.2, 0.3])
  bf.channel_min(r, g, b), np.minimum(np.minimum(r, g), b)            # -> [nan 0.3] [nan nan]
  ```
- Pinned: `tests/test_char_blend_func.py:639`.
- Status: documented, not changed in 0.1.0.

### KI-29 clip_color with only one of min_c / max_c raises TypeError
- Where: `rvt/blend_func.py:269` at `start` (both are computed only when both are None).
- Now: `clip_color` (`rvt/blend_func.py:677`) (lines at `234b156`).
- Repro: `bf.clip_color(np.full((3, 2, 2), 0.5), min_c=np.zeros((2, 2)))   # -> TypeError: '>' not supported ...`
- Pinned: `tests/test_char_blend_func.py:752`.
- Status: documented, not changed in 0.1.0.

### KI-30 clip_color: a gray pixel below 0 becomes NaN (0 / 0); CRIM with a gray colormap gets NaN pixels
- Where: `rvt/blend_func.py:231` at `start` (`matrix_eq_min_lt_zero`, called at `:274-276`); consequence in the
  luminosity layer of `color_relief_image_map`, `rvt/blend.py:1327-1329`.
- Now: `matrix_eq_min_lt_zero` (`rvt/blend_func.py:537`); `clip_color` (`rvt/blend_func.py:682-684`);
  `color_relief_image_map` (`rvt/blend.py:1956-1958`) (lines at `234b156`).
- Description: for R = G = B < 0, `lum == min`, and the correction divides 0 by 0.
- Impact: NaN pixels in luminosity blends onto gray backgrounds instead of 0; `color_relief_image_map` with a gray
  colormap has a larger NaN mask than the DEM.
- Repro:
  ```python
  c = np.zeros((3, 1, 2)); c[:, 0, 0] = -0.2; c[:, 0, 1] = [-0.2, 0.1, 0.4]
  bf.clip_color(c)[:, 0, :]                                          # -> column 0 NaN in all bands; column 1: 0, 0.0531, 0.1062
  yy, xx = np.mgrid[0:40, 0:44]; dem = (3 * np.sin(xx / 3.0) * np.cos(yy / 2.0)).astype(np.float32)
  out = rb.color_relief_image_map(dem, 1.0, default=DefaultValues(), colormap="gray", min_colormap_cut=None, max_colormap_cut=None)
  int(np.isnan(out).any(axis=0).sum())                               # -> 170 (of 1760 finite DEM pixels)
  ```
- Pinned: `tests/test_char_blend_func.py:769`, `tests/test_char_blend.py:1153`.
- Status: documented, not changed in 0.1.0.

### KI-31 blend_soft_light: NaN in the active layer keeps the background value
- Where: `rvt/blend_func.py:407-408` at `start` (reference masks), same masks in `_soft_light_masks` (`:388-389`).
- Now: `_blend_soft_light_ref` (`rvt/blend_func.py:980-981`); `_soft_light_masks` (`rvt/blend_func.py:929-931`) (lines
  at `234b156`).
- Description: the branch is chosen by `active`; NaN matches neither `active < 0.5` nor `active >= 0.5`.
- Impact: no-data in the active layer is not propagated (overlay does propagate NaN of `active`).
- Repro: `bf.blend_soft_light(np.array([np.nan, 0.25]), np.array([0.5, 0.5]))   # -> [0.5   0.375]`
- Pinned: `tests/test_char_blend_func.py:866`.
- Status: documented, not changed in 0.1.0.

### KI-32 equation_blend returns None for "normal", "luminosity" and unknown names
- Where: `rvt/blend_func.py:442-450` at `start` (no `else` branch); consequence in `blend_multi_dim_images`
  (`:474-487`).
- Now: `equation_blend` (`rvt/blend_func.py:1049-1079`); `blend_multi_dim_images` (`rvt/blend_func.py:1153-1166`)
  (lines at `234b156`).
- Impact: `blend_multi_dim_images` with such a mode returns None (2D) or an all-NaN array (RGB) without an error.
  `blend_images` is not affected (it handles normal and luminosity itself).
- Repro:
  ```python
  bf.equation_blend("normal", np.array([0.2]), np.array([0.5]))                  # -> None
  bf.blend_multi_dim_images("bogus", np.zeros((3, 1, 2)), np.zeros((1, 2)))      # -> (3, 1, 2) array of NaN
  ```
- Pinned: `tests/test_char_blend_func.py:883`, `:937`.
- Status: documented, not changed in 0.1.0.

### KI-33 overlay and soft light overwrite the background; their layer opacity has no effect
- Where: `rvt/blend_func.py:379-380`, `:409-411` at `start` (reference), `:363-364` (blocked path); used by
  `BlenderCombination.render_all_images` at `rvt/blend.py:621` and `:626-630` (`top = blend_images(...)`, then
  `_render_images(top, background, ...)`).
- Now: `_blend_overlay_ref` (`rvt/blend_func.py:890-891`); `_blend_soft_light_ref` (`rvt/blend_func.py:982-984`);
  `_blend_in_place_blocked` (`rvt/blend_func.py:866-867`); `BlenderCombination.render_all_images` (`rvt/blend.py:979`,
  `:984-988`) (lines at `234b156`).
- Description: the result is written into `background` and returned; `render_images` then mixes `top` with the
  same array, so `top * p + top * (1 - p) = top` up to rounding.
- Impact: in `render_all_images` the opacity of every overlay and soft_light layer acts as 100 %: CRIM overlay
  50 %, e3MSTP soft light 70 %, e4MSTP MSTP overlay 90 %. The caller's background array changes.
- Repro:
  ```python
  b = np.full((1, 2), 0.25); out = bf.blend_overlay(np.full((1, 2), 0.25), b)
  out is b, b, bf.render_images(out, b, 30)          # -> True [[0.125 0.125]] [[0.125 0.125]] (0.2125 expected for 30 %)
  ```
- Pinned: `tests/test_char_blend_func.py:922`; `tests/test_char_e3e4_recipe.py:79`, `:325` (e3MSTP soft light 70 % and
  65 % differ by less than 1e-6). AUDIT #31.
- Status: fixed in 0.1.1 (c2456b1), see docs/FIXES_0_1_1.md

### KI-34 overlay and soft light of an RGB active layer on a gray background chain the bands
- Where: `rvt/blend_func.py:478-481` at `start` (`blend_multi_dim_images`, case `a_rgb and not b_rgb`), with KI-33.
- Now: `blend_multi_dim_images` (`rvt/blend_func.py:1157-1160`) (lines at `234b156`).
- Description: each band writes its result into the same gray `background`, so band 1 is blended onto the result
  of band 0 and band 2 onto that of band 1.
- Impact: wrong colors for an RGB layer (for example a colored hillshade) in overlay or soft_light mode over a gray
  rendered image; the CRIM, e3MSTP and e4MSTP recipes do not reach this case.
- Repro:
  ```python
  a = np.full((3, 1, 1), 0.25); b = np.full((1, 1), 0.25)
  bf.blend_multi_dim_images("overlay", a, b)[:, 0, 0]   # -> [0.125 0.0625 0.03125] (0.125 per band expected)
  ```
- Pinned: not pinned.
- Status: fixed in 0.1.1 (c2456b1), see docs/FIXES_0_1_1.md

## rvt.blend

### KI-35 create_blender_file_example: the default path has a backslash
- Where: `rvt/blend.py:63` at `start` (`r"settings\blender_file_example.json"`, with `os.makedirs` at `:67-68`).
- Now: `create_blender_file_example` (`rvt/blend.py:138`, `:142-143`) (lines at `234b156`).
- Description: on POSIX `os.path.dirname` of that string is "" and `os.makedirs("")` raises FileNotFoundError.
- Impact: `create_blender_file_example()` without a path fails on Linux and macOS.
- Repro (POSIX path handling on any OS):
  ```python
  import posixpath; p = posixpath.dirname(r"settings\blender_file_example.json"); repr(p)   # -> ''
  os.makedirs(p)                                                                       # -> FileNotFoundError
  ```
- Pinned: `tests/test_char_blend.py:94` (Windows behaviour).
- Status: documented, not changed in 0.1.0.

### KI-36 BlenderLayer.check_data: the opacity check can never fail
- Where: `rvt/blend.py:149` at `start` (`if 0 > self.opacity > 100`).
- Now: `BlenderLayer.check_data` (`rvt/blend.py:280`) (lines at `234b156`).
- Description: the chained comparison needs opacity < 0 and > 100 at once.
- Impact: any opacity is accepted; values above 100 extrapolate in rendering (KI-22).
- Repro:
  ```python
  lay = rb.BlenderLayer("Hillshade", "value", 0, 1, "normal", opacity=150); lay.check_data(); lay.opacity   # -> 150
  c = rb.BlenderCombination(); c.create_layer("x", "value", 0, 1, "normal", 150, image=np.full((1, 2), 0.6))
  c.create_layer("y", "value", 0, 1, "normal", 100, image=np.full((1, 2), 0.4)); c.render_all_images()   # -> [[0.70000005 0.70000005]]
  ```
- Pinned: `tests/test_char_blend.py:157`.
- Status: fixed in 0.1.1 (a37ab44), see docs/FIXES_0_1_1.md

### KI-37 BlenderLayer.check_data accepts min == max for "value"
- Where: `rvt/blend.py:143` at `start` (only `min > max` raises).
- Now: `BlenderLayer.check_data` (`rvt/blend.py:274`) (lines at `234b156`).
- Impact: with "value" `advanced_normalization` raises later; with "Value" (KI-24) a combination file with equal
  limits renders an all-NaN layer without an error.
- Repro:
  ```python
  c = rb.BlenderCombination(); c.create_layer("Hillshade", "Value", 1, 1, image=np.array([[0.0, 1.0, 2.0]]))
  c.check_data(); c.render_all_images()                                    # -> [[nan nan nan]]
  ```
- Pinned: `tests/test_char_blend.py:161`.
- Status: documented, not changed in 0.1.0.

### KI-38 BlenderCombination.layers_info never prints max_colormap_cut
- Where: `rvt/blend.py:225` at `start` (the format string ends with "max_colormap_cut" without "= {}").
- Now: `BlenderCombination.layers_info` (`rvt/blend.py:438`) (lines at `234b156`).
- Impact: cosmetic; the value is missing from the description.
- Repro:
  ```python
  c = rb.BlenderCombination(); c.create_layer("Hillshade", "value", 0, 1, colormap="gray", min_colormap_cut=0.1, max_colormap_cut=0.9)
  c.layers_info()                                                          # -> ... min_colormap_cut = 0.1, max_colormap_cut'
  ```
- Pinned: `tests/test_char_blend.py:237`.
- Status: documented, not changed in 0.1.0.

### KI-39 BlenderCombination.read_from_json stores colormap cuts as strings
- Where: `rvt/blend.py:272`, `:278` at `start` (`str(layer["min_colormap_cut"])`).
- Now: `BlenderCombination.read_from_json` (`rvt/blend.py:521`, `:527`) (lines at `234b156`).
- Impact: a combination file with colormap cuts fails to render (TypeError in `gray_scale_to_color_ramp`); a save /
  read round trip turns numbers into strings.
- Repro:
  ```python
  c = rb.BlenderCombination()
  c.read_from_json({"combination": {"name": "t", "layers": [{"visualization_method": "Hillshade", "norm": "value",
      "min": 0, "max": 1, "blend_mode": "normal", "opacity": 100, "colormap": "gray", "min_colormap_cut": 0.2}]}})
  repr(c.layers[0].min_colormap_cut)                                       # -> "'0.2'"
  bf.gray_scale_to_color_ramp(np.zeros((2, 2)), "gray", c.layers[0].min_colormap_cut)   # -> TypeError
  ```
- Pinned: `tests/test_char_blend.py:287`, `:358`.
- Status: documented, not changed in 0.1.0.

### KI-40 compare_2_combinations ignores normalization, min and max (possibly intended)
- Where: `rvt/blend.py:789-800` at `start` (the three comparisons were commented out upstream at `:791-796`; the
  commented-out code was removed in 0.1.0 by commit 932172f and replaced by a one-line comment).
- Now: `compare_2_combinations` (`rvt/blend.py:1207-1213`; the one-line comment at `rvt/blend.py:1209`) (lines at
  `234b156`).
- Description: only visualization name, blend mode and opacity are compared. This may be intended, because
  `TerrainSettings.apply_terrain` changes the limits of preset combinations, which then still match their preset.
- Impact: `BlenderCombinations.combination_in_combinations` reports combinations with different limits as equal.
- Repro:
  ```python
  a = rb.BlenderCombination(); a.create_layer("Hillshade", "value", 0, 1)
  b = rb.BlenderCombination(); b.create_layer("Hillshade", "perc", 2, 3); rb.compare_2_combinations(a, b)   # -> True
  ```
- Pinned: `tests/test_char_blend.py:407`.
- Status: documented, not changed in 0.1.0.

### KI-41 TerrainSettings.read_from_json writes the sky illumination and local dominance stretches into neg_opns_stretch
- Where: `rvt/blend.py:1119`, `:1124` at `start` (copies of `:1114`).
- Now: `TerrainSettings.read_from_json` (`rvt/blend.py:1675`, `:1680`, `:1670`) (lines at `234b156`).
- Description: `sim_stretch` and `ld_stretch` are never set; `neg_opns_stretch` gets the last of the three sections.
- Impact: with `settings/default_terrains_settings.json`, `apply_terrain` gives negative openness layers the local
  dominance limits (0.5, 1.8) instead of (60, 95), and sky illumination and local dominance layers keep their own.
- Repro:
  ```python
  doc = {"terrain_settings": {"name": "t", "Openness - Negative": {"stretch": {"min": 60, "max": 95}},
         "Sky illumination": {"stretch": {"min": 0.25, "max": 0.3}}, "Local dominance": {"stretch": {"min": 0.5, "max": 1.8}}}}
  ts = rb.TerrainSettings(); ts.read_from_json(doc); ts.neg_opns_stretch, ts.sim_stretch, ts.ld_stretch   # -> (0.5, 1.8) None None
  ```
- Pinned: `tests/test_char_blend.py:563`, `:616`.
- Status: documented, not changed in 0.1.0.

### KI-42 TerrainsSettings.read_from_file appends instead of replacing
- Where: `rvt/blend.py:1247-1256` at `start` (no reset of `terrains_settings`; `BlenderCombinations.read_from_file`
  does reset its list).
- Now: `TerrainsSettings.read_from_file` (`rvt/blend.py:1828-1854`) (lines at `234b156`).
- Impact: reading the file twice holds every preset twice.
- Repro (from the repository root):
  ```python
  ts = rb.TerrainsSettings(); f = "settings/default_terrains_settings.json"; ts.read_from_file(f); ts.read_from_file(f)
  [t.name for t in ts.terrains_settings]          # -> ['general', 'flat', 'steep', 'general', 'flat', 'steep']
  ```
- Pinned: `tests/test_char_blend.py:635`.
- Status: documented, not changed in 0.1.0.

### KI-43 render_all_images: a colormap on a layer without an in-memory image raises AttributeError
- Where: `rvt/blend.py:594` at `start` (`len(image.shape)` with `image = layer.image`, which is None for layers read
  from `image_path` and for layers computed with `save_visualizations=True`).
- Now: `BlenderCombination.render_all_images` (`rvt/blend.py:952`) (lines at `234b156`).
- Impact: colored layers work only with `image` given or computed in memory.
- Repro:
  ```python
  p = write_dem("vis.tif", np.ones((2, 3), np.float32))
  c = rb.BlenderCombination(); c.create_layer("Hillshade", "value", 0, 1, colormap="gray", image_path=p)
  c.render_all_images()                         # -> AttributeError: 'NoneType' object has no attribute 'shape'
  ```
- Pinned: `tests/test_char_blend.py:802`.
- Status: documented, not changed in 0.1.0.

### KI-44 render_all_images(save_visualizations=True): saved to the render directory, read from the DEM directory
- Where: `rvt/blend.py:431-433`, `:442-444` at `start` and the same pattern for every visualization up to `:583`
  (`default.save_*(custom_dir=save_render_directory)`, then `default.get_*_path(self.dem_path)`).
- Now: `BlenderCombination.render_all_images` (`rvt/blend.py:789-791`, `:800-802`, `:941`) (lines at `234b156`).
- Impact: with `save_render_path` in another directory than the DEM, rendering fails after writing the
  visualization (AttributeError, or RuntimeError with GDAL exceptions on).
- Repro:
  ```python
  p = write_dem("dem.tif", np.random.default_rng(0).normal(size=(20, 24)).astype(np.float32)); os.makedirs("out")
  c = rb.BlenderCombination(dem_path=p); c.create_layer("Hillshade", "value", 0, 1)
  c.render_all_images(save_visualizations=True, save_render_path=os.path.join(TMP, "out", "r.tif"))
  # -> ERROR 4: ...dem_HS_A315_H35.tif: No such file or directory; AttributeError: 'NoneType' object has no attribute 'GetGeoTransform'
  ```
- Pinned: `tests/test_char_blend.py:993`.
- Status: documented, not changed in 0.1.0.

### KI-45 color_relief_image_map sets svf_r_max = 10 on the caller's DefaultValues; default instances are shared
- Where: `rvt/blend.py:1303` at `start` (`default.svf_r_max = 10`); default arguments at `:1266`, `:1339`, `:1387`
  (`default=rvt.default.DefaultValues()` is evaluated once, at import).
- Now: `color_relief_image_map` (`rvt/blend.py:1932`, `:1876`); `e3mstp` (`rvt/blend.py:1968`); `e4mstp`
  (`rvt/blend.py:2040`) (lines at `234b156`).
- Description: the function changes the object it receives, and `e3mstp` passes its own `default` on. The default
  argument is one object shared by all calls of each function.
- Impact: after `e3mstp` or `color_relief_image_map`, SVF and openness computed with the same object use
  r = 10 px (5 m at 0.5 m/px) regardless of the user's setting.
- Repro:
  ```python
  d = DefaultValues(); d.svf_r_max = 25
  rb.color_relief_image_map(np.random.default_rng(0).normal(size=(30, 30)).astype(np.float32), 1.0, default=d); d.svf_r_max   # -> 10
  ```
- Pinned: `tests/test_char_blend.py:1186`, `:1259-1267`.
- Status: fixed in 0.1.1 (c72646c), see docs/FIXES_0_1_1.md

### KI-46 render_all_images: the blend mode and opacity of the bottom layer are never used
- Where: `rvt/blend.py:600-605` at `start` (the first rendered layer becomes `rendered_image` as it is).
- Now: `BlenderCombination.render_all_images` (`rvt/blend.py:958-963`) (lines at `234b156`).
- Impact: a bottom layer with opacity below 100 % is rendered at full strength (there is nothing below it to mix
  with).
- Repro:
  ```python
  for op in (100, 0):
      c = rb.BlenderCombination(); c.create_layer("only", "value", 0, 1, "multiply", op, image=np.full((1, 2), 0.6))
      print(op, c.render_all_images())          # -> 100 [[0.6 0.6]] and 0 [[0.6 0.6]]
  ```
- Pinned: `tests/test_char_e3e4_recipe.py:321-323`. AUDIT #35.
- Status: documented, not changed in 0.1.0.

### KI-47 e4mstp: the slope in radians is normalized over 0..55
- Where: `rvt/blend.py:1487` at `start` (`rvt.vis.slope_aspect` with its default `output_units="radian"`) and `:1512`
  (`minimum=0, maximum=55`).
- Now: `e4mstp` (`rvt/blend.py:2160`, `:2185`) (lines at `234b156`).
- Impact: slopes of 0..1.5 rad map to 0..0.027; after the inversion of "Slope gradient" the slope layer stays
  within about 0.97..1, so its Reds_r coloring is almost constant.
- Repro:
  ```python
  dem = np.random.default_rng(0).normal(scale=5.0, size=(40, 40)).astype(np.float32)
  s = v.slope_aspect(dem=dem, resolution_x=1.0, resolution_y=1.0)["slope"]; float(np.nanmax(s)), float(1 - np.nanmax(s) / 55)  # -> 1.5005..., 0.9727...
  ```
- Pinned: `tests/test_char_e3e4_recipe.py:157`. AUDIT #32.
- Status: fixed in 0.1.1 (33a2bd5), see docs/FIXES_0_1_1.md

### KI-48 e3mstp: the upstream docstring named Reds_r(0.5-1); the code uses OrRd 0..1
- Where: docstring `rvt/blend.py:1344` against code `rvt/blend.py:1366-1367` at `start`.
- Now: `e3mstp` (`rvt/blend.py:1968`; the docstring states the discrepancy at `rvt/blend.py:2006-2007`, the code
  is at `rvt/blend.py:2019-2020`) (lines at `234b156`).
- Description: the 0.1.0 docstring describes the code (OrRd, cuts 0..1) and states the discrepancy (commit a9efeb7).
  Which colormap is intended is a question for the upstream authors.
- Repro: `import inspect; 'colormap="OrRd", min_colormap_cut=0, max_colormap_cut=1' in inspect.getsource(rb.e3mstp)   # -> True`
- Pinned: `tests/test_char_e3e4_recipe.py:131`. AUDIT #33.
- Status: documented, not changed in 0.1.0.

### KI-49 ve_factor is not applied to MSTP (DefaultValues.get_mstp) nor to the slope layers of CRIM, e3MSTP, e4MSTP
- Where: `rvt/default.py:2060-2062` at `start` (`get_mstp` calls `rvt.vis.mstp` without `ve_factor`, although `mstp`
  has the parameter); `rvt/blend.py:1368`, `:1442` (e3/e4 use `get_mstp`); `rvt/blend.py:1316-1321` (CRIM slope) and
  `:1487` (e4 slope) call `rvt.vis.slope_aspect` without `ve_factor`; `rvt/blend.py:1436-1439` (e4 flat-terrain SVF
  uses a new `DefaultValues`, ve_factor 1).
- Now: `DefaultValues.get_mstp` (`rvt/default.py:3473-3508`); `e3mstp` (`rvt/blend.py:2021`); `e4mstp`
  (`rvt/blend.py:2115`, `:2160`, `:2109-2112`); `color_relief_image_map` (`rvt/blend.py:1945-1950`) (lines at
  `234b156`).
- Impact: with `default.ve_factor != 1` only some layers of the composites are exaggerated, and MSTP does not follow
  the vertical exaggeration of the other `get_*` methods (its effect on MSTP would be small: about 2e-7 for
  `ve_factor` 5 on a 0-10 m test DEM, see the `mstp` Notes and KI-05).
- Repro:
  ```python
  dem = np.random.default_rng(0).normal(scale=2.0, size=(40, 40)).astype(np.float32)
  d1 = DefaultValues(); d5 = DefaultValues(); d5.ve_factor = 5
  np.array_equal(d1.get_mstp(dem), d5.get_mstp(dem), equal_nan=True)                    # -> True (ignored)
  np.array_equal(d1.get_slope(dem, 1, 1), d5.get_slope(dem, 1, 1), equal_nan=True)      # -> False (get_slope applies it)
  ```
- Pinned: `tests/test_char_e3e4_recipe.py:216`; `tests/test_char_default.py` (get_mstp, line 519). AUDIT #34.
- Status: documented, not changed in 0.1.0.

### KI-50 render_all_images: a missing dem_arr is not reported when dem_resolution is set
- Where: `rvt/blend.py:401` at `start` (`self.dem_arr is None and self.dem_resolution is None`).
- Now: `BlenderCombination.render_all_images` (`rvt/blend.py:759`) (lines at `234b156`).
- Description: the check uses `and`; with only the resolution set, no explanatory exception is raised and the
  computation fails later inside `rvt.vis`.
- Repro:
  ```python
  c = rb.BlenderCombination(dem_resolution=1.0); c.create_layer("Hillshade", "value", 0, 1)
  c.render_all_images()                         # -> AttributeError: 'NoneType' object has no attribute 'ndim'
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

## rvt.default

### KI-51 Tile settings are not saved or read with the settings file
- Where: `rvt/default.py:327` at `start` (`save_default_to_file` writes no `tile_size`, `tile_size_limit`) and
  `rvt/default.py:579` (`read_default_from_file` reads none).
- Now: `DefaultValues.save_default_to_file` (`rvt/default.py:413`); `DefaultValues.read_default_from_file`
  (`rvt/default.py:697`) (lines at `234b156`).
- Impact: a settings round trip silently resets the tile settings to the defaults.
- Repro:
  ```python
  dv = DefaultValues(); dv.tile_size = (100, 100); dv.save_default_to_file("s.json")
  dv2 = DefaultValues(); dv2.read_default_from_file("s.json"); dv2.tile_size, "tile_size" in open("s.json").read()  # -> (4000, 4000) False
  ```
- Pinned: `tests/test_char_default.py::test_json_round_trip_of_every_saved_attribute` (NOT_SAVED).
- Status: documented, not changed in 0.1.0.

### KI-52 save_default_to_file: the default path has a backslash
- Where: `rvt/default.py:569` at `start` (`file_path = r"settings\default_settings.json"`), then `:573-574`
  (`os.makedirs(os.path.dirname(file_path))`).
- Now: `DefaultValues.save_default_to_file` (`rvt/default.py:687`, `:691-692`) (lines at `234b156`).
- Description: on Windows this is `settings/default_settings.json`; on Linux and macOS the backslash is part of the
  file name, `dirname` is "" and `os.makedirs("")` raises FileNotFoundError when the file does not exist yet.
- Impact: `save_default_to_file()` without a path fails on non-Windows systems. The frozen test that pins the Windows
  behaviour is expected to fail on Linux and macOS for this reason (not run there in this project).
- Repro: `posixpath.dirname(r"settings\default_settings.json")   # -> ''` (then `os.makedirs("")` -> FileNotFoundError)
- Pinned: `tests/test_char_default.py::test_json_default_path_is_settings_default_settings_json_in_cwd`.
- Status: documented, not changed in 0.1.0.

### KI-53 .txt settings: "sun_elevation" is tested twice, mhs_sun_el cannot be set
- Where: `rvt/default.py:607` (`hs_sun_el`) and `:613` (`mhs_sun_el`) at `start`, in one if/elif chain.
- Now: `DefaultValues.read_default_from_file` (`rvt/default.py:775`, `:781`) (lines at `234b156`).
- Repro:
  ```python
  open("s.txt", "w").write("sun_elevation = 50\n"); dv = DefaultValues(); dv.read_default_from_file("s.txt")
  dv.hs_sun_el, dv.mhs_sun_el                   # -> 50 35
  ```
- Pinned: `tests/test_char_default.py` (comment at line 288).
- Status: documented, not changed in 0.1.0.

### KI-54 read_default_from_file ignores an unknown extension without an error
- Where: `rvt/default.py:582-583` and `:670` at `start` (only `.txt` and `.json` are handled; no `else`).
- Now: `DefaultValues.read_default_from_file` (`rvt/default.py:750-751`, `:838`) (lines at `234b156`).
- Impact: a typo in the extension, or a missing file with another extension, leaves all parameters unchanged.
- Repro: `dv = DefaultValues(); dv.read_default_from_file("does_not_exist.yaml"); dv.overwrite   # -> 0, no error`
- Pinned: `tests/test_char_default.py::test_txt_wrong_lines_warn_and_unknown_extension_is_ignored`.
- Status: documented, not changed in 0.1.0.

### KI-55 A json round trip turns the MSRM sizes into float and changes the MSRM file name
- Where: `rvt/default.py:713-714` at `start` (`float(...)`; the defaults are int 0 and 20) and `:986-990`
  (formatted with `str`).
- Now: `DefaultValues.read_default_from_file` (`rvt/default.py:881-882`); `DefaultValues.get_msrm_file_name`
  (`rvt/default.py:1520-1524`) (lines at `234b156`).
- Impact: the output name depends on whether the settings came from a file (`M0-20` against `M0.0-20.0`), so
  existing outputs are not found and are computed again under a second name.
- Repro:
  ```python
  dv = DefaultValues(); before = dv.get_msrm_file_name("dem.tif"); dv.save_default_to_file("s.json"); dv.read_default_from_file("s.json")
  before, dv.get_msrm_file_name("dem.tif")      # -> dem_MSRM_F_M0-20_S2.tif dem_MSRM_F_M0.0-20.0_S2.tif
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-56 get_neg_opns_file_name uses "_NRstrong" where the other names use "_NRhigh"
- Where: `rvt/default.py:936` at `start` (`svf_noise == 3`); SVF, ASVF and positive openness use `_NRhigh`
  (`:872`, `:896`, `:916`).
- Now: `DefaultValues.get_neg_opns_file_name` (`rvt/default.py:1374`); `DefaultValues.get_svf_file_name`
  (`rvt/default.py:1209`); `DefaultValues.get_asvf_file_name` (`rvt/default.py:1267`);
  `DefaultValues.get_opns_file_name` (`rvt/default.py:1320`) (lines at `234b156`).
- Repro: `dv = DefaultValues(); dv.svf_noise = 3; dv.get_svf_file_name("dem.tif"), dv.get_neg_opns_file_name("dem.tif")`
  `# -> dem_SVF_R10_D16_NRhigh.tif dem_OPEN-NEG_R10_D16_NRstrong.tif`
- Pinned: `tests/test_char_default.py` (line 388).
- Status: documented, not changed in 0.1.0.

### KI-57 Output file names cut the DEM name at its first dot
- Where: `rvt/default.py:797` at `start` and the same expression in every `get_*_file_name`
  (`os.path.basename(dem_path).split(".")[0]`); `create_log_file` (`:2300`) uses `splitext` instead.
- Now: `DefaultValues.get_shadow_file_name` (`rvt/default.py:979`); `DefaultValues.create_log_file`
  (`rvt/default.py:3885`) (lines at `234b156`).
- Impact: DEMs whose names differ only after the first dot get the same output names; outputs overwrite each other
  or are skipped as existing.
- Repro: `DefaultValues().get_slope_file_name("my.site_v2.tif")   # -> my_SLOPE.tif`
- Pinned: `tests/test_char_default.py::test_dem_name_is_cut_at_the_first_dot`.
- Status: documented, not changed in 0.1.0.

### KI-58 float_to_8bit: the multi hillshade mode test reads the slope stretch setting
- Where: `rvt/default.py:1116` at `start`:
  `if self.mhs_bytscl[0].lower() == "percent" or self.slp_bytscl[0].lower() == "perc":` (`mhs_bytscl` is probably
  meant in the second operand).
- Now: `DefaultValues.float_to_8bit` (`rvt/default.py:1799`) (lines at `234b156`).
- Impact: setting the slope stretch to "perc" switches the 8-bit multi hillshade to a percent stretch with the
  `mhs_bytscl` numbers.
- Repro:
  ```python
  dv = DefaultValues(); a = dv.float_to_8bit(DEM.copy(), V.MULTI_HILLSHADE, x_res=1, y_res=1)
  dv.slp_bytscl = ("perc", 2.0, 2.0); b = dv.float_to_8bit(DEM.copy(), V.MULTI_HILLSHADE, x_res=1, y_res=1)
  dv.mhs_bytscl, np.array_equal(a, b)           # -> ('value', 0.0, 1.0) False
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-59 The 8-bit multi hillshade ignores ve_factor
- Where: `rvt/default.py:1110-1115` at `start` (`rvt.vis.hillshade` called without `ve_factor`); the float multi
  hillshade passes it (`:1400-1402`).
- Now: `DefaultValues.float_to_8bit` (`rvt/default.py:1793-1798`); `DefaultValues.get_multi_hillshade`
  (`rvt/default.py:2307-2309`) (lines at `234b156`).
- Repro:
  ```python
  dv = DefaultValues(); a = dv.float_to_8bit(DEM.copy(), V.MULTI_HILLSHADE, x_res=1, y_res=1)
  dv.ve_factor = 5; b = dv.float_to_8bit(DEM.copy(), V.MULTI_HILLSHADE, x_res=1, y_res=1); np.array_equal(a, b)   # -> True
  ```
- Pinned: `tests/test_char_default.py::test_float_to_8bit_multi_hillshade_from_the_dem`.
- Status: documented, not changed in 0.1.0.

### KI-60 float_to_8bit clips the caller's array; calculate_visualization and tiled float rasters are clipped
- Where: `rvt/default.py:1097` ff. at `start` -> `rvt.blend_func.normalize_image` -> `normalize_lin`, which clips in
  place (`rvt/blend_func.py:146-147` blocked path, `:158-159` reference path); `rvt/default.py:2260-2275`
  (`calculate_visualization`: `vis_float_arr = vis_arr`, then `float_to_8bit(vis_arr)`).
- Now: `DefaultValues.float_to_8bit` (`rvt/default.py:1780`); `normalize_lin` (`rvt/blend_func.py:325-326`,
  `:337-338`); `DefaultValues.calculate_visualization` (`rvt/default.py:3814-3829`) (lines at `234b156`).
- Impact: the float array passed in changes. With `save_float` and `save_8bit` both true, `calculate_visualization`
  returns a float result limited to the stretch range (e.g. slope 0..51°), and so does the float raster written by
  `rvt.tile.save_rvt_visualization_tile_by_tile`. The untiled `save_*` methods are not affected (they write the float
  raster before stretching).
- Repro:
  ```python
  arr = np.array([[0.0, 30.0, 60.0, 90.0]], dtype=np.float32); DefaultValues().float_to_8bit(arr, V.SLOPE); arr  # -> [[ 0. 30. 51. 51.]]
  dv = DefaultValues(); f, _ = dv.calculate_visualization(V.SLOPE, DEM * 10, 1, 1, save_float=True, save_8bit=True)
  float(f.max())                                # -> 51.0
  ```
- Pinned: `tests/test_char_default.py::test_float_to_8bit_modifies_its_argument`,
  `tests/test_char_default.py::test_calculate_visualization_with_both_flags_clips_the_float_result`,
  `tests/test_char_tile.py` (line 271).
- Status: documented, not changed in 0.1.0.

### KI-61 save_*: the "outputs exist" branch for 8-bit only is unreachable
- Where: `rvt/default.py:1245` at `start` (`save_slope`), `:1335` and `:1341` (`save_hillshade`), `:1440`
  (`save_multi_hillshade`), `:1524` (`save_slrm`), `:1769` (`save_neg_opns`), `:1855` (`save_sky_illumination`),
  `:1940` (`save_local_dominance`), `:2018` (`save_msrm`), `:2098` (`save_mstp`):
  `elif not save_float and not save_8bit:` follows a check that already raised for that case;
  `not save_float and save_8bit` was meant (`save_sky_view_factor`, `:1631`, has it). `save_hillshade` also has no
  branch for float + 8-bit without shadow.
- Now: `DefaultValues.save_slope` (`rvt/default.py:1998`); `DefaultValues.save_hillshade` (`rvt/default.py:2208`,
  `:2214`); `DefaultValues.save_multi_hillshade` (`rvt/default.py:2395`); `DefaultValues.save_slrm`
  (`rvt/default.py:2550`); `DefaultValues.save_neg_opns` (`rvt/default.py:2959`);
  `DefaultValues.save_sky_illumination` (`rvt/default.py:3122`); `DefaultValues.save_local_dominance`
  (`rvt/default.py:3279`); `DefaultValues.save_msrm` (`rvt/default.py:3431`); `DefaultValues.save_mstp`
  (`rvt/default.py:3590`); `DefaultValues.save_sky_view_factor` (`rvt/default.py:2748`) (lines at `234b156`).
- Impact: with only the 8-bit output requested and present and `overwrite = 0`, the DEM is read and the
  visualization computed for nothing, and 1 (not 0) is returned.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM); dv = DefaultValues()
  dv.save_slope(p, save_float=False, save_8bit=True), dv.save_slope(p, save_float=False, save_8bit=True)   # -> 1 1 (0 expected the second time)
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-62 save_sky_view_factor returns early only when all three results are requested
- Where: `rvt/default.py:1622-1634` at `start` (`os.path.isfile` of the SVF, ASVF and openness paths; the path of a
  result that is not requested is "", `:1594-1599`).
- Now: `DefaultValues.save_sky_view_factor` (`rvt/default.py:2739-2751`, `:2711-2716`) (lines at `234b156`).
- Impact: as KI-61; SVF is recomputed although its output exists.
- Repro: `p = write_dem("dem.tif", DEM); [DefaultValues().save_sky_view_factor(p, save_svf=True, save_float=True, save_8bit=False) for _ in range(2)]   # -> [1, 1]`
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-63 save_hillshade writes the shadow next to the DEM, not into custom_dir
- Where: `rvt/default.py:1325` at `start`: `shadow_path = os.path.join(custom_dir, self.get_shadow_path(dem_path))`;
  `get_shadow_path` returns a path in the DEM directory, and `os.path.join` with an absolute second argument drops
  `custom_dir`.
- Now: `DefaultValues.save_hillshade` (`rvt/default.py:2198`) (lines at `234b156`).
- Impact: the shadow is written next to the DEM (for a relative `dem_path`: under `custom_dir/<dem dir>/`), while
  `create_log_file` lists it in `custom_dir`.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM); os.mkdir("out")
  DefaultValues().save_hillshade(p, custom_dir="out", save_float=True, save_8bit=False, save_shadow=True)
  sorted(os.listdir("out")), os.path.isfile("dem_shadow_A315_H35.tif")   # -> ['dem_HS_A315_H35.tif'] True
  ```
- Pinned: `tests/test_char_default.py` (line 996).
- Status: documented, not changed in 0.1.0.

### KI-64 save_hillshade: the shadow is all zero when the DEM has no-data pixels
- Where: `rvt/default.py:1373` at `start` (`get_hillshade(dem_arr, no_data=...)` sets the no-data pixels of
  `dem_arr` to NaN in place, KI-17), then `:1391` computes the shadow from the same array; one NaN makes the
  shadow all 0 (KI-07).
- Now: `DefaultValues.save_hillshade` (`rvt/default.py:2246`, `:2264`) (lines at `234b156`).
- Repro:
  ```python
  d = DEM * 10; d[5, 5] = -9999; p = write_dem("dem_nd.tif", d, nodata=-9999)
  DefaultValues().save_hillshade(p, save_float=True, save_8bit=False, save_shadow=True)
  bool(rvt.default.get_raster_arr("dem_nd_shadow_A315_H35.tif")["array"].any())   # -> False
  ```
- Pinned: `tests/test_char_default.py` (line 977).
- Status: documented, not changed in 0.1.0.

### KI-65 get_neg_opns does not recognise no_data
- Where: `rvt/default.py:1728` at `start` (`dem_arr = -1 * dem_arr`), then `:1729-1732` pass the unchanged
  `no_data` to `rvt.vis.sky_view_factor`, which compares it with the negated DEM. `save_neg_opns` inherits it.
- Now: `DefaultValues.get_neg_opns` (`rvt/default.py:2877`, `:2878-2881`) (lines at `234b156`).
- Impact: no-data pixels (-9999 becomes +9999) are processed as a 9999 m peak, which also distorts the openness of
  every pixel within `svf_r_max` px.
- Repro:
  ```python
  d = DEM.copy(); d[30, 30] = -9999; dv = DefaultValues()
  neg = dv.get_neg_opns(d.copy(), 1, no_data=-9999)
  pos = dv.get_sky_view_factor(d.copy(), 1, compute_svf=False, compute_opns=True, no_data=-9999)["opns"]
  bool(np.isnan(neg[30, 30])), bool(np.isnan(pos[30, 30]))   # -> False True
  ```
- Pinned: `tests/test_char_default.py::test_neg_opns_does_not_recognise_no_data`.
- Status: documented, not changed in 0.1.0.

### KI-66 save_mstp writes the 8-bit raster with no_data = NaN
- Where: `rvt/default.py:2141-2147` at `start` (`no_data=np.nan` with `e_type=1`, byte).
- Now: `DefaultValues.save_mstp` (`rvt/default.py:3633-3639`) (lines at `234b156`).
- Description: the GeoTIFF tag then holds "nan", which no byte value matches; GDAL reports the no-data value as NaN,
  rasterio as None. The other `save_*` methods write no no-data value for 8-bit rasters.
- Impact: readers disagree; no pixel is marked as no-data in the 8-bit raster.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM); dv = DefaultValues(); dv.mstp_broad_scale = (10, 20, 10); dv.save_mstp(p, save_float=False, save_8bit=True)
  gdal.Open(dv.get_mstp_path(p, bit8=True)).GetRasterBand(1).GetNoDataValue()   # -> nan (rasterio: None)
  ```
- Pinned: `tests/test_char_default.py` (line 1018, rasterio view).
- Status: documented, not changed in 0.1.0.

### KI-67 create_log_file reads the whole DEM to get its shape
- Where: `rvt/default.py:2281-2283` at `start` (`get_raster_arr(dem_path)`; `get_raster_size` would suffice).
- Now: `DefaultValues.create_log_file` (`rvt/default.py:3866-3868`) (lines at `234b156`).
- Impact: for DEMs above `tile_size_limit`, which the `save_*` methods process tile by tile because they do not fit
  in memory, `save_visualizations` can fail with MemoryError at the end, in the log.
- Repro: wrap `rvt.default.get_raster_arr` with a counter and call `DefaultValues().create_log_file(p)`; it is called
  once.
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-68 Log file: the MSRM output line is doubled; MSTP lists the float file always and the 8-bit file never
- Where: `rvt/default.py:2406-2407` at `start` (">> Output file:" twice for MSRM); `:2530-2532` (MSTP).
- Now: `DefaultValues.create_log_file` (`rvt/default.py:3991-3992`, `:4115-4117`) (lines at `234b156`).
- Impact: the log does not match the outputs.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM); dv = DefaultValues(); dv.hs_compute = 0; dv.msrm_compute = 1
  dv.mstp_compute = 1; dv.mstp_save_float = 0; dv.mstp_save_8bit = 1; os.mkdir("log"); dv.create_log_file(p, "log")
  t = open(os.path.join("log", os.listdir("log")[0])).read()
  t.count(">> Output file:"), dv.get_mstp_file_name(p) in t, dv.get_mstp_file_name(p, bit8=True) in t   # -> 3 True False
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-69 An unknown visualization gives None without an error
- Where: `get_visualization_file_name` / `get_visualization_path` (`rvt/default.py:1018-1084` at `start`) and
  `calculate_visualization` (`:2180-2276`) have no `else`; only `float_to_8bit` raises (`:1204`).
- Now: `DefaultValues.get_visualization_file_name` (`rvt/default.py:1602-1703`);
  `DefaultValues.calculate_visualization` (`rvt/default.py:3690-3830`); `DefaultValues.float_to_8bit`
  (`rvt/default.py:1887`) (lines at `234b156`).
- Impact: a wrong value (e.g. a string) gives None or `(None, None)`; the error surfaces later as AttributeError.
- Repro: `dv = DefaultValues(); dv.get_visualization_file_name("slope", "dem.tif", False), dv.calculate_visualization("slope", DEM.copy(), 1, 1)   # -> None (None, None)`
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

## rvt.tile

### KI-70 The MSRM overlap uses a size in metres as a pixel count
- Where: `rvt/tile.py:289` at `start` (`return int(rvt_default.msrm_feature_max)`, in m).
- Now: `_get_rvt_visualization_overlap` (`rvt/tile.py:450`) (lines at `234b156`).
- Description: the largest filter radius of `rvt.vis.msrm` is `n ** scaling_factor` px with
  `n = ceil(((feature_max - res) / (2 res)) ** (1 / scaling_factor))`: 25 px at 0.5 m/px for the default 20 m, so the
  20 px overlap is too small below 1 m/px (and larger than needed above, e.g. 9 px needed at 2 m/px).
- Impact: tiled MSRM differs from untiled MSRM near tile seams on fine DEMs.
- Repro:
  ```python
  p = write_dem("dem05.tif", DEM, gt=(0, 0.5, 0, 30, 0, -0.5)); dv = DefaultValues(); dv.tile_size = (50, 50)
  rvt.tile.save_rvt_visualization_tile_by_tile(V.MULTI_SCALE_RELIEF_MODEL, dv, Path(p), Path("."), True, False)
  t = rvt.default.get_raster_arr(dv.get_msrm_path(p))["array"]
  rvt.tile._get_rvt_visualization_overlap(V.MULTI_SCALE_RELIEF_MODEL, dv), bool(np.nanmax(np.abs(t - dv.get_msrm(DEM.copy(), 0.5))) > 1e-3)  # -> 20 True
  ```
- Pinned: `tests/test_char_tile.py` (line 422). AUDIT #25.
- Status: documented, not changed in 0.1.0.

### KI-71 The shadow overlap is 1 px
- Where: `rvt/tile.py:270-271` at `start`.
- Now: `_get_rvt_visualization_overlap` (`rvt/tile.py:431-432`) (lines at `234b156`).
- Description: a shadow of a relief of h m reaches h / tan(sun elevation) m (tens of px for a few metres of relief at
  35° sun elevation and 0.25 m/px).
- Impact: the tiled shadow (`save_hillshade` with shadow on DEMs above `tile_size_limit`) differs from the untiled one
  near seams.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM * 10); dv = DefaultValues(); dv.tile_size = (50, 50)
  rvt.tile.save_rvt_visualization_tile_by_tile(V.SHADOW, dv, Path(p), Path("."), True, False)
  t = rvt.default.get_raster_arr(str(dv.get_visualization_path(V.SHADOW, p, Path("."), False)))["array"]
  np.count_nonzero(t != dv.get_shadow(DEM * 10, 1)) > 0     # -> True
  ```
- Pinned: `tests/test_char_tile.py` (line 297).
- Status: documented, not changed in 0.1.0.

### KI-72 Argument checks build an Exception but do not raise it
- Where: `rvt/tile.py:103`, `:105` at `start` (`save_visualization_tile_by_tile`), `:327`, `:329`, `:335`
  (`save_rvt_visualization_tile_by_tile`): `Exception("...")` without `raise`. The texts at `:327`, `:329`, `:335` also
  name the other function. A `msg:` commit cannot change these texts, because they are not inside `raise`.
- Now: `save_visualization_tile_by_tile` (`rvt/tile.py:183`, `:185`); `save_rvt_visualization_tile_by_tile`
  (`rvt/tile.py:524`, `:526`, `:532`) (lines at `234b156`).
- Impact: with no output requested every tile is read and computed and nothing is written; tile sizes below 50 px
  are accepted; a missing DEM fails later in GDAL.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM); dv = DefaultValues(); dv.tile_size = (50, 50)
  rvt.tile.save_rvt_visualization_tile_by_tile(V.SLOPE, dv, Path(p), Path("."), save_float=False, save_8bit=False), sorted(os.listdir("."))  # -> None ['dem.tif']
  ```
- Pinned: `tests/test_char_tile.py::test_tiled_arguments_that_are_not_checked`,
  `tests/test_char_tile.py::test_tile_by_tile_problems_that_are_not_raised`.
- Status: documented, not changed in 0.1.0.

### KI-73 save_visualization_tile_by_tile fills the None entries of the caller's parameter dict
- Where: `rvt/tile.py:118-129` at `start` (writes `resolution*` and `no_data` into `function_parameters`).
- Now: `save_visualization_tile_by_tile` (`rvt/tile.py:198-209`) (lines at `234b156`).
- Impact: reusing the dict for another DEM keeps the first DEM's pixel size and no-data value.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM, gt=(0, 2, 0, 120, 0, -2)); params = {"resolution_x": None, "resolution_y": None}
  rvt.tile.save_visualization_tile_by_tile(v.slope_aspect, params, Path(p), 1, 50, 50, Path("s.tif"), out_visualization_dict_key="slope")
  params                                        # -> {'resolution_x': 2.0, 'resolution_y': 2.0}
  ```
- Pinned: `tests/test_char_tile.py` (line 173).
- Status: documented, not changed in 0.1.0.

### KI-74 save_visualization_tile_by_tile(function_parameters=None) raises TypeError
- Where: `rvt/tile.py:118` at `start` (`if "resolution" in function_parameters:` runs before the None branch at
  `:178-181`, which is therefore unreachable).
- Now: `save_visualization_tile_by_tile` (`rvt/tile.py:198`, `:260-263`) (lines at `234b156`).
- Impact: the `Optional` in the signature does not work; pass `{}`.
- Repro: `rvt.tile.save_visualization_tile_by_tile(lambda dem: dem, None, Path(write_dem("dem.tif", DEM)), 0, 50, 50, Path("o.tif"))   # -> TypeError`
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-75 The y pixel size is -gt[5] without abs; south-up rasters fail when tiled
- Where: `rvt/tile.py:111` and `:344` at `start` (`y_res = -gt[5]`); `get_raster_arr` uses `abs`
  (`rvt/default.py:2560`).
- Now: `save_visualization_tile_by_tile` (`rvt/tile.py:191`); `save_rvt_visualization_tile_by_tile`
  (`rvt/tile.py:541`); `get_raster_arr` (`rvt/default.py:4150`) (lines at `234b156`).
- Impact: a south-up raster (`gt[5] > 0`) works untiled but raises when tiled (negative resolution).
- Repro:
  ```python
  p = write_dem("southup.tif", DEM, gt=(0, 1, 0, 0, 0, 1)); dv = DefaultValues(); dv.tile_size = (50, 50)
  rvt.default.get_raster_arr(p)["resolution"]                                        # -> (1.0, 1.0)
  rvt.tile.save_rvt_visualization_tile_by_tile(V.SLOPE, dv, Path(p), Path("."), True, False)
  # -> Exception: rvt.vis.slope_aspect: resolution must be a positive number!  (at start: "rvt.visualization.slope_aspect: ...")
  ```
- Pinned: not pinned.
- Status: documented, not changed in 0.1.0.

### KI-76 The tiled float MSTP raster has one band for a three-band result
- Where: `rvt/tile.py:237` at `start` (`nr_bands = 1` for every float raster except MULTI_HILLSHADE); `rvt.vis.mstp`
  returns 3 bands, and `:479-482` write band 2 into a 1-band file (GDAL returns None, AttributeError).
- Now: `_create_rvt_visualization_blank_raster` (`rvt/tile.py:346`); `save_rvt_visualization_tile_by_tile`
  (`rvt/tile.py:679-682`) (lines at `234b156`).
- Impact: tiled MSTP float output fails and leaves a partly written file.
- Repro:
  ```python
  p = write_dem("dem.tif", DEM); dv = DefaultValues(); dv.tile_size = (50, 50); dv.mstp_broad_scale = (10, 20, 10)
  rvt.tile.save_rvt_visualization_tile_by_tile(V.MULTI_SCALE_TOPOGRAPHIC_POSITION, dv, Path(p), Path("."), True, False)
  # -> AttributeError; gdal.Open(dv.get_mstp_path(p)).RasterCount -> 1
  ```
- Pinned: `tests/test_char_tile.py::test_tiled_mstp_float_raster_has_one_band_for_a_three_band_result`.
- Status: documented, not changed in 0.1.0.

### KI-77 MSTP (square windows) depends on the absolute elevation in flat terrain and on large tiles
Found in 0.2.0 (module `rvt.vis`; numbered after KI-76 because the list was already published).
- Where: `rvt/vis.py:2146` at `start` (`max_elevation_deviation`: summed-area tables of z and z² in float64 over the
  whole padded tile, then `var = S2/n - (S1/n)²`); the precision fix `03702f4` made the squares float64 but kept
  the sums of raw elevations.
- Now: `max_elevation_deviation` (tables `rvt/vis.py:3231-3237`, variance `:3278` at `v0.2.0`; `:2892-2898`,
  `:2939` at `v0.1.1`), the numba kernel `_mstp_dev_max_kernel` evaluates the same expressions.
- Description: the table of z² reaches about n_px · z², e.g. 4.2 Mpx · 600² m² ≈ 1.5e12 m² on a 2048 x 2048 px tile at
  600 m, where one float64 rounding step is about 2e-4 m²; a small window in flat terrain has a variance of that
  order or less (std 1 cm: 1e-4 m²). The subtraction cancels, std becomes noise, and DEV = (z0 - mean) / (std +
  1e-6) can take any value. A constant elevation offset changes the result although DEV is offset-invariant.
- Impact (measured in 0.2.0 on the three full test DEMs, 10 000 x 8 000 px at 0.25 m/px, cropped 512 px = 128 m;
  MSTP of the DEM against MSTP of the DEM minus its mean rounded to whole metres; DN of the 8-bit display): local
  band (r = 1-5 px = 0.25-1.25 m) p99.9 15 / 137 / 68 DN, max 185 / 250 / 251 DN, more than 5 DN on 1.0 / 19.8 /
  1.8 % of the pixels for DEM_low (111 m) / DEM_medium (585 m) / DEM_high (930 m); meso band max 2 / 155 / 169 DN;
  broad band <= 1 DN. A brute-force float64 reference at the pixels with the largest meso difference agrees with
  the disk path of 0.2.0 (within 8e-7, same DN) and differs from the square output by up to 0.66 (169 DN). The disk path removes the tile
  mean before its sums and does not show the effect (broad and meso 0-1 DN in the same test); the local band stays
  square in every shape (DISK_WINDOWS.md section 3).
- Repro (2048 x 2048 px synthetic terrain with 5 cm of relief on a 1/256 m grid):
  ```python
  rng = np.random.default_rng(0)
  z = (np.round(rng.random((2048, 2048)) * 0.05 * 256) / 256).astype(np.float32)
  d0 = v.max_elevation_deviation(z, 1, 5, 1); d1 = v.max_elevation_deviation(z + np.float32(600), 1, 5, 1)
  np.abs(d1.astype(np.float64) - d0).max(), (np.abs(d1.astype(np.float64) - d0) > 0.1).mean()
  # -> (14320.9..., 0.0141)   (512 x 512 px: max 0.044, share 0)
  ```
- Pinned: not pinned (a fix would change MSTP, e3MSTP and e4MSTP outputs).
- Status: documented in 0.2.0, not changed (rule 4 of AGENTS.md). A fix would subtract the tile mean before the
  summed-area tables (exact in exact arithmetic, as on the disk path); question 8 of
  [DISK_WINDOWS.md](DISK_WINDOWS.md).

## Tiling facts (not defects)

These facts were measured on the optimized code at tag `start` with the frozen evaluation harness of the
optimization run (`eval/overlap.py`), and are identical for the upstream code at tag `baseline-fixed`.

### Minimum tile buffer per visualization

Minimum buffer at which the interior of a tile equals the full-window result, at two window positions per target,
`DefaultValues` defaults, 0.25 m/px. Buffers tried: 0, 8, 16, 32, 64, 128, 192, 256, 384, 512 px and the largest
possible, 569 px. "Within tolerance" uses the 8-bit tolerance of the harness, applied to each output's default
8-bit stretch: 99.9th percentile of the difference <= 5 DN, maximum <= 13 DN, mean <= 0.5 DN, 31 px local mean
<= 1.5 DN.

| visualization | largest default radius | min buffer, exact | min buffer, within tolerance | `rvt.tile` overlap at `start` |
|---|---|---|---|---|
| local_dominance | 20 px = 5 m | 32 px = 8 m | 16 px = 4 m | `ld_max_rad` = 20 px |
| svf, asvf, opns | 10 px = 2.5 m | 16 px = 4 m | 16 px = 4 m | `svf_r_max` = 10 px |
| neg_opns | 10 px = 2.5 m | 16 px = 4 m | 16 px = 4 m | `svf_r_max` = 10 px |
| sky_illumination | 100 px = 25 m | not reached (<= 569 px) | 128 px = 32 m | `sim_shadow_dist` = 100 px |
| msrm | 49 px = 12.25 m | 64 px = 16 m | 64 px = 16 m | `int(msrm_feature_max)` = 20 (m used as px, KI-70) |
| mstp | 500 px = 125 m | not reached (<= 569 px) | 512 px = 128 m | `mstp_broad_scale[1]` = 500 px |
| e3mstp | 500 px = 125 m | not reached (<= 569 px) | 512 px = 128 m | not tiled by `rvt.tile` |
| e4mstp | 500 px = 125 m | not reached (<= 569 px) | 512 px = 128 m | not tiled by `rvt.tile` |
| slrm | 20 px = 5 m | 32 px = 8 m | 32 px = 8 m | `slrm_rad_cell` = 20 px |
| slope | 1 px = 0.25 m | 8 px = 2 m | 8 px = 2 m | 1 px |
| hillshade | 1 px = 0.25 m | 8 px = 2 m | 8 px = 2 m | 1 px |

The `rvt.tile` column is `_get_rvt_visualization_overlap` (`rvt/tile.py:263-292` at `start`; now
`rvt/tile.py:372-453` at `234b156`). Because the harness
tried only the buffers listed above, "32 px exact" means that 16 px was not exact and 32 px was; a value between
the two was not tried. At 1 m/px the px values stay the same for radii defined in px; MSRM and the e4MSTP
flat-terrain SVF use radii in m (`int(10 / resolution)` px).

### Why some interiors never match exactly

- sky_illumination: the horizon pyramid is built from the tile's own corner (KI-03, `rvt/vis.py:1156`
  `horizon_generate_coarse_dem`, `rvt/vis.py:1218` `horizon_generate_pyramids` at `start`; now `rvt/vis.py:1560`,
  `:1665` at `234b156`). Tile origins that are multiples of 8 px give an identical interior with a 128 px buffer
  (synthetic 1024 px window, 512 px tiles); origins congruent to 4, 6, 7 mod 8 differ by up to 1.9e-4, 3.1e-4,
  6.1e-4. The overcast output is also divided by the tile-wide maximum (`rvt/vis.py:1813` at `start`, now
  `rvt/vis.py:2456` in `sky_illumination`); in the synthetic test the maximum was reached inside every tile.
- mstp, e3mstp, e4mstp: exact equality was not reached within 569 px and the cause was not isolated. The summed-area
  tables of the DEM and of its squares are accumulated in float64 from the tile corner (in place, with the same
  cumulative sums as `integral_image`, in `max_elevation_deviation`: `rvt/vis.py:2892-2898` at `234b156`), so
  their rounding depends on the tile origin; this is a likely cause, not a measured one.
- e3mstp, e4mstp: the blend rescales image-wide when an intermediate leaves [0, 1] (`rvt/blend_func.py:555-563` at
  `start`, now `rvt/blend_func.py:1361-1368`, `scale_within_0_and_1`, see KI-19), a tile-wide statistic. Since 0.1.1
  such an operand is clipped per pixel instead (X3); the tiling measurements of this section were made before 0.1.1.
- 8-bit conversion: the default display stretch of sky illumination is a percent cut (`rvt/default.py:319` at
  `start`, now `rvt/default.py:405` in `DefaultValues.__init__`, `sim_bytscl = ("percent", 0.25, 0.00)`), computed
  per image with `np.nanpercentile` (`rvt/blend_func.py:177` at `start`, now `rvt/blend_func.py:378` in
  `lin_cutoff_calc_from_perc`), so per-tile 8-bit output does not match a full-window stretch. The other
  visualizations use value stretches.

### Practical consequences for a tiling pipeline (recommendations, not measured as a pipeline)

- Buffers at 0.25 m/px: 512 px (128 m) for mstp, e3mstp, e4mstp; 128 px (32 m) for sky illumination; 64 px for msrm;
  32 px for local dominance and slrm; 16 px for SVF and openness; 8 px for slope and hillshade.
- Tile origins on multiples of 8 px, so that sky illumination tiles agree exactly with a full-window computation.
- Sky illumination converted to 8 bit with fixed value bounds chosen once, instead of the per-image percent
  stretch, if tiles must match at the seams.
