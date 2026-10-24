# Disk windows in rvt_py_llm 0.2.0 (draft)

Draft written with the feature commits of 0.2.0. The comparison on the three test DEMs (square vs disk vs
equal-area disk per layer, figures, times and memory on 1024² tiles and full DEMs) is added before the release.

0.2.0 adds an opt-in window shape to the functions that use square moving windows: `rvt.vis.mean_filter`, `slrm`,
`msrm`, `max_elevation_deviation` and `mstp` take a keyword `kernel_shape` = `"square"` (default), `"disk"` or
`"disk_equal_area"`. With the default every public function returns the same bits as `v0.1.1` (proof: the frozen
tests of the 0.1.1 freeze run through `equiv_diff` with an empty function allow-list, and `eval/check.py`). SVF,
openness, local dominance and sky illumination are ray or ring based and are not affected.

## 1. Why: the square window is anisotropic

A square window reaches √2 times farther along its diagonals than along its axes, so a linear feature gets a
weaker response when it runs diagonally. Test (review C-15): a straight Gaussian bank 1 m high and 3 m = 12 px
wide at 0.25 m/px, once along the x axis (0°) and once on the diagonal (45°); crest response 45° / 0° (1 =
isotropic):

| r | SLRM square | SLRM disk | DEV square | DEV disk |
|---|---|---|---|---|
| 8 px = 2 m | 0.932 | 0.996 | 0.812 | 0.982 |
| 16 px = 4 m | 0.891 | 0.996 | 0.855 | 0.993 |
| 48 px = 12 m | 0.942 | 1.000 | 0.847 | 0.999 |
| 96 px = 24 m | 0.969 | 1.000 | 0.844 | 0.999 |

So the square path shows a diagonal bank 15-19 % weaker in DEV (MSTP) and 3-11 % weaker in SLRM. The published
DEV / MSTP (Lindsay et al. 2015; WhiteboxTools) is square by design; a disk is an extension of the method, not a
correction. `tests/test_disk_anisotropy.py` repeats the test at r = 16 and 48 px: disk ratios within 2 % of 1,
square below 0.9.

## 2. Definition

A window of radius r px is the exact discrete disk: pixel (dx, dy) from the centre belongs to it iff
dx² + dy² <= r_w², the convention of GRASS `r.neighbors -c`, SAGA and ArcGIS. Two radius mappings:

- `"disk"`: r_w = r, the disk inscribed in the square (area about π/4 of the square);
- `"disk_equal_area"`: r_w = (2r + 1) / √π, the disk with the area of the (2r + 1)² square.

| r (px) | square (px) | disk (px) | r_w equal area (px) | equal-area disk (px) |
|---|---|---|---|---|
| 8 = 2 m | 289 | 197 | 9.59 | 293 |
| 20 = 5 m | 1 681 | 1 257 | 23.13 | 1 685 |
| 50 = 12.5 m | 10 201 | 7 845 | 56.98 | 10 185 |
| 500 = 125 m | 1 002 001 | 785 349 | 564.75 | 1 001 989 |

(pixel counts of `rvt.vis._window_footprint`; m at 0.25 m/px). On the bank test the same-r disk lowers the DEV
contrast to 87-88 % of the square, the equal-area disk to 92-94 % (crest pixel, pre-research of 0.2.0; the
evidence is added with the comparison).

## 3. Threshold and where the disk applies

Windows with r >= `DISK_MIN_RADIUS_PX` = 8 px (2 m at 0.25 m/px) become disks; smaller windows stay square,
because a disk of a few px is dominated by its discretisation (review C-15: disk DEV ratio 1.51 at r = 2 px).

- `mean_filter`: disk iff r >= 8 px; r < 8 px runs the square code (bitwise the default result).
- `max_elevation_deviation` (and the three scales of `mstp`): when the largest radius of the call is >= 8 px,
  every radius goes through the FFT path, with the disk from 8 px and the square footprint below; when all radii
  are < 8 px the square code runs. Defaults: local 1-5 px stays square (bitwise), meso 5-50 px has a square 5 px
  window and disks from 10 px, broad 50-500 px is all disks.
- `slrm`: its radius (10-50 px) is always >= 8 px.
- `msrm`: one shape for all its radii, because consecutive surfaces are subtracted: the disk at every radius
  (also 0-7 px) when the largest radius k^s is >= 8 px, otherwise the square code.
- `e3mstp`, `e4mstp`: through `DefaultValues.get_slrm` / `get_mstp` (attributes below); CRIM has no window filter.

## 4. API, settings, file names, tiling

- Keyword `kernel_shape="square"`, last parameter of `mean_filter`, `max_elevation_deviation`, `slrm`, `msrm`,
  `mstp`. Another value raises `ValueError` in `mean_filter` and `max_elevation_deviation` and `Exception` in
  `slrm`, `msrm`, `mstp` (the type each uses for bad arguments), after the existing checks.
- `DefaultValues.slrm_kernel_shape`, `msrm_kernel_shape`, `mstp_kernel_shape` = `"square"`, passed by `get_slrm`,
  `get_msrm`, `get_mstp`. json settings: one key per group ("Simple local relief model", "Multi-scale relief
  model", "Multi-scale topographic position"); a file without the keys (written before 0.2.0) loads as
  `"square"`. txt settings: the attribute names, e.g. `mstp_kernel_shape = disk_equal_area` (no txt example file
  exists in the repository).
- File names: `_disk` or `_diskea` before `.tif` / `_8bit.tif`, e.g. `dem_01_MSTP_5_50_500_L0.9_diskea.tif`;
  square names unchanged.
- Tile overlap (`rvt.tile`): `"disk_equal_area"` reaches floor(r_w) px, so SLRM gets 23 px instead of 20 px and
  MSTP 564 px = 141 m instead of 500 px = 125 m at the defaults; `"disk"` keeps the square overlap. MSRM keeps
  its overlap, which is a known issue on its own (KNOWN_ISSUES KI-70).

## 5. Algorithm

`rvt.vis._disk_window_sums` (contract in its docstring) takes the padded tile, removes `zbar`, the float64 mean of
its non-NaN pixels, sets NaN to 0, and transforms `z - zbar`, its square and (only with NaN) the valid mask once
per call with `scipy.fft.rfft2` (float64, `workers=1`). Per window the footprint is transformed once and each
product transformed back with `irfft2`; pixel counts are rounded with `np.rint`. The FFT size is `next_fast_len`
of the padded size; the padding of floor(r_w) of the largest window keeps the circular convolution from wrapping
into the output window. Padding modes are those of the square path (edge for the mean filter, symmetric for DEV).

The mean filter is `s1 / count + zbar`; DEV uses the square path's expressions shifted by `zbar`
(`var = s2 / n - mean²`, `std = sqrt(|var|) + 1e-6`), its tie rule and its final float32 rounding.

Why mean removal: sums of z and z² at real elevations carry the rounding of the large values into the small
within-window variance (E[z²] - E[z]² cancels; review C-05). Sums of z - zbar are exact translations of the same
statistics; the tests require DEV at +2 500 m within 1e-7 and the mean within 1e-9 m of the result at 0 m.

## 6. Reproducibility

No numba kernel and no threads on the disk path (`workers=1`), so `RVT_NO_ACCEL` and the thread count do not
change its result; nothing is kept between calls. Checked by `tests/test_disk_composition.py` (in-process) and by
the determinism check of the phase (fresh processes; default threading, all thread variables 1 and 2,
`RVT_NO_ACCEL=1`; mstp, slrm, msrm, e3mstp, e4mstp in both shapes on the 8 dev tiles: identical). Bitwise equality
across operating systems or scipy builds is not promised for FFT results (last-bit differences are possible).

## 7. Cost

To be completed with the comparison runs. Measured during development (commit `2ee3ecf`) on one 2048 x 2048 px
tile = 512 m cut from the centre of `test_DEM/DEM_medium.tif` (0.25 m/px, 541-627 m, no NaN): one run per cell,
fresh process, every thread variable 1, Windows 11, Python 3.11, numpy 2.4.6, scipy 1.17.1. Peak = peak working
set of the process, including about 0.08 GB before the call (interpreter, modules, tile).

| call | square | disk | disk_equal_area |
|---|---|---|---|
| `mstp`, default scales (local 1-5, meso 5-50, broad 50-500 px) | 1.46 s, 0.38 GB | 7.51 s, 0.78 GB | 7.82 s, 0.82 GB |
| `slrm`, r = 20 px = 5 m | 0.06 s, 0.18 GB | not run | 0.19 s, 0.28 GB |
| `msrm`, 0-20 m, factor 2 (radii 0-49 px) | 0.42 s, 0.23 GB | not run | 1.05 s, 0.37 GB |

MSTP with disks stays below the soft target of 10 s per 2048² tile at one thread; a tile with NaN needs one more
inverse transform per window.

Per window the cost is one forward and two (three with NaN) inverse transforms of the padded tile; the MSTP broad
scale (10 windows on a tile padded by 500-564 px) dominates. Memory: about four float64 / complex128 arrays of the
FFT size at a time.

## 8. Rejected alternatives

- Octagon (apothem r) from upright and 45°-rotated summed-area tables: within 1 % of isotropic only from
  r = 32 px (1.04 at 8 px); the numpy prototype took 86.5 s per 2048² MSTP against 24.2 s for the FFT disk
  (pre-research of 0.2.0 on a Linux cloud machine, one thread); a numba octagon was not tested.
- Disks as sums of a few rectangles of one integral image: DEV errors of 0.19-0.24 at r = 20 px.
- Exact row-span sums: 2(2r + 1) lookups per pixel and radius, 2 002 at r = 500 px; too slow at the broad scale.
- `workers = RVT_NUM_THREADS`: bitwise equality across thread counts would have to be proven first.

## 9. Open questions for the upstream authors

1. Default shape: square stays the default. If a disk is wanted, which mapping (`disk` or `disk_equal_area`)?
2. Threshold: 8 px fixed, not user-settable. Is another value, or a disk at every radius, preferable?
3. Should the local MSTP scale (1-5 px) ever be a disk?
4. A radius <= 0 in an MSTP band that takes the FFT path keeps the square-path behaviour of KI-13 (the elevation
   one pixel up and left, not a deviation). Should it?
5. MSRM uses the disk also at its radii below 8 px when its largest radius is >= 8 px (one shape per call). Is
   that the intended reading of "consecutive surfaces share a shape"?
6. The MSTP tile overlap follows the broad scale only (as for the square); the MSRM overlap is not adapted for
   `disk_equal_area` (KI-70 first).
7. Was the SLRM "radius" of Hesse 2010 / RVT ever meant as a circle?
