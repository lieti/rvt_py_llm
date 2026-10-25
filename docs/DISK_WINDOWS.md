# Disk windows in rvt_py_llm 0.2.0

0.2.0 adds an opt-in window shape to the functions that use square moving windows: `rvt.vis.mean_filter`, `slrm`,
`msrm`, `max_elevation_deviation` and `mstp` take a keyword `kernel_shape` = `"square"` (default), `"disk"` or
`"disk_equal_area"`. With the default every public function returns the same bits as `v0.1.1` (proof:
[TESTING.md](TESTING.md) section 8). SVF, openness, local dominance and sky illumination are ray or ring based and
are not affected. The default stays square; whether a disk fits the intent of the methods is a question for the
upstream authors (section 10).

## 1. Why: the square window is anisotropic

A square window reaches √2 times farther along its diagonals than along its axes, so a linear feature gets a weaker
response when it runs diagonally. Test (review C-15): a straight Gaussian bank 1 m high and 3 m = 12 px wide at
0.25 m/px, along the x axis (0°) and on the diagonal (45°); crest response 45° / 0° (1 = isotropic), computed with
0.2.0 ([figure](figures/disk_windows/c15_bank_ratio.png)):

| r | DEV square | DEV disk | DEV equal-area disk | SLRM square | SLRM disk | SLRM equal-area disk |
|---|---|---|---|---|---|---|
| 8 px = 2 m | 0.812 | 0.982 | 0.999 | 0.932 | 0.996 | 1.000 |
| 16 px = 4 m | 0.855 | 0.993 | 0.994 | 0.891 | 0.996 | 0.996 |
| 48 px = 12 m | 0.847 | 0.999 | 0.998 | 0.942 | 1.000 | 0.999 |
| 96 px = 24 m | 0.844 | 0.999 | 1.000 | 0.969 | 1.000 | 1.000 |

The square shows a diagonal bank 15-19 % weaker in DEV (MSTP) and 3-11 % weaker in SLRM. The response at 0°
relative to the square: disk 0.87-0.88 (DEV) and 0.70-0.98 (SLRM, r = 8-96 px), equal-area disk 0.92-0.94 and
0.93-0.99: the same-r disk is smaller (area about π/4 of the square). The published DEV / MSTP (Lindsay et al.
2015; WhiteboxTools) is square by design, so a disk extends the method; it does not correct it.

## 2. Definition

A window of radius r px is the exact discrete disk: pixel (dx, dy) from the centre belongs to it iff
dx² + dy² <= r_w², the convention of GRASS `r.neighbors -c`, SAGA and ArcGIS. Two radius mappings:

- `"disk"`: r_w = r, the disk inscribed in the square;
- `"disk_equal_area"`: r_w = (2r + 1) / √π, the disk with the area of the (2r + 1)² square.

| r (px) | square (px) | disk (px) | r_w equal area (px) | equal-area disk (px) |
|---|---|---|---|---|
| 8 = 2 m | 289 | 197 | 9.59 | 293 |
| 20 = 5 m | 1 681 | 1 257 | 23.13 | 1 685 |
| 50 = 12.5 m | 10 201 | 7 845 | 56.98 | 10 185 |
| 500 = 125 m | 1 002 001 | 785 349 | 564.75 | 1 001 989 |

(pixel counts of `rvt.vis._window_footprint`; m at 0.25 m/px.)

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
  square names unchanged. e3MSTP and e4MSTP have no file-name method in `DefaultValues`; their names do not change.
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
within-window variance (E[z²] - E[z]² cancels). Sums of z - zbar are exact translations of the same statistics;
the tests require DEV at +2 500 m within 1e-7 and the mean within 1e-9 m of the result at 0 m. On the full test
DEMs, MSTP with disks changes by 0 DN in the broad and meso bands (at most 1 DN) when the DEM is shifted by its mean;
the square path does not have this property (section 8.3, KI-77).

## 6. Reproducibility

No numba kernel and no threads on the disk path (`workers=1`), so `RVT_NO_ACCEL` and the thread count do not
change its result; nothing is kept between calls. Checked in-process by `tests/test_disk_composition.py` and in
fresh processes (default threading, every thread variable 1 and 2, `RVT_NO_ACCEL=1`; mstp, slrm, msrm, e3mstp,
e4mstp in both shapes on the 8 tiles of the evaluation set `dev`: bitwise identical). Bitwise equality across
operating systems or scipy builds is not promised for FFT results (last-bit differences are possible).

## 7. Cost

One fresh process per call, every thread variable 1, Windows 11 (Intel i9-14900K, 192 GB), Python 3.11.17, numpy
2.4.6, scipy 1.17.1. Times of the call; peak = peak working set of the process, including the input DEM.

Full test DEMs (10 000 x 8 000 px at 0.25 m/px; range over DEM_low, DEM_medium, DEM_high):

| call | square (v0.1.1) | disk | disk_equal_area |
|---|---|---|---|
| `get_mstp` | 16.4-19.4 s, 3.84 GB | 109.6-113.8 s, 9.36 GB | 109.5-113.2 s, 9.49 GB |
| `get_slrm` (r = 20 px = 5 m) | 1.24-1.29 s, 2.45 GB | 3.56-3.68 s, 4.12 GB | 3.58-3.78 s, 4.12 GB |
| `get_msrm` (0-20 m, factor 2) | 8.8-9.0 s, 3.42 GB | 20.5-20.9 s, 5.72 GB | 20.5-20.8 s, 5.76 GB |
| `e3mstp` | 54.1-57.1 s, 11.71 GB | 150.1-151.7 s, 11.71 GB | 150.3-153.4 s, 11.73 GB |
| `e4mstp` | 80.2-82.8 s, 13.63 GB | 173.3-175.8 s, 13.63 GB | 173.1-176.1 s, 13.63 GB |

One 2048 x 2048 px tile = 512 m cut from DEM_medium (measured during development, commit `2ee3ecf`): `mstp` 1.46 s /
0.38 GB square, 7.51 s / 0.78 GB disk, 7.82 s / 0.82 GB equal-area disk, below the soft target of 10 s per tile.
1024 x 1024 px tiles of `dev` (real_1024_medium, real_1024_high): `get_mstp` 0.23 s square, 2.20-2.28 s disk,
2.46-2.48 s equal-area disk. The square path itself is not slower: e3MSTP 2.99 s against 3.00 s, e4MSTP 4.39 s
against 4.38 s (v0.1.1 against 0.2.0, medium 2048 x 2048 px window, 1 thread, median of 5).

The MSTP broad scale (10 windows on a tile padded by 500-564 px) holds most of the time: per window one forward and
two (three with NaN) inverse transforms of the padded tile. Memory: about four float64 / complex128 arrays of the
FFT size at a time.

## 8. Effect on real terrain

### 8.1 Tiles 1024 x 1024 px, per layer

Inner 512 x 512 px (crop 256 px = 64 m) of `real_1024_medium`, `real_1024_high` (evaluation set `dev`) and
`DEM_low_1024` (cut from DEM_low); DN of the 8-bit display (MSTP 0..1, SLRM -2..2 m, MSRM -2.5..2.5 m); p99.9 / max
of |DN difference| against the square result of v0.1.1:

| layer | medium: disk | medium: equal area | high: disk | high: equal area | low: disk | low: equal area |
|---|---|---|---|---|---|---|
| MSTP R (broad) | 16 / 17 | 4 / 4 | 16 / 20 | 10 / 10 | 29 / 44 | 21 / 26 |
| MSTP G (meso) | 8 / 12 | 4 / 7 | 18 / 23 | 8 / 17 | 26 / 41 | 16 / 40 |
| MSTP B (local) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| SLRM | 1 / 1 | 1 / 1 | 35 / 54 | 12 / 15 | 1 / 2 | 1 / 1 |
| MSRM | 1 / 1 | 1 / 1 | 6 / 7 | 2 / 3 | 1 / 1 | 1 / 1 |
| e3MSTP | 14 / 15 | 5 / 8 | 17 / 32 | 9 / 13 | 22 / 36 | 15 / 33 |
| e4MSTP | 13 / 15 | 3 / 5 | 13 / 17 | 8 / 10 | 19 / 32 | 14 / 28 |

The local MSTP band is bitwise the square result (threshold). With 0.2.0 and the square shape every layer is
bitwise equal to v0.1.1.

### 8.2 Full test DEMs

Crop 512 px = 128 m per side (8 976 x 6 976 px); p99.9 DN / pixels above 5 DN against the square result of v0.1.1,
all bands:

| product | low: disk | low: equal area | medium: disk | medium: equal area | high: disk | high: equal area |
|---|---|---|---|---|---|---|
| MSTP | 33 / 53.6 % | 22 / 32.1 % | 23 / 36.3 % | 12 / 7.8 % | 24 / 36.3 % | 12 / 10.2 % |
| SLRM | 5 / 0.1 % | 2 / 0.0 % | 11 / 1.1 % | 3 / 0.0 % | 39 / 9.6 % | 12 / 1.0 % |
| MSRM | 1 / 0.0 % | 1 / 0.0 % | 3 / 0.0 % | 1 / 0.0 % | 11 / 1.0 % | 3 / 0.0 % |
| e3MSTP | 27 / 42.0 % | 18 / 21.4 % | 21 / 31.8 % | 11 / 5.0 % | 23 / 34.1 % | 11 / 6.5 % |
| e4MSTP | 25 / 38.0 % | 17 / 18.8 % | 17 / 22.1 % | 9 / 3.0 % | 17 / 17.0 % | 8 / 2.8 % |

Overview and zoom: [MSTP DEM_medium](figures/disk_windows/mstp_medium_overview.png),
[MSTP DEM_high](figures/disk_windows/mstp_high_overview.png), [MSTP DEM_high, 100 m zoom at the largest
difference](figures/disk_windows/mstp_high_zoom1.png), [SLRM DEM_high](figures/disk_windows/slrm_high_overview.png).
The equal-area disk changes about half as much as the same-r disk, at the same isotropy.

### 8.3 The largest differences are errors of the square path

MSTP pixels that differ by more than 50 DN are at most 0.06 % (DEM_low broad band, same-r disk). At 114 pixels
(up to 10 per DEM, shape and band: a random sample of the broad-band pixels above 50 DN and the 10 largest meso
differences) a brute-force float64 window computation agrees
with the disk output within 1e-7 (same DN everywhere). The square output agrees in the broad band (within 1.6e-5) but
not at the largest meso differences of DEM_medium and DEM_high: there it is wrong by up to 0.66 (169 DN), in flat
spots (std of the 11 x 11 px neighbourhood below 1 cm). The cause is KNOWN_ISSUES KI-77: the square path computes the
variance from sums of raw elevations over the whole tile, so its result depends on the elevation; shifting the DEM
by its mean changes the local band by up to 250 DN (p99.9 137 DN on DEM_medium). The local band stays square in
0.2.0 in every shape, so it keeps this dependence.

## 9. Rejected alternatives

- Octagon (apothem r) from upright and 45°-rotated summed-area tables: within 1 % of isotropic only from
  r = 32 px (1.04 at 8 px); the numpy prototype took 86.5 s per 2048² MSTP against 24.2 s for the FFT disk
  (pre-research of 0.2.0 on a Linux cloud machine, one thread); a numba octagon was not tested.
- Disks as sums of a few rectangles of one integral image: DEV errors of 0.19-0.24 at r = 20 px.
- Exact row-span sums: 2(2r + 1) lookups per pixel and radius, 2 002 at r = 500 px; too slow at the broad scale.
- `workers = RVT_NUM_THREADS`: bitwise equality across thread counts would have to be proven first.

## 10. Open questions for the upstream authors

1. Default shape: square stays the default. If a disk is wanted, which mapping (`disk` or `disk_equal_area`)?
   The equal-area disk keeps 92-94 % of the DEV contrast of the square and changes real MSTP about half as much.
2. Threshold: 8 px fixed, not user-settable. Is another value, or a disk at every radius, preferable?
3. Should the local MSTP scale (1-5 px) ever be a disk?
4. A radius <= 0 in an MSTP band that takes the FFT path keeps the square-path behaviour of KI-13 (the elevation
   one pixel up and left, not a deviation). Should it?
5. MSRM uses the disk also at its radii below 8 px when its largest radius is >= 8 px (one shape per call). Is
   that the intended reading of "consecutive surfaces share a shape"?
6. The MSTP tile overlap follows the broad scale only (as for the square); the MSRM overlap is not adapted for
   `disk_equal_area` (KI-70 first).
7. Was the SLRM "radius" of Hesse 2010 / RVT ever meant as a circle?
8. KI-77: should the square path of MSTP also remove the tile mean before its sums? It would make MSTP independent
   of the absolute elevation (the change of the local band on DEM_medium is up to 250 DN) and would change MSTP,
   e3MSTP and e4MSTP outputs, so it belongs to a separate fix release.
