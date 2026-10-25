# Disk windows in rvt_py_llm 0.2.0

0.2.0 adds an opt-in window shape to the functions that use square moving windows: `rvt.vis.mean_filter`, `slrm`,
`msrm`, `max_elevation_deviation` and `mstp` take a keyword `kernel_shape` = `"square"` (default), `"disk"` or
`"disk_equal_area"`. With the default every public function returns the same bits as `v0.1.1` (proof:
[TESTING.md](TESTING.md) section 8). SVF, openness, local dominance and sky illumination are ray or ring based and
are not affected. The default stays square; whether a disk fits the intent of the methods is a question for the
upstream authors (section 10). Code places: section 11.

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

Windows with r >= `DISK_MIN_RADIUS_PX` = 8 px (2 m at 0.25 m/px) become disks; smaller windows stay square, because
a disk of a few px is dominated by its discretisation (review C-15: disk DEV ratio 1.51 at r = 2 px).

- `mean_filter`, `slrm` (r = 10-50 px): disk iff r >= 8 px; below, the square code runs.
- `max_elevation_deviation`, `mstp`: if the largest radius of a call is >= 8 px, all its radii use the FFT path,
  square footprint below 8 px; otherwise the square code. Defaults: local 1-5 px square, meso 5 px square and
  10-50 px disks, broad 50-500 px disks.
- `msrm`: one shape for all radii, since consecutive surfaces are subtracted: the disk at every radius (also
  0-7 px) when the largest radius k^s is >= 8 px.
- `e3mstp`, `e4mstp`: through `get_slrm` / `get_mstp`; CRIM has no window filter.

## 4. API, settings, file names, tiling

- Keyword `kernel_shape="square"`, last parameter of the five functions. Another value raises `ValueError` in
  `mean_filter` and `max_elevation_deviation`, `Exception` in `slrm`, `msrm`, `mstp` (each function's type for bad
  arguments), after the existing checks.
- `DefaultValues.slrm_kernel_shape`, `msrm_kernel_shape`, `mstp_kernel_shape` = `"square"`, passed by `get_slrm`,
  `get_msrm`, `get_mstp`. json: one key per group; a file without them loads as `"square"`. txt: the attribute
  names, e.g. `mstp_kernel_shape = disk_equal_area`.
- File names: `_disk` or `_diskea` before `.tif` / `_8bit.tif`; square names unchanged; e3MSTP and e4MSTP have no
  file-name method.
- Tile overlap: `"disk_equal_area"` uses floor(r_w): SLRM 23 px instead of 20 px, MSTP 564 px = 141 m instead of
  500 px = 125 m; `"disk"` keeps the square overlap; MSRM keeps its own (KNOWN_ISSUES KI-70).

## 5. Algorithm

`rvt.vis._disk_window_sums` (contract and cost in its docstring) removes `zbar`, the float64 mean of the padded
tile, sets NaN to 0, and transforms `z - zbar`, its square and (with NaN) the valid mask once per call with
`scipy.fft.rfft2` (float64, `workers=1`); per window it multiplies by the footprint spectrum and transforms back;
counts are rounded with `np.rint`. The padding (mode of the square path, width floor(r_w) of the largest window)
keeps the circular convolution off the output window. DEV and the mean filter use the square path's expressions
shifted by `zbar`, its tie rule and its float32 rounding.

Why mean removal: sums of raw z and z² cancel in E[z²] - E[z]² at real elevations; sums of z - zbar give the same
statistics without that loss. Tests: DEV at +2 500 m within 1e-7, mean within 1e-9 m of the result at 0 m.

## 6. Reproducibility

No numba kernel and no threads on the disk path (`workers=1`), so `RVT_NO_ACCEL` and the thread count do not
change its result; nothing is kept between calls. Checked in-process by `tests/test_disk_composition.py` and in
fresh processes (default threading, every thread variable 1 and 2, `RVT_NO_ACCEL=1`; mstp, slrm, msrm, e3mstp,
e4mstp in both shapes on the 8 tiles of the evaluation set `dev`: bitwise identical). Bitwise equality across
operating systems or scipy builds is not promised for FFT results (last-bit differences are possible).

## 7. Cost

One fresh process per call, every thread variable 1, Windows 11 (Intel i9-14900K, 192 GB), Python 3.11.17, numpy
2.4.6, scipy 1.17.1; time of the call; peak = peak working set of the process, input included.

One 2112 x 2112 px tile (`bench_final/real_2048_medium` of the optimization harness, 0.25 m/px; log
`release_evidence/0.2.0/cost2048.log`), 0.2.0:

| call | square | disk | disk_equal_area |
|---|---|---|---|
| `mstp` | 0.96 s, 0.47 GB (v0.1.1: 0.93 s) | 7.56 s, 0.91 GB | 7.80 s, 0.95 GB |
| `e3mstp` | 3.01 s, 0.83 GB | 9.75 s, 1.04 GB | 9.99 s, 1.08 GB |
| `e4mstp` | 4.39 s, 0.94 GB | 11.05 s, 1.09 GB | 11.25 s, 1.13 GB |

Full test DEMs (10 000 x 8 000 px; range over DEM_low, DEM_medium, DEM_high; `release_evidence/0.2.0/compare_full.md`):

| call | square (v0.1.1) | disk | disk_equal_area |
|---|---|---|---|
| `get_mstp` | 16.4-19.4 s, 3.84 GB | 109.6-113.8 s, 9.36 GB | 109.5-113.2 s, 9.49 GB |
| `get_slrm` (r = 20 px = 5 m) | 1.24-1.29 s, 2.45 GB | 3.56-3.68 s, 4.12 GB | 3.58-3.78 s, 4.12 GB |
| `get_msrm` (0-20 m, factor 2) | 8.8-9.0 s, 3.42 GB | 20.5-20.9 s, 5.72 GB | 20.5-20.8 s, 5.76 GB |
| `e3mstp` | 54.1-57.1 s, 11.71 GB | 150.1-151.7 s, 11.71 GB | 150.3-153.4 s, 11.73 GB |
| `e4mstp` | 80.2-82.8 s, 13.63 GB | 173.3-175.8 s, 13.63 GB | 173.1-176.1 s, 13.63 GB |

MSTP with disks stays below the soft target of 10 s per 2048² tile. The broad scale (10 windows on a tile padded by
500-564 px) holds most of the time. The square path is not slower: e3MSTP x1.002, e4MSTP x0.997 against v0.1.1
(medium 2048 x 2048 px window, median of 5; `release_evidence/0.2.0/timing.log`).

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

Crop 512 px = 128 m per side (8 976 x 6 976 px); p99.9 DN / pixels above 5 DN against the square result of v0.1.1.
The p99.9 is taken over the values of all bands together (`eval/common.py compare_one`); for MSTP, whose local band
does not change, per band: broad 44 / 30 / 27 DN and meso 26 / 19 / 21 DN (disk; low / medium / high), broad
29 / 13 / 13 DN and meso 19 / 12 / 12 DN (equal area).

| product | low: disk | low: equal area | medium: disk | medium: equal area | high: disk | high: equal area |
|---|---|---|---|---|---|---|
| MSTP | 33 / 53.6 % | 22 / 32.1 % | 23 / 36.3 % | 12 / 7.8 % | 24 / 36.3 % | 12 / 10.2 % |
| SLRM | 5 / 0.1 % | 2 / 0.0 % | 11 / 1.1 % | 3 / 0.0 % | 39 / 9.6 % | 12 / 1.0 % |
| MSRM | 1 / 0.0 % | 1 / 0.0 % | 3 / 0.0 % | 1 / 0.0 % | 11 / 1.0 % | 3 / 0.0 % |
| e3MSTP | 27 / 42.0 % | 18 / 21.4 % | 21 / 31.8 % | 11 / 5.0 % | 23 / 34.1 % | 11 / 6.5 % |
| e4MSTP | 25 / 38.0 % | 17 / 18.8 % | 17 / 22.1 % | 9 / 3.0 % | 17 / 17.0 % | 8 / 2.8 % |

Figures: [MSTP DEM_medium](figures/disk_windows/mstp_medium_overview.png),
[MSTP DEM_high](figures/disk_windows/mstp_high_overview.png), [MSTP DEM_high, 100 m zoom at the largest
difference](figures/disk_windows/mstp_high_zoom1.png), [SLRM DEM_high](figures/disk_windows/slrm_high_overview.png).
The equal-area disk changes about half as much as the same-r disk, at the same isotropy.

### 8.3 The largest differences are errors of the square path

MSTP pixels that differ by more than 50 DN are at most 0.06 % (DEM_low broad band, same-r disk). At 114 of them (up
to 10 per DEM, shape and band) a brute-force float64 window computation agrees with the disk output within 8e-7
(111 within 1e-7), with the same DN at all 114. The square output agrees in the broad band (within 1.6e-5), but at
the largest meso differences of DEM_medium and DEM_high it is wrong by up to 0.66 (169 DN), in flat spots. Cause:
KNOWN_ISSUES KI-77, the square path's variance from sums of raw elevations over the whole tile. Shifting the DEM by
its mean changes the square local band by up to 250 DN (p99.9 137 DN, DEM_medium) and the disk's broad and meso
bands by at most 1 DN. The local band stays square in every shape and keeps this dependence.

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
   of the absolute elevation and would change MSTP, e3MSTP and e4MSTP outputs, so it belongs to a separate release.

## 11. Where in the code (`v0.2.0`)

| what | place |
|---|---|
| threshold | `rvt/vis.py:95` `DISK_MIN_RADIUS_PX` |
| radius mapping, footprint | `rvt/vis.py:618` `_window_radius`, `:630` `_window_footprint` |
| FFT window sums | `rvt/vis.py:645` `_disk_window_sums` |
| disk paths | `rvt/vis.py:731` `_mean_filter_disk`, `:2747` `_msrm_radii`, `:2760` `_msrm_disk`, `:3077` `_max_elevation_deviation_disk` |
| dispatch in the public functions | `rvt/vis.py:814` `mean_filter`, `:919` `slrm`, `:2862` `msrm`, `:3202` `max_elevation_deviation`, `:3386` `mstp` |
| settings and file names | `rvt/default.py:61` `_kernel_shape_suffix`, `:69` `_json_kernel_shape` |
| tile overlap | `rvt/tile.py:373` `_window_overlap` |
| square-path variance (KI-77) | `rvt/vis.py:3231-3237`, `:3278` (`v0.1.1`: `:2892-2898`, `:2939`) |
