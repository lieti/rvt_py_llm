# Window-shape experiment report

Generated 2026-10-24 10:03 UTC on Linux-6.18.44-fc-v153-x86_64-with-glibc2.39, x86_64, Python 3.13.17, one thread.

## 1. Validation

```
VALIDATE OK
```

## 2. Anisotropy (synthetic bank)

```
bank 1 m high, 12 px = 3.0 m wide, 0.25 m/px; crest pixels; areas in px
r=  8 px =  2.00 m
  square: area    289, SLRM 45/0 0.918, DEV 45/0 0.825
  disk: area    197, SLRM 45/0 0.995, DEV 45/0 0.982
  disk_eqarea: area    293, SLRM 45/0 1.000, DEV 45/0 0.999
  octagon: area    229, SLRM 45/0 1.013, DEV 45/0 1.041
  contrast at 0 deg relative to square (SLRM, DEV): square 1.000/1.000, disk 0.720/0.878, disk_eqarea 0.951/0.923, octagon 0.797/0.872
r= 16 px =  4.00 m
  square: area   1089, SLRM 45/0 0.893, DEV 45/0 0.857
  disk: area    797, SLRM 45/0 0.996, DEV 45/0 0.994
  disk_eqarea: area   1093, SLRM 45/0 0.996, DEV 45/0 0.994
  octagon: area    869, SLRM 45/0 1.018, DEV 45/0 1.024
  contrast at 0 deg relative to square (SLRM, DEV): square 1.000/1.000, disk 0.847/0.865, disk_eqarea 0.950/0.938, octagon 0.868/0.878
r= 32 px =  8.00 m
  square: area   4225, SLRM 45/0 0.929, DEV 45/0 0.849
  disk: area   3209, SLRM 45/0 0.998, DEV 45/0 0.995
  disk_eqarea: area   4205, SLRM 45/0 1.000, DEV 45/0 1.001
  octagon: area   3465, SLRM 45/0 1.003, DEV 45/0 1.006
  contrast at 0 deg relative to square (SLRM, DEV): square 1.000/1.000, disk 0.933/0.871, disk_eqarea 0.970/0.936, octagon 0.947/0.896
r= 48 px = 12.00 m
  square: area   9409, SLRM 45/0 0.950, DEV 45/0 0.846
  disk: area   7213, SLRM 45/0 1.000, DEV 45/0 0.999
  disk_eqarea: area   9417, SLRM 45/0 0.999, DEV 45/0 0.998
  octagon: area   7669, SLRM 45/0 1.003, DEV 45/0 1.009
  contrast at 0 deg relative to square (SLRM, DEV): square 1.000/1.000, disk 0.958/0.875, disk_eqarea 0.982/0.940, octagon 0.966/0.895
r= 96 px = 24.00 m
  square: area  37249, SLRM 45/0 0.974, DEV 45/0 0.843
  disk: area  28917, SLRM 45/0 1.000, DEV 45/0 0.999
  disk_eqarea: area  37225, SLRM 45/0 1.000, DEV 45/0 1.000
  octagon: area  30637, SLRM 45/0 1.001, DEV 45/0 1.004
  contrast at 0 deg relative to square (SLRM, DEV): square 1.000/1.000, disk 0.980/0.880, disk_eqarea 0.991/0.941, octagon 0.985/0.903
```

## 3. Real samples

Real 1024² samples, 0.25 m/px; metrics on the inner 512² px (DN of the 8-bit display)

## low (110.8–111.8 m)

| output | variant | p99.9 DN | max DN | mean DN | share > 5 DN |
|---|---|---|---|---|---|
| MSTP | prototype square vs rvt.vis.mstp | 1 | 1 | 0.01 | 0.0 % |
| MSTP R_broad | disk vs square | 29 | 44 | 5.13 | 38.0 % |
| MSTP G_meso | disk vs square | 46 | 83 | 6.76 | 44.7 % |
| MSTP B_local | disk vs square | 110 | 183 | 11.90 | 59.4 % |
| MSTP all | disk vs square | 110 | 183 | 15.36 | 85.5 % |
| SLRM | disk vs square | 1 | 2 | 0.16 | 0.0 % |
| MSTP R_broad | disk_t8 vs square | 29 | 44 | 5.13 | 38.0 % |
| MSTP G_meso | disk_t8 vs square | 26 | 41 | 3.07 | 20.8 % |
| MSTP B_local | disk_t8 vs square | 0 | 0 | 0.00 | 0.0 % |
| MSTP all | disk_t8 vs square | 30 | 44 | 6.62 | 51.5 % |
| SLRM | disk_t8 vs square | 1 | 2 | 0.16 | 0.0 % |
| MSTP R_broad | disk_eqarea vs square | 21 | 26 | 3.03 | 14.8 % |
| MSTP G_meso | disk_eqarea vs square | 25 | 40 | 4.22 | 27.3 % |
| MSTP B_local | disk_eqarea vs square | 29 | 52 | 3.39 | 23.3 % |
| MSTP all | disk_eqarea vs square | 30 | 52 | 6.38 | 49.0 % |
| SLRM | disk_eqarea vs square | 1 | 1 | 0.05 | 0.0 % |
| MSTP R_broad | octagon vs square | 23 | 34 | 4.61 | 34.1 % |
| MSTP G_meso | octagon vs square | 34 | 64 | 5.63 | 39.4 % |
| MSTP B_local | octagon vs square | 110 | 183 | 11.41 | 56.7 % |
| MSTP all | octagon vs square | 110 | 183 | 13.95 | 80.6 % |
| SLRM | octagon vs square | 1 | 1 | 0.13 | 0.0 % |

## medium (567.9–601.5 m)

| output | variant | p99.9 DN | max DN | mean DN | share > 5 DN |
|---|---|---|---|---|---|
| MSTP | prototype square vs rvt.vis.mstp | 1 | 4 | 0.04 | 0.0 % |
| MSTP R_broad | disk vs square | 16 | 17 | 5.77 | 47.6 % |
| MSTP G_meso | disk vs square | 27 | 38 | 6.01 | 45.1 % |
| MSTP B_local | disk vs square | 110 | 181 | 13.72 | 60.9 % |
| MSTP all | disk vs square | 110 | 181 | 16.75 | 89.1 % |
| SLRM | disk vs square | 1 | 1 | 0.14 | 0.0 % |
| MSTP R_broad | disk_t8 vs square | 16 | 17 | 5.77 | 47.6 % |
| MSTP G_meso | disk_t8 vs square | 8 | 12 | 0.38 | 0.4 % |
| MSTP B_local | disk_t8 vs square | 0 | 0 | 0.00 | 0.0 % |
| MSTP all | disk_t8 vs square | 16 | 17 | 5.85 | 47.9 % |
| SLRM | disk_t8 vs square | 1 | 1 | 0.14 | 0.0 % |
| MSTP R_broad | disk_eqarea vs square | 4 | 4 | 1.42 | 0.0 % |
| MSTP G_meso | disk_eqarea vs square | 11 | 16 | 2.37 | 8.3 % |
| MSTP B_local | disk_eqarea vs square | 17 | 34 | 1.28 | 6.4 % |
| MSTP all | disk_eqarea vs square | 17 | 34 | 3.30 | 13.5 % |
| SLRM | disk_eqarea vs square | 1 | 1 | 0.06 | 0.0 % |
| MSTP R_broad | octagon vs square | 13 | 14 | 4.67 | 40.6 % |
| MSTP G_meso | octagon vs square | 19 | 27 | 4.52 | 34.8 % |
| MSTP B_local | octagon vs square | 110 | 180 | 14.06 | 62.1 % |
| MSTP all | octagon vs square | 110 | 180 | 15.92 | 85.6 % |
| SLRM | octagon vs square | 1 | 1 | 0.12 | 0.0 % |

## high (929.5–1093.4 m)

| output | variant | p99.9 DN | max DN | mean DN | share > 5 DN |
|---|---|---|---|---|---|
| MSTP | prototype square vs rvt.vis.mstp | 1 | 3 | 0.00 | 0.0 % |
| MSTP R_broad | disk vs square | 16 | 20 | 7.28 | 66.0 % |
| MSTP G_meso | disk vs square | 20 | 67 | 3.20 | 18.9 % |
| MSTP B_local | disk vs square | 48 | 171 | 4.65 | 28.0 % |
| MSTP all | disk vs square | 48 | 171 | 9.38 | 79.8 % |
| SLRM | disk vs square | 35 | 54 | 2.63 | 12.1 % |
| MSTP R_broad | disk_t8 vs square | 16 | 20 | 7.28 | 66.0 % |
| MSTP G_meso | disk_t8 vs square | 18 | 23 | 1.59 | 7.8 % |
| MSTP B_local | disk_t8 vs square | 0 | 0 | 0.00 | 0.0 % |
| MSTP all | disk_t8 vs square | 19 | 23 | 7.76 | 70.2 % |
| SLRM | disk_t8 vs square | 35 | 54 | 2.63 | 12.1 % |
| MSTP R_broad | disk_eqarea vs square | 10 | 10 | 3.07 | 19.7 % |
| MSTP G_meso | disk_eqarea vs square | 9 | 22 | 1.42 | 1.4 % |
| MSTP B_local | disk_eqarea vs square | 9 | 27 | 0.82 | 0.9 % |
| MSTP all | disk_eqarea vs square | 10 | 27 | 3.62 | 21.5 % |
| SLRM | disk_eqarea vs square | 12 | 15 | 0.69 | 1.3 % |
| MSTP R_broad | octagon vs square | 13 | 16 | 6.24 | 63.1 % |
| MSTP G_meso | octagon vs square | 15 | 42 | 2.46 | 10.2 % |
| MSTP B_local | octagon vs square | 47 | 171 | 4.44 | 25.2 % |
| MSTP all | octagon vs square | 47 | 171 | 8.19 | 74.7 % |
| SLRM | octagon vs square | 27 | 42 | 2.04 | 9.5 % |


Figures: `results/png/<sample>_mstp.png`, `results/png/<sample>_slrm.png`.

## 4. Cost, 1024²

n = 1024² px, 0.25 m/px, one thread, median of 3 (s); peak GB by tracemalloc

| band (radii) | square (SAT) | disk (FFT) | octagon (tables) | ratio disk/square | ratio octagon/square |
|---|---|---|---|---|---|
| local (1–5 px, 5 radii) | 0.114 | 0.296 | 2.319 | 2.6 | 20.4 |
| meso (5–50 px, 10 radii) | 0.266 | 1.176 | 4.282 | 4.4 | 16.1 |
| broad (50–500 px, 10 radii) | 0.457 | 6.107 | 6.042 | 13.4 | 13.2 |

| whole MSTP (3 bands, 25 radii) | time (s) | peak GB |
|---|---|---|
| square (this prototype, mean removed) | 1.683 | 0.41 |
| disk (this prototype, mean removed) | 8.298 | 0.81 |
| octagon (this prototype, mean removed) | 13.071 | 0.54 |
| rvt.vis.mstp 0.1.0 square path (numba) | 0.442 | 0.10 |


## 5. Cost, 2048²

n = 2048² px, 0.25 m/px, one thread, median of 3 (s); peak GB by tracemalloc

| band (radii) | square (SAT) | disk (FFT) | octagon (tables) | ratio disk/square | ratio octagon/square |
|---|---|---|---|---|---|
| local (1–5 px, 5 radii) | 0.602 | 2.430 | 16.810 | 4.0 | 27.9 |
| meso (5–50 px, 10 radii) | 1.013 | 3.950 | 29.831 | 3.9 | 29.4 |
| broad (50–500 px, 10 radii) | 1.603 | 14.748 | 34.787 | 9.2 | 21.7 |

| whole MSTP (3 bands, 25 radii) | time (s) | peak GB |
|---|---|---|
| square (this prototype, mean removed) | 11.616 | 1.40 |
| disk (this prototype, mean removed) | 24.167 | 2.03 |
| octagon (this prototype, mean removed) | 86.536 | 1.70 |
| rvt.vis.mstp 0.1.0 square path (numba) | 1.802 | 0.26 |

