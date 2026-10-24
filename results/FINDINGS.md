# Findings: window shape for MSTP / SLRM (square vs disk vs octagon)

`run_all.py --big`, 2026-10-24, Linux x86_64, Python 3.13, numpy 2.5.4, scipy 1.18.1, numba 0.68.0, one thread,
median of 3; `validate.log` ends `VALIDATE OK`. Source: `results/REPORT.md`.
## 1. Cost (s, one thread; ratio = time / rvt.vis.mstp numba square path, whole MSTP)
| band (radii) | 1024² square SAT | disk FFT | octagon | 2048² square SAT | disk FFT | octagon |
|---|---|---|---|---|---|---|
| local (1–5 px, 5) | 0.114 | 0.296 | 2.319 | 0.602 | 2.430 | 16.810 |
| meso (5–50 px, 10) | 0.266 | 1.176 | 4.282 | 1.013 | 3.950 | 29.831 |
| broad (50–500 px, 10) | 0.457 | 6.107 | 6.042 | 1.603 | 14.748 | 34.787 |
| whole MSTP (25 radii) | 1.683 | 8.298 | 13.071 | 11.616 | 24.167 | 86.536 |
| peak memory (GB) | 0.41 | 0.81 | 0.54 | 1.40 | 2.03 | 1.70 |
| ratio to numba square | 3.8× | 18.8× | 29.6× | 6.4× | 13.4× | 48.0× |

rvt.vis.mstp 0.1.0 (numba square): 0.442 s / 0.10 GB at 1024², 1.802 s / 0.26 GB at 2048²; no per-band numba
times in the report, so ratios are whole-MSTP only.
## 2. Real samples (DN of 8-bit display, inner 512² px; p99.9 / max / mean / share > 5 DN)
| sample | variant | MSTP all | SLRM |
|---|---|---|---|
| low (flat) | disk | 110 / 183 / 15.36 / 85.5 % | 1 / 2 / 0.16 / 0.0 % |
| | disk_t8 | 30 / 44 / 6.62 / 51.5 % | 1 / 2 / 0.16 / 0.0 % |
| | disk_eqarea | 30 / 52 / 6.38 / 49.0 % | 1 / 1 / 0.05 / 0.0 % |
| | octagon | 110 / 183 / 13.95 / 80.6 % | 1 / 1 / 0.13 / 0.0 % |
| medium | disk | 110 / 181 / 16.75 / 89.1 % | 1 / 1 / 0.14 / 0.0 % |
| | disk_t8 | 16 / 17 / 5.85 / 47.9 % | 1 / 1 / 0.14 / 0.0 % |
| | disk_eqarea | 17 / 34 / 3.30 / 13.5 % | 1 / 1 / 0.06 / 0.0 % |
| | octagon | 110 / 180 / 15.92 / 85.6 % | 1 / 1 / 0.12 / 0.0 % |
| high (steep) | disk | 48 / 171 / 9.38 / 79.8 % | 35 / 54 / 2.63 / 12.1 % |
| | disk_t8 | 19 / 23 / 7.76 / 70.2 % | 35 / 54 / 2.63 / 12.1 % |
| | disk_eqarea | 10 / 27 / 3.62 / 21.5 % | 12 / 15 / 0.69 / 1.3 % |
| | octagon | 47 / 171 / 8.19 / 74.7 % | 27 / 42 / 2.04 / 9.5 % |

Most of the change is the local band (B_local p99.9 = 110 DN, disk and octagon, low/medium); t8 removes it
(0 DN); the equal-area radius gives the smallest change overall.
## 3. Anisotropy (Gaussian bank, ratio 45°/0° at the crest; SLRM / DEV)
| r (px) | square | disk | disk_eqarea | octagon |
|---|---|---|---|---|
| 8 | 0.918 / 0.825 | 0.995 / 0.982 | 1.000 / 0.999 | 1.013 / 1.041 |
| 32 | 0.929 / 0.849 | 0.998 / 0.995 | 1.000 / 1.001 | 1.003 / 1.006 |
| 96 | 0.974 / 0.843 | 1.000 / 0.999 | 1.000 / 1.000 | 1.001 / 1.004 |

r = 16/48 px in REPORT.md. As expected: square DEV 0.82–0.86, disk ≈ 1.00, octagon ≤ 1.01 from r = 32 px
(1.024–1.041 DEV at r ≤ 16 px).
## 4. Recommendation (2048² tiles; MSTP, e3MSTP and e4MSTP each pay the cost once)
Use the FFT disk only; do not carry the octagon as a second path. In this prototype the octagon is slower than the disk in every band at 2048²:
broad 34.8 vs 14.7 s, whole MSTP 86.5 vs 24.2 s. It is also less isotropic, and on real terrain it changes the
output about as much as the disk. Its "O(1) per pixel" advantage does not show up here. At ~24 s per MSTP, three
products cost ~73 s per 2048² tile against ~5.4 s for today's numba square (≈ 13×), at ~2 GB peak. Pair the disk
with the equal-area radius (smallest DN change, isotropy ≈ 1.00) or with the t8 rule (local band unchanged).
Assumption: e3/e4MSTP use similar radii. Caveat: the broad band needs a ≥ 500 px halo per tile, which pushes the
FFT size toward 3072², so real per-tile cost will be higher than the 2048² figures. A numba octagon was not tested
and could change this choice.
## 5. Things that look off
- Prototype square vs rvt.vis.mstp agrees at p99.9 ≤ 1 DN but not everywhere: max 4 DN (medium) and 3 DN (high).
  Mean ≤ 0.04 DN, so isolated pixels; possibly mean removal or float64 vs float32 (not checked in this run).
- Octagon scales superlinearly: local band 2.32 → 16.81 s (7.2×) and whole MSTP 13.1 → 86.5 s (6.6×) for 4× pixels.
  Possibly cache or memory effects in the anti-diagonal tables (not profiled).
- Square SAT whole MSTP at 2048² (11.6 s) is 3.6× its band sum (3.2 s); at 1024² it is 1.68 s vs 0.84 s. There is
  overhead outside the band loops (cause not identified); disk and octagon show only +3–5 s.
- Octagon over-responds at 45° for small r (DEV 1.041 at r = 8 px), plausibly from round(r√2).
