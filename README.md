# Window-shape experiment for rvt_py_llm (square vs disk vs octagon)

Self-contained: numpy, scipy, matplotlib (numba optional, only for the rvt reference timing). No GDAL. Runs on
any Linux, macOS or Windows machine; the whole run takes a few minutes at 1024² and about 10–20 minutes with
`--big` (2048²), one thread.

```
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "numpy>=2" "scipy>=1.14" matplotlib numba
cd exp && python run_all.py --big
```

Outputs in `results/`: `REPORT.md` (everything), `validate.log` (must end `VALIDATE OK`), `anisotropy.log`,
`samples.md` + `samples.json` + `png/` (the three real 1024² samples: MSTP and SLRM, square vs disk vs
equal-area disk vs octagon, DN metrics on the inner 512²), `cost_1024.md/json` and `cost_2048.md/json` (time and
peak memory per band and per whole MSTP; `rvt.vis.mstp` of the shipped 0.1.0 package as the reference).

## What is here

| path | content |
|---|---|
| `exp/windows.py` | the three window-sum implementations (SAT square; FFT disk with data spectra reused across radii; octagon from a SAT plus anti-diagonal tables, O(1) per pixel), a brute-force oracle, DEV_max and MSTP built on them |
| `exp/validate.py` | oracle check of every shape, with NaN, two padding modes, the threshold rule, offset invariance |
| `exp/anisotropy.py` | Gaussian bank at 0° and 45° (review C-15 setup): crest response ratio and contrast per shape |
| `exp/samples.py` | real samples: MSTP (defaults, lightness 0.9) and SLRM (r = 20 px) per shape, DN metrics, figures |
| `exp/cost.py` | timing and peak memory, one thread |
| `samples/real_1024_{low,medium,high}.npz` | 1024 × 1024 px windows of three ALS DTMs at 0.25 m/px (keys `dem` float32, `resolution`); low ≈ 111 m flat, medium ≈ 580 m, high ≈ 930 m |
| `rvt/`, `settings/` | the rvt_py_llm 0.1.0 package (tag `v0.1.0`), used only for the reference MSTP |

Definitions: square |dx|, |dy| ≤ r; disk dx² + dy² ≤ r²; equal-area disk r_e = (2r + 1)/√π; octagon = square cut by
|dx| + |dy| ≤ round(r√2) (apothem r). All sums in float64 on mean-removed elevations; DEV and MSTP as `rvt.vis`
(std + 1e-6, running max of |DEV| over radii, 1 − exp(−lightness·|DEV|)).

## Questions the run answers

1. Cost: seconds per band and per whole MSTP at 1024² and 2048² for square (SAT and the numba path of rvt),
   FFT disk and octagon; peak memory. Which of disk and octagon is cheaper at the broad scale (50–500 px), and by
   how much against today's numba square path.
2. Effect on real terrain: DN difference distributions of MSTP (per band) and SLRM between square and each
   alternative on flat, medium and steep samples; whether the threshold rule (square below 8 px) and the
   equal-area radius reduce the change at the local/meso scale.
3. Isotropy: the bank test numbers for the alternatives (expected: disk ≈ 1.00, octagon ≤ 1.01 from r = 32 px,
   square 0.82–0.86 for DEV).

## Prompt for a Claude Code session on the cloud machine (token-light; no subagents needed)

```
Repository: this folder (branch circ-exp). Read README.md only. Create a venv, install numpy scipy matplotlib
numba, run "cd exp && python run_all.py --big" in the background and wait for its last line (REPORT ...). If
validate fails, stop and report the failing lines. When done, read results/REPORT.md and write
results/FINDINGS.md (≤ 60 lines, English, numbers with units): (1) the cost table per band and whole MSTP at
2048², with the ratio of disk and octagon to the rvt numba square path; (2) per sample the MSTP "all" and SLRM
rows for disk, disk_t8, disk_eqarea, octagon; (3) the anisotropy ratios; (4) a one-paragraph recommendation:
FFT disk, octagon, or both, for a tiling pipeline on 2048² tiles where MSTP, e3MSTP and e4MSTP each pay the cost
once; (5) anything that looks wrong in the data (e.g. prototype square vs rvt.vis.mstp not within 1 DN). Commit
results/REPORT.md, results/FINDINGS.md, results/*.md, results/*.json and results/png on the branch and push.
Do not modify anything under exp/ or rvt/.
```
