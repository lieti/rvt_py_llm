"""Cost of square, disk (FFT) and octagon (tables) window sums for MSTP's radii; one thread.

Synthetic DEM (plane + waves + noise, 0.25 m/px, offset 600 m) of n x n px. Per scale band (local 1..5, meso 5..50,
broad 50..500 px) the sums of z and z**2 and the counts for all radii of the band: 1 warm-up + 3 repeats, median.
Also the whole MSTP (three bands) per shape, and rvt.vis.mstp of the shipped package (numba if importable) as
the reference of what the square path costs today. Peak memory of each call by tracemalloc (numpy allocations).
Usage: python cost.py [--n 1024] [--repeats 3] [--out ../results/cost_1024.json]
Thread variables are set to 1 before numpy is imported.
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
          "RVT_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import windows as wn  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the shipped rvt package (exp/rvt)
BANDS = {"local": list(range(1, 6)), "meso": list(range(5, 51, 5)), "broad": list(range(50, 501, 50))}


def dem(n, offset=600.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] * 0.25
    z = 0.02 * x + 0.5 * np.sin(x / 15.0) * np.cos(y / 23.0) + rng.normal(0, 0.05, (n, n))
    return (z + offset).astype(np.float32)


def timed(fn, repeats):
    fn()  # warm-up
    ts = []
    tracemalloc.start()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(np.median(ts)), ts, peak / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else HERE.parent / "results" / f"cost_{a.n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    z = dem(a.n)
    zz = z.astype(np.float64) - float(z.mean())
    res = {"n": a.n, "res_m": 0.25, "platform": platform.platform(), "cpu": platform.processor(),
           "numpy": np.__version__, "repeats": a.repeats, "bands": {}, "mstp": {}}
    lines = [f"n = {a.n}² px, 0.25 m/px, one thread, median of {a.repeats} (s); peak GB by tracemalloc", "",
             "| band (radii) | square (SAT) | disk (FFT) | octagon (tables) | ratio disk/square | ratio octagon/square |",
             "|---|---|---|---|---|---|"]
    for band, radii in BANDS.items():
        row = {}
        for shape in wn.SHAPES:
            med, ts, peak = timed(lambda: wn.window_sums(zz, radii, shape), a.repeats)
            row[shape] = {"median_s": med, "runs_s": ts, "peak_gb": peak}
            print(f"{band:6s} {shape:8s} {med:7.3f} s  peak {peak:.2f} GB", flush=True)
        res["bands"][band] = row
        lines.append(f"| {band} ({radii[0]}–{radii[-1]} px, {len(radii)} radii) | {row['square']['median_s']:.3f} "
                     f"| {row['disk']['median_s']:.3f} | {row['octagon']['median_s']:.3f} "
                     f"| {row['disk']['median_s'] / row['square']['median_s']:.1f} "
                     f"| {row['octagon']['median_s'] / row['square']['median_s']:.1f} |")

    lines += ["", "| whole MSTP (3 bands, 25 radii) | time (s) | peak GB |", "|---|---|---|"]
    for shape in wn.SHAPES:
        med, ts, peak = timed(lambda: wn.mstp_rgb(z, shape), a.repeats)
        res["mstp"][shape] = {"median_s": med, "runs_s": ts, "peak_gb": peak}
        lines.append(f"| {shape} (this prototype, mean removed) | {med:.3f} | {peak:.2f} |")
        print(f"mstp   {shape:8s} {med:7.3f} s", flush=True)
    try:
        import rvt.vis
        import rvt._accel
        accel = bool(rvt._accel.enabled())
        med, ts, peak = timed(lambda: rvt.vis.mstp(z.copy()), a.repeats)
        res["mstp"]["rvt_square"] = {"median_s": med, "runs_s": ts, "peak_gb": peak, "numba": accel,
                                     "rvt_file": rvt.__file__}
        lines.append(f"| rvt.vis.mstp 0.1.0 square path ({'numba' if accel else 'numpy'}) | {med:.3f} | {peak:.2f} |")
        print(f"rvt    square   {med:7.3f} s (numba={accel})", flush=True)
    except Exception as e:  # noqa: BLE001
        res["mstp"]["rvt_square"] = {"error": repr(e)}
        lines.append(f"| rvt.vis.mstp | not run: {e!r} | |")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"COST DONE {out}")


if __name__ == "__main__":
    main()
