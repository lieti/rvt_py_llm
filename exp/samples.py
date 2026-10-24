"""Square vs disk vs octagon (and equal-area disk) on the three real 1024 x 1024 px samples (0.25 m/px).

For each sample (samples/real_1024_{low,medium,high}.npz): MSTP (defaults: local 1-5, meso 5-50, broad 50-500 px,
lightness 0.9) and SLRM (r = 20 px = 5 m) with every shape, plus rvt.vis.mstp (the shipped square path) as a
check of the prototype's square. Differences against the square on the 8-bit display (MSTP 0..1 -> 0..255; SLRM
-2..2 m -> 0..255, as rvt's default stretch): p99.9, max, mean, share of pixels > 5 DN (max over bands), on the
window cropped by 512 px? No: the samples are 1024 px, so the broad scale (R = 500) is edge-affected everywhere;
metrics are therefore taken on the inner 512 x 512 px (256 px off each side) and reported as such.
Variants: disk (all radii), disk_t8 (the brief's rule: square below 8 px), disk_eqarea, octagon.
Writes results/samples.json, results/samples.md and results/png/<sample>_{mstp,slrm}.png.
Usage: python samples.py [--crop 256]
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "RVT_NUM_THREADS"):
    os.environ[v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import windows as wn  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
SCALES = ((1, 5, 1), (5, 50, 5), (50, 500, 50))
VARIANTS = {
    "disk": dict(shape="disk"),
    "disk_t8": dict(shape="disk", min_disk_r=8),
    "disk_eqarea": dict(shape="disk", eqarea=True),
    "octagon": dict(shape="octagon"),
}


def to_dn(x, lo, hi):
    return np.rint(np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo), 0, 1) * 255).astype(np.int16)


def metrics(a_dn, b_dn):
    d = np.abs(a_dn.astype(np.int32) - b_dn.astype(np.int32))
    if d.ndim == 3:
        d = d.max(axis=0)
    return {"p999": float(np.percentile(d, 99.9)), "max": int(d.max()), "mean": float(d.mean()),
            "share_gt5_pct": float((d > 5).mean() * 100)}


def mstp_variant(z, v):
    if v.get("eqarea"):
        bands = []
        for lo, hi, step in reversed(SCALES):
            radii = list(range(lo, hi + 1, step))
            r_eff = [wn.equal_area_radius(r) for r in radii]
            d = wn.dev_max(z, radii, "disk", r_eff=r_eff)
            bands.append(np.clip(1.0 - np.exp(-0.9 * np.abs(d)), 0, 1))
        return np.stack(bands).astype(np.float32)
    return wn.mstp_rgb(z, v["shape"], SCALES, 0.9, min_disk_r=v.get("min_disk_r"))


def slrm_variant(z, v, r=20):
    if v.get("eqarea"):
        s1, _, n = wn.window_sums(z - np.nanmean(z), [r], "disk", "edge", r_eff=[wn.equal_area_radius(r)])[0]
        return (z - (s1 / n + np.nanmean(z))).astype(np.float32)
    return (z - wn.mean_filter(z, r, v["shape"])).astype(np.float32)


def figure(path, title, panels, cmap=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    for ax, (name, img, kind) in zip(np.atleast_1d(axs), panels):
        if kind == "rgb":
            ax.imshow(np.clip(img, 0, 1).transpose(1, 2, 0))
        elif kind == "gray":
            ax.imshow(img, cmap="gray", vmin=-2, vmax=2)
        else:
            im = ax.imshow(img, cmap="magma", vmin=0, vmax=40)
            fig.colorbar(im, ax=ax, fraction=0.046, label="DN")
        ax.set_title(name, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=256, help="px removed on each side before the metrics")
    ap.add_argument("--quick", action="store_true", help="smoke test: one sample, short radius ranges")
    a = ap.parse_args()
    global SCALES
    names = ("low", "medium", "high")
    if a.quick:
        SCALES = ((1, 3, 1), (5, 10, 5), (50, 100, 50))
        names = ("low",)
    res_dir = ROOT / "results"
    png = res_dir / "png"
    png.mkdir(parents=True, exist_ok=True)
    c = a.crop
    sl = (slice(c, -c), slice(c, -c))
    results = {}
    lines = [f"Real 1024² samples, 0.25 m/px; metrics on the inner {1024 - 2 * c}² px (DN of the 8-bit display)", ""]
    try:
        import rvt.vis
        have_rvt = True
    except Exception:  # noqa: BLE001
        have_rvt = False
    for name in names:
        f = np.load(ROOT / "samples" / f"real_1024_{name}.npz")
        z = f["dem"].astype(np.float64)
        r = {"elev_min_m": float(np.nanmin(z)), "elev_max_m": float(np.nanmax(z))}
        sq = wn.mstp_rgb(z, "square", SCALES, 0.9)
        sq_dn = to_dn(sq, 0, 1)
        if have_rvt:
            ref = rvt.vis.mstp(z.astype(np.float32).copy(), local_scale=SCALES[0], meso_scale=SCALES[1],
                               broad_scale=SCALES[2], lightness=0.9)
            r["prototype_square_vs_rvt_mstp"] = metrics(sq_dn[(slice(None),) + sl], to_dn(ref, 0, 1)[(slice(None),) + sl])
        slrm_sq = slrm_variant(z, dict(shape="square"))
        slrm_sq_dn = to_dn(slrm_sq, -2, 2)
        panels_m = [("square", sq[:, sl[0], sl[1]], "rgb")]
        panels_s = [("square", slrm_sq[sl], "gray")]
        for vname, v in VARIANTS.items():
            m = mstp_variant(z, v)
            m_dn = to_dn(m, 0, 1)
            r[f"mstp_{vname}"] = {band: metrics(m_dn[i][sl], sq_dn[i][sl]) for i, band in
                                  enumerate(("R_broad", "G_meso", "B_local"))}
            r[f"mstp_{vname}"]["all"] = metrics(m_dn[(slice(None),) + sl], sq_dn[(slice(None),) + sl])
            s = slrm_variant(z, v)
            r[f"slrm_{vname}"] = metrics(to_dn(s, -2, 2)[sl], slrm_sq_dn[sl])
            if vname in ("disk", "octagon"):
                panels_m += [(vname, m[:, sl[0], sl[1]], "rgb"),
                             (f"|{vname} - square| DN", np.abs(m_dn - sq_dn).max(axis=0)[sl], "diff")]
                panels_s += [(vname, s[sl], "gray"), (f"|{vname} - square| DN", np.abs(to_dn(s, -2, 2) - slrm_sq_dn)[sl], "diff")]
            print(f"{name:6s} {vname:12s} mstp all: {r[f'mstp_{vname}']['all']}  slrm: {r[f'slrm_{vname}']}", flush=True)
        figure(png / f"{name}_mstp.png", f"MSTP, sample {name} ({r['elev_min_m']:.0f}–{r['elev_max_m']:.0f} m)", panels_m)
        figure(png / f"{name}_slrm.png", f"SLRM r = 20 px = 5 m, sample {name}", panels_s)
        results[name] = r
        lines += [f"## {name} ({r['elev_min_m']:.1f}–{r['elev_max_m']:.1f} m)", "",
                  "| output | variant | p99.9 DN | max DN | mean DN | share > 5 DN |", "|---|---|---|---|---|---|"]
        if have_rvt:
            q = r["prototype_square_vs_rvt_mstp"]
            lines.append(f"| MSTP | prototype square vs rvt.vis.mstp | {q['p999']:.0f} | {q['max']} | {q['mean']:.2f} | {q['share_gt5_pct']:.1f} % |")
        for vname in VARIANTS:
            for band in ("R_broad", "G_meso", "B_local", "all"):
                q = r[f"mstp_{vname}"][band]
                lines.append(f"| MSTP {band} | {vname} vs square | {q['p999']:.0f} | {q['max']} | {q['mean']:.2f} | {q['share_gt5_pct']:.1f} % |")
            q = r[f"slrm_{vname}"]
            lines.append(f"| SLRM | {vname} vs square | {q['p999']:.0f} | {q['max']} | {q['mean']:.2f} | {q['share_gt5_pct']:.1f} % |")
        lines.append("")
    (res_dir / "samples.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (res_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print("SAMPLES DONE")


if __name__ == "__main__":
    main()
