"""Pre-research: anisotropy and contrast of window shapes for local mean / DEV (MSTP, SLRM).

Shapes (half-width r px): square |dx|,|dy| <= r; disk dx^2 + dy^2 <= r^2; equal-area disk (area of the square);
regular octagon (square cut by |dx| + |dy| <= r * sqrt(2)); cross-free "rect-3" approximation (3 rectangles).
Input: a bank 1 m high, 12 px = 3 m wide (0.25 m/px) at 0 deg and at 45 deg on a flat 0 m plane; statistics at
crest pixels only (brute force, float64), so the run takes seconds and no harness lock is needed.
Output: per r the crest response (mean removed: z - mean; DEV: (z - mean) / std) and the ratio 45/0 (1 = isotropic),
plus the ratio of each shape's 0-deg response to the square's (contrast continuity).
Usage: pixi run python shape_anisotropy.py
"""
import numpy as np

RES = 0.25
N = 513  # odd: the centre pixel lies on the bank axis for both orientations (as the review's C-15 repro)
W_PX = 12  # full width at half maximum of the Gaussian bank, 3 m
RADII = (8, 16, 32, 48, 96)


def bank(angle_deg):
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    cx = cy = (N - 1) / 2
    a = np.deg2rad(angle_deg)
    d = -(x - cx) * np.sin(a) + (y - cy) * np.cos(a)  # signed distance from the bank axis (px)
    sigma = W_PX / (2 * np.sqrt(2 * np.log(2)))
    z = np.exp(-0.5 * (d / sigma) ** 2)
    return z, d


def masks(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    sq = np.ones_like(x, dtype=bool)
    disk = x * x + y * y <= r * r
    re = np.sqrt((2 * r + 1) ** 2 / np.pi)
    R = int(np.ceil(re))
    ye, xe = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    disk_ea = xe * xe + ye * ye <= re * re
    octa = np.abs(x) + np.abs(y) <= r * np.sqrt(2)
    return {"square": sq, "disk": disk, "disk_eqarea": disk_ea, "octagon": octa}


def stats(z, m, cy, cx):
    r = m.shape[0] // 2
    w = z[cy - r:cy + r + 1, cx - r:cx + r + 1][m]
    mean = w.mean()
    std = w.std()
    return z[cy, cx] - mean, (z[cy, cx] - mean) / std if std > 0 else np.nan


def main():
    print(f"bank 1 m high, {W_PX} px = {W_PX * RES} m wide, 0.25 m/px; crest pixels; areas in px")
    for r in RADII:
        ms = masks(r)
        row = [f"r={r:3d} px = {r * RES:5.2f} m"]
        resp = {}
        for name, m in ms.items():
            vals = {}
            for ang in (0, 45):
                z, d = bank(ang)
                c = (N - 1) // 2
                vals[ang] = np.array(stats(z, m, c, c))
            resp[name] = vals
            row.append(f"{name}: area {int(m.sum()):6d}, SLRM 45/0 {vals[45][0] / vals[0][0]:.3f}, "
                       f"DEV 45/0 {vals[45][1] / vals[0][1]:.3f}")
        print("\n  ".join(row))
        sq = resp["square"][0]
        print("  contrast at 0 deg relative to square (SLRM, DEV): " + ", ".join(
            f"{k} {v[0][0] / sq[0]:.3f}/{v[0][1] / sq[1]:.3f}" for k, v in resp.items()))


if __name__ == "__main__":
    main()
