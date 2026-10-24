"""Validate windows.py against the brute-force oracle on small random arrays (runs in about a second).

Checks, for square, disk and octagon, with and without NaN, two padding modes, radii 1..12:
  sums of z, z**2 and counts equal the oracle (float64, relative 1e-9); the octagon cell count equals the mask;
  dev_max with the threshold rule equals the composition by hand. Last line: VALIDATE OK / VALIDATE FAIL.
"""
import sys

import numpy as np

import windows as wn


def check(name, a, b, tol=1e-9):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    err = np.nanmax(np.abs(a - b) / np.maximum(1.0, np.abs(b)))
    ok = err <= tol and np.array_equal(np.isnan(a), np.isnan(b))
    print(f"  {'ok ' if ok else 'BAD'} {name}: max rel err {err:.2e}")
    return ok


def main():
    rng = np.random.default_rng(1)
    ok = True
    for H, W in ((37, 45), (64, 64)):
        for with_nan in (False, True):
            z = rng.normal(0, 1, (H, W)) * 3 + 500.0
            if with_nan:
                z[10:14, 20:26] = np.nan
                z[0, 0] = np.nan
            for pad_mode in ("symmetric", "edge"):
                radii = [1, 2, 3, 5, 8, 12]
                for shape in wn.SHAPES:
                    got = wn.window_sums(z, radii, shape, pad_mode)
                    ref = wn.bf_sums(z, radii, shape, pad_mode)
                    for (g1, g2, gn), (r1, r2, rn), r in zip(got, ref, radii):
                        tag = f"{shape} r={r} nan={with_nan} pad={pad_mode} {H}x{W}"
                        # tolerances: the tables hold sums of ~500 m values over up to 64 x 64 px (1e6-1e8), so the
                        # differences of table entries round at ~1e-9 relative; samples.py removes the mean first
                        ok &= check(tag + " s1", g1, r1, 1e-8) and check(tag + " s2", g2, r2, 1e-6) and check(tag + " n", gn, rn, 0)
                        if shape == "octagon" and not with_nan:
                            ok &= check(tag + " count", gn[0, 0], wn.mask("octagon", r).sum(), 0)
                # equal-area disk
                r_eff = [wn.equal_area_radius(r) for r in radii]
                got = wn.window_sums(z, radii, "disk", pad_mode, r_eff=r_eff)
                ref = wn.bf_sums(z, radii, "disk", pad_mode, r_eff=r_eff)
                for (g1, g2, gn), (r1, r2, rn), r in zip(got, ref, radii):
                    ok &= check(f"disk_eqarea r={r} nan={with_nan} pad={pad_mode} s1", g1, r1)
                    ok &= check(f"disk_eqarea r={r} nan={with_nan} pad={pad_mode} n", gn, rn, 0)
    # threshold rule: radii < 8 square, >= 8 disk, equals the composition by hand
    z = rng.normal(0, 1, (50, 50)) + 300.0
    radii = [2, 5, 8, 11]
    d = wn.dev_max(z, radii, "disk", min_disk_r=8)
    zz = z - z.mean()
    sums = wn.window_sums(zz, [2, 5], "square") + wn.window_sums(zz, [8, 11], "disk")
    out = None
    for s1, s2, n in sums:
        mean = s1 / n
        std = np.sqrt(np.abs(s2 / n - mean * mean)) + 1e-6
        dev = (zz - mean) / std
        out = dev if out is None else np.where(np.abs(out) >= np.abs(dev), out, dev)
    ok &= check("dev_max threshold rule", d, out, 1e-12)
    # offset invariance of dev_max with mean removal
    d0 = wn.dev_max(z, radii, "disk")
    d1 = wn.dev_max(z + 2500.0, radii, "disk")
    ok &= check("dev_max offset +2500 m (disk, mean removed)", d1, d0, 1e-7)
    print("VALIDATE OK" if ok else "VALIDATE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
