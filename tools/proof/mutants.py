"""Mutation smoke test: show that the tests and equiv.py detect small wrong changes to the code.

No mutation library is available in the frozen environment, so the mutants are written by hand:
each one replaces one exact snippet of ``rvt/<file>.py`` by a plausible mistake (wrong padding mode,
float32 instead of float64 accumulation, off-by-one, swapped operands, changed NaN rule, changed
default). For every mutant a scratch tree is built from ``--ref`` (default ``main``), the snippet is
replaced, and the frozen tests run once under the recording plugin of ``equiv.py``:

- **killed by tests**: the test run fails;
- **killed by equiv**: the run's record log differs from the unmutated tree's log.

Two control edits that must *not* change behaviour (a local rename, a docstring edit) check that
equiv.py does not report false differences.

Thresholds (fixed before the first run, never lowered): tests kill >= 90 % of the mutants; equiv
kills 100 %; both controls are equivalent. Last line ``MUTANTS PASS`` or ``MUTANTS FAIL``.

Usage (from the repository root)::

    python tools/proof/mutants.py [--ref main] [--tests-ref tests-frozen] [--only m01,m07]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equiv  # noqa: E402

TEST_KILL_MIN = 0.90

# (id, file, old snippet, new snippet, occurrence (1-based), description)
MUTANTS = [
    ("m01", "vis.py", 'dem = np.pad(array=dem, pad_width=1, mode="edge")',
     'dem = np.pad(array=dem, pad_width=1, mode="reflect")', 1, "slope_aspect pads with reflect"),
    ("m02", "vis.py", "slope_out = np.rad2deg(np.arctan(tan_slope))",
     "slope_out = np.rad2deg(np.arctan(tan_slope * 1.0001))", 1, "slope in degrees off by 0.01 %"),
    ("m03", "vis.py", "sun_zenith_rad = np.pi / 2 - sun_elevation_rad",
     "sun_zenith_rad = np.pi / 2 - sun_elevation_rad * 1.01", 1, "hillshade zenith off by 1 %"),
    ("m04", "vis.py", "dem_i1 = dem_pad.astype(np.float64, copy=dem_pad.dtype != np.float64)",
     "dem_i1 = dem_pad.astype(np.float32, copy=True)", 1, "mean_filter integral image in float32"),
    ("m05", "vis.py", "slrm_out = dem - dem_mean_filter", "slrm_out = dem_mean_filter - dem", 1,
     "slrm sign swapped"),
    ("m06", "vis.py", "q = np.float64(s_r[c] - h_r[c]) / rad",
     "q = (np.float64(s_r[c]) - np.float64(h_r[c])) / rad", 1,
     "SVF numba kernel: difference in float64 instead of the height dtype"),
    ("m07", "vis.py", "ms_r[c] = a if (a > q or a != a) else q", "ms_r[c] = a if (a > q) else q", 1,
     "sky-illumination numba kernel: NaN rule of np.maximum dropped"),
    ("m08", "vis.py", "norma = np.sum((observer_height / distances) * (2 * distances + rad_inc)) * n_ang",
     "norma = np.sum((observer_height / distances) * (2 * distances + rad_inc + 1)) * n_ang", 1,
     "local_dominance normalisation area"),
    ("m09", "vis.py", "dem_convolve = np.maximum(dem_convolve, np.roll(dem_fine, (i, j), axis=(0, 1)))",
     "dem_convolve = np.minimum(dem_convolve, np.roll(dem_fine, (i, j), axis=(0, 1)))", 1,
     "coarse pyramid DEM takes the minimum"),
    ("m10", "vis.py", "cutoff = lightness", "cutoff = lightness * 1.1", 1, "mstp lightness"),
    ("m11", "vis.py", "opns_out = np.rad2deg(0.5 * np.pi - (opns_win / num_directions))",
     "opns_out = np.rad2deg(0.5 * np.pi - (opns_win / (num_directions + 1)))", 1, "openness divisor"),
    ("m12", "vis.py", "byte_data[np.isnan(byte_data)] = 255", "byte_data[np.isnan(byte_data)] = 0", 1,
     "byte_scale no-data value"),
    ("m13", "vis.py", "n = int(np.ceil(((feature_max - resolution) / (2 * resolution)) ** (1 / scaling_factor)))",
     "n = int(np.floor(((feature_max - resolution) / (2 * resolution)) ** (1 / scaling_factor)))", 1,
     "msrm number of scales"),
    ("m14", "blend_func.py", "return 1 - (1 - active) * (1 - background)",
     "return 1 - (1 - active) * (1 - background) * 0.999", 1, "screen blend"),
    ("m15", "blend_func.py", "opacity = opacity / 100", "opacity = opacity / 101", 1, "opacity scale"),
    ("m16", "blend_func.py", "return active * opacity + background * (1 - opacity)",
     "return background * opacity + active * (1 - opacity)", 1, "apply_opacity operands swapped"),
    ("m17", "tile.py", "                else:\n                    left_offset = overlap",
     "                else:\n                    left_offset = overlap - 1", 1, "tile left overlap off by one"),
    ("m18", "default.py", "self.svf_r_max = 10", "self.svf_r_max = 11", 1, "default SVF radius"),
    ("m19", "default.py", "self.hs_sun_el = 35", "self.hs_sun_el = 36", 1, "default sun elevation"),
    ("m20", "_accel.py", '("", "0", "false", "no", "off")', '("", "0", "false", "no")', 1,
     "RVT_NO_ACCEL=off no longer accepted"),
    ("m21", "blend.py", 'maximum=0.5, blend_mode="screen", opacity=25', 'maximum=0.5, blend_mode="screen", opacity=30',
     1, "e3mstp/e4mstp SLRM layer opacity"),
    # m22-m23 target the helpers introduced by the 0.1.0 refactor commits (release audit, finding I4)
    ("m22", "vis.py", "    box -= table[hi:hi + n_rows, lo:lo + n_cols]\n    box -= table[lo:lo + n_rows, hi:hi + n_cols]",
     "    box -= table[lo:lo + n_rows, hi:hi + n_cols]\n    box -= table[hi:hi + n_rows, lo:lo + n_cols]", 1,
     "_box_sum: the two subtractions in the other order"),
    ("m23", "vis.py", "if acc.dtype == np.result_type(acc.dtype, term.dtype):", "if True:", 1,
     "_accumulate: always in place (a float32 accumulator stays float32)"),
]

CONTROLS = [
    ("c01", "vis.py", "nan_dem", "nan_mask_renamed", 0, "local rename in vis.py (all occurrences)"),
    ("c02", "vis.py", "Compute slope and aspect with central finite differences.",
     "Compute the slope and the aspect with central finite differences.", 1,
     "docstring edit"),
]


def apply(root, file, old, new, occurrence):
    path = root / "rvt" / file
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0 or count < occurrence:
        raise SystemExit(f"snippet not found ({count} occurrences) in {file}: {old[:60]!r}")
    if occurrence == 0:
        text = text.replace(old, new)
    else:
        idx = -1
        for _ in range(occurrence):
            idx = text.index(old, idx + 1)
        text = text[:idx] + new + text[idx + len(old):]
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="main")
    ap.add_argument("--tests-ref", default="tests-frozen")
    ap.add_argument("--only")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None

    equiv.SCRATCH.mkdir(parents=True, exist_ok=True)
    base_root = equiv.assemble("MBASE", ref=a.ref, tests_ref=a.tests_ref)
    base_log, base_tail = equiv.run(base_root, "MBASE", "accel")
    if base_log["exitstatus"] != 0 or "skipped" in base_tail:
        print(f"unmutated tree fails its tests: {base_tail}\nMUTANTS FAIL")
        return 1
    print(f"base {a.ref}, tests {a.tests_ref}: {base_tail}")

    results = {}
    for kind, items in (("mutant", MUTANTS), ("control", CONTROLS)):
        for mid, file, old, new, occ, desc in items:
            if only and mid not in only:
                continue
            root = equiv.assemble(f"M_{mid}", ref=a.ref, tests_ref=a.tests_ref)
            apply(root, file, old, new, occ)
            log, tail = equiv.run(root, f"M_{mid}", "accel")
            bad, _ = equiv.compare(base_log, log)
            by_tests = log["exitstatus"] != 0
            by_equiv = bool(bad) or by_tests
            results[mid] = (kind, by_tests, by_equiv)
            if kind == "mutant":
                verdict = f"tests {'KILLED' if by_tests else 'survived'}, equiv {'KILLED' if by_equiv else 'survived'}"
            else:
                verdict = "equivalent (ok)" if not by_equiv else f"FALSE DIFFERENCE: {bad[:1]}"
            print(f"{mid} {desc:<62} {verdict}  [{tail}]")

    muts = [r for r in results.values() if r[0] == "mutant"]
    ctrls = [r for r in results.values() if r[0] == "control"]
    t_rate = sum(r[1] for r in muts) / len(muts) if muts else 1.0
    e_rate = sum(r[2] for r in muts) / len(muts) if muts else 1.0
    ctrl_ok = all(not r[2] for r in ctrls)
    print(f"mutants: {len(muts)}; killed by tests {t_rate:.0%} (min {TEST_KILL_MIN:.0%}); "
          f"killed by equiv {e_rate:.0%} (min 100%); controls equivalent: {ctrl_ok}")
    ok = t_rate >= TEST_KILL_MIN and e_rate == 1.0 and ctrl_ok
    print("MUTANTS PASS" if ok else "MUTANTS FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
