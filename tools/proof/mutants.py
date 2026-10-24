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

The mutants m24-m28 (``MUTANTS_0_1_1``) target the code of the 0.1.1 fixes X1-X5 (docs/FIXES_0_1_1.md,
docs/TESTING.md section 7) and apply only to a tree that contains them. Each is a list of regular-expression edits limited to the source of one
function or method (``scope``), so that it does not depend on the exact layout of the fixed lines; an edit that
matches nothing stops the run with an error naming the scope.

The mutants m29-m33 (``MUTANTS_0_2_0``) target the disk path of 0.2.0 (``kernel_shape="disk"``,
docs/DISK_WINDOWS.md) in the same form; their scopes are the private names of that path in ``rvt/vis.py``
(``_window_footprint``, ``_disk_window_sums``, ``_mean_filter_disk``, and the module-level assignment of
``DISK_MIN_RADIUS_PX``), so they apply only to a tree that contains it: window radius off by one, mean removal
dropped, FFT pixel counts not rounded, square window at r = 8 px, ``zbar`` not added back by the mean filter.
The unrounded counts change the outputs only at the rounding level (bits of DEV where a window holds one valid
pixel), below every tolerance of the tests, so that mutant is expected to be killed by equiv only.

Thresholds (fixed before the first run, never lowered): tests kill >= 90 % of the mutants; equiv
kills 100 %; both controls are equivalent. Last line ``MUTANTS PASS`` or ``MUTANTS FAIL``.

Usage (from the repository root)::

    python tools/proof/mutants.py [--ref main] [--tests-ref tests-frozen] [--only m01,m07]
"""

import argparse
import ast
import re
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
    # m14-m16 and m21 (blend_func.blend_screen, apply_opacity; e3mstp SLRM layer) are outside the 0.1.1 changes and
    # still apply to the fixed code; no mutant of m01-m23 had to be retired for 0.1.1.
]

# 0.1.1 fixes: (id, [(file, scope, pattern, replacement), ...], description). `scope` is a top-level function or
# "Class.method"; the pattern is applied (all matches) to the source lines of that scope only. The replacement is a
# re.sub replacement string or a function of the match.
_CLIP = r"np\.clip\(\s*(\w+)\s*,\s*0(?:\.0*)?\s*,\s*1(?:\.0*)?\s*\)|\b_clip\w*\(\s*(\w+)\s*\)"  # clip to [0, 1]


def _rescale(prefix):
    return lambda m: "{}scale_0_to_1({})".format(prefix, m.group(1) or m.group(2))


MUTANTS_0_1_1 = [
    ("m24", [("blend.py", "BlenderCombination.render_all_images",
              r"(_render_images\(\s*)(\w+)(\s*,\s*)(\w+)(\s*,)", r"\1\2\3\2\5")],
     "X1 opacity mix dropped: render_all_images mixes the blend result with itself"),
    ("m25", [("blend_func.py", "blend_multi_dim_images",
              r"(_assign_elementwise\(\s*blended_image\[i, :, :\],\s*func,\s*active\[i, :, :\],\s*)background(\s*\))",
              r"\1background if i == 0 else blended_image[i - 1, :, :]\2")],
     "X1 band aliasing restored: RGB over gray blends band i onto band i-1"),
    ("m26", [("blend.py", "e4mstp", r"""output_units\s*=\s*(["'])degree\1""", 'output_units="radian"')],
     "X2 slope unit flipped: e4mstp slope in radians again"),
    ("m27", [("blend_func.py", "_render_images", _CLIP, _rescale("")),
             ("blend.py", "BlenderCombination.render_all_images", _CLIP, _rescale("rvt.blend_func."))],
     "X3 clip replaced by the rescale (scale_0_to_1) in both sites"),
    ("m28", [("blend.py", "color_relief_image_map", r"(?:copy\.)?deepcopy\(\s*default\s*\)", "default")],
     "X4 deep copy removed: CRIM writes svf_r_max into the caller's default"),
]


def _drop_zbar(m):
    """m33: remove the term that adds zbar back (``+ zbar``, ``zbar +``, ``+= zbar``, ``np.add(x, zbar``)."""
    if m.group(1):
        return m.group(1) + "0.0"
    return "+= 0.0" if m.group(0).startswith("+=") else ""


# 0.2.0 disk windows (kernel_shape="disk" / "disk_equal_area", docs/DISK_WINDOWS.md), same form as MUTANTS_0_1_1.
# The scopes are the private names of the disk path in rvt/vis.py; "DISK_MIN_RADIUS_PX" is the module-level
# assignment of that constant. They apply only to a tree that contains the disk path.
MUTANTS_0_2_0 = [
    ("m29", [("vis.py", "_window_footprint", r"(def _window_footprint\((\w+)[^)]*\)[^:\n]*:[ \t]*\n)",
              r"\1    \2 = \2 + 1\n")],
     "disk path: window radius off by one (footprint of r_w + 1)"),
    ("m30", [("vis.py", "_disk_window_sums", r"(\bz_?bar\s*=\s*)[^\n]+", r"\g<1>0.0")],
     "disk path: mean removal dropped (zbar = 0, sums of z and z^2 at full elevation)"),
    ("m31", [("vis.py", "_disk_window_sums", r"np\.(?:rint|round|around)\(", "(lambda a, *args, **kwargs: a)(")],
     "disk path: pixel counts of the FFT not rounded to integers"),
    ("m32", [("vis.py", "DISK_MIN_RADIUS_PX", r"(\bDISK_MIN_RADIUS_PX\s*(?::[^=\n]+)?=\s*)8\b", r"\g<1>9")],
     "disk path: square window used at r = 8 px (threshold 9 px)"),
    ("m33", [("vis.py", "_mean_filter_disk",
              r"\+=\s*z_?bar\b|\s*\+\s*z_?bar\b|\bz_?bar\s*\+(?!=)\s*|(np\.add\([^,]+,\s*)z_?bar\b", _drop_zbar)],
     "disk path: mean filter does not add zbar back"),
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


def _scope_lines(text, scope):
    """Return (first, last) 1-based line numbers of a top-level function, of "Class.method", or of the module-level
    assignment of a name (``NAME = ...``) in `text`."""
    names = scope.split(".")
    body = ast.parse(text).body
    if len(names) == 1:
        for n in body:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
            if any(isinstance(t, ast.Name) and t.id == scope for t in targets):
                return n.lineno, n.end_lineno
    node = None
    for name in names:
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == name), None)
        if node is None:
            raise SystemExit(f"scope {scope!r} not found")
        body = node.body
    first = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return first, node.end_lineno


def apply_scoped(root, edits):
    """Apply regular-expression edits, each limited to the source lines of one function or method."""
    for file, scope, pattern, repl in edits:
        path = root / "rvt" / file
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        first, last = _scope_lines(text, scope)
        segment, count = re.subn(pattern, repl, "".join(lines[first - 1:last]))
        if count == 0:
            raise SystemExit(f"pattern not found in {file}::{scope}: {pattern[:60]!r}")
        path.write_text("".join(lines[:first - 1]) + segment + "".join(lines[last:]), encoding="utf-8")


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

    # (kind, id, description, function applying the edit to a scratch tree)
    plan = [("mutant", mid, desc, lambda root, e=(file, old, new, occ): apply(root, *e))
            for mid, file, old, new, occ, desc in MUTANTS]
    plan += [("mutant", mid, desc, lambda root, e=edits: apply_scoped(root, e))
             for mid, edits, desc in MUTANTS_0_1_1 + MUTANTS_0_2_0]
    plan += [("control", mid, desc, lambda root, e=(file, old, new, occ): apply(root, *e))
             for mid, file, old, new, occ, desc in CONTROLS]

    results = {}
    for kind, mid, desc, edit in plan:
        if only and mid not in only:
            continue
        root = equiv.assemble(f"M_{mid}", ref=a.ref, tests_ref=a.tests_ref)
        edit(root)
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
