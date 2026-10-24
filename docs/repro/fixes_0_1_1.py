"""Repro of the five defects fixed in 0.1.1 (docs/FIXES_0_1_1.md): one line of numbers per check.

Each check is the repro of its KNOWN_ISSUES entry with an expected value computed here, independently of rvt:

- X1 (KI-33): overlay of 0.25 on 0.25 at opacity 30 % against o * B(a, b) + (1 - o) * b = 0.2125; whether the
  caller's background changed.
- X1 (KI-34): overlay of an RGB layer of 0.25 on a gray background of 0.25, per band (expected 0.125 in each band).
- X2 (KI-47): the units in which e4mstp asks rvt.vis.slope_aspect for the slope (recorded by wrapping the function
  for one call on a random 40 x 40 px DEM, relief sigma 5 m, 1 m/px), and the span of the normalized, inverted
  slope layer over 0..55 for both units (radians give a span of about 0.03).
- X3 (KI-19, KI-20, KI-21): scale_0_to_1 of [-2, 3] (expected [0, 1]); scale_within_0_and_1 of [-1, NaN, 2]
  (expected [0, NaN, 1], argument unchanged); scale_strict_0_to_1 of [1, 2, NaN] (expected [0, 1, NaN], no error).
- X4 (KI-45): svf_r_max of a DefaultValues set to 25 px, after color_relief_image_map (expected 25).
- X5 (KI-36): BlenderLayer.check_data with opacity 150 (expected: raises).

Usage, from the repository root or any other directory: ``python docs/repro/fixes_0_1_1.py [<checkout>]``. Without
an argument the package is imported from the repository root of this file; with one, from that checkout (for
example a worktree of tag v0.1.0), checked through ``rvt.__file__``. Exit code 0 when every check gives the
expected value, 10 otherwise. Runtime: about 3 s.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import rvt  # noqa: E402
import rvt.blend as rb  # noqa: E402
import rvt.blend_func as bf  # noqa: E402
import rvt.default  # noqa: E402
import rvt.vis  # noqa: E402

if not Path(rvt.__file__).resolve().is_relative_to(ROOT):
    raise SystemExit(f"rvt imported from {rvt.__file__}, not from {ROOT}")


def check(name, ok, text):
    print(f"{'ok  ' if ok else 'DIFF'} {name}: {text}")
    return ok


def main():
    print(f"rvt from {Path(rvt.__file__).parent}")
    results = []

    b = np.full((1, 2), 0.25)
    b_before = b.copy()
    top = bf.blend_overlay(np.full((1, 2), 0.25), b)
    out = bf.render_images(top, b, 30)
    expected = 0.3 * 0.125 + 0.7 * 0.25
    results.append(check("X1 KI-33 overlay 30 %", np.allclose(out, expected) and np.array_equal(b, b_before),
                         f"result {out[0, 0]:.4f} (expected {expected:.4f}); background changed: "
                         f"{not np.array_equal(b, b_before)}"))

    rgb = bf.blend_multi_dim_images("overlay", np.full((3, 1, 1), 0.25), np.full((1, 1), 0.25))[:, 0, 0]
    results.append(check("X1 KI-34 RGB overlay on gray", np.allclose(rgb, 0.125),
                         f"bands {np.round(rgb, 5).tolist()} (expected 0.125 each)"))

    dem = np.random.default_rng(0).normal(scale=5.0, size=(40, 40)).astype(np.float32)
    deg = rvt.vis.slope_aspect(dem=dem, resolution_x=1.0, resolution_y=1.0, output_units="degree")["slope"]
    rad = rvt.vis.slope_aspect(dem=dem, resolution_x=1.0, resolution_y=1.0)["slope"]
    layers = {}
    for unit, s in (("degree", deg), ("radian", rad)):
        layers[unit] = bf.normalize_image("Slope gradient", s.copy(), 0, 55, "value")
    # the units e4mstp asks for: slope_aspect is wrapped for one call (rvt itself is not modified)
    seen = []
    orig = rvt.vis.slope_aspect

    def spy(*args, **kwargs):
        seen.append(kwargs.get("output_units", "radian"))
        return orig(*args, **kwargs)

    rvt.vis.slope_aspect = spy
    try:
        rb.e4mstp(dem.copy(), 1.0, default=rvt.default.DefaultValues())
    finally:
        rvt.vis.slope_aspect = orig
    uses_degree = seen == ["degree"]
    used = layers["degree" if uses_degree else "radian"]
    span = float(np.nanmax(used) - np.nanmin(used))
    expected_span = float(np.nanmax(layers["degree"]) - np.nanmin(layers["degree"]))
    results.append(check("X2 KI-47 e4MSTP slope layer", uses_degree,
                         f"e4mstp slope in {'degrees' if uses_degree else 'radians'}; inverted layer span {span:.4f} "
                         f"(degrees give {expected_span:.4f}, radians "
                         f"{float(np.nanmax(layers['radian']) - np.nanmin(layers['radian'])):.4f})"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s01 = bf.scale_0_to_1(np.array([-2.0, 3.0]))
        results.append(check("X3 KI-19 scale_0_to_1([-2, 3])", np.allclose(s01, [0.0, 1.0]),
                             f"{s01.tolist()} (expected [0.0, 1.0])"))
        arg = np.array([-1.0, np.nan, 2.0])
        sw = bf.scale_within_0_and_1(arg)
        ok = np.allclose(sw, [0.0, np.nan, 1.0], equal_nan=True) and np.isnan(arg[1])
        results.append(check("X3 KI-20 scale_within_0_and_1([-1, NaN, 2])", ok,
                             f"{sw.tolist()} (expected [0.0, nan, 1.0]); argument afterwards {arg.tolist()}"))
        arg = np.array([1.0, 2.0, np.nan])
        try:
            ss = bf.scale_strict_0_to_1(arg)
            ok = np.allclose(ss, [0.0, 1.0, np.nan], equal_nan=True) and np.isnan(arg[2])
            text = f"{ss.tolist()} (expected [0.0, 1.0, nan]); argument afterwards {arg.tolist()}"
        except ValueError as e:
            ok, text = False, f"ValueError: {str(e)[:60]}; argument afterwards {arg.tolist()}"
        results.append(check("X3 KI-21 scale_strict_0_to_1([1, 2, NaN])", ok, text))

    d = rvt.default.DefaultValues()
    d.svf_r_max = 25
    rb.color_relief_image_map(np.random.default_rng(0).normal(size=(30, 30)).astype(np.float32), 1.0, default=d)
    results.append(check("X4 KI-45 svf_r_max after CRIM", d.svf_r_max == 25,
                         f"{d.svf_r_max} px (set to 25 px before the call)"))

    lay = rb.BlenderLayer("Hillshade", "value", 0, 1, "normal", opacity=150)
    try:
        lay.check_data()
        ok, text = False, "accepted"
    except Exception as e:  # the neighbouring checks raise Exception
        ok, text = True, f"raises {type(e).__name__}: {e}"
    results.append(check("X5 KI-36 check_data(opacity=150)", ok, text))

    print(f"{sum(results)} / {len(results)} checks give the expected value")
    return 0 if all(results) else 10


if __name__ == "__main__":
    sys.exit(main())
