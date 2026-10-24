"""Run the whole experiment and assemble results/REPORT.md.

Steps: validate.py (oracle check) -> anisotropy.py -> samples.py (three real 1024² samples) -> cost.py at 1024²
and, with --big, at 2048². Each step runs in a fresh process with the thread variables at 1. Fails fast if the
validation fails. Usage: python run_all.py [--big] [--repeats 3]
"""
import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"


def run(args, log_name):
    RES.mkdir(parents=True, exist_ok=True)
    log = RES / log_name
    print(f"== {' '.join(args)} -> {log.name}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run([sys.executable] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, text=True)
    tail = log.read_text(encoding="utf-8").strip().splitlines()[-1:]
    print("   " + (tail[0] if tail else "(no output)"), flush=True)
    return p.returncode, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="also time 2048² (several minutes)")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    rc, _ = run(["validate.py"], "validate.log")
    if rc != 0:
        print("VALIDATE FAIL: stop (see results/validate.log)")
        return 1
    run(["anisotropy.py"], "anisotropy.log")
    run(["samples.py"], "samples.log")
    run(["cost.py", "--n", "1024", "--repeats", str(a.repeats)], "cost_1024.log")
    if a.big:
        run(["cost.py", "--n", "2048", "--repeats", str(a.repeats)], "cost_2048.log")
    parts = [f"# Window-shape experiment report", "",
             f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} on {platform.platform()}, "
             f"{platform.processor() or 'cpu n/a'}, Python {platform.python_version()}, one thread.", "",
             "## 1. Validation", "", "```", (RES / "validate.log").read_text(encoding="utf-8").strip().splitlines()[-1], "```", "",
             "## 2. Anisotropy (synthetic bank)", "", "```", (RES / "anisotropy.log").read_text(encoding="utf-8").strip(), "```", "",
             "## 3. Real samples", "", (RES / "samples.md").read_text(encoding="utf-8") if (RES / "samples.md").exists() else "(samples.md missing)",
             "", "Figures: `results/png/<sample>_mstp.png`, `results/png/<sample>_slrm.png`.", "",
             "## 4. Cost, 1024²", "", (RES / "cost_1024.md").read_text(encoding="utf-8") if (RES / "cost_1024.md").exists() else "(missing)"]
    if a.big and (RES / "cost_2048.md").exists():
        parts += ["", "## 5. Cost, 2048²", "", (RES / "cost_2048.md").read_text(encoding="utf-8")]
    (RES / "REPORT.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"REPORT {RES / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
