"""Self-test of astcheck.py: honest commits pass, disguised behaviour changes fail.

Builds a throw-away git repository in a temporary directory with one package file ``rvt/a.py`` and one
runtime data file ``settings/s.json``, commits a sequence of changes with class prefixes, and checks that
``astcheck.check_range`` accepts exactly the honest ones. Last line ``ASTCHECK SELFTEST PASS`` or ``FAIL``.

Usage (from the repository root)::

    python tools/proof/astcheck_selftest.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import astcheck  # noqa: E402

BASE = '''def f(x, d):
    """Doc."""
    if x < 0:
        raise ValueError("negative x")
    if x > 9:
        raise ValueError(f"x = {x} too large")
    if d is None:
        raise KeyError(d["a"])
    return x + 1
'''

# (subject, new content of rvt/a.py or None, new content of settings/s.json or None, expected ok)
STEPS = [
    ("docs: reword the docstring and add a comment", BASE.replace('"""Doc."""', '"""New doc.\n\n    More."""\n    # why'), None, True),
    ("msg: reword a literal message", BASE.replace("negative x", "x must be >= 0"), None, True),
    ("msg: reword the literal part of an f-string", BASE.replace("too large", "is above 9"), None, True),
    ("docs: sneaky code change", BASE.replace("return x + 1", "return x + 2"), None, False),
    ("msg: sneaky exception type", BASE.replace("raise ValueError(\"negative x\")", "raise TypeError(\"negative x\")"), None, False),
    ("msg: sneaky subscript inside raise", BASE.replace('d["a"]', 'd["b"]'), None, False),
    ("msg: sneaky formatted expression", BASE.replace("{x}", "{x + 1}"), None, False),
    ("docs: sneaky settings change", None, '{"r": 11}\n', False),
    ("meta: sneaky settings change", None, '{"r": 12}\n', False),
    ("refactor: real change (needs equiv, not judged here)", BASE.replace("return x + 1", "return 1 + x"), None, True),
]


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "selftest@example.invalid")
        git(repo, "config", "user.name", "selftest")
        git(repo, "config", "core.autocrlf", "false")
        (repo / "rvt").mkdir()
        (repo / "settings").mkdir()
        (repo / "rvt" / "a.py").write_text(BASE, encoding="utf-8")
        (repo / "settings" / "s.json").write_text('{"r": 10}\n', encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "meta: initial")
        for subject, code, data, expected in STEPS:
            git(repo, "reset", "-q", "--hard", "HEAD")
            (repo / "rvt" / "a.py").write_text(code if code is not None else BASE, encoding="utf-8")
            (repo / "settings" / "s.json").write_text(data if data is not None else '{"r": 10}\n', encoding="utf-8")
            git(repo, "add", "-A")
            git(repo, "commit", "-q", "--allow-empty", "-m", subject)
            cls = subject.split(":", 1)[0]
            problems = astcheck.check(str(repo), "HEAD^", "HEAD", cls)
            got = not problems
            verdict = "ok" if got == expected else "WRONG"
            ok &= got == expected
            print(f"{verdict:<5} expected {'pass' if expected else 'fail'}, got {'pass' if got else 'fail'}: {subject}")
            # return to the base state so that every step is judged against the same parent content
            (repo / "rvt" / "a.py").write_text(BASE, encoding="utf-8")
            (repo / "settings" / "s.json").write_text('{"r": 10}\n', encoding="utf-8")
            git(repo, "add", "-A")
            git(repo, "commit", "-q", "--allow-empty", "-m", "refactor: back to base")
    print("ASTCHECK SELFTEST PASS" if ok else "ASTCHECK SELFTEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
