"""Self-test of astcheck.py: honest commits pass, disguised behaviour changes fail.

Builds a throw-away git repository in a temporary directory with one package file ``rvt/a.py``, one runtime data
file ``settings/s.json``, two test files (``tests/test_a.py`` and ``tests/conftest.py`` frozen in
``tools/proof/frozen_manifest.json``, ``tests/test_b.py`` not frozen) and one tool file ``tools/proof/t.py``. Every
step starts from that base commit, commits one change with a class prefix, and checks that ``astcheck.check``
accepts exactly the honest ones. Last line ``ASTCHECK SELFTEST PASS`` or ``FAIL``.

Usage (from the repository root)::

    python tools/proof/astcheck_selftest.py
"""

import json
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


def g(y):
    """Other doc."""
    # a comment
    return y * 2


class C:
    """Class doc."""

    def m(self):
        """Method doc."""
        return 3
'''

A = "rvt/a.py"
S = "settings/s.json"
TEST_A = "tests/test_a.py"
TEST_B = "tests/test_b.py"
CONFTEST = "tests/conftest.py"
TOOL = "tools/proof/t.py"
BASE_FILES = {
    A: BASE,
    S: '{"r": 10}\n',
    TEST_A: "def test_a():\n    assert 1 == 1\n",
    TEST_B: "def test_b():\n    assert 2 == 2\n",
    CONFTEST: "import sys\n",
    TOOL: "print('tool')\n",
    "tools/proof/frozen_manifest.json": json.dumps({"commit": "0" * 40, "files": [CONFTEST, TEST_A]}) + "\n",
}

CODE = BASE.replace("return x + 1", "return x + 2")  # a real change of f

# (subject, {path: new content, or None to delete}, expected ok)
STEPS = [
    ("docs: reword the docstring and add a comment", {A: BASE.replace('"""Doc."""', '"""New doc.\n\n    More."""\n    # why')}, True),
    ("msg: reword a literal message", {A: BASE.replace("negative x", "x must be >= 0")}, True),
    ("msg: reword the literal part of an f-string", {A: BASE.replace("too large", "is above 9")}, True),
    ("docs: sneaky code change", {A: CODE}, False),
    ("msg: sneaky exception type", {A: BASE.replace("raise ValueError(\"negative x\")", "raise TypeError(\"negative x\")")}, False),
    ("msg: sneaky subscript inside raise", {A: BASE.replace('d["a"]', 'd["b"]')}, False),
    ("msg: sneaky formatted expression", {A: BASE.replace("{x}", "{x + 1}")}, False),
    ("docs: sneaky settings change", {S: '{"r": 11}\n'}, False),
    ("meta: sneaky settings change", {S: '{"r": 12}\n'}, False),
    ("refactor: real change (needs equiv, not judged here)", {A: BASE.replace("return x + 1", "return 1 + x")}, True),
    # test: (0.1.0 class)
    ("test: new test file", {"tests/test_c.py": "def test_c():\n    pass\n"}, True),
    ("test: sneaky edit of an existing test file", {TEST_B: "def test_b():\n    assert 3 == 3\n"}, False),
    # fix: code under rvt/ plus the docstrings and comments of the changed functions
    ("fix: real code change", {A: CODE}, True),
    ("fix: code change with its own docstring and comment", {A: CODE.replace('"""Doc."""', '"""Fixed doc."""\n    # why')}, True),
    ("fix: code change and an added import", {A: "import copy\n\n\n" + CODE}, True),
    ("fix: sneaky docstring edit of an unchanged function", {A: CODE.replace("Other doc.", "Other new doc.")}, False),
    ("fix: sneaky comment edit of an unchanged function", {A: CODE.replace("# a comment", "# another comment")}, False),
    ("fix: sneaky method docstring edit", {A: CODE.replace("Method doc.", "New method doc.")}, False),
    ("fix: sneaky class docstring edit", {A: CODE.replace("Class doc.", "New class doc.")}, False),
    ("fix: sneaky module docstring", {A: '"""Module doc."""\n' + CODE}, False),
    ("fix: no code change at all", {A: BASE.replace('"""Doc."""', '"""New doc."""')}, False),
    ("fix: sneaky settings change", {A: CODE, S: '{"r": 13}\n'}, False),
    ("fix: sneaky test change", {A: CODE, TEST_A: "def test_a():\n    assert 0 == 0\n"}, False),
    ("fix: sneaky new package file", {A: CODE, "rvt/b.py": "X = 1\n"}, False),
    # retire: edits of frozen test files only
    ("retire: edit of a frozen test file", {TEST_A: "def test_a():\n    assert 1 == 1  # 0.1.1: fixed, KI-1\n"}, True),
    ("retire: sneaky code change", {TEST_A: "def test_a():\n    pass\n", A: CODE}, False),
    ("retire: sneaky edit of a test file outside the frozen set", {TEST_B: "def test_b():\n    pass\n"}, False),
    ("retire: sneaky new test file", {"tests/test_d.py": "def test_d():\n    pass\n"}, False),
    ("retire: sneaky deletion of a frozen test file", {TEST_A: None}, False),
    ("retire: sneaky conftest change", {CONFTEST: "import os\n"}, False),
    ("retire: sneaky pytest.ini", {TEST_A: "def test_a():\n    pass\n", "pytest.ini": "[pytest]\n"}, False),
    ("retire: sneaky manifest change", {"tools/proof/frozen_manifest.json": "{}\n"}, False),
    # tool: tools/proof/** only
    ("tool: proof tool change", {TOOL: "print('tool 2')\n", "tools/proof/u.py": "print('new')\n"}, True),
    ("tool: sneaky test change", {TOOL: "print('tool 3')\n", TEST_B: "def test_b():\n    pass\n"}, False),
    ("tool: sneaky code change", {TOOL: "print('tool 4')\n", A: CODE}, False),
    ("tool: sneaky file next to tools/proof", {"tools/other.py": "print('x')\n"}, False),
]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def write(repo, files):
    for path, content in files.items():
        target = repo / path
        if content is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "selftest@example.invalid")
        git(repo, "config", "user.name", "selftest")
        git(repo, "config", "core.autocrlf", "false")
        write(repo, BASE_FILES)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "meta: initial")
        base = git(repo, "rev-parse", "HEAD").strip()
        for subject, changes, expected in STEPS:
            # every step is judged against the same parent: the base commit
            git(repo, "reset", "-q", "--hard", base)
            git(repo, "clean", "-q", "-fdx")
            write(repo, changes)
            git(repo, "add", "-A")
            git(repo, "commit", "-q", "--allow-empty", "-m", subject)
            cls = subject.split(":", 1)[0]
            problems = astcheck.check(str(repo), "HEAD^", "HEAD", cls)
            got = not problems
            verdict = "ok" if got == expected else "WRONG"
            ok &= got == expected
            print(f"{verdict:<5} expected {'pass' if expected else 'fail'}, got {'pass' if got else 'fail'}: {subject}")
            if got != expected:
                for p in problems:
                    print(f"      {p}")
        # check_range classifies by the subject prefix and names the further check of the new classes
        git(repo, "reset", "-q", "--hard", base)
        for subject, changes in (("fix: real code change", {A: CODE}),
                                 ("retire: edit of a frozen test file", {TEST_A: "def test_a():\n    pass\n"}),
                                 ("tool: proof tool change", {TOOL: "print('t')\n"})):
            write(repo, changes)
            git(repo, "add", "-A")
            git(repo, "commit", "-q", "-m", subject)
        range_ok = astcheck.check_range(str(repo), f"{base}..HEAD")
        print(f"{'ok' if range_ok else 'WRONG':<5} expected pass, got {'pass' if range_ok else 'fail'}: "
              f"check_range over fix:, retire:, tool:")
        ok &= range_ok
    print("ASTCHECK SELFTEST PASS" if ok else "ASTCHECK SELFTEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
