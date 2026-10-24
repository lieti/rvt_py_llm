"""Classify commits of rvt_py_llm by what they change in the package AST.

Proves without running anything that a commit cannot change behaviour:

- ``docs``: the AST of every ``rvt/**/*.py`` file, with docstrings removed, is identical before and
  after (comments and blank lines are not in the AST at all).
- ``msg``: as ``docs``, but the message arguments of exception constructors in ``raise`` statements and
  of ``warn(...)`` calls may differ (literal text only; see ``_MaskMessages``). Every other node must be
  identical.
- ``meta``: no file under ``rvt/`` changes.
- ``docs``, ``msg`` and ``meta`` also may not change ``settings/`` or ``test_data/``, which the package reads
  at run time.
- ``test``: only files under ``tests/`` are added; no existing file under ``tests/`` changes.
- ``refactor``: any change is allowed here; such commits must be proven by ``equiv.py`` and the tests.

Classes of the 0.1.1 fix phase (docs/FIXES_0_1_1.md; docs/TESTING.md section 7); each also needs the check named
in brackets, which this tool does not run:

- ``fix``: only existing ``rvt/**/*.py`` files are modified, and the AST without docstrings of at least one of them
  differs. A function or method whose code (AST without docstrings) is unchanged keeps its docstring and its source
  text (comments); the module and class docstrings are unchanged. Docstrings and comments may change only in the
  functions whose code changes. [full tests with and without ``RVT_NO_ACCEL=1``; the scoped record comparison of
  docs/TESTING.md section 7, EQUIV_DIFF OK]
- ``retire``: only files of the frozen test set (``tools/proof/frozen_manifest.json`` at the parent commit) are
  modified; no file is added or deleted; no pytest configuration file changes. [the retire check of docs/TESTING.md
  section 7, RETIRE OK: only the listed nodes change]
- ``tool``: only files under ``tools/proof/`` change. [the tool's own self-test; review of the diff]

Usage (from the repository root; ``--repo`` defaults to this repository)::

    python tools/proof/astcheck.py --base start --head HEAD --class docs
    python tools/proof/astcheck.py --repo . --range start..HEAD

``--range`` checks every commit in the range against its parent, using the class named by the commit
subject prefix (``docs:``, ``msg:``, ``meta:``, ``test:``, ``refactor:``, ``fix:``, ``retire:``, ``tool:``).
Merge commits are skipped. Commits of the classes ``refactor``, ``fix``, ``retire`` and ``tool`` that pass are
listed with the further check they need (``needs-equiv``, ``needs-tests+equiv_diff``, ``needs-retire_check``,
``needs-selftest``). The last line is ``ASTCHECK PASS`` or ``ASTCHECK FAIL``.
"""

import argparse
import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repository root
CLASSES = ("docs", "msg", "meta", "test", "refactor", "fix", "retire", "tool")
NEEDS = {"refactor": "needs-equiv", "fix": "needs-tests+equiv_diff", "retire": "needs-retire_check",
         "tool": "needs-selftest"}
RUNTIME_DATA = ("settings/", "test_data/")  # read by rvt.default / rvt.blend at run time
MANIFEST = "tools/proof/frozen_manifest.json"  # frozen test set: {"commit": ..., "files": [...], ...}
PYTEST_CONFIG = ("conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")  # by file name
MSG_PLACEHOLDER = "<message>"


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def changed_files(repo, base, head):
    """Return [(status, path)] between two commits (rename detection off)."""
    out = git(repo, "diff", "--no-renames", "--name-status", base, head)
    return [tuple(line.split("\t", 1)) for line in out.splitlines() if line.strip()]


def strip_docstrings(tree):
    """Remove the docstring of the module and of every class and function, in place."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _mask_message(expr):
    """Mask a message argument: a str literal, or the literal parts of an f-string.

    Expressions inside an f-string (``{name}``) and any other expression stay as they are, so a change of
    what is formatted into the message, or of a non-literal argument, is still a difference.
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return ast.copy_location(ast.Constant(MSG_PLACEHOLDER), expr)
    if isinstance(expr, ast.JoinedStr):
        values = [ast.Constant(MSG_PLACEHOLDER) if isinstance(v, ast.Constant) and isinstance(v.value, str) else v
                  for v in expr.values]
        return ast.copy_location(ast.JoinedStr(values=values), expr)
    return expr


class _MaskMessages(ast.NodeTransformer):
    """Mask only the message arguments of exception constructors in ``raise`` and of ``warn(...)`` calls.

    ``raise E("text")``: the positional and keyword arguments of the constructor call ``E(...)`` that are
    literals or f-strings. ``warn("text", ...)`` / ``warnings.warn(message="text")``: the first positional
    argument and the ``message`` keyword. Strings nested deeper (dict keys, subscripts, arguments of other
    calls) are not masked, so ``raise E(d["a"])`` -> ``raise E(d["b"])`` is a difference.
    """

    def visit_Raise(self, node):
        self.generic_visit(node)
        if isinstance(node.exc, ast.Call):
            node.exc.args = [_mask_message(a) for a in node.exc.args]
            for kw in node.exc.keywords:
                kw.value = _mask_message(kw.value)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == "warn":
            if node.args:
                node.args[0] = _mask_message(node.args[0])
            for kw in node.keywords:
                if kw.arg == "message":
                    kw.value = _mask_message(kw.value)
        return node


def normalized_dump(source, cls):
    tree = strip_docstrings(ast.parse(source))
    if cls == "msg":
        tree = _MaskMessages().visit(tree)
    return ast.dump(tree)


def _units(source):
    """Return {name: (code dump without docstrings, docstring, source text)} of the module docstring ("<module>"),
    every class (docstring only) and every function and method (top level and in top-level classes)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out = {"<module>": (None, ast.get_docstring(tree, clean=False), None)}

    def add(name, node):
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        code = ast.dump(strip_docstrings(copy.deepcopy(node)))
        out[name] = (code, ast.get_docstring(node, clean=False), "".join(lines[start - 1:node.end_lineno]))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, node)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = (None, ast.get_docstring(node, clean=False), None)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(f"{node.name}.{sub.name}", sub)
    return out


def _check_fix(repo, base, head, files):
    """Rules of the fix: class (module docstring)."""
    problems = []
    code_changed = False
    for s, p in files:
        if not p.startswith("rvt/"):
            problems.append(f"{s} {p}: fix commit changes a file outside rvt/")
            continue
        if s != "M" or not p.endswith(".py"):
            problems.append(f"{s} {p}: fix commit may only modify existing Python files of the package")
            continue
        before_src = git(repo, "show", f"{base}:{p}")
        after_src = git(repo, "show", f"{head}:{p}")
        code_changed |= normalized_dump(before_src, "docs") != normalized_dump(after_src, "docs")
        before, after = _units(before_src), _units(after_src)
        for name in sorted(set(before) & set(after)):
            (code_b, doc_b, text_b), (code_a, doc_a, text_a) = before[name], after[name]
            if code_b is None:  # module or class: its docstring belongs to docs: commits
                if doc_b != doc_a:
                    problems.append(f"{p}::{name}: fix commit changes a module or class docstring")
            elif code_b == code_a and doc_b != doc_a:
                problems.append(f"{p}::{name}: fix commit changes the docstring of a function whose code is unchanged")
            elif code_b == code_a and text_b != text_a:
                problems.append(f"{p}::{name}: fix commit changes comments or layout of a function whose code is "
                                f"unchanged")
    if not code_changed and not problems:
        problems.append("fix commit changes no code under rvt/ (use docs:)")
    return problems


def _check_retire(repo, base, files):
    """Rules of the retire: class (module docstring)."""
    try:
        frozen = set(json.loads(git(repo, "show", f"{base}:{MANIFEST}"))["files"])
    except (subprocess.CalledProcessError, ValueError, KeyError):
        return [f"{MANIFEST}: missing or unreadable at {base}"]
    problems = []
    for s, p in files:
        if p.rsplit("/", 1)[-1] in PYTEST_CONFIG:
            problems.append(f"{s} {p}: retire commit changes a pytest configuration file")
        elif p not in frozen:
            problems.append(f"{s} {p}: retire commit changes a file outside the frozen test set")
        elif s != "M":
            problems.append(f"{s} {p}: retire commit adds or deletes a frozen test file")
    return problems


def check(repo, base, head, cls):
    """Return a list of violation strings (empty = the commit satisfies its class)."""
    problems = []
    files = changed_files(repo, base, head)
    pkg = [(s, p) for s, p in files if p.startswith("rvt/")]
    # files the package reads at run time (settings json, example data): only refactor: may change them
    data = [(s, p) for s, p in files if p.startswith(RUNTIME_DATA)]
    if cls == "refactor":
        return problems
    if cls == "fix":
        return _check_fix(repo, base, head, files)
    if cls == "retire":
        return _check_retire(repo, base, files)
    if cls == "tool":
        return [f"{s} {p}: tool commit changes a file outside tools/proof/" for s, p in files
                if not p.startswith("tools/proof/")]
    if cls in ("docs", "msg", "meta"):
        problems += [f"{s} {p}: {cls} commit changes data read by the package at run time" for s, p in data]
    if cls == "meta":
        return problems + [f"{s} {p}: meta commit changes the package" for s, p in pkg]
    if cls == "test":
        for s, p in files:
            if not p.startswith("tests/"):
                problems.append(f"{s} {p}: test commit changes a file outside tests/")
            elif s != "A":
                problems.append(f"{s} {p}: test commit modifies or deletes an existing test file")
        return problems
    for s, p in pkg:
        if s != "M":
            problems.append(f"{s} {p}: {cls} commit adds or removes a package file")
            continue
        if not p.endswith(".py"):
            problems.append(f"{s} {p}: {cls} commit changes a non-Python package file")
            continue
        before = normalized_dump(git(repo, "show", f"{base}:{p}"), cls)
        after = normalized_dump(git(repo, "show", f"{head}:{p}"), cls)
        if before != after:
            problems.append(f"{p}: AST differs beyond what class '{cls}' allows")
    return problems


def check_range(repo, rng):
    commits = git(repo, "rev-list", "--reverse", "--no-merges", rng).split()
    failed = 0
    for c in commits:
        subject = git(repo, "log", "-1", "--format=%s", c).strip()
        cls = subject.split(":", 1)[0].strip() if ":" in subject else ""
        if cls not in CLASSES:
            print(f"{c[:8]} FAIL  no class prefix: {subject}")
            failed += 1
            continue
        problems = check(repo, f"{c}^", c, cls)
        tag = "FAIL" if problems else NEEDS.get(cls, "ok")
        print(f"{c[:8]} {tag:<11} {subject[:70]}")
        for p in problems[:10]:
            print(f"    {p}")
        failed += bool(problems)
    return failed == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--base")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--class", dest="cls", choices=CLASSES)
    ap.add_argument("--range")
    a = ap.parse_args()
    if a.range:
        ok = check_range(a.repo, a.range)
    else:
        if not (a.base and a.cls):
            ap.error("--base and --class are required without --range")
        problems = check(a.repo, a.base, a.head, a.cls)
        for p in problems:
            print(p)
        ok = not problems
    print("ASTCHECK PASS" if ok else "ASTCHECK FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
