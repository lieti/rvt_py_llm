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

Usage (from the repository root; ``--repo`` defaults to this repository)::

    python tools/proof/astcheck.py --base start --head HEAD --class docs
    python tools/proof/astcheck.py --repo . --range start..HEAD

``--range`` checks every commit in the range against its parent, using the class named by the commit
subject prefix (``docs:``, ``msg:``, ``meta:``, ``test:``, ``refactor:``). Merge commits are skipped.
The last line is ``ASTCHECK PASS`` or ``ASTCHECK FAIL``.
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repository root
CLASSES = ("docs", "msg", "meta", "test", "refactor")
RUNTIME_DATA = ("settings/", "test_data/")  # read by rvt.default / rvt.blend at run time
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


def check(repo, base, head, cls):
    """Return a list of violation strings (empty = the commit satisfies its class)."""
    problems = []
    files = changed_files(repo, base, head)
    pkg = [(s, p) for s, p in files if p.startswith("rvt/")]
    # files the package reads at run time (settings json, example data): only refactor: may change them
    data = [(s, p) for s, p in files if p.startswith(RUNTIME_DATA)]
    if cls == "refactor":
        return problems
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
        tag = "needs-equiv" if cls == "refactor" else ("FAIL" if problems else "ok")
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
