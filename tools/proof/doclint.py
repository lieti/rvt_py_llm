"""Docstring lint for rvt: every public item documented, numpy sections consistent with the signature.

Rules (docs/CONVENTIONS.md §1), checked per public module, class, function and method
(``__init__`` is covered by the class docstring):

- a docstring exists and its first line is a sentence (non-empty);
- if the function has parameters (except ``self``/``cls``): a ``Parameters`` section with a
  ``----------`` underline that names every parameter, in signature order, as ``name : type`` lines;
  no documented parameter that is not in the signature;
- if the function body has ``return <value>``: a ``Returns`` section.

Usage (from the repository root; ``--dir`` defaults to this repository)::

    python tools/proof/doclint.py [--dir .] [--module vis.py] [--verbose]

Prints counts per module and, with ``--verbose``, every problem. Last line ``DOCLINT PASS`` or
``DOCLINT FAIL (<n> problems)``.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repository root

MODULES = ("vis.py", "blend.py", "blend_func.py", "default.py", "tile.py", "_accel.py", "_blocking.py",
           "__init__.py")


def section(doc, name):
    """Return the body lines of a numpy section, or None if absent."""
    lines = doc.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.strip() == name and set(lines[i + 1].strip()) == {"-"} and len(lines[i + 1].strip()) >= len(name):
            body = []
            for nxt in lines[i + 2:]:
                if nxt.strip() and set(nxt.strip()) == {"-"} and body:
                    body.pop()  # that was the next section's title
                    break
                body.append(nxt)
            return body
    return None


def documented_params(body):
    indent = min((len(l) - len(l.lstrip()) for l in body if l.strip()), default=0)
    names = []
    for l in body:
        if l.strip() and len(l) - len(l.lstrip()) == indent:
            m = re.match(r"\s*\*{0,2}([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*(:|$)", l)
            if m:
                names += [n.strip() for n in m.group(1).split(",")]
    return names


def returns_value(fn):
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value is None):
            return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def check_function(qual, fn, problems):
    doc = ast.get_docstring(fn)
    if not doc or not doc.strip().splitlines()[0].strip():
        problems.append(f"{qual}: no docstring")
        return
    a = fn.args
    params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs if p.arg not in ("self", "cls")]
    params += [p.arg for p in (a.vararg, a.kwarg) if p is not None]
    if params:
        body = section(doc, "Parameters")
        if body is None:
            problems.append(f"{qual}: no Parameters section")
        else:
            got = documented_params(body)
            missing = [p for p in params if p not in got]
            extra = [g for g in got if g not in params]
            if missing:
                problems.append(f"{qual}: parameters not documented: {', '.join(missing)}")
            if extra:
                problems.append(f"{qual}: documented but not in signature: {', '.join(extra)}")
            if not missing and [g for g in got if g in params] != params:
                problems.append(f"{qual}: parameters documented out of signature order")
    if returns_value(fn) and section(doc, "Returns") is None and section(doc, "Yields") is None:
        problems.append(f"{qual}: returns a value but has no Returns section")


def lint_module(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems, items = [], 0
    items += 1
    if not ast.get_docstring(tree):
        problems.append(f"{path.name}: no module docstring")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            items += 1
            check_function(f"{path.name}:{node.name}", node, problems)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            items += 1
            if not ast.get_docstring(node):
                problems.append(f"{path.name}:{node.name}: no class docstring")
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and not m.name.startswith("_"):
                    items += 1
                    check_function(f"{path.name}:{node.name}.{m.name}", m, problems)
    return items, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(REPO))
    ap.add_argument("--module", action="append")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    total = 0
    for mod in a.module or MODULES:
        items, problems = lint_module(Path(a.dir) / "rvt" / mod)
        total += len(problems)
        print(f"{mod:<14} public items {items:>4}  problems {len(problems):>4}")
        if a.verbose:
            for p in problems:
                print(f"    {p}")
    print("DOCLINT PASS" if total == 0 else f"DOCLINT FAIL ({total} problems)")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
