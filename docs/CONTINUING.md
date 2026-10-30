# Continuing the work

How to pick this repository up on another machine and keep going, with Claude Code (or another LLM agent) or by
hand. Written for two readers at once: the author, who is not a software developer, and the LLM session that
does the work. A person reads sections 1, 2 and 4; an LLM session reads everything and follows
[AGENTS.md](../AGENTS.md) above all.

## 1. Setup on a new machine

What you need: git, [pixi](https://pixi.sh) (one installer, no admin rights), and Claude Code if you want to
work with an LLM. Then, in a terminal (bash on Linux and macOS, PowerShell on Windows):

```
git clone https://github.com/lieti/rvt_py_llm
cd rvt_py_llm
pixi install --locked
pixi run python -m pytest tests -q
```

`pixi install --locked` builds the pinned environment from `pixi.lock`: the exact versions of Python 3.11, numpy,
scipy, GDAL, matplotlib, numba and pytest that every test and measurement in this repository ran in, for Windows,
Linux and macOS (Apple silicon). It lives in `.pixi/` inside the clone and touches nothing else on the machine.
The last line of the test run must be `1430 passed` (a few warnings are normal). If it is, the machine is ready;
the same run with `RVT_NO_ACCEL=1` in the environment must also end with `1430 passed`.

Every command in this document and in the other documents is run from the repository root and, in the pinned
environment, with the prefix `pixi run` (for example `pixi run python tools/proof/doclint.py`). Without pixi, any
Python 3.11 with the packages of `setup.py` extras `test` and `accel` works too; the numbers in the documents were
measured in the pinned environment.

To work with Claude Code: open the repository root in Claude Code. It reads [CLAUDE.md](../CLAUDE.md), which
sends it to [AGENTS.md](../AGENTS.md) and to this file. No other setup is needed; the rules, the checks and the
layout are all in the repository.

## 2. What you must know before changing anything

Five things explain almost every rule in [AGENTS.md](../AGENTS.md):

1. **Upstream is the reference.** rvt_py_llm exists to try things on top of RVT_py, not to replace it. Anything
   that changes an output on purpose is a proposal to the upstream authors, documented as such
   ([docs/CHANGES_VS_UPSTREAM.md](CHANGES_VS_UPSTREAM.md), [docs/FIXES_0_1_1.md](FIXES_0_1_1.md),
   [docs/DISK_WINDOWS.md](DISK_WINDOWS.md)), never a quiet edit.
2. **Bitwise identity is the default contract.** A change that does not say "this output changes" must give the
   same bits as the reference tag for every public call, with and without numba, for any thread count. "Almost the
   same" is a failure. The tool that checks this is `tools/proof/equiv.py`.
3. **The frozen tests are the judge.** The test files that exist at tag `tests-frozen` are never edited, skipped
   or deleted. New tests go into new files. If a frozen test fails, the change is wrong, not the test.
4. **Every commit has a class** (`docs:`, `msg:`, `refactor:`, `test:`, `meta:`, ...) that says what it may
   touch, and `tools/proof/astcheck.py` checks it from the git history without running any code. One logical
   change per commit.
5. **Behaviour that looks wrong is documented, not fixed.** [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists 77
   such issues with a reproduction each. Fixing one is a decision of the upstream authors or of a planned fix
   phase (section 3.3), never a side effect of another change.

What is in the repository and what is not: the code (`rvt/`), the example settings files (`settings/`), the tests
(`tests/`, synthetic DEMs built inside the test code, no data files needed), the proof tools (`tools/proof/`),
the documents (`docs/`) and the release evidence (`docs/release_evidence/`). Not in the repository: the real DEM
tiles and the acceptance harness that measured speed and equivalence during the optimization run
([docs/TESTING.md](TESTING.md) section 9), and the method kit (briefs, agent definitions, the harness code and
the research on disk windows) that the author keeps in a separate, unpublished directory. Nothing in this
repository depends on either; they are needed only to reproduce the published speed numbers or to run a whole
new phase the way the first ones were run.

## 3. The four kinds of change

Decide which kind a change is before starting; it fixes the commit class, the checks and how much planning is
needed. For an LLM session: state the kind in the first message of the work, and stop and ask when a request
does not fit one of them.

### 3.1 Documents and messages (classes `docs:`, `msg:`, `meta:`)

Docstrings, comments, exception texts, the markdown documents, `setup.py`, the CI workflow. No behaviour
changes. Steps: edit, then

```
pixi run python tools/proof/doclint.py                                   # DOCLINT PASS (docstrings)
pixi run python tools/proof/astcheck.py --base HEAD~1 --head HEAD --class <class>   # after the commit: ASTCHECK PASS
```

A `meta:` commit must not touch anything under `rvt/`; a `docs:` commit must leave the code's syntax tree
identical; a `msg:` commit may change only string constants inside `raise` and `warn`. Documents follow
[docs/CONVENTIONS.md](CONVENTIONS.md) section 4: plain sentences, numbers with units and the data they were
measured on, `file:line` at a named tag for every claim about code, no superlatives.

### 3.2 Code that must not change any output (class `refactor:`, also `test:`)

A speed-up, a cleanup, a portability fix like the 0.2.2 paths. The contract is bitwise identity with the newest
version tag. Steps:

1. Write the change. If it touches a numba kernel or an inner block loop (AGENTS.md rule 8), keep the arithmetic
   in the same order; a different summation order moves bits.
2. Run the full tests twice, with the numba paths and with `RVT_NO_ACCEL=1`; both must end `1430 passed`.
3. Run the bitwise comparison of the working tree against the newest version tag, for example
   `pixi run python tools/proof/equiv.py --a v0.2.2 --b-dir .`; the last line must be `EQUIV PASS`. It runs the
   frozen tests three times on each tree (about 2 minutes per tree) and compares every return value, every array
   argument after the call and every file written.
4. Commit as `refactor:`; new tests for the change go into a new file in a separate `test:` commit; then
   `astcheck.py` over the range.
5. Add the change to [CHANGELOG.md](../CHANGELOG.md) and, if it is a performance change, to
   [docs/CHANGES_VS_UPSTREAM.md](CHANGES_VS_UPSTREAM.md) with the measurement.

### 3.3 Code that changes an output on purpose (a phase: classes `fix:` or `feat:`, `retire:`, `tool:`)

A bug fix, a new parameter, a new method. This is not a single commit but a phase, because it breaks the two
contracts above in a controlled way: some frozen tests pin the old behaviour and must be retired, and the
reference tag for bitwise identity moves. 0.1.1 (five blend fixes) and 0.2.0 (disk windows) were such phases;
[docs/TESTING.md](TESTING.md) sections 7 and 8 record how their scope was bounded and proven. The steps that
worked:

1. **Write the brief first**, as a document in the repository (`docs/<NAME>_BRIEF.md` or the decision section of
   the feature document): which functions may change their outputs and which may not, which frozen test nodes
   pin the old behaviour and will be retired, which new tests must exist before the code is written, how the new
   behaviour is verified (an independent reference, not the code itself), what the commit classes of the phase
   may touch, and what "done" is. Every decision a person would be asked for during the phase goes in here
   before any code is written. An LLM session must not start the code without this document.
2. **Tests before code**: the new tests are written and committed (`test:`) and fail against the current code
   for the right reason.
3. **The code**, one `fix:` or `feat:` commit per listed function, with the docstrings of the changed functions
   in the same commit.
4. **Retire** the listed frozen nodes in one `retire:` commit that changes only those nodes, and refreeze: new
   annotated tag `tests-frozen-<version>` on the old set first, then `tests-frozen` moved to the new set and
   `tools/proof/frozen_manifest.json` regenerated with `frozen_check.py --freeze` (this is the one `tool:` edit a
   phase always needs).
5. **Prove the scope**: the full tests twice; `equiv.py` against the previous version tag must fail only in the
   records of the listed functions (TESTING.md section 7 shows the scoped comparison); `mutants.py` with new
   mutants of the changed code; `astcheck.py` over the range with the phase's allowed classes.
6. **Document**: the feature or fix document (what changed, file:line, evidence, risk, open questions for the
   upstream authors), KNOWN_ISSUES entries marked fixed, CHANGELOG, README TL;DR, AGENTS.md (the new reference
   tag and the new allowed classes), then the version (section 3.4).

Expect a phase to take days, not hours, and to need the author's decisions at step 1 and at the end. The
heavier machinery that ran the earlier phases (one brief per phase, separate agents for writing, testing and
verifying, a path-ownership hook, an external harness) is in the author's method kit, not here; it is worth it
for a phase of 0.1.1 or 0.2.0 size, not for one fix.

### 3.4 A release

When `main` has something worth a tag: `setup.py` version, the TL;DR lines of [README.md](../README.md) in both
languages, a [CHANGELOG.md](../CHANGELOG.md) entry dated today with the commits and the evidence, the Versions
table of the README, then an annotated tag `v<major>.<minor>.<patch>` on the `meta:` commit. Patch for
documents and hygiene, minor for a change of behaviour or a new parameter. Push `main` with tags; the CI
workflow (`.github/workflows/tests.yml`) runs the suite on Ubuntu and Windows and must be green on both.

## 4. Working with an LLM session

For the author: open the repository in Claude Code and say what you want in plain words, with the kind of change
from section 3 when you know it. Three requests that match how this repository is meant to be used:

- "Explain what the function `rvt.vis.sky_illumination` does, where its time goes, and what KNOWN_ISSUES says
  about it. No code changes."
- "Make `rvt.vis.slope_aspect` faster without changing any output. Kind 3.2: full tests twice, `equiv.py`
  against `v0.2.2`, `refactor:` commit, then show me the before/after time on a 2048 x 2048 px DEM."
- "I want to fix KI-12. Write the brief for a kind 3.3 phase first (which outputs change, which frozen tests
  pin the old behaviour, which new tests come first) and stop there; I will read it before anything else."

What to look at when the session reports: the last line of each check (`1430 passed`, `EQUIV PASS`,
`ASTCHECK PASS`, `DOCLINT PASS`) and the list of commits with their classes. A session that reports a change as
done without those lines has not finished. A session that proposes to edit a frozen test, skip a test, add a
dependency, change a signature or default, or "fix" something it found on the way has left the rules; say no
and point it to AGENTS.md.

For the LLM session, in addition to AGENTS.md:

- Read [README.md](../README.md), [AGENTS.md](../AGENTS.md) and this file first, then
  [docs/ARCHITECTURE.md](ARCHITECTURE.md) for where things are. Do not read the upstream `.rst` documentation
  under `docs/` unless the task is about it; it is the upstream user manual, kept unchanged.
- Name the kind of change (section 3) before starting, and the reference tag you will compare against.
- Run the checks yourself and quote their last lines; do not assume a check passed because the edit looks right.
- Long runs (`equiv.py`, `mutants.py`, benchmarks) run alone on the machine; benchmark numbers taken while
  anything else runs are not reported.
- Never `git push`, never change `pixi.toml` or `pixi.lock`, never create files outside the repository, unless
  the author asks for exactly that.
- The author is not a developer: report in short plain sentences, say what changed and what was checked, and
  put long outputs into files under `.proof_scratch/` (ignored by git) rather than into the conversation.

## 5. Checklist before a change is called done

- [ ] The kind of change (3.1 to 3.4) was named and the commit classes match it.
- [ ] `pixi run python -m pytest tests -q` ends `1430 passed`, with and without `RVT_NO_ACCEL=1`.
- [ ] For 3.2: `equiv.py --a <newest version tag> --b-dir .` ends `EQUIV PASS`.
- [ ] For 3.3: the brief exists, the new tests were committed before the code, the scoped `equiv.py` differs only
      in the listed functions, the frozen set was refrozen with a new manifest.
- [ ] `astcheck.py` over the new commits ends `ASTCHECK PASS`; `doclint.py` ends `DOCLINT PASS`.
- [ ] No frozen test file was touched (`frozen_check.py` ends `FROZEN PASS`).
- [ ] CHANGELOG, and the document that owns the change, updated; numbers carry units and the data set.
