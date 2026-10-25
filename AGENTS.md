# AGENTS.md — rules for LLM sessions (and other contributors) working on rvt_py_llm

## What this repository is

rvt_py_llm is the upstream Relief Visualization Toolbox Python library (rvt_py, EarthObservation/RVT_py, by Žiga
Kokalj et al.) with one MSTP precision fix, eleven performance commits that keep every output bitwise identical, a
documentation and test pass (0.1.0), five fixes of the blend path that change CRIM, e3MSTP and e4MSTP on
purpose (0.1.1, [docs/FIXES_0_1_1.md](docs/FIXES_0_1_1.md)), and opt-in disk windows for MSTP, SLRM and MSRM
(0.2.0, keyword `kernel_shape`, [docs/DISK_WINDOWS.md](docs/DISK_WINDOWS.md)). The import name is `rvt`. Read
[README.md](README.md) first, then [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for where things are.

Tags: `baseline` (upstream `bab109c`), `baseline-fixed` (`03702f4`, + precision fix), `start` (`8413d3e`, optimized
code + equivalence tests), `v0.1.0` (`74c63d0`), `tests-frozen` (annotated, on `b691587`, the frozen test set of
0.2.0), `tests-frozen-0.1.0` (the 0.1.0 freeze, `c341ec8`), `tests-frozen-0.1.1` (the 0.1.1 freeze, `5b5cbad`),
`v0.1.1`, `v0.2.0`.

The reference for rule 1 below is `v0.2.0` from 0.2.0 on (`v0.1.1` for 0.1.1, `start` up to 0.1.0); the 0.1.1 fixes
changed the outputs of the functions listed in [docs/FIXES_0_1_1.md](docs/FIXES_0_1_1.md) and nothing else, and
0.2.0 changed no output of a call that does not pass `kernel_shape` or set a `*_kernel_shape` attribute. The disk
path itself (`kernel_shape` other than `"square"`) is held by its own tests (`tests/test_disk_*.py`) against a
brute-force reference, not by a bitwise reference tag.

## Hard rules

1. **Bitwise identity.** Every public function, method and class returns bitwise identical results to tag `start`,
   with and without numba, for any thread count; in-place effects on arguments, files written, exception types and
   warnings stay the same. "Close" is not enough: a change that moves one bit inside a test tolerance is wrong.
2. **Frozen tests.** Files that exist under `tests/` at tag `tests-frozen` are never modified, renamed or deleted, and
   no configuration may skip, deselect or xfail them. New tests go into new files. A failing frozen test means the
   code change is wrong.
3. **No API change.** Names, signatures, defaults, return shapes and dtypes, `DefaultValues` attributes, settings
   files and output file names stay as they are. `tests/test_api_snapshot.py` checks this. The one planned
   exception so far is 0.2.0: the keyword `kernel_shape` of five functions, three `DefaultValues` attributes, their
   settings keys and the `_disk` / `_diskea` file-name suffixes, all with defaults that keep the old behaviour.
4. **Document latent bugs, do not fix them.** Behaviour that looks wrong is described in the docstring Notes and in
   [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) (file:line at a tag, impact, repro), and pinned by a test with the
   comment `# pins current behaviour`. Fixing is the upstream authors' decision.
5. **numba fallback contract.** numba stays optional and is imported lazily (never at `import rvt`). Every kernel is
   declared with `@_accel.jit_kernel`, run through `_accel.call`, and has a numpy path that gives the same bits.
   `RVT_NO_ACCEL=1` must switch every kernel off. No fastmath, no on-disk cache, no `parallel=True`.
6. **Determinism.** Results must not depend on the thread count, the block size, the machine load or the order of
   calls. No process pools (rvt also runs inside QGIS), no caching of results between calls, no work after a
   function returns.
7. **No new dependencies.** Runtime: numpy, scipy, matplotlib, GDAL; optional numba. Portable code only (pathlib, no
   OS-specific calls).
8. **Hot code.** Bodies of numba kernels and inner block loops change only in a `refactor:` commit that is proven by
   the full test suite and `equiv.py`.

## Commit classes

The subject line starts with the class; one logical change per commit; subject <= 72 characters; the body says what
and why.

| class | allowed content | machine check |
|---|---|---|
| `docs:` | docstrings, comments, blank lines | AST without docstrings identical |
| `msg:` | text of exception or warning messages only | AST differs only in string constants inside `raise` / `warn` |
| `refactor:` | behaviour-neutral code edits | full test suite (accel and `RVT_NO_ACCEL=1`) + `equiv.py` PASS |
| `test:` | new test files only | frozen check PASS |
| `meta:` | packaging, repository documents (`*.md`, `setup.py`, requirements) | no change under `rvt/` |
| `fix:` | 0.1.1 only: code under `rvt/` of one listed fix, with the docstrings and comments of the changed functions | full tests (accel and `RVT_NO_ACCEL=1`) PASS; record differences only in the listed functions |
| `retire:` | 0.1.1 and 0.2.0 only: edits of frozen test files limited to listed nodes that pinned a fixed behaviour (0.1.1) or the API that 0.2.0 extends | only listed nodes changed |
| `tool:` | `tools/proof/**` only | the tool's own self-test |
| `feat:` | 0.2.0 only: code under `rvt/` and `settings/` that adds the disk path and its parameters, with the docstrings of the changed functions | full tests (accel and `RVT_NO_ACCEL=1`) PASS; records of every default call unchanged; determinism of the new path |

`fix:`, `retire:` and `tool:` were used for the 0.1.1 fixes, under a written list of the allowed functions and
nodes (see [docs/FIXES_0_1_1.md](docs/FIXES_0_1_1.md) and [docs/TESTING.md](docs/TESTING.md)), and `feat:`,
`retire:` and `tool:` for the 0.2.0 disk windows (TESTING.md section 8); rule 4 below still applies to every other
behaviour.

## How to verify a change

From the repository root:

```
python -m pytest tests -q                                   # 1428 passed
RVT_NO_ACCEL=1 python -m pytest tests -q                    # bash
$env:RVT_NO_ACCEL="1"; python -m pytest tests -q; Remove-Item Env:RVT_NO_ACCEL   # PowerShell
```

Proof tools, also from the repository root (they are in [tools/proof/](tools/proof/README.md); in the development
workspace, the parent directory with `pixi.toml`, prefix every command with `pixi run`):

```
python tools/proof/astcheck.py --repo . --range main..HEAD      # every commit against its class
python tools/proof/equiv.py --b-dir .                           # bitwise records vs tag start, 3 variants
python tools/proof/frozen_check.py --ref <branch>               # frozen tests untouched and PASSED
python tools/proof/doclint.py                                   # docstrings complete
```

`equiv.py` is required for every `refactor:` commit. A test must derive its expected values independently of the
code under test (analytic results, properties, a plain reference implementation in the test file); no golden files.
Details: [docs/TESTING.md](docs/TESTING.md). Never run plain `pytest` at the repository root (upstream scripts
`test_tile.py`, `test_rvt_tile.py` are there).

## Where things are

Line numbers refer to commit `234b156`; the function names are the stable references.

| need | file, entry points |
|---|---|
| algorithms | `rvt/vis.py`; kernels `vis._svf_max_slope_kernel` (`rvt/vis.py:857`), `vis._local_dominance_kernel` (`:1355`), `vis._sky_illum_search_kernel` (`:1864`), `vis._sky_illum_search_literal_kernel` (`:1893`), `vis._mstp_dev_max_kernel` (`:2758`) |
| default parameters, `get_*` / `save_*`, GeoTIFF I/O | `rvt/default.py`; `DefaultValues.__init__` (`rvt/default.py:317`), `get_raster_arr` (`:4125`), `save_raster` (`:4194`) |
| blending, e3MSTP, e4MSTP | `rvt/blend.py`, `rvt/blend_func.py`; `BlenderCombination.render_all_images` (`rvt/blend.py:616`), `blend.e3mstp` (`:1968`), `blend.e4mstp` (`:2040`), `blend_func.blend_images` (`rvt/blend_func.py:1171`) |
| tiling | `rvt/tile.py`; `tile._get_rvt_visualization_overlap` (`rvt/tile.py:372`) |
| disk windows (0.2.0) | `rvt/vis.py`: `DISK_MIN_RADIUS_PX`, `_window_radius`, `_window_footprint`, `_disk_window_sums`, `_mean_filter_disk`, `_msrm_disk`, `_max_elevation_deviation_disk`; [docs/DISK_WINDOWS.md](docs/DISK_WINDOWS.md) |
| numba switch, kernels registry | `rvt/_accel.py`; `_accel.enabled` (`rvt/_accel.py:49`), `_accel.call` (`:174`), `_accel.jit_kernel` (`:116`); per call site: docs/ARCHITECTURE.md section 4 |
| block size of blocked loops | `rvt/_blocking.py`; `BLOCK_ELEMS` (`rvt/_blocking.py:19`) |
| settings files | `settings/` |
| proof tools | [tools/proof/](tools/proof/README.md) |
| module map, call graphs, "change X look at Y" | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| units, dtypes, NaN, edges, docstring standard | [docs/CONVENTIONS.md](docs/CONVENTIONS.md) |
| latent bugs | [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) |
| what changed against upstream and why it is equivalent | [docs/CHANGES_VS_UPSTREAM.md](docs/CHANGES_VS_UPSTREAM.md) |

## Writing standard

[docs/CONVENTIONS.md](docs/CONVENTIONS.md): numpy-style docstrings with units, dtypes, shapes, NaN and edge handling;
comments say why, not what; no commented-out code; no references to files outside this repository in code comments.
Documents: English, plain declarative sentences, numbers with units and the data set they were measured on, radii as
px and m ("r = 10 px = 2.5 m at 0.25 m/px"), every claim about code with `file:line` at a named tag or commit, no
superlatives or marketing words. Do not explain SVF, MSTP and the other methods beyond one sentence with a literature
pointer.
