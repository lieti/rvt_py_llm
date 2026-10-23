# rvt_py_llm — Relief Visualization Toolbox in Python, optimized and documented

rvt_py_llm is a drop-in replacement for rvt_py, the Relief Visualization Toolbox Python library by Žiga Kokalj and
co-authors (ZRC SAZU and University of Ljubljana, https://github.com/EarthObservation/RVT_py). It computes relief
visualizations of digital elevation models (slope, hillshade, sky-view factor, openness, local dominance, sky
illumination, SLRM, MSRM, MSTP) and blends them into composites (VAT, e3MSTP, e4MSTP). The import name is still
`rvt`, the API is unchanged, and the outputs are bitwise identical to the upstream code with one precision fix. The
code is faster, every function is documented, and 1089 tests guard the behaviour. The upstream README is kept in
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## TL;DR (English)

- What: upstream rvt_py (master `bab109c`, version 2.2.3) + one MSTP precision fix + 11 performance commits +
  documentation and tests. Package `rvt_py_llm` 0.1.0, import `rvt`.
- Drop-in: same functions, signatures, defaults, settings files and output dtypes; existing scripts keep working.
- Speed: the 11 visualizations of a tiling pipeline together run x11.44 [95 % interval 11.35-11.46] faster than the
  fixed baseline on `bench_final` (three 2048 x 2048 px windows at 0.25 m/px), 1 thread per process.
- Outputs: bitwise identical to that baseline, with and without numba, for any thread count.
- numba is optional: if it is missing or fails, the numpy code runs with the same results; `RVT_NO_ACCEL=1` turns it off.
- Proof: frozen tests, AST check per commit, bitwise comparison of every public call against tag `start`, mutants.
- Read next: docs/CHANGES_VS_UPSTREAM.md, docs/ARCHITECTURE.md, docs/KNOWN_ISSUES.md, docs/TESTING.md.

## TL;DR (slovensky)

- Čo: pôvodná knižnica rvt_py (master `bab109c`, verzia 2.2.3) + oprava presnosti MSTP + 11 zrýchlení + dokumentácia
  a testy. Balík `rvt_py_llm` 0.1.0, import `rvt`.
- Náhrada 1:1: rovnaké funkcie, parametre, predvolené hodnoty a výstupy; existujúce skripty fungujú bez zmeny.
- Rýchlosť: 11 vizualizácií spolu ×11,44 [11,35–11,46] oproti opravenej baseline na `bench_final`
  (2048 × 2048 px, 0,25 m/px), 1 vlákno.
- Výstupy: bitovo zhodné s baseline, s numbou aj bez nej, pri ľubovoľnom počte vlákien.
- Numba je voliteľná: bez nej beží numpy s rovnakým výsledkom; `RVT_NO_ACCEL=1` ju vypne.
- Dôkaz: zmrazené testy, AST kontrola commitov, bitové porovnanie každého volania voči tagu `start`, mutačné testy.
- Ďalej čítaj: `docs/CHANGES_VS_UPSTREAM.md`, `docs/ARCHITECTURE.md`, `docs/KNOWN_ISSUES.md`.

## What changed compared to the original rvt_py, in plain words

- **Speed.** Local dominance, MSTP, sky-view factor and openness, sky illumination, SLRM, MSRM, e3MSTP and e4MSTP
  run 4 to 79 times faster (local dominance 33.97 s -> 0.43 s on a 2048 x 2048 px window). Slope and hillshade are
  unchanged. Memory use per call is lower or equal.
- **How.** The code computes only the pixels it returns instead of shifting whole padded arrays, works in blocks that
  fit the CPU cache, and runs four inner loops through numba when it is installed. It performs the same arithmetic
  in the same order, which is why the results are identical bit for bit.
- **One precision fix.** MSTP (and e3MSTP, e4MSTP, which use it) squared elevations in 32-bit floats, so the result
  depended on the absolute elevation: the same terrain 17.3 m higher changed the 8-bit e3MSTP/e4MSTP by up to 210 DN.
  The fix squares in 64-bit floats. These three outputs therefore differ from raw upstream; everything else is
  identical to raw upstream.
- **numba optional.** No new required dependency. numba costs about 0.2 s to import and 0.24-0.40 s to compile per
  kernel, once per process.
- **Documentation.** Every public module, class and function has a numpy-style docstring (units, dtypes, edges,
  NaN, in-place effects); new documents describe the architecture, the tests, the known issues and the changes.
- **Tests.** 223 equivalence tests of the optimized code and 866 characterization tests that pin the current
  behaviour of every public function, including the API itself.
- **Messages.** 40 exception texts were corrected (they named the old module `rvt.visualization`, a wrong function or
  a wrong bound). Exception types are unchanged.
- **Not changed.** API, defaults, settings files, file names and outputs (except the precision fix). Behaviour that
  looks wrong is documented in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), not fixed.

## Install

The package needs numpy, scipy, matplotlib and GDAL (`osgeo`); numba is optional. Tested only with Python 3.11.17,
numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, GDAL 3.12.3, numba 0.68.0 on Windows 11. `setup.py` allows Python >= 3.9.

- Development workspace: the environment is pinned by `pixi.lock` in the parent workspace of this repository
  (platforms win-64, linux-64, osx-arm64); `pixi install --locked`, then run Python through `pixi run`.
- pip: install GDAL first (it is easiest from conda-forge), then
  `pip install .` (or `pip install ".[accel]"` for numba, `pip install ".[test]"` for pytest and rasterio, which the
  tests use).
- conda: `environment.yml` lists the upstream environment; add `numba` for the accelerated paths.

## Quick start

```python
import numpy as np
import rvt.vis
import rvt.default

# a synthetic 200 x 300 px DEM at 0.5 m/px: elevations in m, row 0 is north
y, x = np.mgrid[0:200, 0:300]
dem = (100.0 + 5.0 * np.sin(x / 15.0) * np.cos(y / 20.0)).astype(np.float32)
resolution = 0.5  # m/px

# direct calls to the algorithms
slope = rvt.vis.slope_aspect(dem, resolution_x=resolution, resolution_y=resolution,
                             output_units="degree")["slope"]          # float32 (200, 300), degrees
svf = rvt.vis.sky_view_factor(dem, resolution=resolution, compute_svf=True,
                              svf_n_dir=16, svf_r_max=10)["svf"]      # float64 (200, 300); r = 10 px = 5 m

# the same with the default parameters of DefaultValues
default = rvt.default.DefaultValues()
slope_d = default.get_slope(dem, resolution_x=resolution, resolution_y=resolution)
svf_d = default.get_sky_view_factor(dem, resolution=resolution, compute_svf=True)["svf"]
```

`DefaultValues.save_<vis>(dem_path)` reads a DEM GeoTIFF and writes the result as GeoTIFF; `rvt.blend.e3mstp` and
`rvt.blend.e4mstp` compute the composites. More examples: the upstream documentation, https://rvt-py.readthedocs.io/.

## Environment variables

| variable | effect |
|---|---|
| `RVT_NO_ACCEL` | any value except empty, `0`, `false`, `no`, `off` switches the numba paths off; the numpy code runs, with the same results. Read on every call. |
| `RVT_NUM_THREADS` | thread count for parallel numba kernels; all kernels in rvt are serial, so it has no effect on results |
| `NUMBA_NUM_THREADS` | used when `RVT_NUM_THREADS` is not set; also numba's own setting |

For one tile per process, set `RVT_NUM_THREADS`, `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` to 1; rvt uses one CPU per call anyway. For a process that computes one cheap visualization on
a small tile, `RVT_NO_ACCEL=1` avoids the one-off numba cost.

## Tests

From the repository root (always pass the `tests` path; the upstream scripts `test_tile.py` and `test_rvt_tile.py`
at the root are not part of the suite):

```
python -m pytest tests -q                                   # 1089 passed
RVT_NO_ACCEL=1 python -m pytest tests -q                    # bash: numpy fallback path
$env:RVT_NO_ACCEL="1"; python -m pytest tests -q; Remove-Item Env:RVT_NO_ACCEL   # PowerShell
```

The proof tools (AST check, bitwise record comparison, frozen check, mutants) are described in
[docs/TESTING.md](docs/TESTING.md).

## Documents

| document | content |
|---|---|
| [docs/CHANGES_VS_UPSTREAM.md](docs/CHANGES_VS_UPSTREAM.md) | every change against upstream: precision fix, each performance commit with its equivalence argument and measured gain, the 0.1.0 pass, all changed exception texts |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, call graphs, data flow, accelerated paths, where to change what |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | 76 documented, unchanged issues with repros; tile buffers |
| [docs/TESTING.md](docs/TESTING.md) | test suites, frozen-test rule, proof chain, commands |
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md) | units, dtypes, NaN, edges; docstring and writing standard |
| [AGENTS.md](AGENTS.md) | rules for LLM sessions and other contributors |
| [CHANGELOG.md](CHANGELOG.md) | release history |
| [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md) | the upstream README, unchanged |

## License and credits

Apache License 2.0 ([LICENSE](LICENSE)), as upstream.

The Relief Visualization Toolbox and the rvt_py library are by Žiga Kokalj, Krištof Oštir, Klemen Zakšek, Peter
Pehani, Klemen Čotar, Maja Somrak, Žiga Maroh and Nejc Čož (Research Centre of the Slovenian Academy of Sciences and
Arts, and University of Ljubljana, Faculty of Civil and Geodetic Engineering). When using the methods, cite the
publications listed in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

rvt_py_llm (performance work, documentation, tests) is maintained by Tibor Lieskovský. The work was done with an LLM
(Claude) under his direction; every change is verified by the tests and proofs in [docs/TESTING.md](docs/TESTING.md).
