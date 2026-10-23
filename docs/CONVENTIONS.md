# Conventions of rvt_py_llm

This file is the shared standard for code comments, docstrings and documents in this repository. It is
written for two readers: a developer who knows Python and relief visualization but not this code base,
and an LLM session that has to change the code without breaking it.

## 1. Docstrings (numpy style)

Every public module, class, function and method has a docstring in numpy style. Private helpers
(`_name`) have at least a one-paragraph docstring when they hold numerics that are not obvious.

```python
def slope_aspect(dem, resolution_x=1, resolution_y=1, output_units="radian", ve_factor=1, no_data=None):
    """
    Compute slope and aspect with central finite differences.

    One paragraph on what is computed, at which pixels, and how edges and NaN are handled.

    Parameters
    ----------
    dem : numpy.ndarray
        2D elevation array (m). Converted to float32 internally; the caller's array is not modified
        unless stated under ``no_data``.
    resolution_x : float
        Pixel size in x (m/px).
    ...

    Returns
    -------
    dict
        ``{"slope": numpy.ndarray float32 (H, W), "aspect": numpy.ndarray float32 (H, W)}``.

    Raises
    ------
    Exception
        If ``dem`` is not 2D or ``ve_factor`` is outside [-10000, 10000].

    Notes
    -----
    Implementation notes: algorithm, accelerated path, why a numeric choice was made.

    See Also
    --------
    hillshade
    """
```

Rules:
- The first line is one sentence in the imperative ("Compute …", "Return …").
- `Parameters` lists every parameter of the signature, in order, with type, unit and default meaning.
- `Returns` states type, dtype and shape relative to the input (`(H, W)`, `(3, H, W)`, `(16, H-2, W-2)`).
- `Notes` explains *why* for non-obvious implementation choices and names the accelerated path, if any.
- Docstrings describe behaviour as it is. If the behaviour looks wrong, describe it and point to
  `docs/KNOWN_ISSUES.md`; do not change it.

## 2. Comments

- Comments say *why*, not *what*. A comment above a block states its purpose in one line.
- Numeric choices carry their reason: float64 accumulation, block sizes (cache), tie rules, padding.
- No references to files outside this repository (no `work/...` paths of the optimization run); cite a
  document in `docs/` or a commit instead.
- No commented-out code.

## 3. Units and conventions used throughout `rvt`

| Topic | Convention |
|---|---|
| Elevation | metres, 2D `numpy.ndarray`, row 0 is north (top of the raster) |
| Pixel size | `resolution`, `resolution_x`, `resolution_y` in m/px |
| Radii, distances | in **px** unless the parameter name or docstring says m (e.g. MSRM feature sizes are in m). Documents state both: "r = 10 px = 2.5 m at 0.25 m/px" |
| Angles | `sun_azimuth` in degrees clockwise from north; `sun_elevation` in degrees above the horizon; outputs in the unit named by `output_units` or the docstring |
| No-data | `no_data` value is replaced by NaN; several functions do this **in the caller's array** before converting dtype (stated per function) |
| Dtype | inputs converted to float32; some outputs are float64 (stated per function) |
| NaN | NaN in the input stays NaN in the output at the same pixels unless a function fills it (stated per function) |
| Edges | each function pads with its own mode (reflect, edge, symmetric); the mode is stated in its docstring |
| `ve_factor` | vertical exaggeration, checked to lie in [-10000, 10000] |
| Accelerators | optional numba kernels registered through `rvt/_accel.py`, with a numpy fallback; `RVT_NO_ACCEL=1` switches them off; results are bitwise identical either way |
| Threads | `RVT_NUM_THREADS`, then `NUMBA_NUM_THREADS`; results do not depend on the thread count |

## 4. Documents

- English; plain declarative sentences; numbers with units and the data set they were measured on.
- Every claim about code carries `file:line` at a named commit or tag.
- No superlatives and no marketing words.
