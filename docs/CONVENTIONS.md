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
| Window shape | `kernel_shape` of `mean_filter`, `slrm`, `msrm`, `max_elevation_deviation`, `mstp` (0.2.0): `"square"` (default, (2r + 1) x (2r + 1) px), `"disk"` (pixels with dx² + dy² <= r²) or `"disk_equal_area"` (dx² + dy² <= r_w², r_w = (2r + 1) / sqrt(pi)); disks only for windows of r >= 8 px (`DISK_MIN_RADIUS_PX`), MSRM one shape for all its radii. Documents state the shape with the radius, e.g. "disk, r = 20 px = 5 m at 0.25 m/px" ([DISK_WINDOWS.md](DISK_WINDOWS.md)) |
| `ve_factor` | vertical exaggeration, checked to lie in [-10000, 10000]; `mstp` accepts only [-1000, 10000] (KNOWN_ISSUES KI-05) |
| Accelerators | optional numba kernels registered through `rvt/_accel.py`, with a numpy fallback; `RVT_NO_ACCEL=1` switches them off; results are bitwise identical either way |
| Threads | `RVT_NUM_THREADS`, then `NUMBA_NUM_THREADS`; results do not depend on the thread count |

### 3.1 Which public `rvt.vis` functions modify the caller's array when `no_data` is given

Derived from the `no_data` entries of the docstrings and checked by calling each function on a 48 x 48 px float32
array with one pixel equal to `no_data` (and on the same array as int16) at commit `234b156`. "Modified" means that
the pixels equal to `no_data` are NaN in the caller's array after the call. The in-place replacement happens before
the float32 conversion, so float64 arrays are modified too, and an integer array raises ValueError ("cannot convert
float NaN to integer"; KNOWN_ISSUES KI-17).

| function | caller's float array modified | integer array with `no_data` |
|---|---|---|
| `byte_scale`, 2D input | no (copied first) | ValueError |
| `byte_scale`, 3D input | yes (the bands are views of the input; KI-15) | ValueError |
| `slope_aspect` | no (converted to float32 first) | works, input unchanged |
| `hillshade` | yes | ValueError |
| `multi_hillshade` | yes | ValueError |
| `slrm` | yes | ValueError |
| `sky_view_factor` | no (converted to float32 first) | works, input unchanged |
| `local_dominance` | yes | ValueError |
| `sky_illumination` | yes | ValueError |
| `shadow_horizon` | yes | ValueError |
| `msrm` | yes | ValueError |
| `mstp` | yes | ValueError |

Functions without a `no_data` parameter (`mean_filter`, `sky_view_factor_compute`, `max_elevation_deviation`,
`integral_image`, `topographic_dev`, `fill_where_nan`) have no such replacement; `mean_filter`,
`max_elevation_deviation` and `fill_where_nan` were checked to leave a float32 input with NaN unchanged.
`DefaultValues.get_*` pass the caller's array and `no_data` on, so they inherit the row of the function they call
(checked: `get_hillshade`, `get_slrm` modify, `get_sky_view_factor` does not); `get_neg_opns` ignores `no_data`
(KI-65) and does not modify. Other in-place effects (not caused by `no_data`): `DefaultValues.float_to_8bit` clips its
argument (KI-60). Up to 0.1.0 `blend_func.scale_within_0_and_1` and `scale_strict_0_to_1` overwrote NaN (KI-20,
KI-21), overlay and soft light wrote into the background (KI-33), and `blend.color_relief_image_map` (also through
`e3mstp`) set `svf_r_max` on the caller's `DefaultValues` (KI-45); these effects are gone in 0.1.1
(docs/FIXES_0_1_1.md). The `no_data` replacement in the caller's DEM by `color_relief_image_map`, `e3mstp` and
`e4mstp` is unchanged.

## 4. Documents

- English; plain declarative sentences; numbers with units and the data set they were measured on.
- Every claim about code carries `file:line` at a named commit or tag.
- No superlatives and no marketing words.
