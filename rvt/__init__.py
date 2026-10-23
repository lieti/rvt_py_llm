"""
Relief Visualization Toolbox – python library (package ``rvt``).

rvt computes relief visualizations of digital elevation models (DEM), for example slope, hillshade, sky-view
factor, openness, local dominance, sky illumination, MSRM and MSTP, and blends them into composite images.
Importing the package imports no submodule; import the one you need (``import rvt.vis``).

Modules:

- ``rvt.vis``: the visualization algorithms on numpy arrays (no file I/O);
- ``rvt.default``: ``DefaultValues``, the default settings of every visualization, settings files, and
  ``get_*`` / ``save_*`` methods that read a DEM raster, call ``rvt.vis`` and write GeoTIFFs;
- ``rvt.blend``: layer blending (``BlenderCombination``, ``BlenderLayer``), JSON combination files, and the
  e3MSTP / e4MSTP recipes;
- ``rvt.blend_func``: blend modes, normalization and opacity on arrays, used by ``rvt.blend``;
- ``rvt.tile``: tile-by-tile processing of rasters larger than memory;
- ``rvt._accel``: private; optional numba kernels with a numpy fallback (``RVT_NO_ACCEL``,
  ``RVT_NUM_THREADS``);
- ``rvt._blocking``: private; the block size shared by the row-blocked loops of ``rvt.vis`` and
  ``rvt.blend_func``.

The module structure, the data flow and the accelerated paths are described in docs/ARCHITECTURE.md; units,
NaN, dtype and edge conventions in docs/CONVENTIONS.md; known issues in docs/KNOWN_ISSUES.md.

Credits:
    Žiga Kokalj (ziga.kokalj@zrc-sazu.si)
    Krištof Oštir (kristof.ostir@fgg.uni-lj.si)
    Klemen Zakšek
    Klemen Čotar
    Maja Somrak
    Žiga Maroh
    Nejc Čož

Copyright:
    2010-2020 Research Centre of the Slovenian Academy of Sciences and Arts
    2016-2020 University of Ljubljana, Faculty of Civil and Geodetic Engineering
"""
