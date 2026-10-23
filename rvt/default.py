"""
Relief Visualization Toolbox – DefaultValues: default parameters, computing and saving visualizations

``DefaultValues`` holds one parameter set for every visualization of ``rvt.vis`` (slope, hillshade, shadow,
multi-direction hillshade, SLRM, SVF, anisotropic SVF, positive and negative openness, sky illumination, local
dominance, MSRM, MSTP) together with the output options (float and/or 8-bit GeoTIFF, 8-bit stretch) and the
tile settings. Its methods:

- ``get_<vis>``: compute a visualization from a numpy array with ``rvt.vis`` and the stored parameters;
- ``save_<vis>``: read a DEM GeoTIFF, compute, and write GeoTIFFs whose file names encode the parameters
  (``get_<vis>_file_name``, ``get_<vis>_path``); DEMs above ``tile_size_limit`` px are processed tile by tile
  with ``rvt.tile``; ``save_visualizations`` runs every visualization whose ``<vis>_compute`` flag is set and
  writes a log file (``create_log_file``);
- ``save_default_to_file`` / ``read_default_from_file``: settings file (json; a legacy ``.txt`` format is read);
- ``float_to_8bit``: stretch a float result to 0-255 with the ``<vis>_bytscl`` setting of that visualization;
- ``calculate_visualization``: compute float and 8-bit result for one ``RVTVisualization`` (used by
  ``rvt.tile``).

Module-level raster I/O helpers (GDAL): ``get_raster_arr``, ``get_raster_size``, ``save_raster``.

Relation to the other modules: ``rvt.vis`` holds the numerics and is called with the parameters stored here;
``rvt.blend_func.normalize_image`` and ``rvt.vis.byte_scale`` give the 8-bit stretch; ``rvt.tile`` uses
``DefaultValues.calculate_visualization`` and the file-name methods; ``rvt.blend`` (``BlenderCombination``)
uses a ``DefaultValues`` instance to compute (``get_<vis>``) or save and reload (``save_<vis>``,
``get_<vis>_path``) the layers of a blend that are not given as image files.

Units: elevation in m, pixel size (resolution) in m/px, radii and distances in px unless the attribute says m
(``msrm_feature_min``, ``msrm_feature_max``, ``ld_observer_h``), angles in degrees.

Credits:
    Žiga Kokalj (ziga.kokalj@zrc-sazu.si)
    Krištof Oštir (kristof.ostir@fgg.uni-lj.si)
    Klemen Zakšek
    Peter Pehani
    Klemen Čotar
    Maja Somrak
    Žiga Maroh
    Nejc Čož

Copyright:
    2010-2022 Research Centre of the Slovenian Academy of Sciences and Arts
    2016-2022 University of Ljubljana, Faculty of Civil and Geodetic Engineering
"""

import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import rvt.vis
import rvt.blend_func
import rvt.tile
import os
from osgeo import gdal
import numpy as np
import json
import datetime
import time


class RVTVisualization(Enum):
    """
    Identify one visualization; the value is the short name used as attribute prefix in ``DefaultValues``.

    ==================================  ============  ==========================================================
    Member                              Value         Visualization (``rvt.vis`` function)
    ==================================  ============  ==========================================================
    SLOPE                               "slp"         slope gradient (``slope_aspect``, key "slope")
    HILLSHADE                           "hs"          hillshade (``hillshade``)
    SHADOW                              "shd"         binary cast shadow (``shadow_horizon``, key "shadow");
                                                      uses the ``hs_`` sun position
    MULTI_HILLSHADE                     "mhs"         hillshade from ``mhs_nr_dir`` directions
                                                      (``multi_hillshade``)
    SIMPLE_LOCAL_RELIEF_MODEL           "slrm"        simple local relief model (``slrm``)
    SKY_VIEW_FACTOR                     "svf"         sky-view factor (``sky_view_factor``, key "svf")
    ANISOTROPIC_SKY_VIEW_FACTOR         "asvf"        anisotropic SVF (``sky_view_factor``, key "asvf")
    POSITIVE_OPENNESS                   "pos_opns"    positive openness (``sky_view_factor``, key "opns")
    NEGATIVE_OPENNESS                   "neg_opns"    negative openness (positive openness of the negated DEM)
    SKY_ILLUMINATION                    "sim"         sky illumination (``sky_illumination``)
    LOCAL_DOMINANCE                     "ld"          local dominance (``local_dominance``)
    MULTI_SCALE_RELIEF_MODEL            "msrm"        multi-scale relief model (``msrm``)
    MULTI_SCALE_TOPOGRAPHIC_POSITION    "mstp"        multi-scale topographic position, RGB (``mstp``)
    ==================================  ============  ==========================================================

    Used by ``DefaultValues.calculate_visualization``, ``DefaultValues.float_to_8bit``,
    ``DefaultValues.get_visualization_file_name`` / ``get_visualization_path`` and ``rvt.tile``. The value of
    SHADOW ("shd") is not an attribute prefix; the shadow parameters are ``hs_sun_azi`` and ``hs_sun_el``.
    """
    SLOPE = "slp"
    HILLSHADE = "hs"
    SHADOW = "shd"
    MULTI_HILLSHADE = "mhs"
    SIMPLE_LOCAL_RELIEF_MODEL = "slrm"
    SKY_VIEW_FACTOR = "svf"
    ANISOTROPIC_SKY_VIEW_FACTOR = "asvf"
    POSITIVE_OPENNESS = "pos_opns"
    NEGATIVE_OPENNESS = "neg_opns"
    SKY_ILLUMINATION = "sim"
    LOCAL_DOMINANCE = "ld"
    MULTI_SCALE_RELIEF_MODEL = "msrm"
    MULTI_SCALE_TOPOGRAPHIC_POSITION = "mstp"


class DefaultValues:
    """
    Hold the default parameters of every RVT visualization and compute or save visualizations with them.

    One instance is one parameter set. Change attributes directly, or read them from a settings file
    (``read_default_from_file``) and write them with ``save_default_to_file``. Method families (``<vis>`` is
    slope, hillshade, shadow, multi_hillshade, slrm, sky_view_factor, neg_opns, sky_illumination,
    local_dominance, msrm or mstp):

    - ``get_<vis>(dem_arr, ...)``: compute from a 2D numpy array with ``rvt.vis`` and return the result;
    - ``save_<vis>(dem_path, custom_dir, save_float, save_8bit)``: read a DEM GeoTIFF, compute and write the
      float32 and/or 8-bit GeoTIFF; above ``tile_size_limit`` px the DEM is processed tile by tile (``rvt.tile``);
    - ``get_<vis>_file_name(dem_path, bit8)`` / ``get_<vis>_path(dem_path, bit8)``: output name encoding the
      parameters / that name in the DEM directory;
    - ``float_to_8bit``, ``calculate_visualization``, ``save_visualizations``, ``create_log_file``.

    Flags are int 0/1 (bool works too). Radii and distances are in px unless marked m. The attributes below are
    grouped by visualization; each entry gives the default and the unit or the allowed values.

    Attributes
    ----------
    [General]
    overwrite : int
        Default 0 (0/1). 1 recomputes and replaces existing output files; 0 keeps existing files (a
        ``save_<vis>`` call returns 0 without computing when all requested outputs exist).
    ve_factor : float
        Default 1. Vertical exaggeration factor passed to every visualization (``rvt.vis`` accepts
        -10000..10000); not passed by ``get_mstp`` (known issue, see ``get_mstp``).

    [Slope gradient, prefix slp_]
    slp_compute : int
        Default 0 (0/1). Compute the slope in ``save_visualizations``.
    slp_output_units : str
        Default "degree"; "degree", "radian" or "percent". Unit of the slope.

    [Hillshade, prefix hs_, and shadow]
    hs_compute : int
        Default 1 (0/1). Compute the hillshade in ``save_visualizations``.
    hs_sun_azi : int
        Default 315 (deg). Sun azimuth, clockwise from north; also the shadow azimuth.
    hs_sun_el : int
        Default 35 (deg). Sun elevation above the horizon; also the shadow elevation.
    hs_shadow : int
        Default 0 (0/1). ``save_hillshade`` also writes the binary shadow raster.

    [Multi-direction hillshade, prefix mhs_]
    mhs_compute : int
        Default 0 (0/1). Compute in ``save_visualizations``.
    mhs_nr_dir : int
        Default 16 (count). Number of sun azimuths, 360 / mhs_nr_dir deg apart starting at 0 deg; bands of the
        float output.
    mhs_sun_el : int
        Default 35 (deg). Sun elevation.

    [Simple local relief model, prefix slrm_]
    slrm_compute : int
        Default 0 (0/1). Compute in ``save_visualizations``.
    slrm_rad_cell : int
        Default 20 (px). Radius of the trend (mean) filter; ``rvt.vis.slrm`` accepts 10..50 px. Example:
        r = 20 px = 10 m at 0.5 m/px.

    [Sky-view factor (svf_), anisotropic SVF (asvf_), positive (pos_opns_) and negative (neg_opns_) openness;
    the svf_ search parameters apply to all four]
    svf_compute : int
        Default 0 (0/1). Compute the SVF in ``save_visualizations``.
    svf_n_dir : int
        Default 16 (count). Number of search directions.
    svf_r_max : int
        Default 10 (px). Search radius. Example: r = 10 px = 2.5 m at 0.25 m/px.
    svf_noise : int
        Default 0 (0..3). Noise removal: 0 none, 1 low, 2 medium, 3 high.
    asvf_compute : int
        Default 0 (0/1). Compute the anisotropic SVF in ``save_visualizations``.
    asvf_dir : int
        Default 315 (deg). Main direction of anisotropy, clockwise from north.
    asvf_level : int
        Default 1 (1 or 2). Level of anisotropy: 1 low, 2 high.
    pos_opns_compute : int
        Default 0 (0/1). Compute the positive openness in ``save_visualizations``.
    neg_opns_compute : int
        Default 0 (0/1). Compute the negative openness in ``save_visualizations``.

    [Sky illumination, prefix sim_]
    sim_compute : int
        Default 0 (0/1). Compute in ``save_visualizations``.
    sim_sky_mod : str
        Default "overcast"; "overcast" or "uniform". Sky model.
    sim_compute_shadow : int
        Default 0 (0/1). Add the cast shadow of the sun at ``sim_shadow_az`` / ``sim_shadow_el``.
    sim_nr_dir : int
        Default 32 (count). Number of horizon search directions.
    sim_shadow_dist : int
        Default 100 (px). Horizon search distance on the full-resolution DEM (``max_fine_radius`` of
        ``rvt.vis.sky_illumination``).
    sim_shadow_az : int
        Default 315 (deg). Sun azimuth for the shadow, clockwise from north.
    sim_shadow_el : int
        Default 35 (deg). Sun elevation for the shadow.

    [Local dominance, prefix ld_]
    ld_compute : int
        Default 0 (0/1). Compute in ``save_visualizations``.
    ld_min_rad : int
        Default 10 (px). Smallest radial distance of the observed ring.
    ld_max_rad : int
        Default 20 (px). Largest radial distance of the observed ring.
    ld_rad_inc : int
        Default 1 (px). Step of the radial distance.
    ld_anglr_res : int
        Default 15 (deg). Angular step between directions.
    ld_observer_h : float
        Default 1.7 (m). Observer height above the surface.

    [Multi-scale relief model, prefix msrm_]
    msrm_compute : int
        Default 0 (0/1). Compute in ``save_visualizations``.
    msrm_feature_min : float
        Default 0 (m). Smallest feature size to detect.
    msrm_feature_max : float
        Default 20 (m). Largest feature size to detect.
    msrm_scaling_factor : int
        Default 2. Exponent of the filter-radius sequence of ``rvt.vis.msrm``.

    [Multi-scale topographic position, prefix mstp_]
    mstp_compute : int
        Default 0 (0/1). Compute in ``save_visualizations``.
    mstp_local_scale : tuple of int
        Default (1, 5, 1) (px). (minimum radius, maximum radius, step) of the local scale (one RGB band).
    mstp_meso_scale : tuple of int
        Default (5, 50, 5) (px). Same for the meso scale.
    mstp_broad_scale : tuple of int
        Default (50, 500, 50) (px). Same for the broad scale.
    mstp_lightness : float
        Default 0.9. Lightness of the RGB result.

    [Outputs: float32 GeoTIFF (no-data NaN) and 8-bit GeoTIFF; the svf_ flags also apply to asvf and positive
    openness, which have no flags of their own]
    slp_save_float : int
        Default 1 (0/1). Write the float slope.
    hs_save_float : int
        Default 1 (0/1). Write the float hillshade.
    mhs_save_float : int
        Default 1 (0/1). Write the float multi-direction hillshade.
    slrm_save_float : int
        Default 1 (0/1). Write the float SLRM.
    svf_save_float : int
        Default 1 (0/1). Write the float SVF, anisotropic SVF and positive openness.
    neg_opns_save_float : int
        Default 1 (0/1). Write the float negative openness.
    sim_save_float : int
        Default 1 (0/1). Write the float sky illumination.
    ld_save_float : int
        Default 1 (0/1). Write the float local dominance.
    msrm_save_float : int
        Default 1 (0/1). Write the float MSRM.
    mstp_save_float : int
        Default 1 (0/1). Write the float MSTP.
    slp_save_8bit : int
        Default 0 (0/1). Write the 8-bit slope.
    hs_save_8bit : int
        Default 0 (0/1). Write the 8-bit hillshade.
    mhs_save_8bit : int
        Default 0 (0/1). Write the 8-bit multi-direction hillshade (3-band RGB, see ``float_to_8bit``).
    slrm_save_8bit : int
        Default 0 (0/1). Write the 8-bit SLRM.
    svf_save_8bit : int
        Default 0 (0/1). Write the 8-bit SVF, anisotropic SVF and positive openness.
    neg_opns_save_8bit : int
        Default 0 (0/1). Write the 8-bit negative openness.
    sim_save_8bit : int
        Default 0 (0/1). Write the 8-bit sky illumination.
    ld_save_8bit : int
        Default 0 (0/1). Write the 8-bit local dominance.
    msrm_save_8bit : int
        Default 0 (0/1). Write the 8-bit MSRM.
    mstp_save_8bit : int
        Default 0 (0/1). Write the 8-bit MSTP (3 bands).

    [8-bit stretch (mode, min, max). Mode "value": linear stretch from min to max in the unit of the
    visualization. Mode "percent": cut off the lowest min % and the highest max % of the values, then linear.
    Values outside the range are clipped; slope and negative openness are inverted (high values dark)]
    slp_bytscl : tuple(str, float, float)
        Default ("value", 0, 51), unit of ``slp_output_units``.
    hs_bytscl : tuple(str, float, float)
        Default ("value", 0, 1).
    mhs_bytscl : tuple(str, float, float)
        Default ("value", 0, 1).
    slrm_bytscl : tuple(str, float, float)
        Default ("value", -2, 2) (m).
    svf_bytscl : tuple(str, float, float)
        Default ("value", 0.6375, 1).
    asvf_bytscl : tuple(str, float, float)
        Default ("value", 0.7, 0.9).
    pos_opns_bytscl : tuple(str, float, float)
        Default ("value", 60, 95) (deg).
    neg_opns_bytscl : tuple(str, float, float)
        Default ("value", 60, 95) (deg).
    sim_bytscl : tuple(str, float, float)
        Default ("percent", 0.25, 0) (%).
    ld_bytscl : tuple(str, float, float)
        Default ("value", 0.5, 1.8).
    msrm_bytscl : tuple(str, float, float)
        Default ("value", -2.5, 2.5) (m).
    mstp_bytscl : tuple(str, float, float)
        Default ("value", 0, 1).

    [Tiling; not written to or read from settings files (known issue, see ``save_default_to_file``)]
    tile_size_limit : int
        Default 10000 * 10000 (px). A ``save_<vis>`` call processes the DEM tile by tile (``rvt.tile``) when its
        number of pixels exceeds this value.
    tile_size : tuple(int, int)
        Default (4000, 4000) (px). (x_size, y_size) of one tile.
    """

    def __init__(self):
        self.overwrite = 0  # (0=False, 1=True)
        self.ve_factor = 1
        # slope gradient
        self.slp_compute = 0
        self.slp_output_units = "degree"
        # hillshade
        self.hs_compute = 1
        self.hs_sun_azi = 315
        self.hs_sun_el = 35
        self.hs_shadow = 0
        # multi hillshade
        self.mhs_compute = 0
        self.mhs_nr_dir = 16
        self.mhs_sun_el = 35
        # simple local relief model
        self.slrm_compute = 0
        self.slrm_rad_cell = 20
        # sky view factor
        self.svf_compute = 0
        self.svf_n_dir = 16
        self.svf_r_max = 10
        self.svf_noise = 0
        # anisotropic sky-view factor
        self.asvf_compute = 0
        self.asvf_dir = 315
        self.asvf_level = 1
        # positive openness
        self.pos_opns_compute = 0
        # negative openness
        self.neg_opns_compute = 0
        # sky_illum
        self.sim_compute = 0
        self.sim_sky_mod = "overcast"
        self.sim_compute_shadow = 0
        self.sim_shadow_dist = 100
        self.sim_nr_dir = 32
        self.sim_shadow_az = 315
        self.sim_shadow_el = 35
        # local dominance
        self.ld_compute = 0
        self.ld_min_rad = 10
        self.ld_max_rad = 20
        self.ld_rad_inc = 1
        self.ld_anglr_res = 15
        self.ld_observer_h = 1.7
        # multi-scale relief model
        self.msrm_compute = 0
        self.msrm_feature_min = 0
        self.msrm_feature_max = 20
        self.msrm_scaling_factor = 2
        # multi-scale topographic position
        self.mstp_compute = 0
        self.mstp_local_scale = (1, 5, 1)
        self.mstp_meso_scale = (5, 50, 5)
        self.mstp_broad_scale = (50, 500, 50)
        self.mstp_lightness = 0.9
        # save float
        self.slp_save_float = 1
        self.hs_save_float = 1
        self.mhs_save_float = 1
        self.slrm_save_float = 1
        self.svf_save_float = 1
        self.neg_opns_save_float = 1
        self.sim_save_float = 1
        self.ld_save_float = 1
        self.msrm_save_float = 1
        self.mstp_save_float = 1
        # save 8bit
        self.slp_save_8bit = 0
        self.hs_save_8bit = 0
        self.mhs_save_8bit = 0
        self.slrm_save_8bit = 0
        self.svf_save_8bit = 0
        self.neg_opns_save_8bit = 0
        self.sim_save_8bit = 0
        self.ld_save_8bit = 0
        self.msrm_save_8bit = 0
        self.mstp_save_8bit = 0
        # 8-bit bytescale parameters
        self.slp_bytscl = ("value", 0.00, 51.00)
        self.hs_bytscl = ("value", 0.00, 1.00)
        self.mhs_bytscl = ("value", 0.00, 1.00)
        self.slrm_bytscl = ("value", -2.00, 2.00)
        self.svf_bytscl = ("value", 0.6375, 1.00)
        self.asvf_bytscl = ("value", 0.70, 0.90)
        self.pos_opns_bytscl = ("value", 60.00, 95.00)
        self.neg_opns_bytscl = ("value", 60.00, 95.00)
        self.sim_bytscl = ("percent", 0.25, 0.00)
        self.ld_bytscl = ("value", 0.50, 1.80)
        self.msrm_bytscl = ("value", -2.50, 2.50)
        self.mstp_bytscl = ("value", 0.00, 1.00)
        # tile
        self.tile_size_limit = 10000 * 10000  # if arr size > tile_size limit, it uses tile module
        self.tile_size = (4000, 4000)  # size of single tile when using tile module (x_size, y_size)

    def save_default_to_file(self, file_path=None):
        """
        Write the parameters of this instance to a json settings file.

        The file has one object ``"default_settings"`` with ``overwrite``, ``ve_factor`` and one group per
        visualization ("Hillshade", "Multiple directions hillshade", "Slope gradient", "Simple local relief model",
        "Multi-scale relief model", "Sky-View Factor", "Anisotropic Sky-View Factor", "Openness - Positive",
        "Openness - Negative", "Sky illumination", "Local dominance", "Multi-scale topographic position"). Each
        parameter is ``{"value": ..., "description": ...}``; a ``*_bytscl`` tuple is ``{"mode", "min", "max",
        "description"}`` and an ``mstp_*_scale`` tuple is ``{"min", "max", "step", "description"}``.
        ``read_default_from_file`` reads this format back.

        Parameters
        ----------
        file_path : str or None
            Path of the json file; an existing file is replaced. None writes ``settings\\default_settings.json``
            relative to the current working directory and creates the ``settings`` directory if needed.

        Notes
        -----
        Known issue: ``tile_size`` and ``tile_size_limit`` are not written (and not read), so they do not survive
        a settings round trip. See docs/KNOWN_ISSUES.md.

        Known issue: the default path is the literal ``r"settings\\default_settings.json"`` with a backslash. On
        Windows it is a path in the directory ``settings``; on Linux/macOS it is one file name with an empty
        directory part, and ``os.makedirs("")`` raises FileNotFoundError when the file does not exist yet. See
        docs/KNOWN_ISSUES.md.

        The file is opened with the platform default encoding; all texts written are ASCII.
        """
        # Structure of this method:
        #   1. build the nested dict "default_settings" (general values, then one group per visualization)
        #   2. choose the default path and create its directory if file_path is None
        #   3. write the dict as json with indent 4
        data = {"default_settings": {
            "overwrite": {
                "value": self.overwrite,
                "description": "When saving visualisation functions and file already exists, if 0 "
                               "it doesn't compute it, if 1 it overwrites it."},
            "ve_factor": {
                "value": self.ve_factor,
                "description": "Vertical exaggeration."},
            "Hillshade": {
                "hs_compute": {"value": self.hs_compute,
                               "description": "If compute Hillshade. Parameter for GUIs."},
                "hs_sun_azi": {"value": self.hs_sun_azi,
                               "description": "Solar azimuth angle (clockwise from North) in "
                                              "degrees."},
                "hs_sun_el": {"value": self.hs_sun_el,
                              "description": "Solar vertical angle (above the horizon) in "
                                             "degrees."},
                "hs_save_float": {"value": self.hs_save_float,
                                  "description": "If 1 it saves float raster, if 0 it doesn't."},
                "hs_save_8bit": {"value": self.hs_save_8bit,
                                 "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "hs_shadow": {"value": self.hs_shadow,
                              "description": "If 1 it saves shadow binary raster, if 0 it doesn't."},
                "hs_bytscl": {"mode": self.hs_bytscl[0], "min": self.hs_bytscl[1], "max": self.hs_bytscl[2],
                              "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                             "Mode can be 'value' or 'percent' (cut-off units). "
                                             "Values min and max define stretch borders (in mode units)."}
            },
            "Multiple directions hillshade": {
                "mhs_compute": {"value": self.mhs_compute,
                                "description": "If compute Multiple directions hillshade."
                                               " Parameter for GUIs."},
                "mhs_nr_dir": {"value": self.mhs_nr_dir,
                               "description": "Number of solar azimuth angles (clockwise "
                                              "from North)."},
                "mhs_sun_el": {"value": self.mhs_sun_el,
                               "description": "Solar vertical angle (above the horizon) in "
                                              "degrees."},
                "mhs_save_float": {"value": self.mhs_save_float,
                                   "description": "If 1 it saves float raster, if 0 it doesn't."},
                "mhs_save_8bit": {"value": self.mhs_save_8bit,
                                  "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "mhs_bytscl": {"mode": self.mhs_bytscl[0], "min": self.mhs_bytscl[1], "max": self.mhs_bytscl[2],
                               "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                              "Mode can be 'value' or 'percent' (cut-off units). "
                                              "Values min and max define stretch borders (in mode units)."}
            },
            "Slope gradient": {
                "slp_compute": {"value": self.slp_compute,
                                "description": "If compute Slope. Parameter for GUIs."},
                "slp_output_units": {"value": self.slp_output_units,
                                     "description": "Slope output units [radian, degree, "
                                                    "percent]."},
                "slp_save_float": {"value": self.slp_save_float,
                                   "description": "If 1 it saves float raster, if 0 it doesn't."},
                "slp_save_8bit": {"value": self.slp_save_8bit,
                                  "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "slp_bytscl": {"mode": self.slp_bytscl[0], "min": self.slp_bytscl[1], "max": self.slp_bytscl[2],
                               "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                              "Mode can be 'value' or 'percent' (cut-off units). "
                                              "Values min and max define stretch borders (in mode units)."}
            },
            "Simple local relief model": {
                "slrm_compute": {"value": self.slrm_compute,
                                 "description": "If compute Simple local relief model. "
                                                "Parameter for GUIs."},
                "slrm_rad_cell": {"value": self.slrm_rad_cell,
                                  "description": "Radius for trend assessment in pixels."},
                "slrm_save_float": {"value": self.slrm_save_float,
                                    "description": "If 1 it saves float raster, if 0 it doesn't."},
                "slrm_save_8bit": {"value": self.slrm_save_8bit,
                                   "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "slrm_bytscl": {"mode": self.slrm_bytscl[0], "min": self.slrm_bytscl[1], "max": self.slrm_bytscl[2],
                                "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                               "Mode can be 'value' or 'percent' (cut-off units). "
                                               "Values min and max define stretch borders (in mode units)."}
            },
            "Multi-scale relief model": {
                "msrm_compute": {"value": self.msrm_compute,
                                 "description": "If compute Multi-scale relief model. "
                                                "Parameter for GUIs."},
                "msrm_feature_min": {"value": self.msrm_feature_min,
                                     "description": "Minimum size of the feature you want to detect in meters."},
                "msrm_feature_max": {"value": self.msrm_feature_max,
                                     "description": "Maximum size of the feature you want to detect in meters."},
                "msrm_scaling_factor": {"value": self.msrm_scaling_factor,
                                        "description": "Scaling factor."},
                "msrm_save_float": {"value": self.msrm_save_float,
                                    "description": "If 1 it saves float raster, if 0 it doesn't."},
                "msrm_save_8bit": {"value": self.msrm_save_8bit,
                                   "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "msrm_bytscl": {"mode": self.msrm_bytscl[0], "min": self.msrm_bytscl[1], "max": self.msrm_bytscl[2],
                                "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                               "Mode can be 'value' or 'percent' (cut-off units). "
                                               "Values min and max define stretch borders (in mode units)."}
            },
            "Sky-View Factor": {
                "svf_compute": {"value": self.svf_compute,
                                "description": "If compute Sky-View Factor."
                                               " Parameter for GUIs."},
                "svf_n_dir": {"value": self.svf_n_dir, "description": "Number of directions."},
                "svf_r_max": {"value": self.svf_r_max, "description": "Maximal search "
                                                                      "radious in pixels."},
                "svf_noise": {"value": self.svf_noise,
                              "description": "The level of noise remove [0-don't remove, "
                                             "1-low, 2-med, 3-high]."},
                "svf_save_float": {"value": self.svf_save_float,
                                   "description": "If 1 it saves float raster, if 0 it doesn't."},
                "svf_save_8bit": {"value": self.svf_save_8bit,
                                  "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "svf_bytscl": {"mode": self.svf_bytscl[0], "min": self.svf_bytscl[1], "max": self.svf_bytscl[2],
                               "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                              "Mode can be 'value' or 'percent' (cut-off units). "
                                              "Values min and max define stretch borders (in mode units)."}
            },
            "Anisotropic Sky-View Factor": {
                "asvf_compute": {"value": self.asvf_compute,
                                 "description": "If compute Anisotropic Sky-View Factor."
                                                " Parameter for GUIs."},
                "asvf_dir": {"value": self.asvf_dir,
                             "description": "Direction of anisotropy in degrees."},
                "asvf_level": {"value": self.asvf_level,
                               "description": "Level of anisotropy [1-low, 2-high]."},
                "asvf_bytscl": {"mode": self.asvf_bytscl[0], "min": self.asvf_bytscl[1], "max": self.asvf_bytscl[2],
                                "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                               "Mode can be 'value' or 'percent' (cut-off units). "
                                               "Values min and max define stretch borders (in mode units)."}
            },
            "Openness - Positive": {
                "pos_opns_compute": {"value": self.pos_opns_compute,
                                     "description": "If compute Openness - Positive. "
                                                    "Parameter for GUIs."},
                "pos_opns_bytscl": {"mode": self.pos_opns_bytscl[0], "min": self.pos_opns_bytscl[1],
                                    "max": self.pos_opns_bytscl[2],
                                    "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                                   "Mode can be 'value' or 'percent' (cut-off units). "
                                                   "Values min and max define stretch borders (in mode units)."}
            },
            "Openness - Negative": {
                "neg_opns_compute": {"value": self.neg_opns_compute,
                                     "description": "If compute Openness - Negative. "
                                                    "Parameter for GUIs."},
                "neg_opns_save_float": {"value": self.neg_opns_save_float,
                                        "description": "If 1 it saves float raster, if 0 it doesn't."},
                "neg_opns_save_8bit": {"value": self.neg_opns_save_8bit,
                                       "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "neg_opns_bytscl": {"mode": self.neg_opns_bytscl[0], "min": self.neg_opns_bytscl[1],
                                    "max": self.neg_opns_bytscl[2],
                                    "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                                   "Mode can be 'value' or 'percent' (cut-off units). "
                                                   "Values min and max define stretch borders (in mode units)."}
            },
            "Sky illumination": {
                "sim_compute": {"value": self.sim_compute,
                                "description": "If compute Sky illumination. Parameter for "
                                               "GUIs."},
                "sim_sky_mod": {"value": self.sim_sky_mod,
                                "description": "Sky model [overcast, uniform]."},
                "sim_compute_shadow": {"value": self.sim_compute_shadow,
                                       "description": "If 1 it computes shadows, if 0 it doesn't."},
                "sim_shadow_dist": {"value": self.sim_shadow_dist,
                                    "description": "Max shadow modeling distance in pixels."},
                "sim_nr_dir": {"value": self.sim_nr_dir,
                               "description": "Number of directions to search for horizon"},
                "sim_shadow_az": {"value": self.sim_shadow_az, "description": "Shadow "
                                                                              "azimuth in "
                                                                              "degrees."},
                "sim_shadow_el": {"value": self.sim_shadow_el, "description": "Shadow "
                                                                              "elevation in "
                                                                              "degrees."},
                "sim_save_float": {"value": self.sim_save_float,
                                   "description": "If 1 it saves float raster, if 0 it doesn't."},
                "sim_save_8bit": {"value": self.sim_save_8bit,
                                  "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "sim_bytscl": {"mode": self.sim_bytscl[0], "min": self.sim_bytscl[1], "max": self.sim_bytscl[2],
                               "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                              "Mode can be 'value' or 'percent' (cut-off units). "
                                              "Values min and max define stretch borders (in mode units)."}
            },
            "Local dominance": {
                "ld_compute": {"value": self.ld_compute,
                               "description": "If compute Local dominance. Parameter for "
                                              "GUIs."},
                "ld_min_rad": {"value": self.ld_min_rad,
                               "description": "Minimum radial distance (in pixels) at which "
                                              "the algorithm starts with visualization "
                                              "computation."},
                "ld_max_rad": {"value": self.ld_max_rad,
                               "description": "Maximum radial distance (in pixels) at which "
                                              "the algorithm ends with visualization "
                                              "computation."},
                "ld_rad_inc": {"value": self.ld_rad_inc, "description": "Radial distance "
                                                                        "steps in pixels."},
                "ld_anglr_res": {"value": self.ld_anglr_res,
                                 "description": "Angular step for determination of number of "
                                                "angular directions."},
                "ld_observer_h": {"value": self.ld_observer_h,
                                  "description": "Height at which we observe the terrain."},
                "ld_save_float": {"value": self.ld_save_float,
                                  "description": "If 1 it saves float raster, if 0 it doesn't."},
                "ld_save_8bit": {"value": self.ld_save_8bit,
                                 "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "ld_bytscl": {"mode": self.ld_bytscl[0], "min": self.ld_bytscl[1], "max": self.ld_bytscl[2],
                              "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                             "Mode can be 'value' or 'percent' (cut-off units). "
                                             "Values min and max define stretch borders (in mode units)."}
            },
            "Multi-scale topographic position": {
                "mstp_compute": {"value": self.mstp_compute,
                                 "description": "If compute Multi-scale topographic position. "
                                                "Parameter for GUIs."},
                "mstp_local_scale": {"min": self.mstp_local_scale[0], "max": self.mstp_local_scale[1],
                                     "step": self.mstp_local_scale[2],
                                     "description": "Local scale minimum radius, maximum radius and step in pixels to"
                                                    " calculate maximum mean deviation from elevation."
                                                    " All have to be integers!"},
                "mstp_meso_scale": {"min": self.mstp_meso_scale[0], "max": self.mstp_meso_scale[1],
                                    "step": self.mstp_meso_scale[2],
                                    "description": "Meso scale minimum radius, maximum radius and step in pixels to"
                                                   " calculate maximum mean deviation from elevation."
                                                   " All have to be integers!"},
                "mstp_broad_scale": {"min": self.mstp_broad_scale[0], "max": self.mstp_broad_scale[1],
                                     "step": self.mstp_broad_scale[2],
                                     "description": "Broad scale minimum radius, maximum radius and step in pixels to"
                                                    " calculate maximum mean deviation from elevation."
                                                    " All have to be integers!"},
                "mstp_lightness": {"value": self.mstp_lightness,
                                   "description": "Lightness factor to adjust MSTP visibility."},
                "mstp_save_float": {"value": self.mstp_save_float,
                                  "description": "If 1 it saves float raster, if 0 it doesn't."},
                "mstp_save_8bit": {"value": self.mstp_save_8bit,
                                 "description": "If 1 it saves 8bit raster, if 0 it doesn't."},
                "mstp_bytscl": {"mode": self.mstp_bytscl[0], "min": self.mstp_bytscl[1], "max": self.mstp_bytscl[2],
                              "description": "Linear stretch and byte scale (0-255) for 8bit raster. "
                                             "Mode can be 'value' or 'percent' (cut-off units). "
                                             "Values min and max define stretch borders (in mode units)."}
            }

        }}
        if file_path is None:
            file_path = r"settings\default_settings.json"
            if os.path.isfile(file_path):
                pass
            else:
                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path))
        dat = open(file_path, "w")
        dat.write(json.dumps(data, indent=4))
        dat.close()

    def read_default_from_file(self, file_path):
        """
        Set the parameters of this instance from a settings file (``.json`` or legacy ``.txt``).

        ``.json``: the format written by ``save_default_to_file``. Every key of that format must be present
        (a missing key raises KeyError). Values are converted with ``int``, ``float`` or ``str`` as listed in the
        code; ``ve_factor``, ``msrm_feature_min``, ``msrm_feature_max``, ``mstp_lightness`` and ``ld_observer_h``
        become float, all other numbers int.

        ``.txt``: lines ``name = value``, ``#`` starts a comment. Recognised names (case-insensitive):
        overwrite, exaggeration_factor, slope_gradient, hillshading, sun_azimuth, sun_elevation,
        multiple_hillshading, hillshade_directions, simple_local_relief, trend_radius, sky_view_factor,
        svf_directions, search_radius, remove_noise, noise_removal (low/medium/high), anisotropic_svf,
        anisotropy_direction, anisotropy_level (low/high), positive_openness, negative_openness,
        sky_illumination, sky_model, max_shadow_dist, local_dominance, min_radius, max_radius. Other names are
        ignored; a line without exactly one ``=`` gives a warning.

        Parameters
        ----------
        file_path : str
            Path of the settings file; the extension selects the format.

        Raises
        ------
        FileNotFoundError
            If a ``.json`` or ``.txt`` file does not exist.
        KeyError
            If a ``.json`` file lacks a key of the format.

        Warns
        -----
        UserWarning
            For every ``.txt`` line that does not have the form ``name = value``.

        Notes
        -----
        Known issue: any other extension (or a missing file with another extension) is ignored without error;
        the instance keeps its values. See docs/KNOWN_ISSUES.md.

        Known issue: in ``.txt`` files ``sun_elevation`` is tested twice (hillshade first), so the second branch
        is unreachable and ``mhs_sun_el`` is never set from a ``.txt`` file. See docs/KNOWN_ISSUES.md.

        Known issue: after a json round trip ``msrm_feature_min`` and ``msrm_feature_max`` are float (0 -> 0.0),
        so ``get_msrm_file_name`` changes from ``..._MSRM_F_M0-20_S2.tif`` to ``..._MSRM_F_M0.0-20.0_S2.tif``.
        ``tile_size`` and ``tile_size_limit`` are not read. See docs/KNOWN_ISSUES.md.
        """
        # Structure of this method:
        #   1. ".txt": parse "name = value" lines one by one; remove_noise gates noise_removal
        #   2. ".json": load the file and set the attributes group by group (same order as
        #      save_default_to_file: general, slope, hillshade, multi hillshade, SLRM, MSRM, SVF, ASVF,
        #      positive and negative openness, sky illumination, local dominance, MSTP)
        #   3. any other extension: nothing happens
        # Example file in dir settings: default_settings.json
        extension = os.path.splitext(file_path)[1]
        if extension == ".txt":
            dat = open(file_path, "r")
            remove_noise = 0
            for line in dat:
                line = line.strip()
                if line == "":
                    continue
                if line[0] == "#":
                    continue
                line_list = line.split("#")  # remove comments
                line_list = line_list[0].split("=")
                if len(line_list) == 2:
                    parameter_name = line_list[0].strip().lower()
                    parameter_value = line_list[1].strip()
                    if parameter_name == "overwrite":
                        self.overwrite = int(parameter_value)
                    elif parameter_name == "exaggeration_factor":
                        self.ve_factor = float(parameter_value)
                    elif parameter_name == "slope_gradient":
                        self.slp_compute = int(parameter_value)
                    elif parameter_name == "hillshading":
                        self.hs_compute = int(parameter_value)
                    elif parameter_name == "sun_azimuth":
                        self.hs_sun_azi = int(parameter_value)
                    elif parameter_name == "sun_elevation":
                        self.hs_sun_el = int(parameter_value)
                    elif parameter_name == "multiple_hillshading":
                        self.mhs_compute = int(parameter_value)
                    elif parameter_name == "hillshade_directions":
                        self.mhs_nr_dir = int(parameter_value)
                    elif parameter_name == "sun_elevation":
                        self.mhs_sun_el = int(parameter_value)
                    elif parameter_name == "simple_local_relief":
                        self.slrm_compute = int(parameter_value)
                    elif parameter_name == "trend_radius":
                        self.slrm_rad_cell = int(parameter_value)
                    elif parameter_name == "sky_view_factor":
                        self.svf_compute = int(parameter_value)
                    elif parameter_name == "svf_directions":
                        self.svf_n_dir = int(parameter_value)
                    elif parameter_name == "search_radius":
                        self.svf_r_max = int(parameter_value)
                    elif parameter_name == "remove_noise":
                        if int(parameter_value) == 0:
                            remove_noise = 0
                            self.svf_noise = 0
                        else:
                            remove_noise = 1
                    elif parameter_name == "noise_removal":
                        if remove_noise != 0:
                            if parameter_value == "low":
                                self.svf_noise = 1
                            elif parameter_value == "medium":
                                self.svf_noise = 2
                            elif parameter_value == "high":
                                self.svf_noise = 3
                    elif parameter_name == "anisotropic_svf":
                        self.asvf_compute = int(parameter_value)
                    elif parameter_name == "anisotropy_direction":
                        self.asvf_dir = int(parameter_value)
                    elif parameter_name == "anisotropy_level":
                        if parameter_value == "low":
                            self.asvf_level = 1
                        elif parameter_value == "high":
                            self.asvf_level = 2
                    elif parameter_name == "positive_openness":
                        self.pos_opns_compute = int(parameter_value)
                    elif parameter_name == "negative_openness":
                        self.neg_opns_compute = int(parameter_value)
                    elif parameter_name == "sky_illumination":
                        self.sim_compute = int(parameter_value)
                    elif parameter_name == "sky_model":
                        self.sim_sky_mod = parameter_value
                    elif parameter_name == "max_shadow_dist":
                        self.sim_shadow_dist = int(parameter_value)
                    elif parameter_name == "local_dominance":
                        self.ld_compute = int(parameter_value)
                    elif parameter_name == "min_radius":
                        self.ld_min_rad = int(parameter_value)
                    elif parameter_name == "max_radius":
                        self.ld_max_rad = int(parameter_value)
                    # else:
                    #     warnings.warn("rvt.default.read_default_from_file: Line '{}' not used.".format(line))
                else:
                    warnings.warn("rvt.default.read_default_from_file: Wrong line '{}'".format(line))
                    continue
            dat.close()
        elif extension == ".json":
            dat = open(file_path, "r")
            data = json.load(dat)
            default_data = data["default_settings"]
            self.overwrite = int(default_data["overwrite"]["value"])
            self.ve_factor = float(default_data["ve_factor"]["value"])
            # Slope gradient
            self.slp_compute = int(default_data["Slope gradient"]["slp_compute"]["value"])
            self.slp_output_units = str(default_data["Slope gradient"]["slp_output_units"]["value"])
            self.slp_save_float = int(default_data["Slope gradient"]["slp_save_float"]["value"])
            self.slp_save_8bit = int(default_data["Slope gradient"]["slp_save_8bit"]["value"])
            self.slp_bytscl = (str(default_data["Slope gradient"]["slp_bytscl"]["mode"]),
                               float(default_data["Slope gradient"]["slp_bytscl"]["min"]),
                               float(default_data["Slope gradient"]["slp_bytscl"]["max"]))
            # Hillshade
            self.hs_compute = int(default_data["Hillshade"]["hs_compute"]["value"])
            self.hs_sun_azi = int(default_data["Hillshade"]["hs_sun_azi"]["value"])
            self.hs_sun_el = int(default_data["Hillshade"]["hs_sun_el"]["value"])
            self.hs_save_float = int(default_data["Hillshade"]["hs_save_float"]["value"])
            self.hs_save_8bit = int(default_data["Hillshade"]["hs_save_8bit"]["value"])
            self.hs_shadow = int(default_data["Hillshade"]["hs_shadow"]["value"])
            self.hs_bytscl = (str(default_data["Hillshade"]["hs_bytscl"]["mode"]),
                              float(default_data["Hillshade"]["hs_bytscl"]["min"]),
                              float(default_data["Hillshade"]["hs_bytscl"]["max"]))
            # Multiple directions hillshade
            self.mhs_compute = int(default_data["Multiple directions hillshade"]["mhs_compute"]["value"])
            self.mhs_nr_dir = int(default_data["Multiple directions hillshade"]["mhs_nr_dir"]["value"])
            self.mhs_sun_el = int(default_data["Multiple directions hillshade"]["mhs_sun_el"]["value"])
            self.mhs_save_float = int(default_data["Multiple directions hillshade"]["mhs_save_float"]["value"])
            self.mhs_save_8bit = int(default_data["Multiple directions hillshade"]["mhs_save_8bit"]["value"])
            self.mhs_bytscl = (str(default_data["Multiple directions hillshade"]["mhs_bytscl"]["mode"]),
                               float(default_data["Multiple directions hillshade"]["mhs_bytscl"]["min"]),
                               float(default_data["Multiple directions hillshade"]["mhs_bytscl"]["max"]))
            # Simple local relief model
            self.slrm_compute = int(default_data["Simple local relief model"]["slrm_compute"]["value"])
            self.slrm_rad_cell = int(default_data["Simple local relief model"]["slrm_rad_cell"]["value"])
            self.slrm_save_float = int(default_data["Simple local relief model"]["slrm_save_float"]["value"])
            self.slrm_save_8bit = int(default_data["Simple local relief model"]["slrm_save_8bit"]["value"])
            self.slrm_bytscl = (str(default_data["Simple local relief model"]["slrm_bytscl"]["mode"]),
                                float(default_data["Simple local relief model"]["slrm_bytscl"]["min"]),
                                float(default_data["Simple local relief model"]["slrm_bytscl"]["max"]))
            # Mulit-scale relief model
            self.msrm_compute = int(default_data["Multi-scale relief model"]["msrm_compute"]["value"])
            self.msrm_feature_min = float(default_data["Multi-scale relief model"]["msrm_feature_min"]["value"])
            self.msrm_feature_max = float(default_data["Multi-scale relief model"]["msrm_feature_max"]["value"])
            self.msrm_scaling_factor = int(default_data["Multi-scale relief model"]["msrm_scaling_factor"]["value"])
            self.msrm_save_float = int(default_data["Multi-scale relief model"]["msrm_save_float"]["value"])
            self.msrm_save_8bit = int(default_data["Multi-scale relief model"]["msrm_save_8bit"]["value"])
            self.msrm_bytscl = (str(default_data["Multi-scale relief model"]["msrm_bytscl"]["mode"]),
                                float(default_data["Multi-scale relief model"]["msrm_bytscl"]["min"]),
                                float(default_data["Multi-scale relief model"]["msrm_bytscl"]["max"]))
            # Sky-View Factor
            self.svf_compute = int(default_data["Sky-View Factor"]["svf_compute"]["value"])
            self.svf_n_dir = int(default_data["Sky-View Factor"]["svf_n_dir"]["value"])
            self.svf_r_max = int(default_data["Sky-View Factor"]["svf_r_max"]["value"])
            self.svf_noise = int(default_data["Sky-View Factor"]["svf_noise"]["value"])
            self.svf_save_float = int(default_data["Sky-View Factor"]["svf_save_float"]["value"])
            self.svf_save_8bit = int(default_data["Sky-View Factor"]["svf_save_8bit"]["value"])
            self.svf_bytscl = (str(default_data["Sky-View Factor"]["svf_bytscl"]["mode"]),
                               float(default_data["Sky-View Factor"]["svf_bytscl"]["min"]),
                               float(default_data["Sky-View Factor"]["svf_bytscl"]["max"]))
            # Anisotropic Sky-View Factor
            self.asvf_compute = int(default_data["Anisotropic Sky-View Factor"]["asvf_compute"]["value"])
            self.asvf_dir = int(default_data["Anisotropic Sky-View Factor"]["asvf_dir"]["value"])
            self.asvf_level = int(default_data["Anisotropic Sky-View Factor"]["asvf_level"]["value"])
            self.asvf_bytscl = (str(default_data["Anisotropic Sky-View Factor"]["asvf_bytscl"]["mode"]),
                                float(default_data["Anisotropic Sky-View Factor"]["asvf_bytscl"]["min"]),
                                float(default_data["Anisotropic Sky-View Factor"]["asvf_bytscl"]["max"]))
            # Openness - Positive
            self.pos_opns_compute = int(default_data["Openness - Positive"]["pos_opns_compute"]["value"])
            self.pos_opns_bytscl = (str(default_data["Openness - Positive"]["pos_opns_bytscl"]["mode"]),
                                    float(default_data["Openness - Positive"]["pos_opns_bytscl"]["min"]),
                                    float(default_data["Openness - Positive"]["pos_opns_bytscl"]["max"]))
            # Openness - Negative
            self.neg_opns_compute = int(default_data["Openness - Negative"]["neg_opns_compute"]["value"])
            self.neg_opns_save_float = int(default_data["Openness - Negative"]["neg_opns_save_float"]["value"])
            self.neg_opns_save_8bit = int(default_data["Openness - Negative"]["neg_opns_save_8bit"]["value"])
            self.neg_opns_bytscl = (str(default_data["Openness - Negative"]["neg_opns_bytscl"]["mode"]),
                                    float(default_data["Openness - Negative"]["neg_opns_bytscl"]["min"]),
                                    float(default_data["Openness - Negative"]["neg_opns_bytscl"]["max"]))
            # Sky illumination
            self.sim_compute = int(default_data["Sky illumination"]["sim_compute"]["value"])
            self.sim_sky_mod = str(default_data["Sky illumination"]["sim_sky_mod"]["value"])
            self.sim_compute_shadow = int(default_data["Sky illumination"]["sim_compute_shadow"]["value"])
            self.sim_nr_dir = int(default_data["Sky illumination"]["sim_nr_dir"]["value"])
            self.sim_shadow_dist = int(default_data["Sky illumination"]["sim_shadow_dist"]["value"])
            self.sim_shadow_az = int(default_data["Sky illumination"]["sim_shadow_az"]["value"])
            self.sim_shadow_el = int(default_data["Sky illumination"]["sim_shadow_el"]["value"])
            self.sim_save_float = int(default_data["Sky illumination"]["sim_save_float"]["value"])
            self.sim_save_8bit = int(default_data["Sky illumination"]["sim_save_8bit"]["value"])
            self.sim_bytscl = (str(default_data["Sky illumination"]["sim_bytscl"]["mode"]),
                               float(default_data["Sky illumination"]["sim_bytscl"]["min"]),
                               float(default_data["Sky illumination"]["sim_bytscl"]["max"]))
            # Local dominance
            self.ld_compute = int(default_data["Local dominance"]["ld_compute"]["value"])
            self.ld_min_rad = int(default_data["Local dominance"]["ld_min_rad"]["value"])
            self.ld_max_rad = int(default_data["Local dominance"]["ld_max_rad"]["value"])
            self.ld_rad_inc = int(default_data["Local dominance"]["ld_rad_inc"]["value"])
            self.ld_anglr_res = int(default_data["Local dominance"]["ld_anglr_res"]["value"])
            self.ld_observer_h = float(default_data["Local dominance"]["ld_observer_h"]["value"])
            self.ld_save_float = int(default_data["Local dominance"]["ld_save_float"]["value"])
            self.ld_save_8bit = int(default_data["Local dominance"]["ld_save_8bit"]["value"])
            self.ld_bytscl = (str(default_data["Local dominance"]["ld_bytscl"]["mode"]),
                              float(default_data["Local dominance"]["ld_bytscl"]["min"]),
                              float(default_data["Local dominance"]["ld_bytscl"]["max"]))
            # Multi-scale topographic position
            self.mstp_compute = int(default_data["Multi-scale topographic position"]["mstp_compute"]["value"])
            self.mstp_local_scale = (int(default_data["Multi-scale topographic position"]["mstp_local_scale"]["min"]),
                                     int(default_data["Multi-scale topographic position"]["mstp_local_scale"]["max"]),
                                     int(default_data["Multi-scale topographic position"]["mstp_local_scale"]["step"]))
            self.mstp_meso_scale = (int(default_data["Multi-scale topographic position"]["mstp_meso_scale"]["min"]),
                                    int(default_data["Multi-scale topographic position"]["mstp_meso_scale"]["max"]),
                                    int(default_data["Multi-scale topographic position"]["mstp_meso_scale"]["step"]))
            self.mstp_broad_scale = (int(default_data["Multi-scale topographic position"]["mstp_broad_scale"]["min"]),
                                     int(default_data["Multi-scale topographic position"]["mstp_broad_scale"]["max"]),
                                     int(default_data["Multi-scale topographic position"]["mstp_broad_scale"]["step"]))
            self.mstp_lightness = float(default_data["Multi-scale topographic position"]["mstp_lightness"]["value"])
            self.mstp_save_float = int(default_data["Multi-scale topographic position"]["mstp_save_float"]["value"])
            self.mstp_save_8bit = int(default_data["Multi-scale topographic position"]["mstp_save_8bit"]["value"])
            self.mstp_bytscl = (str(default_data["Multi-scale topographic position"]["mstp_bytscl"]["mode"]),
                              float(default_data["Multi-scale topographic position"]["mstp_bytscl"]["min"]),
                              float(default_data["Multi-scale topographic position"]["mstp_bytscl"]["max"]))
            dat.close()

    def get_shadow_file_name(self, dem_path):
        """
        Return the file name of the shadow raster (sun position from the hillshade parameters).

        Pattern: ``<dem>_shadow_A<hs_sun_azi>_H<hs_sun_el>.tif``. There is no 8-bit variant (the shadow is binary).

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        return "{}_shadow_A{}_H{}.tif".format(dem_name, self.hs_sun_azi, self.hs_sun_el)

    def get_shadow_path(self, dem_path):
        """
        Return the path of the shadow raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_shadow_file_name(dem_path)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_shadow_file_name(dem_path)))

    def get_hillshade_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the hillshade raster.

        Pattern: ``<dem>_HS_A<hs_sun_azi>_H<hs_sun_el>[_8bit].tif``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        if bit8:
            return "{}_HS_A{}_H{}_8bit.tif".format(dem_name, self.hs_sun_azi, self.hs_sun_el)
        else:
            return "{}_HS_A{}_H{}.tif".format(dem_name, self.hs_sun_azi, self.hs_sun_el)

    def get_hillshade_path(self, dem_path, bit8=False):
        """
        Return the path of the hillshade raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_hillshade_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_hillshade_file_name(dem_path, bit8)))

    def get_slope_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the slope raster.

        Pattern: ``<dem>_SLOPE[_8bit].tif``. The output unit is not part of the name.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        if bit8:
            return "{}_SLOPE_8bit.tif".format(dem_name)
        else:
            return "{}_SLOPE.tif".format(dem_name)

    def get_slope_path(self, dem_path, bit8=False):
        """
        Return the path of the slope raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_slope_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_slope_file_name(dem_path, bit8)))

    def get_multi_hillshade_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the multi-direction hillshade raster.

        Pattern: ``<dem>_MULTI-HS_D<mhs_nr_dir>_H<mhs_sun_el>[_8bit].tif``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        if bit8:
            return "{}_MULTI-HS_D{}_H{}_8bit.tif".format(dem_name, self.mhs_nr_dir, self.mhs_sun_el)
        else:
            return "{}_MULTI-HS_D{}_H{}.tif".format(dem_name, self.mhs_nr_dir, self.mhs_sun_el)

    def get_multi_hillshade_path(self, dem_path, bit8=False):
        """
        Return the path of the multi-direction hillshade raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_multi_hillshade_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path),
                                             self.get_multi_hillshade_file_name(dem_path, bit8)))

    def get_slrm_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the simple local relief model raster.

        Pattern: ``<dem>_SLRM_R<slrm_rad_cell>[_8bit].tif``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        if bit8:
            return "{}_SLRM_R{}_8bit.tif".format(dem_name, self.slrm_rad_cell)
        else:
            return "{}_SLRM_R{}.tif".format(dem_name, self.slrm_rad_cell)

    def get_slrm_path(self, dem_path, bit8=False):
        """
        Return the path of the simple local relief model raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_slrm_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_slrm_file_name(dem_path, bit8)))

    def get_svf_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the sky-view factor raster.

        Pattern: ``<dem>_SVF_R<svf_r_max>_D<svf_n_dir><NR><8>.tif``.
        ``<NR>`` is ``_NRlow``, ``_NRmedium`` or ``_NRhigh`` for ``svf_noise`` 1, 2, 3 and empty otherwise;
        ``<8>`` is ``_8bit`` for ``bit8``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        out_name = "{}_SVF_R{}_D{}".format(dem_name, self.svf_r_max, self.svf_n_dir)
        if self.svf_noise == 1:
            out_name += "_NRlow"
        elif self.svf_noise == 2:
            out_name += "_NRmedium"
        elif self.svf_noise == 3:
            out_name += "_NRhigh"
        if bit8:
            out_name += "_8bit"
        return out_name + ".tif"

    def get_svf_path(self, dem_path, bit8=False):
        """
        Return the path of the sky-view factor raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_svf_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_svf_file_name(dem_path, bit8)))

    def get_asvf_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the anisotropic sky-view factor raster.

        Pattern: ``<dem>_SVF-A_R<svf_r_max>_D<svf_n_dir>_A<asvf_dir><AL><NR><8>.tif``.
        ``<AL>`` is ``_ALlow`` or ``_ALhigh`` for ``asvf_level`` 1 or 2 and empty otherwise; ``<NR>`` is ``_NRlow``,
        ``_NRmedium`` or ``_NRhigh`` for ``svf_noise`` 1, 2, 3 and empty otherwise;
        ``<8>`` is ``_8bit`` for ``bit8``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        out_name = "{}_SVF-A_R{}_D{}_A{}".format(dem_name, self.svf_r_max, self.svf_n_dir, self.asvf_dir)
        if self.asvf_level == 1:
            out_name += "_ALlow"
        elif self.asvf_level == 2:
            out_name += "_ALhigh"
        if self.svf_noise == 1:
            out_name += "_NRlow"
        elif self.svf_noise == 2:
            out_name += "_NRmedium"
        elif self.svf_noise == 3:
            out_name += "_NRhigh"
        if bit8:
            out_name += "_8bit"
        return out_name + ".tif"

    def get_asvf_path(self, dem_path, bit8=False):
        """
        Return the path of the anisotropic sky-view factor raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_asvf_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_asvf_file_name(dem_path, bit8)))

    def get_opns_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the positive openness raster.

        Pattern: ``<dem>_OPEN-POS_R<svf_r_max>_D<svf_n_dir><NR><8>.tif``.
        ``<NR>`` is ``_NRlow``, ``_NRmedium`` or ``_NRhigh`` for ``svf_noise`` 1, 2, 3 and empty otherwise;
        ``<8>`` is ``_8bit`` for ``bit8``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        out_name = "{}_OPEN-POS_R{}_D{}".format(dem_name, self.svf_r_max, self.svf_n_dir)
        if self.svf_noise == 1:
            out_name += "_NRlow"
        elif self.svf_noise == 2:
            out_name += "_NRmedium"
        elif self.svf_noise == 3:
            out_name += "_NRhigh"
        if bit8:
            out_name += "_8bit"
        return out_name + ".tif"

    def get_opns_path(self, dem_path, bit8=False):
        """
        Return the path of the positive openness raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_opns_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_opns_file_name(dem_path, bit8)))

    def get_neg_opns_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the negative openness raster.

        Pattern: ``<dem>_OPEN-NEG_R<svf_r_max>_D<svf_n_dir><NR><8>.tif``.
        ``<NR>`` is ``_NRlow``, ``_NRmedium`` or ``_NRstrong`` for ``svf_noise`` 1, 2, 3 and empty
        otherwise (known issue: the other visualizations use ``_NRhigh`` for 3, see docs/KNOWN_ISSUES.md);
        ``<8>`` is ``_8bit`` for ``bit8``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        out_name = "{}_OPEN-NEG_R{}_D{}".format(dem_name, self.svf_r_max, self.svf_n_dir)
        if self.svf_noise == 1:
            out_name += "_NRlow"
        elif self.svf_noise == 2:
            out_name += "_NRmedium"
        elif self.svf_noise == 3:
            out_name += "_NRstrong"
        if bit8:
            out_name += "_8bit"
        return out_name + ".tif"

    def get_neg_opns_path(self, dem_path, bit8=False):
        """
        Return the path of the negative openness raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_neg_opns_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_neg_opns_file_name(dem_path, bit8)))

    def get_sky_illumination_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the sky illumination raster.

        Pattern: ``<dem>_SIM_<sim_sky_mod>_D<sim_nr_dir>_<sim_shadow_dist>px[_8bit].tif``.
        The shadow settings are not part of the name.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        if bit8:
            return "{}_SIM_{}_D{}_{}px_8bit.tif".format(dem_name, self.sim_sky_mod, self.sim_nr_dir,
                                                        self.sim_shadow_dist)
        else:
            return "{}_SIM_{}_D{}_{}px.tif".format(dem_name, self.sim_sky_mod, self.sim_nr_dir,
                                                   self.sim_shadow_dist)

    def get_sky_illumination_path(self, dem_path, bit8=False):
        """
        Return the path of the sky illumination raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_sky_illumination_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path),
                                             self.get_sky_illumination_file_name(dem_path, bit8)))

    def get_local_dominance_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the local dominance raster.

        Pattern: ``<dem>_LD_R_M<ld_min_rad>-<ld_max_rad>_DI<ld_rad_inc>_A<ld_anglr_res>_OH<ld_observer_h>``
        ``[_8bit].tif``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        if bit8:
            return "{}_LD_R_M{}-{}_DI{}_A{}_OH{}_8bit.tif".format(dem_name, self.ld_min_rad, self.ld_max_rad,
                                                                  self.ld_rad_inc, self.ld_anglr_res,
                                                                  self.ld_observer_h)
        else:
            return "{}_LD_R_M{}-{}_DI{}_A{}_OH{}.tif".format(dem_name, self.ld_min_rad, self.ld_max_rad,
                                                             self.ld_rad_inc, self.ld_anglr_res, self.ld_observer_h)

    def get_local_dominance_path(self, dem_path, bit8=False):
        """
        Return the path of the local dominance raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_local_dominance_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path),
                                             self.get_local_dominance_file_name(dem_path, bit8)))

    def get_msrm_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the multi-scale relief model raster.

        Pattern: ``<dem>_MSRM_F_M<msrm_feature_min>-<msrm_feature_max>_S<msrm_scaling_factor>[_8bit].tif``.
        Numbers are formatted with ``str``, so an int 20 gives ``20`` and a float 20.0 gives ``20.0``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension
        if bit8:
            return "{}_MSRM_F_M{}-{}_S{}_8bit.tif".format(dem_name, self.msrm_feature_min, self.msrm_feature_max,
                                                          self.msrm_scaling_factor)
        else:
            return "{}_MSRM_F_M{}-{}_S{}.tif".format(dem_name, self.msrm_feature_min, self.msrm_feature_max,
                                                     self.msrm_scaling_factor)

    def get_msrm_path(self, dem_path, bit8=False):
        """
        Return the path of the multi-scale relief model raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_msrm_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_msrm_file_name(dem_path, bit8)))

    def get_mstp_file_name(self, dem_path, bit8=False):
        """
        Return the file name of the multi-scale topographic position raster.

        Pattern: ``<dem>_MSTP_<local>_<meso>_<broad>_L<mstp_lightness>[_8bit].tif`` with ``<local>`` =
        ``mstp_local_scale[1]``, ``<meso>`` = ``mstp_meso_scale[1]``, ``<broad>`` = ``mstp_broad_scale[1]``: only
        the maximum radius (px) of each scale is part of the name.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM. Only its base name up to the first dot is used (known issue: "my.site_v2.tif"
            gives "my", see docs/KNOWN_ISSUES.md).
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            File name without directory.
        """
        dem_name = os.path.basename(dem_path).split(".")[0]  # base name without extension

        mstp_file_name = "{}_MSTP_{}_{}_{}_L{}.tif".format(
            dem_name,
            self.mstp_local_scale[1],
            self.mstp_meso_scale[1],
            self.mstp_broad_scale[1],
            self.mstp_lightness
        )

        if bit8:
            mstp_file_name = mstp_file_name.replace(".tif", "_8bit.tif")

        return mstp_file_name

    def get_mstp_path(self, dem_path, bit8=False):
        """
        Return the path of the multi-scale topographic position raster next to the DEM.

        ``os.path.normpath(os.path.join(os.path.dirname(dem_path), get_mstp_file_name(dem_path, bit8)))``.

        Parameters
        ----------
        dem_path : str or pathlib.Path
            Path of the DEM; the output is placed in its directory and named after its base name.
        bit8 : bool
            True gives the name of the 8-bit raster (``_8bit`` before ``.tif``), False the float raster.

        Returns
        -------
        str
            Normalized path in the DEM directory.
        """
        return os.path.normpath(os.path.join(os.path.dirname(dem_path), self.get_mstp_file_name(dem_path, bit8)))

    def get_visualization_file_name(self,
                                    rvt_visualization: RVTVisualization,
                                    dem_path: Path,
                                    path_8bit: bool
                                    ) -> str:
        """
        Return the output file name of ``rvt_visualization`` by calling its ``get_<vis>_file_name`` method.

        Parameters
        ----------
        rvt_visualization : RVTVisualization
            Visualization whose file name is wanted.
        dem_path : str or pathlib.Path
            Path of the DEM; its base name up to the first dot starts the file name.
        path_8bit : bool
            True gives the name of the 8-bit raster; ignored for SHADOW (no 8-bit variant).

        Returns
        -------
        str or None
            File name without directory, e.g. ``dem_SLOPE.tif``; None if ``rvt_visualization`` is not a member
            of ``RVTVisualization``.
        """
        if rvt_visualization == rvt.default.RVTVisualization.SLOPE:
            return self.get_slope_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.HILLSHADE:
            return self.get_hillshade_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.SHADOW:
            return self.get_shadow_file_name(dem_path=dem_path)
        elif rvt_visualization == rvt.default.RVTVisualization.MULTI_HILLSHADE:
            return self.get_multi_hillshade_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL:
            return self.get_slrm_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.SKY_VIEW_FACTOR:
            return self.get_svf_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR:
            return self.get_asvf_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.POSITIVE_OPENNESS:
            return self.get_opns_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.NEGATIVE_OPENNESS:
            return self.get_neg_opns_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.SKY_ILLUMINATION:
            return self.get_sky_illumination_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.LOCAL_DOMINANCE:
            return self.get_local_dominance_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.MULTI_SCALE_RELIEF_MODEL:
            return self.get_msrm_file_name(dem_path=dem_path, bit8=path_8bit)
        elif rvt_visualization == rvt.default.RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION:
            return self.get_mstp_file_name(dem_path=dem_path, bit8=path_8bit)

    def get_visualization_path(
            self,
            rvt_visualization: RVTVisualization,
            dem_path: Path,
            output_dir_path: Path,
            path_8bit: bool
    ) -> Path:
        """
        Return ``output_dir_path / <file name>`` for ``rvt_visualization`` (see ``get_visualization_file_name``).

        Parameters
        ----------
        rvt_visualization : RVTVisualization
            Visualization whose output path is wanted.
        dem_path : str or pathlib.Path
            Path of the DEM; its base name up to the first dot starts the file name.
        output_dir_path : pathlib.Path
            Output directory (a str works too, ``str / Path`` gives a ``Path``).
        path_8bit : bool
            True gives the path of the 8-bit raster; ignored for SHADOW.

        Returns
        -------
        pathlib.Path or None
            Output path; None if ``rvt_visualization`` is not a member of ``RVTVisualization``.
        """
        if rvt_visualization == rvt.default.RVTVisualization.SLOPE:
            return output_dir_path / Path(self.get_slope_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.HILLSHADE:
            return output_dir_path / Path(self.get_hillshade_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.SHADOW:
            return output_dir_path / Path(self.get_shadow_file_name(dem_path=dem_path))
        elif rvt_visualization == rvt.default.RVTVisualization.MULTI_HILLSHADE:
            return output_dir_path / Path(self.get_multi_hillshade_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL:
            return output_dir_path / Path(self.get_slrm_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.SKY_VIEW_FACTOR:
            return output_dir_path / Path(self.get_svf_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR:
            return output_dir_path / Path(self.get_asvf_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.POSITIVE_OPENNESS:
            return output_dir_path / Path(self.get_opns_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.NEGATIVE_OPENNESS:
            return output_dir_path / Path(self.get_neg_opns_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.SKY_ILLUMINATION:
            return output_dir_path / Path(self.get_sky_illumination_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.LOCAL_DOMINANCE:
            return output_dir_path / Path(self.get_local_dominance_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.MULTI_SCALE_RELIEF_MODEL:
            return output_dir_path / Path(self.get_msrm_file_name(dem_path=dem_path, bit8=path_8bit))
        elif rvt_visualization == rvt.default.RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION:
            return output_dir_path / Path(self.get_mstp_file_name(dem_path=dem_path, bit8=path_8bit))

    def float_to_8bit(
            self,
            float_arr: np.array,
            visualization: RVTVisualization,
            x_res: float = None,
            y_res: float = None,
            no_data: Optional[float] = None
    ):
        """
        Stretch a float visualization to 0..255 with the ``<vis>_bytscl`` setting of that visualization.

        For every visualization except SHADOW and MULTI_HILLSHADE: ``rvt.blend_func.normalize_image`` with
        ``<vis>_bytscl`` = (mode, min, max) maps the values to 0..1 (mode "value": linear between min and max;
        "percent": cut off min % and max % of the histogram, then linear; values outside are clipped; slope and
        negative openness are inverted, 1 - x), then ``rvt.vis.byte_scale(c_min=0, c_max=1)`` gives
        ``floor(255.9999 * x)``; NaN becomes 255.

        ===========================  ===================  ================================================
        visualization                setting              note
        ===========================  ===================  ================================================
        SLOPE                        ``slp_bytscl``       inverted (steep is dark)
        HILLSHADE                    ``hs_bytscl``
        SHADOW                       none                 ``float_arr`` is returned unchanged
        MULTI_HILLSHADE              ``mhs_bytscl``       ``float_arr`` is the DEM, see below
        SIMPLE_LOCAL_RELIEF_MODEL    ``slrm_bytscl``
        SKY_VIEW_FACTOR              ``svf_bytscl``
        ANISOTROPIC_SKY_VIEW_FACTOR  ``asvf_bytscl``
        POSITIVE_OPENNESS            ``pos_opns_bytscl``
        NEGATIVE_OPENNESS            ``neg_opns_bytscl``  inverted
        SKY_ILLUMINATION             ``sim_bytscl``
        LOCAL_DOMINANCE              ``ld_bytscl``
        MULTI_SCALE_RELIEF_MODEL     ``msrm_bytscl``
        MULTI_SCALE_TOPOGRAPHIC_     ``mstp_bytscl``      3 bands in, 3 bands out
        POSITION
        ===========================  ===================  ================================================

        MULTI_HILLSHADE: ``float_arr`` must be the DEM (m). Three hillshades with sun azimuth 315 (red), 22.5
        (green) and 90 deg (blue) and sun elevation ``mhs_sun_el`` are computed with ``rvt.vis.hillshade`` and
        each is stretched with ``rvt.blend_func.normalize_perc`` (mode "percent") or ``normalize_lin`` (any other
        mode) and ``byte_scale``; no inversion.

        Parameters
        ----------
        float_arr : numpy.ndarray
            Float result of ``visualization``, (H, W) or (3, H, W) for MSTP; the DEM (H, W) for MULTI_HILLSHADE.
        visualization : RVTVisualization
            Selects the setting and the special cases above.
        x_res : float or None
            Pixel size in x (m/px); used only for MULTI_HILLSHADE.
        y_res : float or None
            Pixel size in y (m/px); used only for MULTI_HILLSHADE.
        no_data : float or None
            No-data value of the DEM; used only for MULTI_HILLSHADE.

        Returns
        -------
        numpy.ndarray
            uint8, same shape as ``float_arr``; (3, H, W) for MULTI_HILLSHADE; for SHADOW the input array itself.

        Raises
        ------
        Exception
            If ``visualization`` is not a member of ``RVTVisualization``; from ``rvt.blend_func`` if min and max
            of a "value" stretch are equal or reversed, or the percentages are outside 0..100.

        Notes
        -----
        Known issue: the stretch clips ``float_arr`` in place to the stretch range (the caller's array changes).
        See docs/KNOWN_ISSUES.md.

        Known issue: MULTI_HILLSHADE ignores ``ve_factor`` (the float multi hillshade uses it), and its mode test
        is ``mhs_bytscl[0].lower() == "percent" or slp_bytscl[0].lower() == "perc"``: the slope setting can switch
        the multi hillshade to the percent stretch. See docs/KNOWN_ISSUES.md.
        """
        if visualization == RVTVisualization.HILLSHADE:
            norm_arr = rvt.blend_func.normalize_image(visualization="hs", image=float_arr,
                                                      min_norm=self.hs_bytscl[1], max_norm=self.hs_bytscl[2],
                                                      normalization=self.hs_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.SLOPE:
            norm_arr = rvt.blend_func.normalize_image(visualization="slp", image=float_arr,
                                                      min_norm=self.slp_bytscl[1], max_norm=self.slp_bytscl[2],
                                                      normalization=self.slp_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.SHADOW:
            return float_arr
        elif visualization == RVTVisualization.MULTI_HILLSHADE:
            # Be careful when multihillshade we input dem, because we have to calculate hillshade in 3 directions
            red_band_arr = rvt.vis.hillshade(dem=float_arr, resolution_x=x_res, resolution_y=y_res,
                                             sun_elevation=self.mhs_sun_el, sun_azimuth=315, no_data=no_data)
            green_band_arr = rvt.vis.hillshade(dem=float_arr, resolution_x=x_res, resolution_y=y_res,
                                               sun_elevation=self.mhs_sun_el, sun_azimuth=22.5, no_data=no_data)
            blue_band_arr = rvt.vis.hillshade(dem=float_arr, resolution_x=x_res, resolution_y=y_res,
                                              sun_elevation=self.mhs_sun_el, sun_azimuth=90, no_data=no_data)
            if self.mhs_bytscl[0].lower() == "percent" or self.slp_bytscl[0].lower() == "perc":
                red_band_arr = rvt.blend_func.normalize_perc(
                    image=red_band_arr, minimum=self.mhs_bytscl[1], maximum=self.mhs_bytscl[2]
                )
                red_band_arr = rvt.vis.byte_scale(data=red_band_arr, no_data=np.nan, c_min=0, c_max=1)
                green_band_arr = rvt.blend_func.normalize_perc(
                    image=green_band_arr, minimum=self.mhs_bytscl[1], maximum=self.mhs_bytscl[2]
                )
                green_band_arr = rvt.vis.byte_scale(data=green_band_arr, no_data=np.nan, c_min=0, c_max=1)
                blue_band_arr = rvt.blend_func.normalize_perc(
                    image=blue_band_arr, minimum=self.mhs_bytscl[1], maximum=self.mhs_bytscl[2]
                )
                blue_band_arr = rvt.vis.byte_scale(data=blue_band_arr, no_data=np.nan, c_min=0, c_max=1)
            else:  # self.mhs_bytscl[0] == "value"
                red_band_arr = rvt.blend_func.normalize_lin(
                    image=red_band_arr, minimum=self.mhs_bytscl[1], maximum=self.mhs_bytscl[2]
                )
                red_band_arr = rvt.vis.byte_scale(
                    data=red_band_arr, no_data=np.nan, c_min=0, c_max=1
                )
                green_band_arr = rvt.blend_func.normalize_lin(
                    image=green_band_arr, minimum=self.mhs_bytscl[1], maximum=self.mhs_bytscl[2]
                )
                green_band_arr = rvt.vis.byte_scale(
                    data=green_band_arr, no_data=np.nan, c_min=0, c_max=1
                )
                blue_band_arr = rvt.blend_func.normalize_lin(
                    image=blue_band_arr, minimum=self.mhs_bytscl[1], maximum=self.mhs_bytscl[2]
                )
                blue_band_arr = rvt.vis.byte_scale(
                    data=blue_band_arr, no_data=np.nan, c_min=0, c_max=1
                )
            multi_hillshade_8bit_arr = np.array([red_band_arr, green_band_arr, blue_band_arr])
            return multi_hillshade_8bit_arr
        elif visualization == RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL:
            norm_arr = rvt.blend_func.normalize_image(visualization="slrm", image=float_arr,
                                                      min_norm=self.slrm_bytscl[1], max_norm=self.slrm_bytscl[2],
                                                      normalization=self.slrm_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.SKY_VIEW_FACTOR:
            norm_arr = rvt.blend_func.normalize_image(visualization="svf", image=float_arr,
                                                      min_norm=self.svf_bytscl[1], max_norm=self.svf_bytscl[2],
                                                      normalization=self.svf_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR:
            norm_arr = rvt.blend_func.normalize_image(visualization="asvf", image=float_arr,
                                                      min_norm=self.asvf_bytscl[1], max_norm=self.asvf_bytscl[2],
                                                      normalization=self.asvf_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.POSITIVE_OPENNESS:
            norm_arr = rvt.blend_func.normalize_image(visualization="pos_opns", image=float_arr,
                                                      min_norm=self.pos_opns_bytscl[1],
                                                      max_norm=self.pos_opns_bytscl[2],
                                                      normalization=self.pos_opns_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.NEGATIVE_OPENNESS:
            norm_arr = rvt.blend_func.normalize_image(visualization="neg_opns", image=float_arr,
                                                      min_norm=self.neg_opns_bytscl[1],
                                                      max_norm=self.neg_opns_bytscl[2],
                                                      normalization=self.neg_opns_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.SKY_ILLUMINATION:
            norm_arr = rvt.blend_func.normalize_image(visualization="sim", image=float_arr,
                                                      min_norm=self.sim_bytscl[1], max_norm=self.sim_bytscl[2],
                                                      normalization=self.sim_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.LOCAL_DOMINANCE:
            norm_arr = rvt.blend_func.normalize_image(visualization="ld", image=float_arr,
                                                      min_norm=self.ld_bytscl[1], max_norm=self.ld_bytscl[2],
                                                      normalization=self.ld_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.MULTI_SCALE_RELIEF_MODEL:
            norm_arr = rvt.blend_func.normalize_image(visualization="msrm", image=float_arr,
                                                      min_norm=self.msrm_bytscl[1], max_norm=self.msrm_bytscl[2],
                                                      normalization=self.msrm_bytscl[0])
            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        elif visualization == RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION:
            # This might not be necessary, as all mstp data should already be between 0 and 1
            norm_arr = rvt.blend_func.normalize_image(
                visualization="mstp",
                image=float_arr,
                min_norm=self.mstp_bytscl[1],
                max_norm=self.mstp_bytscl[2],
                normalization=self.mstp_bytscl[0]
            )

            return rvt.vis.byte_scale(data=norm_arr, no_data=np.nan, c_min=0, c_max=1)
        else:
            raise Exception("rvt.default.DefaultValues.float_to_8bit: Wrong visualization (visualization) parameter!")

    def get_slope(self, dem_arr, resolution_x, resolution_y, no_data=None):
        """
        Compute the slope gradient with ``rvt.vis.slope_aspect`` and the stored parameters.

        Uses ``slp_output_units`` and ``ve_factor``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution_x : float
            Pixel size in x (m/px).
        resolution_y : float
            Pixel size in y (m/px).
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float32 (H, W), slope in ``slp_output_units``.

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.slope_aspect`` works on a float32 copy, so ``dem_arr`` is not modified.
        """
        slope_arr = rvt.vis.slope_aspect(dem=dem_arr, resolution_x=resolution_x, resolution_y=resolution_y,
                                         ve_factor=self.ve_factor, output_units=self.slp_output_units,
                                         no_data=no_data)["slope"]
        return slope_arr

    def save_slope(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the slope of a DEM GeoTIFF with the stored parameters and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_slope`` and writes GeoTIFFs with the projection and geotransform of the DEM,
        named by ``get_slope_file_name`` (pattern there). ``save_visualizations`` calls this method when
        ``slp_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (slope in ``slp_output_units``, no-data NaN); None takes
            ``slp_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``slp_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.slp_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.slp_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_slope: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_slope: dem_path doesn't exist!")

        if custom_dir is None:
            slope_path = self.get_slope_path(dem_path)
            slope_8bit_path = self.get_slope_path(dem_path, bit8=True)
        else:
            slope_path = os.path.join(custom_dir, self.get_slope_file_name(dem_path))
            slope_8bit_path = os.path.join(custom_dir, self.get_slope_file_name(dem_path, bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(slope_8bit_path) and os.path.isfile(slope_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(slope_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(slope_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile calculation
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.SLOPE,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            x_res = dict_arr_res["resolution"][0]
            y_res = dict_arr_res["resolution"][1]
            slope_arr = self.get_slope(dem_arr=dem_arr, resolution_x=x_res, resolution_y=y_res, no_data=no_data)
            if save_float:
                if os.path.isfile(slope_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=slope_path, out_raster_arr=slope_arr,
                                no_data=np.nan)
            if save_8bit:
                if os.path.isfile(slope_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    slope_8bit_arr = self.float_to_8bit(float_arr=slope_arr, visualization=RVTVisualization.SLOPE)
                    save_raster(src_raster_path=dem_path, out_raster_path=slope_8bit_path,
                                out_raster_arr=slope_8bit_arr, e_type=1)
            return 1

    def get_shadow(self, dem_arr, resolution, no_data=None):
        """
        Compute the binary cast shadow with ``rvt.vis.shadow_horizon`` for the hillshade sun position.

        Sun azimuth ``hs_sun_azi`` and elevation ``hs_sun_el`` (deg), ``ve_factor``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution : float
            Pixel size (m/px); the x pixel size is used for both axes.
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            int64 (H, W), 1 in shadow, 0 otherwise.

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.shadow_horizon`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).

        Known issue: one NaN anywhere in the DEM (also a no-data pixel) makes the whole shadow 0. See
        docs/KNOWN_ISSUES.md.
        """
        shadow_arr = rvt.vis.shadow_horizon(dem=dem_arr, resolution=resolution, shadow_az=self.hs_sun_azi,
                                            shadow_el=self.hs_sun_el, ve_factor=self.ve_factor,
                                            no_data=no_data)["shadow"]
        return shadow_arr

    def get_hillshade(self, dem_arr, resolution_x, resolution_y, no_data=None):
        """
        Compute the hillshade with ``rvt.vis.hillshade`` and the stored parameters.

        Sun azimuth ``hs_sun_azi`` and elevation ``hs_sun_el`` (deg), ``ve_factor``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution_x : float
            Pixel size in x (m/px).
        resolution_y : float
            Pixel size in y (m/px).
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float64 (H, W), values 0..1.

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.hillshade`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).
        """
        hillshade_arr = rvt.vis.hillshade(dem=dem_arr, resolution_x=resolution_x, resolution_y=resolution_y,
                                          sun_azimuth=self.hs_sun_azi, sun_elevation=self.hs_sun_el,
                                          ve_factor=self.ve_factor, no_data=no_data)
        return hillshade_arr

    def save_hillshade(self, dem_path, custom_dir=None, save_float=None, save_8bit=None, save_shadow=None):
        """
        Compute the hillshade (and optionally the shadow) of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_hillshade`` and writes GeoTIFFs with the projection and geotransform of the
        DEM, named by ``get_hillshade_file_name`` (pattern there). ``save_visualizations`` calls this method when
        ``hs_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (values 0..1, no-data NaN); None takes
            ``hs_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``hs_save_8bit``.
        save_shadow : bool or None
            Also write the binary shadow (``get_shadow``, float32 0/1, name of ``get_shadow_file_name``);
            None takes ``hs_shadow``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: ``rvt.vis.hillshade`` assigns NaN to the no-data pixels before converting to float32, so a DEM
        GeoTIFF with an integer data type and a no-data value raises ValueError. See docs/KNOWN_ISSUES.md.

        Known issue: the "outputs exist" test covers only some combinations of the three flags; for
        8-bit only, float + 8-bit without shadow and 8-bit + shadow it never returns 0 (the two branches
        meant for the 8-bit cases test ``not save_float and not save_8bit`` and are unreachable; float + 8-bit
        without shadow has no branch). See docs/KNOWN_ISSUES.md.

        Known issue: with ``custom_dir`` the shadow path is ``os.path.join(custom_dir, get_shadow_path(dem_path))``;
        ``get_shadow_path`` already holds the DEM directory, so the shadow is written next to the DEM
        (for an absolute ``dem_path``). See docs/KNOWN_ISSUES.md.

        Known issue: the shadow is computed from the array in which ``get_hillshade`` has already set the
        no-data pixels to NaN, and one NaN makes the whole shadow 0, so a DEM with no-data pixels gives
        an all-zero shadow. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.hs_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.hs_save_8bit
        # if save_shadow is None it takes boolean from default (self)
        if save_shadow is None:
            save_shadow = self.hs_shadow

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_hillshade: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_hillshade: dem_path doesn't exist!")

        if custom_dir is None:
            hillshade_path = self.get_hillshade_path(dem_path)
            hillshade_8bit_path = self.get_hillshade_path(dem_path, bit8=True)
            shadow_path = self.get_shadow_path(dem_path)
        else:
            hillshade_path = os.path.join(custom_dir, self.get_hillshade_file_name(dem_path))
            hillshade_8bit_path = os.path.join(custom_dir, self.get_hillshade_file_name(dem_path, bit8=True))
            shadow_path = os.path.join(custom_dir, self.get_shadow_path(dem_path))

        # if file already exists and overwrite=0
        if save_float and save_8bit and save_shadow:
            if os.path.isfile(hillshade_8bit_path) and os.path.isfile(hillshade_path) \
                    and os.path.isfile(shadow_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit and not save_shadow:
            if os.path.isfile(hillshade_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit and not save_shadow:
            if os.path.isfile(hillshade_8bit_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit and save_shadow:
            if os.path.isfile(hillshade_path) and os.path.isfile(shadow_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit and save_shadow:
            if os.path.isfile(hillshade_8bit_path) and os.path.isfile(shadow_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.HILLSHADE,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            if save_shadow:
                rvt.tile.save_rvt_visualization_tile_by_tile(
                    rvt_visualization=RVTVisualization.SHADOW,
                    rvt_default=self,
                    dem_path=Path(dem_path),
                    output_dir_path=Path(custom_dir),
                    save_float=True,
                    save_8bit=False
                )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            x_res = dict_arr_res["resolution"][0]
            y_res = dict_arr_res["resolution"][1]
            hillshade_arr = self.get_hillshade(dem_arr=dem_arr, resolution_x=x_res, resolution_y=y_res,
                                               no_data=no_data).astype('float32')
            if save_float:
                if os.path.isfile(hillshade_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=hillshade_path, out_raster_arr=hillshade_arr,
                                no_data=np.nan)
            if save_8bit:
                if os.path.isfile(hillshade_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    hillshade_8_bit_arr = self.float_to_8bit(
                        float_arr=hillshade_arr, visualization=RVTVisualization.HILLSHADE
                    )
                    save_raster(src_raster_path=dem_path, out_raster_path=hillshade_8bit_path,
                                out_raster_arr=hillshade_8_bit_arr, e_type=1)
            if save_shadow:
                shadow_arr = self.get_shadow(dem_arr=dem_arr, resolution=x_res)
                if os.path.isfile(shadow_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=shadow_path, out_raster_arr=shadow_arr,
                                no_data=np.nan)
            return 1

    def get_multi_hillshade(self, dem_arr, resolution_x, resolution_y, no_data=None):
        """
        Compute hillshades from ``mhs_nr_dir`` sun azimuths with ``rvt.vis.multi_hillshade``.

        Azimuths 0, 360 / mhs_nr_dir, ... deg; sun elevation ``mhs_sun_el`` (deg); ``ve_factor``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution_x : float
            Pixel size in x (m/px).
        resolution_y : float
            Pixel size in y (m/px).
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float64 (mhs_nr_dir, H-2, W-2), values 0..1: ``rvt.vis.multi_hillshade`` drops the
            edge pixels (see its docstring).

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.multi_hillshade`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).
        """
        multi_hillshade_arr = rvt.vis.multi_hillshade(dem=dem_arr, resolution_x=resolution_x, resolution_y=resolution_y,
                                                      nr_directions=self.mhs_nr_dir, sun_elevation=self.mhs_sun_el,
                                                      ve_factor=self.ve_factor, no_data=no_data)
        return multi_hillshade_arr

    def save_multi_hillshade(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the multi-direction hillshade of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_multi_hillshade`` and writes GeoTIFFs with the projection and geotransform of
        the DEM, named by ``get_multi_hillshade_file_name`` (pattern there). ``save_visualizations`` calls this
        method when ``mhs_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (``mhs_nr_dir`` bands, values 0..1, no-data NaN); None takes
            ``mhs_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``mhs_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: ``rvt.vis.multi_hillshade`` assigns NaN to the no-data pixels before converting to float32, so
        a DEM GeoTIFF with an integer data type and a no-data value raises ValueError. See docs/KNOWN_ISSUES.md.

        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        The float raster is ``(mhs_nr_dir, H-2, W-2)`` (``rvt.vis.multi_hillshade`` drops the edge pixels)
        but is written with the geotransform of the DEM, so it is 2 px smaller and its pixel [i, j]
        belongs to DEM pixel [i + 1, j + 1] (known issue, see docs/KNOWN_ISSUES.md). The 8-bit raster is not
        derived from the float result: it is a 3-band RGB of hillshades at 315, 22.5 and 90 deg computed from
        the DEM by ``float_to_8bit``.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.mhs_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.mhs_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_multi_hillshade: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_multi_hillshade: dem_path doesn't exist!")

        if custom_dir is None:
            multi_hillshade_path = self.get_multi_hillshade_path(dem_path)
            multi_hillshade_8bit_path = self.get_multi_hillshade_path(dem_path, bit8=True)
        else:
            multi_hillshade_path = os.path.join(custom_dir, self.get_multi_hillshade_file_name(dem_path))
            multi_hillshade_8bit_path = os.path.join(custom_dir, self.get_multi_hillshade_file_name(dem_path,
                                                                                                    bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(multi_hillshade_8bit_path) and os.path.isfile(multi_hillshade_path) \
                    and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(multi_hillshade_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(multi_hillshade_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.MULTI_HILLSHADE,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            x_res = dict_arr_res["resolution"][0]
            y_res = dict_arr_res["resolution"][1]
            if save_float:
                if os.path.isfile(multi_hillshade_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    multi_hillshade_arr = self.get_multi_hillshade(dem_arr=dem_arr, resolution_x=x_res,
                                                                   resolution_y=y_res,
                                                                   no_data=no_data).astype('float32')
                    save_raster(src_raster_path=dem_path, out_raster_path=multi_hillshade_path,
                                out_raster_arr=multi_hillshade_arr, no_data=np.nan)
            if save_8bit:
                if os.path.isfile(multi_hillshade_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    multi_hillshade_8bit_arr = self.float_to_8bit(
                        float_arr=dem_arr,
                        visualization=RVTVisualization.MULTI_HILLSHADE,
                        x_res=x_res,
                        y_res=y_res,
                        no_data=no_data
                    )
                    save_raster(src_raster_path=dem_path, out_raster_path=multi_hillshade_8bit_path,
                                out_raster_arr=multi_hillshade_8bit_arr, e_type=1)
            return 1

    def get_slrm(self, dem_arr, no_data=None):
        """
        Compute the simple local relief model with ``rvt.vis.slrm`` and the stored parameters.

        Trend radius ``slrm_rad_cell`` (px), ``ve_factor``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float32 (H, W), elevation minus trend (m).

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.slrm`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).
        """
        slrm_arr = rvt.vis.slrm(dem=dem_arr, radius_cell=self.slrm_rad_cell, ve_factor=self.ve_factor, no_data=no_data)
        return slrm_arr

    def save_slrm(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the simple local relief model of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_slrm`` and writes GeoTIFFs with the projection and geotransform of the DEM,
        named by ``get_slrm_file_name`` (pattern there). ``save_visualizations`` calls this method when
        ``slrm_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (m, no-data NaN); None takes
            ``slrm_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``slrm_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: ``rvt.vis.slrm`` assigns NaN to the no-data pixels before converting to float32, so a DEM
        GeoTIFF with an integer data type and a no-data value raises ValueError. See docs/KNOWN_ISSUES.md.

        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.slrm_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.slrm_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_slrm: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_slrm: dem_path doesn't exist!")

        if custom_dir is None:
            slrm_path = self.get_slrm_path(dem_path)
            slrm_8bit_path = self.get_slrm_path(dem_path, bit8=True)
        else:
            slrm_path = os.path.join(custom_dir, self.get_slrm_file_name(dem_path))
            slrm_8bit_path = os.path.join(custom_dir, self.get_slrm_file_name(dem_path, bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(slrm_8bit_path) and os.path.isfile(slrm_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(slrm_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(slrm_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            slrm_arr = self.get_slrm(dem_arr=dem_arr, no_data=no_data).astype('float32')
            if save_float:
                if os.path.isfile(slrm_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=slrm_path, out_raster_arr=slrm_arr,
                                no_data=np.nan)
            if save_8bit:
                if os.path.isfile(slrm_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    slrm_8bit_arr = self.float_to_8bit(
                        float_arr=slrm_arr, visualization=RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL
                    )
                    save_raster(src_raster_path=dem_path, out_raster_path=slrm_8bit_path, out_raster_arr=slrm_8bit_arr,
                                e_type=1)
            return 1

    def get_sky_view_factor(self, dem_arr, resolution, compute_svf=True, compute_asvf=False, compute_opns=False,
                            no_data=None):
        """
        Compute sky-view factor, anisotropic SVF and/or positive openness with ``rvt.vis.sky_view_factor``.

        Uses ``svf_n_dir``, ``svf_r_max`` (px), ``svf_noise``, ``asvf_dir`` (deg), ``asvf_level`` and ``ve_factor``;
        the three results come from one horizon search.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution : float
            Pixel size (m/px); the x pixel size is used for both axes.
        compute_svf : bool
            Compute the sky-view factor (key "svf").
        compute_asvf : bool
            Compute the anisotropic sky-view factor (key "asvf").
        compute_opns : bool
            Compute the positive openness (key "opns").
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        dict
            ``{"svf", "asvf", "opns"}`` for the requested results: float64 (H, W); SVF and ASVF
            0..1, openness in deg.

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.sky_view_factor`` works on a float32 copy, so ``dem_arr`` is not modified.
        """
        dict_svf_asvf_opns = rvt.vis.sky_view_factor(dem=dem_arr, resolution=resolution, compute_svf=compute_svf,
                                                     compute_opns=compute_opns, compute_asvf=compute_asvf,
                                                     svf_n_dir=self.svf_n_dir, svf_r_max=self.svf_r_max,
                                                     svf_noise=self.svf_noise, asvf_dir=self.asvf_dir,
                                                     asvf_level=self.asvf_level, ve_factor=self.ve_factor,
                                                     no_data=no_data)
        return dict_svf_asvf_opns

    def save_sky_view_factor(self, dem_path, save_svf=True, save_asvf=False, save_opns=False, custom_dir=None,
                             save_float=None, save_8bit=None):
        """
        Compute SVF, anisotropic SVF and/or positive openness of a DEM GeoTIFF and write the rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_sky_view_factor`` and writes GeoTIFFs with the projection and geotransform of
        the DEM, named by ``get_svf_file_name``, ``get_asvf_file_name`` and ``get_opns_file_name`` (patterns there).
        ``save_visualizations`` calls this method when ``svf_compute``, ``asvf_compute`` or ``pos_opns_compute`` is
        set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        save_svf : bool
            Write the sky-view factor.
        save_asvf : bool
            Write the anisotropic sky-view factor.
        save_opns : bool
            Write the positive openness.
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (SVF and ASVF 0..1, openness in deg, no-data NaN); None takes
            ``svf_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``svf_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        The three results come from one call of ``rvt.vis.sky_view_factor``; the flags ``svf_save_float``
        and ``svf_save_8bit`` apply to all three.

        Known issue: the "outputs exist" test checks the paths of all three results; the path of a result
        that is not requested is "", so the method returns 0 only when all three are requested and
        present. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.svf_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.svf_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_sky_view_factor: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_sky_view_factor: dem_path doesn't exist!")

        svf_path = ""
        asvf_path = ""
        opns_path = ""
        svf_8bit_path = ""
        asvf_8bit_path = ""
        opns_8bit_path = ""
        if custom_dir is None:
            if save_svf:
                svf_path = self.get_svf_path(dem_path)
                svf_8bit_path = self.get_svf_path(dem_path, bit8=True)
            if save_asvf:
                asvf_path = self.get_asvf_path(dem_path)
                asvf_8bit_path = self.get_asvf_path(dem_path, bit8=True)
            if save_opns:
                opns_path = self.get_opns_path(dem_path)
                opns_8bit_path = self.get_opns_path(dem_path, bit8=True)
        else:
            if save_svf:
                svf_path = os.path.join(custom_dir, self.get_svf_file_name(dem_path))
                svf_8bit_path = os.path.join(custom_dir, self.get_svf_file_name(dem_path, bit8=True))
            if save_asvf:
                asvf_path = os.path.join(custom_dir, self.get_asvf_file_name(dem_path))
                asvf_8bit_path = os.path.join(custom_dir, self.get_asvf_file_name(dem_path, bit8=True))
            if save_opns:
                opns_path = os.path.join(custom_dir, self.get_opns_file_name(dem_path))
                opns_8bit_path = os.path.join(custom_dir, self.get_opns_file_name(dem_path, bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(svf_path) and os.path.isfile(asvf_path) and os.path.isfile(opns_path) and \
                    os.path.isfile(svf_8bit_path) and os.path.isfile(asvf_8bit_path) and \
                    os.path.isfile(opns_8bit_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(svf_path) and os.path.isfile(asvf_path) and os.path.isfile(opns_path) \
                    and not self.overwrite:
                return 0
        elif not save_float and save_8bit:
            if os.path.isfile(svf_8bit_path) and os.path.isfile(asvf_8bit_path) and os.path.isfile(opns_8bit_path) \
                    and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            if save_svf:
                rvt.tile.save_rvt_visualization_tile_by_tile(
                    rvt_visualization=RVTVisualization.SKY_VIEW_FACTOR,
                    rvt_default=self,
                    dem_path=Path(dem_path),
                    output_dir_path=Path(custom_dir),
                    save_float=save_float,
                    save_8bit=save_8bit
                )
            if save_asvf:
                rvt.tile.save_rvt_visualization_tile_by_tile(
                    rvt_visualization=RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR,
                    rvt_default=self,
                    dem_path=Path(dem_path),
                    output_dir_path=Path(custom_dir),
                    save_float=save_float,
                    save_8bit=save_8bit
                )
            if save_opns:
                rvt.tile.save_rvt_visualization_tile_by_tile(
                    rvt_visualization=RVTVisualization.POSITIVE_OPENNESS,
                    rvt_default=self,
                    dem_path=Path(dem_path),
                    output_dir_path=Path(custom_dir),
                    save_float=save_float,
                    save_8bit=save_8bit
                )
            return 1
        else:
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            x_res = dict_arr_res["resolution"][0]
            y_res = dict_arr_res["resolution"][1]
            dict_svf_asvf_opns = self.get_sky_view_factor(dem_arr=dem_arr, resolution=x_res, compute_svf=save_svf,
                                                          compute_asvf=save_asvf, compute_opns=save_opns,
                                                          no_data=no_data)
            if save_float:
                if save_svf:
                    if os.path.isfile(svf_path) and not self.overwrite:  # file exists and overwrite=0
                        pass
                    else:  # svf_path, file doesn't exists or exists and overwrite=1
                        save_raster(src_raster_path=dem_path, out_raster_path=svf_path,
                                    out_raster_arr=dict_svf_asvf_opns["svf"].astype('float32'), no_data=np.nan)
                if save_asvf:
                    if os.path.isfile(asvf_path) and not self.overwrite:  # file exists and overwrite=0
                        pass
                    else:  # asvf_path, file doesn't exists or exists and overwrite=1
                        save_raster(src_raster_path=dem_path, out_raster_path=asvf_path,
                                    out_raster_arr=dict_svf_asvf_opns["asvf"].astype('float32'), no_data=np.nan)
                if save_opns:
                    if os.path.isfile(opns_path) and not self.overwrite:  # file exists and overwrite=0
                        pass
                    else:  # opns_path, file doesn't exists or exists and overwrite=1
                        save_raster(src_raster_path=dem_path, out_raster_path=opns_path,
                                    out_raster_arr=dict_svf_asvf_opns["opns"].astype('float32'), no_data=np.nan)
            if save_8bit:
                if save_svf:
                    if os.path.isfile(svf_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                        pass
                    else:  # svf_8bit_path, file doesn't exists or exists and overwrite=1
                        svf_8bit_arr = self.float_to_8bit(
                            float_arr=dict_svf_asvf_opns["svf"], visualization=RVTVisualization.SKY_VIEW_FACTOR
                        )
                        save_raster(src_raster_path=dem_path, out_raster_path=svf_8bit_path,
                                    out_raster_arr=svf_8bit_arr, e_type=1)
                if save_asvf:
                    if os.path.isfile(asvf_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                        pass
                    else:  # asvf_8bit_path, file doesn't exists or exists and overwrite=1
                        asvf_8bit_arr = self.float_to_8bit(
                            float_arr=dict_svf_asvf_opns["asvf"],
                            visualization=RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR
                        )
                        save_raster(src_raster_path=dem_path, out_raster_path=asvf_8bit_path,
                                    out_raster_arr=asvf_8bit_arr, e_type=1)
                if save_opns:
                    if os.path.isfile(opns_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                        pass
                    else:  # opns_8bit_path, file doesn't exists or exists and overwrite=1
                        opns_8bit_arr = self.float_to_8bit(
                            float_arr=dict_svf_asvf_opns["opns"], visualization=RVTVisualization.POSITIVE_OPENNESS
                        )
                        save_raster(src_raster_path=dem_path, out_raster_path=opns_8bit_path,
                                    out_raster_arr=opns_8bit_arr, e_type=1)
            return 1

    def get_neg_opns(self, dem_arr, resolution, no_data=None):
        """
        Compute the negative openness: positive openness of ``-1 * dem_arr`` with ``rvt.vis.sky_view_factor``.

        Uses ``svf_n_dir``, ``svf_r_max`` (px), ``svf_noise`` and ``ve_factor``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution : float
            Pixel size (m/px); the x pixel size is used for both axes.
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float64 (H, W), openness in deg.

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``dem_arr`` is not modified (the negation makes a new array).

        Known issue: ``no_data`` is compared with the negated DEM, so a no-data value -9999 (now +9999) is
        not recognised and is processed as a 9999 m peak. See docs/KNOWN_ISSUES.md.
        """
        dem_arr = -1 * dem_arr
        dict_neg_opns = rvt.vis.sky_view_factor(dem=dem_arr, resolution=resolution, svf_n_dir=self.svf_n_dir,
                                                svf_r_max=self.svf_r_max, svf_noise=self.svf_noise,
                                                compute_svf=False, compute_asvf=False, compute_opns=True,
                                                ve_factor=self.ve_factor, no_data=no_data)
        neg_opns_arr = dict_neg_opns["opns"]
        return neg_opns_arr

    def save_neg_opns(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the negative openness of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_neg_opns`` and writes GeoTIFFs with the projection and geotransform of the
        DEM, named by ``get_neg_opns_file_name`` (pattern there). ``save_visualizations`` calls this method when
        ``neg_opns_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (deg, no-data NaN); None takes
            ``neg_opns_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``neg_opns_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        Known issue: the no-data value is not recognised (see ``get_neg_opns``). See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.neg_opns_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.neg_opns_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_neg_opns: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_neg_opns: dem_path doesn't exist!")

        if custom_dir is None:
            neg_opns_path = self.get_neg_opns_path(dem_path)
            neg_opns_8bit_path = self.get_neg_opns_path(dem_path, bit8=True)
        else:
            neg_opns_path = os.path.join(custom_dir, self.get_neg_opns_file_name(dem_path))
            neg_opns_8bit_path = os.path.join(custom_dir, self.get_neg_opns_file_name(dem_path, bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(neg_opns_8bit_path) and os.path.isfile(neg_opns_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(neg_opns_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(neg_opns_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.NEGATIVE_OPENNESS,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            x_res = dict_arr_res["resolution"][0]
            y_res = dict_arr_res["resolution"][1]

            neg_opns_arr = self.get_neg_opns(dem_arr=dem_arr, resolution=x_res, no_data=no_data).astype('float32')
            if save_float:
                if os.path.isfile(neg_opns_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=neg_opns_path, out_raster_arr=neg_opns_arr,
                                no_data=np.nan)
            if save_8bit:
                if os.path.isfile(neg_opns_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    neg_opns_8bit_arr = self.float_to_8bit(
                        float_arr=neg_opns_arr, visualization=RVTVisualization.NEGATIVE_OPENNESS
                    )
                    save_raster(src_raster_path=dem_path, out_raster_path=neg_opns_8bit_path,
                                out_raster_arr=neg_opns_8bit_arr, e_type=1)
            return 1

    def get_sky_illumination(self, dem_arr, resolution, no_data=None):
        """
        Compute the sky illumination with ``rvt.vis.sky_illumination`` and the stored parameters.

        Uses ``sim_sky_mod``, ``sim_compute_shadow``, ``sim_shadow_dist`` (px, ``max_fine_radius``),
        ``sim_nr_dir``, ``sim_shadow_az`` and ``sim_shadow_el`` (deg) and ``ve_factor``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution : float
            Pixel size (m/px); the x pixel size is used for both axes.
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float64 (H, W).

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.sky_illumination`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).

        Known issue: a single NaN (or no-data) pixel turns a large part of the output into NaN, not only
        that pixel. See docs/KNOWN_ISSUES.md.
        """
        sky_illumination_arr = rvt.vis.sky_illumination(dem=dem_arr, resolution=resolution, sky_model=self.sim_sky_mod,
                                                        compute_shadow=bool(self.sim_compute_shadow),
                                                        max_fine_radius=self.sim_shadow_dist,
                                                        num_directions=self.sim_nr_dir, shadow_az=self.sim_shadow_az,
                                                        shadow_el=self.sim_shadow_el, ve_factor=self.ve_factor,
                                                        no_data=no_data)
        return sky_illumination_arr

    def save_sky_illumination(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the sky illumination of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_sky_illumination`` and writes GeoTIFFs with the projection and geotransform
        of the DEM, named by ``get_sky_illumination_file_name`` (pattern there). ``save_visualizations`` calls this
        method when ``sim_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (no-data NaN); None takes
            ``sim_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``sim_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: ``rvt.vis.sky_illumination`` assigns NaN to the no-data pixels before converting to float32, so
        a DEM GeoTIFF with an integer data type and a no-data value raises ValueError. See docs/KNOWN_ISSUES.md.

        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.sim_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.sim_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_sky_illumination: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_sky_illumination: dem_path doesn't exist!")

        if custom_dir is None:
            sky_illumination_path = self.get_sky_illumination_path(dem_path)
            sky_illumination_8bit_path = self.get_sky_illumination_path(dem_path, bit8=True)
        else:
            sky_illumination_path = os.path.join(custom_dir, self.get_sky_illumination_file_name(dem_path))
            sky_illumination_8bit_path = os.path.join(custom_dir, self.get_sky_illumination_file_name(dem_path,
                                                                                                      bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(sky_illumination_8bit_path) and os.path.isfile(sky_illumination_path) \
                    and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(sky_illumination_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(sky_illumination_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.SKY_ILLUMINATION,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            x_res = dict_arr_res["resolution"][0]
            y_res = dict_arr_res["resolution"][1]

            sky_illumination_arr = self.get_sky_illumination(dem_arr=dem_arr, resolution=x_res,
                                                             no_data=no_data).astype('float32')
            if save_float:
                if os.path.isfile(sky_illumination_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=sky_illumination_path,
                                out_raster_arr=sky_illumination_arr, no_data=np.nan)
            if save_8bit:
                if os.path.isfile(sky_illumination_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    sky_illumination_8bit_arr = self.float_to_8bit(
                        float_arr=sky_illumination_arr, visualization=RVTVisualization.SKY_ILLUMINATION
                    )
                    save_raster(src_raster_path=dem_path, out_raster_path=sky_illumination_8bit_path,
                                out_raster_arr=sky_illumination_8bit_arr, e_type=1)
            return 1

    def get_local_dominance(self, dem_arr, no_data=None):
        """
        Compute the local dominance with ``rvt.vis.local_dominance`` and the stored parameters.

        Uses ``ld_min_rad``, ``ld_max_rad``, ``ld_rad_inc`` (px), ``ld_anglr_res`` (deg), ``ld_observer_h`` (m) and
        ``ve_factor``. The pixel size is not needed.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float64 (H, W).

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.local_dominance`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).
        """
        local_dominance_arr = rvt.vis.local_dominance(dem=dem_arr, min_rad=self.ld_min_rad, max_rad=self.ld_max_rad,
                                                      rad_inc=self.ld_rad_inc, angular_res=self.ld_anglr_res,
                                                      observer_height=self.ld_observer_h, ve_factor=self.ve_factor,
                                                      no_data=no_data)
        return local_dominance_arr

    def save_local_dominance(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the local dominance of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_local_dominance`` and writes GeoTIFFs with the projection and geotransform of
        the DEM, named by ``get_local_dominance_file_name`` (pattern there). ``save_visualizations`` calls this
        method when ``ld_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (no-data NaN); None takes
            ``ld_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``ld_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: ``rvt.vis.local_dominance`` assigns NaN to the no-data pixels before converting to float32, so
        a DEM GeoTIFF with an integer data type and a no-data value raises ValueError. See docs/KNOWN_ISSUES.md.

        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.ld_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.ld_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_local_dominance: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_local_dominance: dem_path doesn't exist!")

        if custom_dir is None:
            local_dominance_path = self.get_local_dominance_path(dem_path)
            local_dominance_8bit_path = self.get_local_dominance_path(dem_path, bit8=True)
        else:
            local_dominance_path = os.path.join(custom_dir, self.get_local_dominance_file_name(dem_path))
            local_dominance_8bit_path = os.path.join(custom_dir, self.get_local_dominance_file_name(dem_path,
                                                                                                    bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(local_dominance_8bit_path) and os.path.isfile(
                    local_dominance_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(local_dominance_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(local_dominance_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.LOCAL_DOMINANCE,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            local_dominance_arr = self.get_local_dominance(dem_arr=dem_arr, no_data=no_data).astype('float32')
            if save_float:
                if os.path.isfile(local_dominance_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=local_dominance_path,
                                out_raster_arr=local_dominance_arr, no_data=np.nan)
            if save_8bit:
                if os.path.isfile(local_dominance_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    local_dominance_8bit_arr = self.float_to_8bit(
                        float_arr=local_dominance_arr, visualization=RVTVisualization.LOCAL_DOMINANCE
                    )
                    save_raster(src_raster_path=dem_path, out_raster_path=local_dominance_8bit_path,
                                out_raster_arr=local_dominance_8bit_arr, e_type=1)
            return 1

    def get_msrm(self, dem_arr, resolution, no_data=None):
        """
        Compute the multi-scale relief model with ``rvt.vis.msrm`` and the stored parameters.

        Uses ``msrm_feature_min``, ``msrm_feature_max`` (m), ``msrm_scaling_factor`` and ``ve_factor``;
        ``resolution`` converts the feature sizes from m to px.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        resolution : float
            Pixel size (m/px); the x pixel size is used for both axes.
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float64 (H, W), relief in m.

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.msrm`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).
        """
        msrm_arr = rvt.vis.msrm(dem=dem_arr, resolution=resolution, feature_min=self.msrm_feature_min,
                                feature_max=self.msrm_feature_max, scaling_factor=self.msrm_scaling_factor,
                                ve_factor=self.ve_factor, no_data=no_data)
        return msrm_arr

    def save_msrm(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the multi-scale relief model of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_msrm`` and writes GeoTIFFs with the projection and geotransform of the DEM,
        named by ``get_msrm_file_name`` (pattern there). ``save_visualizations`` calls this method when
        ``msrm_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (m, no-data NaN); None takes
            ``msrm_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``msrm_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: ``rvt.vis.msrm`` assigns NaN to the no-data pixels before converting to float32, so a DEM
        GeoTIFF with an integer data type and a no-data value raises ValueError. See docs/KNOWN_ISSUES.md.

        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.msrm_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.msrm_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_msrm: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")
        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_msrm: dem_path doesn't exist!")

        if custom_dir is None:
            msrm_path = self.get_msrm_path(dem_path)
            msrm_8bit_path = self.get_msrm_path(dem_path, bit8=True)
        else:
            msrm_path = os.path.join(custom_dir, self.get_msrm_file_name(dem_path))
            msrm_8bit_path = os.path.join(custom_dir, self.get_msrm_file_name(dem_path, bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(msrm_8bit_path) and os.path.isfile(msrm_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(msrm_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(msrm_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.MULTI_SCALE_RELIEF_MODEL,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]
            x_res = dict_arr_res["resolution"][0]
            y_res = dict_arr_res["resolution"][1]

            msrm_arr = self.get_msrm(dem_arr=dem_arr, resolution=x_res, no_data=no_data).astype('float32')
            if save_float:
                if os.path.isfile(msrm_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(src_raster_path=dem_path, out_raster_path=msrm_path, out_raster_arr=msrm_arr,
                                no_data=np.nan)
            if save_8bit:
                if os.path.isfile(msrm_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    msrm_8bit_arr = self.float_to_8bit(
                        float_arr=msrm_arr, visualization=RVTVisualization.MULTI_SCALE_RELIEF_MODEL
                    )
                    save_raster(src_raster_path=dem_path, out_raster_path=msrm_8bit_path, out_raster_arr=msrm_8bit_arr,
                                e_type=1)
            return 1

    def get_mstp(self, dem_arr, no_data=None):
        """
        Compute the multi-scale topographic position (RGB) with ``rvt.vis.mstp`` and the stored parameters.

        Uses ``mstp_local_scale``, ``mstp_meso_scale``, ``mstp_broad_scale`` (px) and ``mstp_lightness``.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D elevation array (m), shape (H, W).
        no_data : float or None
            Value marking no-data pixels; they are set to NaN before the computation. None: no masking.

        Returns
        -------
        numpy.ndarray
            float32 (3, H, W), bands R = broad, G = meso, B = local scale,
            values 0..1.

        Raises
        ------
        Exception
            From ``rvt.vis`` if a stored parameter is out of range (e.g. ``ve_factor`` outside
            [-10000, 10000]).

        Notes
        -----
        ``rvt.vis.mstp`` sets the no-data pixels to NaN in the caller's array before converting it to
        float32, so a float ``dem_arr`` is modified in place when ``no_data`` is given; an integer
        ``dem_arr`` with ``no_data`` raises ValueError (known issue, see docs/KNOWN_ISSUES.md).

        Known issue: ``ve_factor`` is not passed to ``rvt.vis.mstp`` (which then uses 1), unlike every
        other ``get_<vis>``. See docs/KNOWN_ISSUES.md.
        """
        mstp_arr = rvt.vis.mstp(dem=dem_arr, local_scale=self.mstp_local_scale, meso_scale=self.mstp_meso_scale,
                                broad_scale=self.mstp_broad_scale, lightness=self.mstp_lightness, no_data=no_data)
        return mstp_arr

    def save_mstp(self, dem_path, custom_dir=None, save_float=None, save_8bit=None):
        """
        Compute the multi-scale topographic position (RGB) of a DEM GeoTIFF and write the result rasters.

        Reads the DEM with ``get_raster_arr`` (whole raster, one band expected; pixel size and no-data value from
        the file), computes with ``get_mstp`` and writes GeoTIFFs with the projection and geotransform of the DEM,
        named by ``get_mstp_file_name`` (pattern there). ``save_visualizations`` calls this method when
        ``mstp_compute`` is set.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory; None writes next to the DEM.
        save_float : bool or None
            Write the float32 GeoTIFF (3 bands R, G, B = broad, meso, local, values 0..1, no-data NaN); None takes
            ``mstp_save_float``.
        save_8bit : bool or None
            Write the 8-bit GeoTIFF (stretch of ``float_to_8bit``, NaN becomes 255); None takes
            ``mstp_save_8bit``.

        Returns
        -------
        int
            0 if every requested output file exists and ``overwrite`` is 0 (nothing is computed),
            otherwise 1. With ``overwrite`` 0 an existing output file is never replaced.

        Raises
        ------
        Exception
            If both ``save_float`` and ``save_8bit`` are false (after the defaults are applied), or
            ``dem_path`` is not a file; also exceptions of ``rvt.vis`` for parameters out of range.

        Notes
        -----
        Known issue: ``rvt.vis.mstp`` assigns NaN to the no-data pixels before converting to float32, so a DEM
        GeoTIFF with an integer data type and a no-data value raises ValueError. See docs/KNOWN_ISSUES.md.

        Known issue: the "outputs exist" test for the 8-bit-only case is written as
        ``elif not save_float and not save_8bit`` (unreachable after the check above), so with only the 8-bit
        output requested and present and ``overwrite`` 0 the DEM is still read and computed, nothing is
        written and 1 is returned. See docs/KNOWN_ISSUES.md.

        Known issue: the 8-bit raster is written with ``no_data=np.nan``, which a byte band cannot hold;
        the file records no no-data value. See docs/KNOWN_ISSUES.md.

        Above ``tile_size_limit`` pixels the work is done by ``rvt.tile.save_rvt_visualization_tile_by_tile``,
        which replaces existing files and has its own known issues (see there).
        """

        # if save_float is None it takes boolean from default (self)
        if save_float is None:
            save_float = self.mstp_save_float
        # if save_8bit is None it takes boolean from default (self)
        if save_8bit is None:
            save_8bit = self.mstp_save_8bit

        if not save_float and not save_8bit:
            raise Exception("rvt.default.DefaultValues.save_mstp: Both save_float and save_8bit are False,"
                            " at least one of them has to be True!")

        if not os.path.isfile(dem_path):
            raise Exception("rvt.default.DefaultValues.save_mstp: dem_path doesn't exist!")

        if custom_dir is None:
            mstp_path = self.get_mstp_path(dem_path)
            mstp_8bit_path = self.get_mstp_path(dem_path, bit8=True)
        else:
            mstp_path = os.path.join(custom_dir, self.get_mstp_file_name(dem_path))
            mstp_8bit_path = os.path.join(custom_dir, self.get_mstp_file_name(dem_path, bit8=True))

        # if file already exists and overwrite=0
        if save_float and save_8bit:
            if os.path.isfile(mstp_8bit_path) and os.path.isfile(mstp_path) and not self.overwrite:
                return 0
        elif save_float and not save_8bit:
            if os.path.isfile(mstp_path) and not self.overwrite:
                return 0
        elif not save_float and not save_8bit:
            if os.path.isfile(mstp_8bit_path) and not self.overwrite:
                return 0

        dem_size = get_raster_size(raster_path=dem_path)
        if dem_size[0] * dem_size[1] > self.tile_size_limit:  # tile by tile calculation
            if custom_dir is None:
                custom_dir = Path(dem_path).parent
            rvt.tile.save_rvt_visualization_tile_by_tile(
                rvt_visualization=RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION,
                rvt_default=self,
                dem_path=Path(dem_path),
                output_dir_path=Path(custom_dir),
                save_float=save_float,
                save_8bit=save_8bit
            )
            return 1
        else:  # singleprocess
            dict_arr_res = get_raster_arr(raster_path=dem_path)
            dem_arr = dict_arr_res["array"]
            no_data = dict_arr_res["no_data"]

            mstp_arr = self.get_mstp(dem_arr=dem_arr, no_data=no_data)

            if save_float:
                if os.path.isfile(mstp_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    save_raster(
                        src_raster_path=dem_path,
                        out_raster_path=mstp_path,
                        out_raster_arr=mstp_arr,
                        no_data=np.nan,
                        e_type=6
                    )
            if save_8bit:
                if os.path.isfile(mstp_8bit_path) and not self.overwrite:  # file exists and overwrite=0
                    pass
                else:
                    mstp_8bit_arr = self.float_to_8bit(
                        float_arr=mstp_arr,
                        visualization=RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION
                    )
                    save_raster(
                        src_raster_path=dem_path,
                        out_raster_path=mstp_8bit_path,
                        out_raster_arr=mstp_8bit_arr,
                        no_data=np.nan,
                        e_type=1
                    )

            return 1

    def save_visualizations(self, dem_path, custom_dir=None):
        """
        Compute and save every visualization whose ``<vis>_compute`` flag is set, then write a log file.

        Calls, in this order and with the stored ``<vis>_save_float`` / ``<vis>_save_8bit`` flags:
        ``save_slope``, ``save_hillshade``, ``save_multi_hillshade``, ``save_slrm``, ``save_sky_view_factor``
        (once for SVF, anisotropic SVF and positive openness), ``save_neg_opns``, ``save_sky_illumination``,
        ``save_local_dominance``, ``save_msrm``, ``save_mstp``; then ``create_log_file`` with the elapsed time.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF (elevation in m).
        custom_dir : str or None
            Output directory for rasters and log; None uses the DEM directory.

        Raises
        ------
        Exception
            From the ``save_<vis>`` methods, e.g. when both save flags of a computed visualization are 0.
        """
        start_time = time.time()
        if self.slp_compute:
            self.save_slope(dem_path, custom_dir=custom_dir)
        if self.hs_compute:
            self.save_hillshade(dem_path, custom_dir=custom_dir)
        if self.mhs_compute:
            self.save_multi_hillshade(dem_path, custom_dir=custom_dir)
        if self.slrm_compute:
            self.save_slrm(dem_path, custom_dir=custom_dir)
        if self.svf_compute or self.asvf_compute or self.pos_opns_compute:
            self.save_sky_view_factor(dem_path, save_svf=bool(self.svf_compute), save_asvf=bool(self.asvf_compute),
                                      save_opns=bool(self.pos_opns_compute), custom_dir=custom_dir)
        if self.neg_opns_compute:
            self.save_neg_opns(dem_path, custom_dir=custom_dir)
        if self.sim_compute:
            self.save_sky_illumination(dem_path, custom_dir=custom_dir)
        if self.ld_compute:
            self.save_local_dominance(dem_path, custom_dir=custom_dir)
        if self.msrm_compute:
            self.save_msrm(dem_path, custom_dir=custom_dir)
        if self.mstp_compute:
            self.save_mstp(dem_path, custom_dir=custom_dir)
        end_time = time.time()
        compute_time = end_time - start_time
        self.create_log_file(dem_path=dem_path, custom_dir=custom_dir, compute_time=compute_time)

    def calculate_visualization(
            self,
            visualization: RVTVisualization,
            dem: np.array,
            resolution_x: float,
            resolution_y: float,
            no_data: Optional[float] = None,
            save_float: bool = True,
            save_8bit: bool = False
    ) -> Optional[Tuple[np.array, np.array]]:  # tuple[vis_float_arr, vis_8bit_arr]
        """
        Compute the float and/or 8-bit result of one visualization from a DEM array with the stored parameters.

        Dispatches to the ``get_<vis>`` method of ``visualization`` and to ``float_to_8bit``. ``rvt.tile`` calls
        it once per tile.

        Parameters
        ----------
        visualization : RVTVisualization
            Visualization to compute.
        dem : numpy.ndarray
            2D elevation array (m), shape (H, W). May be modified in place (no-data set to NaN, see the
            ``get_<vis>`` methods).
        resolution_x : float
            Pixel size in x (m/px); the only pixel size for visualizations that take one ``resolution``.
        resolution_y : float
            Pixel size in y (m/px); used by SLOPE, HILLSHADE and MULTI_HILLSHADE only.
        no_data : float or None
            No-data value of the DEM; None: no masking.
        save_float : bool
            Return the float result.
        save_8bit : bool
            Return the 8-bit result (``float_to_8bit``; for MULTI_HILLSHADE computed from ``dem``).

        Returns
        -------
        tuple
            ``(vis_float_arr, vis_8bit_arr)``; an entry is None when its flag is false. Dtypes and shapes are
            those of the ``get_<vis>`` method (e.g. float32 (H, W) for SLOPE, float64 (16, H-2, W-2) for
            MULTI_HILLSHADE, float32 (3, H, W) for MSTP) and of ``float_to_8bit`` (uint8; int64 for SHADOW).
            For a value that is not a member of ``RVTVisualization`` the float entry is None.

        Raises
        ------
        Exception
            From ``float_to_8bit`` for an unknown visualization when ``save_8bit`` is true, and from ``rvt.vis``
            for parameters out of range.

        Notes
        -----
        Known issue: with ``save_float`` and ``save_8bit`` both true the float result is the same array object
        that ``float_to_8bit`` clips in place, so the returned float result is clipped to the stretch range
        (e.g. slope limited to 0..51 deg). See docs/KNOWN_ISSUES.md.
        """
        vis_arr = None
        vis_float_arr = None
        vis_8bit_arr = None
        if visualization == RVTVisualization.SLOPE:
            vis_arr = self.get_slope(
                dem_arr=dem, resolution_x=resolution_x, resolution_y=resolution_y, no_data=no_data
            )
        elif visualization == RVTVisualization.SHADOW:
            vis_arr = self.get_shadow(
                dem_arr=dem, resolution=resolution_x, no_data=no_data
            )
        elif visualization == RVTVisualization.HILLSHADE:
            vis_arr = self.get_hillshade(
                dem_arr=dem, resolution_x=resolution_x, resolution_y=resolution_y, no_data=no_data
            )
        elif visualization == RVTVisualization.MULTI_HILLSHADE:
            vis_arr = self.get_multi_hillshade(
                dem_arr=dem, resolution_x=resolution_x, resolution_y=resolution_y, no_data=no_data
            )
        elif visualization == RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL:
            vis_arr = self.get_slrm(
                dem_arr=dem, no_data=no_data
            )
        elif visualization == RVTVisualization.SKY_VIEW_FACTOR:
            vis_arr = self.get_sky_view_factor(
                dem_arr=dem,
                resolution=resolution_x,
                compute_svf=True,
                compute_asvf=False,
                compute_opns=False,
                no_data=no_data
            )["svf"]
        elif visualization == RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR:
            vis_arr = self.get_sky_view_factor(
                dem_arr=dem,
                resolution=resolution_x,
                compute_svf=False,
                compute_asvf=True,
                compute_opns=False,
                no_data=no_data
            )["asvf"]
        elif visualization == RVTVisualization.POSITIVE_OPENNESS:
            vis_arr = self.get_sky_view_factor(
                dem_arr=dem,
                resolution=resolution_x,
                compute_svf=False,
                compute_asvf=False,
                compute_opns=True,
                no_data=no_data
            )["opns"]
        elif visualization == RVTVisualization.NEGATIVE_OPENNESS:
            vis_arr = self.get_neg_opns(
                dem_arr=dem, resolution=resolution_x, no_data=no_data
            )
        elif visualization == RVTVisualization.SKY_ILLUMINATION:
            vis_arr = self.get_sky_illumination(
                dem_arr=dem, resolution=resolution_x, no_data=no_data
            )
        elif visualization == RVTVisualization.LOCAL_DOMINANCE:
            vis_arr = self.get_local_dominance(
                dem_arr=dem, no_data=no_data
            )
        elif visualization == RVTVisualization.MULTI_SCALE_RELIEF_MODEL:
            vis_arr = self.get_msrm(
                dem_arr=dem, resolution=resolution_x, no_data=no_data
            )
        elif visualization == RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION:
            vis_arr = self.get_mstp(
                dem_arr=dem, no_data=no_data
            )
        if save_float:
            vis_float_arr = vis_arr
        if save_8bit:
            if visualization == RVTVisualization.MULTI_HILLSHADE:
                vis_8bit_arr = self.float_to_8bit(
                    float_arr=dem,
                    visualization=visualization,
                    x_res=resolution_x,
                    y_res=resolution_y,
                    no_data=no_data
                )
            else:
                vis_8bit_arr = self.float_to_8bit(
                    float_arr=vis_arr,
                    visualization=visualization
                )
        return vis_float_arr, vis_8bit_arr

    def create_log_file(self, dem_path, custom_dir=None, compute_time=None):
        """
        Write a text log of the DEM metadata, the selected parameters and the output file names.

        The log is ``<dem>_vis_log_<YYYY-mm-dd_HH-MM-SS>.txt`` (local time; ``<dem>`` is the base name without
        its last extension) in ``custom_dir`` or the DEM directory. It lists, for every visualization whose
        ``<vis>_compute`` flag is set, its parameters and the absolute paths of the outputs selected by the
        ``<vis>_save_float`` / ``<vis>_save_8bit`` flags. It describes the current attributes; it does not check
        which files were written.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF.
        custom_dir : str or None
            Directory of the log and of the listed outputs; None uses the DEM directory.
        compute_time : float or None
            Processing time (s) written at the end; None omits the line.

        Notes
        -----
        Known issue: the whole DEM is read into memory (``get_raster_arr``) to get its shape, also for rasters
        above ``tile_size_limit`` that the ``save_<vis>`` methods process tile by tile. See docs/KNOWN_ISSUES.md.

        Known issue: for MSRM the line ">> Output file:" is written twice; for MSTP the float file is listed
        whatever ``mstp_save_float`` is and the 8-bit file is never listed. See docs/KNOWN_ISSUES.md.
        """
        # Structure of this method:
        #   1. read the DEM (array shape, resolution); number of bands, rows, columns
        #   2. log directory and time-stamped log name
        #   3. header, input metadata, general parameters (overwrite, ve_factor, tiling on/off)
        #   4. one block per visualization with *_compute set: hillshade (+ shadow), multi hillshade, slope,
        #      SLRM, MSRM, SVF, ASVF, positive and negative openness, sky illumination, local dominance, MSTP
        #   5. computation time, close the file
        dict_arr_res = get_raster_arr(raster_path=dem_path)
        resolution = dict_arr_res["resolution"]
        arr_shape = np.array(dict_arr_res["array"]).shape
        del dict_arr_res
        nr_bands = 0
        nr_cols = 0
        nr_rows = 0
        if len(arr_shape) == 3:
            nr_bands = arr_shape[0]
            nr_rows = arr_shape[1]
            nr_cols = arr_shape[2]
        elif len(arr_shape) == 2:
            nr_bands = 1
            nr_rows = arr_shape[0]
            nr_cols = arr_shape[1]
        dem_dir = os.path.dirname(dem_path)
        log_dir = dem_dir
        if custom_dir is not None:
            log_dir = custom_dir
        dem_name = os.path.splitext(os.path.basename(dem_path))[0]
        log_file_time = datetime.datetime.now()
        log_file_time_str = log_file_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_name = "{}_vis_log_{}.txt".format(dem_name, log_file_time_str)
        log_path = os.path.join(log_dir, log_name)
        dat = open(log_path, "w")
        dat.write(
            "===============================================================================================\n"
            "Relief Visualization Toolbox (python), visualizations log\n"
            "Copyright:\n"
            "\tResearch Centre of the Slovenian Academy of Sciences and Arts\n"
            "\tUniversity of Ljubljana, Faculty of Civil and Geodetic Engineering\n"
            "===============================================================================================\n")
        dat.write("\n\n\n")

        dat.write("Processing info about visualizations\n"
                  "===============================================================================================\n\n")
        dat.write("# Metadata of the input file\n\n")
        dat.write("\tInput filename:\t\t{}\n".format(dem_path))
        dat.write("\tNumber of rows:\t\t{}\n".format(nr_rows))
        dat.write("\tNumber of columns:\t{}\n".format(nr_cols))
        dat.write("\tNumber of bands:\t{}\n".format(nr_bands))
        dat.write("\tResolution (x, y):\t{}, {}\n".format(resolution[0], resolution[1]))
        dat.write("\n")

        dat.write("# Selected visualization parameters\n")
        dat.write("\tOverwrite: {}\n".format(self.overwrite))
        dat.write("\tVertical exaggeration factor: {}\n".format(self.ve_factor))
        if nr_rows * nr_cols > self.tile_size_limit:
            dat.write("\tCalculating tile by tile: {}\n".format("ON"))
            dat.write("\t\tTile block size: {}x{}\n".format(self.tile_size[0],
                                                            self.tile_size[1]))
        else:
            dat.write("\tCalculating tile by tile: {}\n".format("OFF"))

        dat.write("\n")

        dat.write("# The following visualizations have been performed:\n\n")
        if self.hs_compute:
            dat.write("\tHillshade\n")
            dat.write("\t\ths_sun_el=\t\t{}\n".format(self.hs_sun_el))
            dat.write("\t\ths_sun_azi=\t\t{}\n".format(self.hs_sun_azi))
            if self.hs_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_hillshade_file_name(dem_path)))))
            if self.hs_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\ths_bytscl=\t\t({}, {}, {})\n".format(self.hs_bytscl[0], self.hs_bytscl[1],
                                                                    self.hs_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_hillshade_file_name(dem_path, bit8=True)))))
            if self.hs_shadow:
                dat.write("\t\t>> Output shadow file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_shadow_file_name(dem_path)))))
            dat.write("\n")
        if self.mhs_compute:
            dat.write("\tMultiple directions hillshade\n")
            dat.write("\t\tmhs_sun_el=\t\t{}\n".format(self.mhs_sun_el))
            dat.write("\t\tmhs_nr_dir=\t\t{}\n".format(self.mhs_nr_dir))
            if self.mhs_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_multi_hillshade_file_name(dem_path)))))
            if self.mhs_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tmhs_bytscl=\t\t({}, {}, {})\n".format(self.mhs_bytscl[0], self.mhs_bytscl[1],
                                                                     self.mhs_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_multi_hillshade_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.slp_compute:
            dat.write("\tSlope gradient\n")
            dat.write("\t\tslp_output_units=\t\t{}\n".format(self.slp_output_units))
            if self.slp_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_slope_file_name(dem_path)))))
            if self.slp_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tslp_bytscl=\t\t({}, {}, {})\n".format(self.slp_bytscl[0], self.slp_bytscl[1],
                                                                     self.slp_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_slope_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.slrm_compute:
            dat.write("\tSimple local relief model\n")
            dat.write("\t\tslrm_rad_cell=\t\t{}\n".format(self.slrm_rad_cell))
            if self.slrm_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_slrm_file_name(dem_path)))))
            if self.slrm_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tslrm_bytscl=\t\t({}, {}, {})\n".format(self.slrm_bytscl[0], self.slrm_bytscl[1],
                                                                      self.slrm_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_slrm_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.msrm_compute:
            dat.write("\tMulti-scale relief model\n")
            dat.write("\t\tmsrm_feature_min=\t\t{}\n".format(self.msrm_feature_min))
            dat.write("\t\tmsrm_feature_max=\t\t{}\n".format(self.msrm_feature_max))
            dat.write("\t\tmsrm_scaling_factor=\t\t{}\n".format(self.msrm_scaling_factor))
            if self.msrm_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_msrm_file_name(dem_path)))))
            if self.msrm_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tmsrm_bytscl=\t\t({}, {}, {})\n".format(self.msrm_bytscl[0], self.msrm_bytscl[1],
                                                                      self.msrm_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_msrm_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.svf_compute:
            dat.write("\tSky-View Factor\n")
            dat.write("\t\tsvf_n_dir=\t\t{}\n".format(self.svf_n_dir))
            dat.write("\t\tsvf_noise=\t\t{}\n".format(self.svf_noise))
            dat.write("\t\tsvf_r_max=\t\t{}\n".format(self.svf_r_max))
            if self.svf_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(os.path.join(log_dir, self.get_svf_file_name(dem_path)))))
            if self.svf_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tsvf_bytscl=\t\t({}, {}, {})\n".format(self.svf_bytscl[0], self.svf_bytscl[1],
                                                                     self.svf_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_svf_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.asvf_compute:
            dat.write("\tAnisotropic Sky-View Factor\n")
            dat.write("\t\tsvf_n_dir=\t\t{}\n".format(self.svf_n_dir))
            dat.write("\t\tsvf_noise=\t\t{}\n".format(self.svf_noise))
            dat.write("\t\tsvf_r_max=\t\t{}\n".format(self.svf_r_max))
            dat.write("\t\tasvf_level=\t\t{}\n".format(self.asvf_level))
            dat.write("\t\tasvf_dir=\t\t{}\n".format(self.asvf_dir))
            if self.svf_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_asvf_file_name(dem_path)))))
            if self.svf_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tasvf_bytscl=\t\t({}, {}, {})\n".format(self.asvf_bytscl[0], self.asvf_bytscl[1],
                                                                      self.asvf_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_asvf_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.pos_opns_compute:
            dat.write("\tOpenness - Positive\n")
            dat.write("\t\tsvf_n_dir=\t\t{}\n".format(self.svf_n_dir))
            dat.write("\t\tsvf_noise=\t\t{}\n".format(self.svf_noise))
            dat.write("\t\tsvf_r_max=\t\t{}\n".format(self.svf_r_max))
            if self.svf_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_opns_file_name(dem_path)))))
            if self.svf_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tpos_opns_bytscl=\t\t({}, {}, {})\n".format(self.pos_opns_bytscl[0],
                                                                          self.pos_opns_bytscl[1],
                                                                          self.pos_opns_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_opns_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.neg_opns_compute:
            dat.write("\tOpenness - Negative\n")
            dat.write("\t\tsvf_n_dir=\t\t{}\n".format(self.svf_n_dir))
            dat.write("\t\tsvf_noise=\t\t{}\n".format(self.svf_noise))
            dat.write("\t\tsvf_r_max=\t\t{}\n".format(self.svf_r_max))
            if self.neg_opns_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_neg_opns_file_name(dem_path)))))
            if self.neg_opns_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tneg_opns_bytscl=\t\t({}, {}, {})\n".format(self.neg_opns_bytscl[0],
                                                                          self.neg_opns_bytscl[1],
                                                                          self.neg_opns_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_neg_opns_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.sim_compute:
            dat.write("\tSky illumination\n")
            dat.write("\t\tsim_sky_mod=\t\t{}\n".format(self.sim_sky_mod))
            dat.write("\t\tsim_compute_shadow=\t\t{}\n".format(self.sim_compute_shadow))
            dat.write("\t\tsim_shadow_az=\t\t{}\n".format(self.sim_shadow_az))
            dat.write("\t\tsim_shadow_el=\t\t{}\n".format(self.sim_shadow_el))
            dat.write("\t\tsim_nr_dir=\t\t{}\n".format(self.sim_nr_dir))
            dat.write("\t\tsim_shadow_dist=\t\t{}\n".format(self.sim_shadow_dist))
            if self.sim_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_sky_illumination_file_name(dem_path)))))
            if self.sim_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tsim_bytscl=\t\t({}, {}, {})\n".format(self.sim_bytscl[0], self.sim_bytscl[1],
                                                                     self.sim_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_sky_illumination_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.ld_compute:
            dat.write("\tLocal dominance\n")
            dat.write("\t\tld_rad_inc=\t\t{}\n".format(self.ld_rad_inc))
            dat.write("\t\tld_min_rad=\t\t{}\n".format(self.ld_min_rad))
            dat.write("\t\tld_max_rad=\t\t{}\n".format(self.ld_max_rad))
            dat.write("\t\tld_anglr_res=\t\t{}\n".format(self.ld_anglr_res))
            dat.write("\t\tld_observer_h=\t\t{}\n".format(self.ld_observer_h))
            if self.ld_save_float:
                dat.write("\t\t>> Output file:\n")
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_local_dominance_file_name(dem_path)))))
            if self.ld_save_8bit:
                dat.write("\t\t>> Output 8bit file:\n")
                dat.write("\t\tld_bytscl=\t\t({}, {}, {})\n".format(self.ld_bytscl[0], self.ld_bytscl[1],
                                                                    self.ld_bytscl[2]))
                dat.write("\t\t\t{}\n".format(os.path.abspath(
                    os.path.join(log_dir, self.get_local_dominance_file_name(dem_path, bit8=True)))))
            dat.write("\n")
        if self.mstp_compute:
            dat.write("\tMulti-scale topographic position\n")
            dat.write("\t\tmstp_local_scale=\t({}, {}, {})\n".format(
                self.mstp_local_scale[0], self.mstp_local_scale[1], self.mstp_local_scale[2]))
            dat.write("\t\tmstp_meso_scale=\t({}, {}, {})\n".format(
                self.mstp_meso_scale[0], self.mstp_meso_scale[1], self.mstp_meso_scale[2]))
            dat.write("\t\tmstp_broad_scale=\t({}, {}, {})\n".format(
                self.mstp_broad_scale[0], self.mstp_broad_scale[1], self.mstp_broad_scale[2]))
            dat.write("\t\tmstp_lightness=\t\t{}\n".format(self.mstp_lightness))
            dat.write("\t\t>> Output file:\n")
            dat.write("\t\t\t{}\n".format(os.path.abspath(
                os.path.join(log_dir, self.get_mstp_file_name(dem_path)))))
            dat.write("\n")

        if compute_time is not None:
            dat.write("# Computation time: {:.3f}s".format(compute_time))
        dat.close()


def get_raster_arr(raster_path):
    """
    Read a raster with GDAL and return its pixel values, pixel size and no-data value.

    Parameters
    ----------
    raster_path : str
        Path of the raster (any format GDAL reads, normally a GeoTIFF).

    Returns
    -------
    dict
        ``{"array": numpy.ndarray, "resolution": (x_res, y_res), "no_data": float or None}``. The array has the
        data type of the file (not converted to float32): (H, W) for one band, (B, H, W) for several bands.
        ``x_res = abs(gt[1])`` and ``y_res = abs(gt[5])`` (m/px for a projected raster). ``no_data`` is the
        no-data value of band 1 (assumed for all bands), None if the file has none.

    Notes
    -----
    The whole raster is read into memory. ``gdal.Open`` returns None for a missing file, so a missing path
    raises AttributeError here (or RuntimeError when GDAL exceptions are enabled).
    """
    data_set = gdal.Open(raster_path)
    gt = data_set.GetGeoTransform()
    x_res = abs(gt[1])
    y_res = abs(-gt[5])
    bands = []
    no_data = data_set.GetRasterBand(1).GetNoDataValue()  # we assume that all the bands have same no_data val
    if data_set.RasterCount == 1:  # only one band
        array = np.array(data_set.GetRasterBand(1).ReadAsArray())
        data_set = None
        return {"array": array, "resolution": (x_res, y_res), "no_data": no_data}
    else:  # multiple bands
        for i_band in range(data_set.RasterCount):
            i_band += 1
            band = np.array(data_set.GetRasterBand(i_band).ReadAsArray())
            if band is None:
                continue
            else:
                bands.append(band)
        data_set = None  # close dataset
        return {"array": np.array(bands), "resolution": (x_res, y_res), "no_data": no_data}


def get_raster_size(raster_path, band=1):
    """
    Return the size in px of one band of a raster without reading its pixels.

    Parameters
    ----------
    raster_path : str
        Path of the raster.
    band : int
        Band number, starting at 1.

    Returns
    -------
    tuple of int
        ``(x_size, y_size)`` = (number of columns, number of rows).
    """
    data_set = gdal.Open(raster_path)  # Open dem raster
    band = data_set.GetRasterBand(band)
    x_size = band.XSize  # number of columns
    y_size = band.YSize  # number of rows
    del band
    data_set = None  # close data_set
    return x_size, y_size


def save_raster(src_raster_path, out_raster_path, out_raster_arr: np.ndarray, no_data=None, e_type=6):
    """
    Write an array to a GeoTIFF with the projection and geotransform of a source raster.

    Parameters
    ----------
    src_raster_path : str
        Path of the raster whose projection and geotransform are copied (normally the DEM).
    out_raster_path : str
        Path of the GeoTIFF to write; an existing file is replaced.
    out_raster_arr : numpy.ndarray
        (H, W) for one band or (B, H, W) for B bands. Its size is used as is; it is not checked against the
        source raster.
    no_data : float or None
        No-data value stored in the file (set on band 1; a GeoTIFF holds one value for all bands). None stores
        none. NaN cannot be stored for a byte raster.
    e_type : int
        GDAL data type code: 6 = ``GDT_Float32`` (default), 1 = ``GDT_Byte``; see
        https://gdal.org/api/raster_c_api.html#_CPPv412GDALDataType. Values are converted by GDAL.

    Raises
    ------
    Exception
        If ``out_raster_arr`` is not 2D or 3D.

    Notes
    -----
    Creation options ``COMPRESS=LZW`` and ``BIGTIFF=IF_SAFER``.
    """
    src_data_set = gdal.Open(src_raster_path)
    gtiff_driver = gdal.GetDriverByName("GTiff")
    if len(out_raster_arr.shape) == 2:  # 2D array, one band
        out_data_set = gtiff_driver.Create(out_raster_path, xsize=out_raster_arr.shape[1],
                                           ysize=out_raster_arr.shape[0],
                                           bands=1,
                                           eType=e_type,  # eType: 6 = GDT_Float32
                                           options=['COMPRESS=LZW', 'BIGTIFF=IF_SAFER'])
        out_data_set.SetProjection(src_data_set.GetProjection())
        out_data_set.SetGeoTransform(src_data_set.GetGeoTransform())
        out_data_set.GetRasterBand(1).WriteArray(out_raster_arr)
        if no_data is not None:
            out_data_set.GetRasterBand(1).SetNoDataValue(no_data)

    elif len(out_raster_arr.shape) == 3:  # 3D array, more bands
        out_data_set = gtiff_driver.Create(out_raster_path, xsize=out_raster_arr.shape[2],
                                           ysize=out_raster_arr.shape[1],
                                           bands=out_raster_arr.shape[0],
                                           eType=e_type,  # eType: 6 = GDT_Float32
                                           options=['COMPRESS=LZW', 'BIGTIFF=IF_SAFER'])
        out_data_set.SetProjection(src_data_set.GetProjection())
        out_data_set.SetGeoTransform(src_data_set.GetGeoTransform())
        for i_band in range(out_raster_arr.shape[0]):
            out_data_set.GetRasterBand(i_band + 1).WriteArray(out_raster_arr[i_band, :, :])
        if no_data is not None:
            out_data_set.GetRasterBand(1).SetNoDataValue(no_data)
    else:
        raise Exception("rvt.default.save_raster: You have to input 2D or 3D numpy array!")
    out_data_set.FlushCache()
    src_data_set = None  # Close source data set
    out_data_set = None  # Close output data set
