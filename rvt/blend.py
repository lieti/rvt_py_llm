"""
Relief Visualization Toolbox – layer blending, blender settings files and composite visualizations.

Relation to the other modules
-----------------------------
- `rvt.vis` computes single visualizations from a DEM array (slope, hillshade, SVF, MSTP, ...).
- `rvt.default.DefaultValues` holds the parameters of every visualization and computes (``get_*``) or saves
  (``save_*``) them; `rvt.blend` uses it to compute a layer image that was not supplied.
- `rvt.blend_func` holds the per-array operations used here (normalization, blend modes, opacity, color ramps).
- `rvt.blend` (this module) combines visualizations into one image.

Layer model
-----------
BlenderLayer
    One layer: a visualization (given as array, as raster path, or computed from the DEM), its normalization
    ("value" or "perc" with min / max), a blend mode (normal, multiply, screen, overlay, soft_light, luminosity),
    an opacity in percent (0-100) and an optional matplotlib colormap.
BlenderCombination
    An ordered list of layers. ``layers[0]`` is the top layer, ``layers[-1]`` the bottom layer.
    render_all_images composes from the bottom up: the bottom layer is the start image, then each layer above is
    blended onto the result with its own blend mode and opacity.
BlenderCombinations
    A list of named combinations, e.g. the presets of ``settings/default_blender_combinations.json``.

Settings files (JSON, directory ``settings/`` of the repository)
---------------------------------------------------------------
- One combination (``blender_file_example.json``, ``blender_VAT.json``, ``blender_custom_layers.json``)::

      {"combination": {"name": str, "layers": [
          {"layer": "1", "visualization_method": str, "norm": "Value" | "Perc", "min": float, "max": float,
           "blend_mode": str, "opacity": int, "colormap": str (optional), "min_colormap_cut": float (optional),
           "max_colormap_cut": float (optional)}, ...]}}

  A layer with ``"visualization_method": "None"`` (or null) is skipped. Read by BlenderCombination.read_from_file,
  written by BlenderCombination.save_to_file.
- Several combinations (``default_blender_combinations.json``): ``{"combinations": [<one combination>, ...]}``,
  read and written by BlenderCombinations.
- Terrain presets (``default_terrains_settings.json``): ``{"terrains_settings": [{"terrain_settings": {"name": str,
  "<visualization name>": {"stretch": {"min", "max"}, "<parameter>": {"value": ...}}, ...}}, ...]}``, read by
  TerrainsSettings / TerrainSettings. TerrainSettings.apply_terrain copies the parameters into a DefaultValues
  instance and the stretches into the min / max of the matching layers of a combination.

Composite visualizations
------------------------
color_relief_image_map (CRIM), e3mstp and e4mstp compute their input visualizations with `rvt.vis` and
DefaultValues and combine them with fixed BlenderCombination recipes, documented in each function. They return
``(3, H, W)`` float64 RGB arrays with values in [0, 1] and NaN at no-data pixels (e4mstp writes the ``no_data``
value there instead, when one is given).

Optimized path
--------------
render_all_images evaluates the range test "any value outside [0, 1]" (``rvt.blend_func._outside_0_1``) once per
array and passes the result on instead of recomputing it on the same, unchanged array (commit 8e81581, "K-11").
The test is a pure function of the array values, so reusing it is exact; where the array may have changed (overlay
and soft_light write into the background, a rescaled operand) the test is recomputed. The blend and opacity kernels
themselves are evaluated in row blocks in `rvt.blend_func`; outputs are bitwise identical to the original code.

Known issues
------------
Latent bugs (unused opacity of the bottom layer, no visible effect of the opacity of overlay / soft_light layers,
slope units of e4mstp, copy-paste errors in TerrainSettings.read_from_json, ...) are described in the Notes of the
affected functions and listed in docs/KNOWN_ISSUES.md; the behaviour is kept as it is.

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
    2010-2020 Research Centre of the Slovenian Academy of Sciences and Arts
    2016-2020 University of Ljubljana, Faculty of Civil and Geodetic Engineering
"""

# TODO: more testing, find and fix bugs if they exists

import copy
import datetime
import json
import os
import warnings

import numpy as np

import rvt.default
import rvt.vis
from rvt.blend_func import (
    blend_images,
    gray_scale_to_color_ramp,
    normalize_image,
    render_images,
    scale_0_to_1,
)
from rvt.blend_func import _outside_0_1, _render_images


def create_blender_file_example(file_path=None):
    """
    Write an example combination file (the "VAT - Archaeological" combination) in JSON.

    The file has the format read by BlenderCombination.read_from_file: four layers (sky-view factor multiply 25 %,
    positive openness overlay 50 %, slope luminosity 50 %, hillshade normal 100 %) and an empty fifth layer.

    Parameters
    ----------
    file_path : str or None
        Output path; an existing file is overwritten. None writes ``settings\\blender_file_example.json`` relative
        to the current working directory and creates the directory ``settings`` if it is missing.

    Notes
    -----
    Known issue: the default path is written with a backslash. On POSIX ``os.path.dirname`` of it is "" and
    ``os.makedirs("")`` raises FileNotFoundError, so ``file_path=None`` works on Windows only. See
    docs/KNOWN_ISSUES.md.
    """
    data = {"combination": {"name": "VAT - Archaeological",
                            "layers":
                                [
                                    {"layer": "1", "visualization_method": "Sky-View Factor", "norm": "Value",
                                     "min": 0.7, "max": 1.0,
                                     "blend_mode": "Multiply", "opacity": 25},
                                    {"layer": "2", "visualization_method": "Openness - Positive",
                                     "norm": "Value", "min": 68, "max": 93,
                                     "blend_mode": "Overlay", "opacity": 50},
                                    {"layer": "3", "visualization_method": "Slope gradient", "norm": "Value",
                                     "min": 0, "max": 50,
                                     "blend_mode": "Luminosity", "opacity": 50},
                                    {"layer": "4", "visualization_method": "Hillshade", "norm": "Value",
                                     "min": 0, "max": 1,
                                     "blend_mode": "Normal", "opacity": 100},
                                    {"layer": "5", "visualization_method": "None"}
                                ]
                            }}
    if file_path is None:
        file_path = r"settings\blender_file_example.json"
        if os.path.isfile(file_path):
            pass
        else:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))

    dat = open(file_path, "w")
    dat.write(json.dumps(data, indent=4))
    dat.close()


class BlenderLayer:
    """
    One layer of a blending combination: a visualization with its normalization, blend mode and opacity.

    BlenderLayer is the element of ``BlenderCombination.layers``. The constructor stores its arguments without
    checks; BlenderCombination.render_all_images calls check_data before rendering.

    Parameters
    ----------
    vis_method : str or None
        Visualization name, stored as ``vis``. To be computed from the DEM it must be one of the names listed in
        check_data (case-insensitive); with ``image`` or ``image_path`` any name is accepted (the names
        "slope gradient" and "openness - negative" still invert the layer, see
        rvt.blend_func.normalize_image). None makes an empty layer, which BlenderCombination skips.
    normalization : str
        "value" (``minimum`` / ``maximum`` are values) or "perc" (they are percentages cut off at the low / high
        end); case-insensitive. "percent" (lower case only) is accepted and renamed to "perc" by check_data.
    minimum : float or None
        Lower normalization limit (value, or percent 0-100).
    maximum : float or None
        Upper normalization limit (value, or percent 0-100).
    blend_mode : str
        "normal", "multiply", "screen", "overlay", "soft_light" or "luminosity", case-insensitive.
    opacity : int or float
        Opacity of the layer in percent, 0-100 (see rvt.blend_func.apply_opacity for values <= 1).
    colormap : str or None
        matplotlib colormap name applied to a gray layer after normalization; None keeps it gray.
    min_colormap_cut : float or None
        Lower end of the used part of the colormap, in [0, 1].
    max_colormap_cut : float or None
        Upper end of the used part of the colormap, in [0, 1].
    image : numpy.ndarray or None
        Precomputed visualization ``(H, W)`` or ``(3, H, W)``. Takes precedence over ``image_path``. Clipped in
        place by the normalization during rendering.
    image_path : str or None
        Path of a visualization raster (GeoTIFF) read when ``image`` is None.

    Attributes
    ----------
    vis : str or None
        Visualization name (``vis_method``).
    normalization : str or None
        Normalization type.
    min : float or None
        Normalization minimum.
    max : float or None
        Normalization maximum.
    blend_mode : str or None
        Blend mode.
    opacity : int or float or None
        Opacity in percent.
    colormap : str or None
        Colormap name.
    min_colormap_cut : float or None
        Lower colormap cut.
    max_colormap_cut : float or None
        Upper colormap cut.
    image : numpy.ndarray or None
        Visualization array.
    image_path : str or None
        Visualization raster path.
    """

    def __init__(self, vis_method=None, normalization="value", minimum=None, maximum=None,
                 blend_mode="normal", opacity=100, colormap=None, min_colormap_cut=None, max_colormap_cut=None,
                 image=None, image_path=None):
        self.vis = vis_method
        self.normalization = normalization
        self.min = minimum
        self.max = maximum
        self.blend_mode = blend_mode
        self.opacity = opacity
        self.colormap = colormap
        self.min_colormap_cut = min_colormap_cut
        self.max_colormap_cut = max_colormap_cut
        self.image_path = image_path
        self.image = image

    def check_data(self):
        """
        Validate the layer attributes; normalize some of them in place.

        A layer with ``vis`` None is emptied (every other attribute set to None). Otherwise "percent" is renamed
        to "perc" and the attributes are checked.

        Raises
        ------
        Exception
            If the normalization is not "value" / "perc" (case-insensitive); for "perc" if ``min + max >= 100``;
            for "value" if ``min > max``; if the blend mode is unknown; if a colormap cut is set without a
            colormap; if neither ``image`` nor ``image_path`` is set and ``vis`` is not one of: slope gradient,
            hillshade, shadow, multiple directions hillshade, simple local relief model, sky-view factor,
            anisotropic sky-view factor, openness - positive, openness - negative, sky illumination, local
            dominance, multi-scale relief model, multi-scale topographic position (case-insensitive).
        AttributeError
            If ``normalization`` or ``blend_mode`` is None while ``vis`` is set.
        TypeError
            If ``min`` or ``max`` is None while ``vis`` is set.

        Notes
        -----
        Known issues (docs/KNOWN_ISSUES.md): the opacity check ``0 > self.opacity > 100`` can never be true, so
        any opacity is accepted (values above 100 extrapolate in rendering); ``min == max`` is accepted for
        "value" (advanced_normalization then raises for "value", or gives all NaN for "Value").
        """
        if self.normalization == "percent":
            self.normalization = "perc"
        if self.vis is None:  # if visualization is None everything is None
            self.normalization = None
            self.min = None
            self.max = None
            self.blend_mode = None
            self.opacity = None
            self.colormap = None  # leave None if you don't want to apply colormap to grayscale
            self.min_colormap_cut = None  # if None it equals to 0
            self.max_colormap_cut = None  # if None it equals to 1
            self.image = None  # leave None if you wish for blender to compute visualization
            self.image_path = None  # leave None if you wish for blender to compute visualization
        else:
            if not (self.normalization.lower() == "value" or self.normalization.lower() == "perc"):
                raise Exception("rvt.blend.BlenderLayer.check_data: normalization value incorrect!")
            if self.normalization.lower() == "perc" and (self.min + self.max) >= 100:
                raise Exception("rvt.blend.BlenderLayer.check_data: when normalization is perc, min + max has "
                                "to smaller then 100%!")
            if self.min > self.max and self.normalization.lower() == "value":
                raise Exception("rvt.blend.BlenderLayer.check_data: min bigger than max!")
            if self.blend_mode.lower() != "normal" and self.blend_mode.lower() != "multiply" and \
                    self.blend_mode.lower() != "overlay" and self.blend_mode.lower() != "luminosity" and \
                    self.blend_mode.lower() != "screen" and self.blend_mode.lower() != "soft_light":
                raise Exception("rvt.blend.BlenderLayer.check_data: blend_mode incorrect!")
            if not 0 <= self.opacity <= 100:
                raise Exception("rvt.blend.BlenderLayer.check_data: opacity incorrect [0-100]!")
            if self.colormap is None and (self.min_colormap_cut is not None or self.max_colormap_cut is not None):
                raise Exception("rvt.blend.BlenderLayer.check_data: colormap not defined but min_colormap_cut or "
                                "max_colormap_cut are!")
            if self.image is None and self.image_path is None:
                if self.vis.lower() != "slope gradient" and self.vis.lower() != "hillshade" and \
                        self.vis.lower() != "shadow" and self.vis.lower() != "multiple directions hillshade" and \
                        self.vis.lower() != "simple local relief model" and self.vis.lower() != "sky-view factor" and \
                        self.vis.lower() != "anisotropic sky-view factor" and \
                        self.vis.lower() != "openness - positive" and self.vis.lower() != "openness - negative" and \
                        self.vis.lower() != "sky illumination" and self.vis.lower() != "local dominance" and \
                        self.vis.lower() != "multi-scale relief model" and \
                        self.vis.lower() != "multi-scale topographic position":
                    raise Exception("rvt.blend.BlenderLayer.check_data: Incorrect vis, if you don't input image or "
                                    "image_path you have to input known visualization method (vis)!")


class BlenderCombination:
    """
    An ordered stack of layers and the DEM they are computed from; renders the blended image.

    ``layers[0]`` is the top layer and ``layers[-1]`` the bottom layer. render_all_images starts from the bottom
    layer and blends each layer above onto the result with its blend mode and opacity.

    Parameters
    ----------
    dem_arr : numpy.ndarray or None
        2D DEM ``(H, W)`` in m, used to compute layers in memory.
    dem_resolution : float or None
        Pixel size of ``dem_arr`` in m/px (the same in x and y).
    dem_path : str or None
        Path of the DEM raster, used when visualizations are computed and saved as files, and as the georeference
        of saved renders.

    Attributes
    ----------
    dem_arr : numpy.ndarray or None
        DEM array.
    dem_resolution : float or None
        DEM pixel size in m/px.
    dem_path : str or None
        DEM raster path.
    name : str
        Name of the combination ("" after construction).
    layers : list of BlenderLayer
        Layers, top first.
    """

    def __init__(self, dem_arr=None, dem_resolution=None, dem_path=None):
        self.dem_arr = dem_arr
        self.dem_resolution = dem_resolution
        self.dem_path = dem_path
        self.name = ""
        self.layers = []

    def add_dem_arr(self, dem_arr, dem_resolution):
        """
        Set the DEM array and its pixel size.

        Parameters
        ----------
        dem_arr : numpy.ndarray
            2D DEM ``(H, W)`` in m.
        dem_resolution : float
            Pixel size in m/px.
        """
        self.dem_arr = dem_arr
        self.dem_resolution = dem_resolution

    def add_dem_path(self, dem_path):
        """
        Set the path of the DEM raster.

        Parameters
        ----------
        dem_path : str
            Path of the DEM GeoTIFF.
        """
        self.dem_path = dem_path

    def create_layer(self, vis_method=None, normalization="value", minimum=None, maximum=None,
                     blend_mode="normal", opacity=100, colormap=None, min_colormap_cut=None, max_colormap_cut=None,
                     image=None, image_path=None):
        """
        Create a BlenderLayer and append it below the existing layers.

        Nothing is added when ``vis_method`` is None. The arguments are those of BlenderLayer and are not checked
        here.

        Parameters
        ----------
        vis_method : str or None
            Visualization name, see BlenderLayer.
        normalization : str
            "value" or "perc".
        minimum : float or None
            Lower normalization limit.
        maximum : float or None
            Upper normalization limit.
        blend_mode : str
            Blend mode name.
        opacity : int or float
            Opacity in percent, 0-100.
        colormap : str or None
            matplotlib colormap name.
        min_colormap_cut : float or None
            Lower colormap cut in [0, 1].
        max_colormap_cut : float or None
            Upper colormap cut in [0, 1].
        image : numpy.ndarray or None
            Precomputed visualization (stored by reference, clipped in place during rendering).
        image_path : str or None
            Path of a visualization raster.
        """
        if vis_method is not None:
            layer = BlenderLayer(vis_method=vis_method, normalization=normalization, minimum=minimum, maximum=maximum,
                                 blend_mode=blend_mode, opacity=opacity, colormap=colormap,
                                 min_colormap_cut=min_colormap_cut, max_colormap_cut=max_colormap_cut,
                                 image=image, image_path=image_path)
            self.layers.append(layer)

    def add_layer(self, layer: BlenderLayer):
        """
        Append a layer below the existing layers, unless its ``vis`` is None.

        Parameters
        ----------
        layer : BlenderLayer
            Layer to append (stored by reference, not copied).
        """
        if layer.vis is not None:
            self.layers.append(layer)

    def remove_all_layers(self):
        """Remove all layers (``layers`` becomes a new empty list)."""
        self.layers = []

    def layers_info(self):
        """
        Return a one-line text description of every layer.

        Returns
        -------
        list of str
            One string per layer, top first: "layer <n>, visualization = ..., normalization = ..., min = ...,
            max = ..., blend_mode = ..., opacity = ..., colormap = ..., min_colormap_cut = ..., max_colormap_cut".

        Notes
        -----
        Known issue: the format string ends with "max_colormap_cut" without "= {}", so the value of
        max_colormap_cut is never printed (the extra format argument is ignored). See docs/KNOWN_ISSUES.md.
        """
        layer_nr = 1
        layers_info = []
        for layer in self.layers:
            layers_info.append("layer {}, visualization = {}, normalization = {}, min = {}, max = {}, blend_mode = {},"
                               " opacity = {}, colormap = {},"
                               " min_colormap_cut = {}, max_colormap_cut".format(layer_nr, layer.vis,
                                                                                 layer.normalization,
                                                                                 layer.min,
                                                                                 layer.max, layer.blend_mode,
                                                                                 layer.opacity,
                                                                                 layer.colormap,
                                                                                 layer.min_colormap_cut,
                                                                                 layer.max_colormap_cut))
            layer_nr += 1
        return layers_info

    def read_from_file(self, file_path):
        """
        Replace the name and the layers with those of a combination JSON file.

        Parameters
        ----------
        file_path : str
            Path of a file in the one-combination format (module docstring), e.g.
            ``settings/blender_file_example.json``.

        Raises
        ------
        KeyError
            If a required key is missing (see read_from_json).
        """
        dat = open(file_path, "r")
        json_data = json.load(dat)
        self.read_from_json(json_data)
        dat.close()

    def read_from_json(self, json_data):
        """
        Replace the name and the layers with those of a parsed combination JSON document.

        Layers whose "visualization_method" is null, "None" or "null" (case-insensitive) are skipped. For the other
        layers "norm" and "blend_mode" are read as str, "min" and "max" as float, "opacity" as int; "colormap" is
        optional ("None" / "null" mean no colormap), and the cuts are read only when a colormap is set. The values
        are not checked here (render_all_images calls check_data).

        Parameters
        ----------
        json_data : dict
            ``{"combination": {"name": str, "layers": [...]}}`` as described in the module docstring.

        Raises
        ------
        KeyError
            If "combination", "name", "layers" or a required layer key ("visualization_method", "norm", "min",
            "max", "blend_mode", "opacity") is missing.

        Notes
        -----
        Known issue: the colormap cuts are converted with ``str()``, so numbers from the JSON become strings such
        as "0.2"; rendering such a layer raises TypeError in gray_scale_to_color_ramp. A save_to_file /
        read_from_file round trip therefore turns numeric cuts into strings. See docs/KNOWN_ISSUES.md.
        """
        self.layers = []
        self.name = json_data["combination"]["name"]
        layers_data = json_data["combination"]["layers"]
        for layer in layers_data:
            if layer["visualization_method"] is None:
                continue
            if layer["visualization_method"].lower() == "none" or layer["visualization_method"].lower() == "null":
                continue
            else:
                vis_method = str(layer["visualization_method"])
            norm = str(layer["norm"])
            norm_min = float(layer["min"])
            norm_max = float(layer["max"])
            blend_mode = str(layer["blend_mode"])
            opacity = int(layer["opacity"])
            colormap = None
            min_colormap_cut = None
            max_colormap_cut = None
            try:
                colormap = str(layer["colormap"])
                if colormap.lower() == "null" or colormap.lower() == "none":
                    colormap = None
            except KeyError:
                colormap = None
            if colormap is not None:
                try:
                    min_colormap_cut = str(layer["min_colormap_cut"])
                    if min_colormap_cut.lower() == "null" or min_colormap_cut.lower() == "none":
                        min_colormap_cut = None
                except KeyError:
                    min_colormap_cut = None
                try:
                    max_colormap_cut = str(layer["max_colormap_cut"])
                    if max_colormap_cut.lower() == "null" or max_colormap_cut.lower() == "none":
                        max_colormap_cut = None
                except KeyError:
                    max_colormap_cut = None

            self.add_layer(BlenderLayer(vis_method=vis_method, normalization=norm, minimum=norm_min, maximum=norm_max,
                                        blend_mode=blend_mode, opacity=opacity, colormap=colormap,
                                        min_colormap_cut=min_colormap_cut, max_colormap_cut=max_colormap_cut))

    def save_to_file(self, file_path):
        """
        Write the combination as a JSON file (format of to_json, indent 4).

        ``image`` and ``image_path`` of the layers are not written, so the file describes how to compute the
        layers, not their data.

        Parameters
        ----------
        file_path : str
            Output path; an existing file is overwritten.
        """
        json_data = self.to_json()
        dat = open(file_path, "w")
        dat.write(json.dumps(json_data, indent=4))
        dat.close()

    def to_json(self):
        """
        Return the combination as a JSON-serializable dict.

        Returns
        -------
        dict
            ``{"combination": {"name": name, "layers": [{"layer": "1", "visualization_method", "norm", "min",
            "max", "blend_mode", "opacity"}, ...]}}``, layers top first, numbered from "1". "colormap",
            "min_colormap_cut" and "max_colormap_cut" are added only when they are not None. Values are written as
            stored in the layers.
        """
        json_data = {"combination": {"name": self.name,
                                     "layers": []
                                     }}
        i_layer = 1
        for layer in self.layers:
            if layer.colormap is None:
                json_data["combination"]["layers"].append({"layer": str(i_layer), "visualization_method": layer.vis,
                                                           "norm": layer.normalization, "min": layer.min,
                                                           "max": layer.max, "blend_mode": layer.blend_mode,
                                                           "opacity": layer.opacity})
            else:
                if layer.min_colormap_cut is None and layer.max_colormap_cut is None:
                    json_data["combination"]["layers"].append({"layer": str(i_layer), "visualization_method": layer.vis,
                                                               "norm": layer.normalization, "min": layer.min,
                                                               "max": layer.max, "blend_mode": layer.blend_mode,
                                                               "opacity": layer.opacity, "colormap": layer.colormap})
                elif layer.min_colormap_cut is not None and layer.max_colormap_cut is None:
                    json_data["combination"]["layers"].append({"layer": str(i_layer), "visualization_method": layer.vis,
                                                               "norm": layer.normalization, "min": layer.min,
                                                               "max": layer.max, "blend_mode": layer.blend_mode,
                                                               "opacity": layer.opacity, "colormap": layer.colormap,
                                                               "min_colormap_cut": layer.min_colormap_cut})
                elif layer.min_colormap_cut is None and layer.max_colormap_cut is not None:
                    json_data["combination"]["layers"].append({"layer": str(i_layer), "visualization_method": layer.vis,
                                                               "norm": layer.normalization, "min": layer.min,
                                                               "max": layer.max, "blend_mode": layer.blend_mode,
                                                               "opacity": layer.opacity, "colormap": layer.colormap,
                                                               "max_colormap_cut": layer.max_colormap_cut})
                elif layer.min_colormap_cut is not None and layer.max_colormap_cut is not None:
                    json_data["combination"]["layers"].append({"layer": str(i_layer), "visualization_method": layer.vis,
                                                               "norm": layer.normalization, "min": layer.min,
                                                               "max": layer.max, "blend_mode": layer.blend_mode,
                                                               "opacity": layer.opacity, "colormap": layer.colormap,
                                                               "min_colormap_cut": layer.min_colormap_cut,
                                                               "max_colormap_cut": layer.max_colormap_cut})
            i_layer += 1
        return json_data

    def check_data(self):
        """
        Run BlenderLayer.check_data on every layer (may rename "percent" to "perc" in place).

        Raises
        ------
        Exception
            From BlenderLayer.check_data for the first invalid layer.
        """
        for layer in self.layers:
            layer.check_data()

    def render_all_images(self, default=None, save_visualizations=False, save_render_path=None, save_float=True,
                          save_8bit=False, no_data=None):
        """
        Render all layers into one blended image, optionally saving it as GeoTIFF.

        The image of each layer is taken from ``layer.image``; if that is None, read from ``layer.image_path``;
        if both are None, computed from the DEM with the parameters of ``default``: in memory from ``dem_arr`` /
        ``dem_resolution`` (``save_visualizations=False``), or saved as a GeoTIFF from ``dem_path`` and read back
        (``save_visualizations=True``). Each image is normalized with rvt.blend_func.normalize_image and, if the
        layer has a colormap and the image is gray, colored. The bottom layer is the start image; every layer
        above is blended onto the result (rvt.blend_func.blend_images) and mixed with it according to its
        opacity (rvt.blend_func.render_images).

        Parameters
        ----------
        default : rvt.default.DefaultValues or None
            Visualization parameters for layers computed from the DEM; None uses a new DefaultValues().
        save_visualizations : bool
            True: compute missing layers from ``dem_path`` and save them as files (float GeoTIFF; the multiple
            directions hillshade as 8-bit). False: compute them in memory from ``dem_arr``.
        save_render_path : str or None
            Path of the float GeoTIFF of the render (requires ``dem_path`` for the georeference); None does not
            save. The 8-bit render is written next to it as ``<name>_8bit.tif``.
        save_float : bool
            Save the render as float32 GeoTIFF at ``save_render_path``.
        save_8bit : bool
            Save the render as 8-bit GeoTIFF (``rvt.vis.byte_scale`` of [0, 1]).
        no_data : float or None
            No-data value of ``dem_arr``, passed to the visualization functions for layers computed in memory
            (they replace it by NaN). Not used for files.

        Returns
        -------
        numpy.ndarray
            Blended image, ``(H, W)`` if every layer is gray, ``(3, H, W)`` if any layer is RGB (colormap,
            multiple directions hillshade, 3-band image). float32 for gray float32 layers, float64 as soon as an RGB
            layer is involved. Values in [0, 1] for inputs in range (a warning is issued otherwise); NaN where the
            layers are NaN.

        Raises
        ------
        Exception
            From check_data; if ``save_render_path`` is set without ``dem_path``; if ``save_render_path`` is set
            with ``save_float`` and ``save_8bit`` both False; if a layer has to be computed and the DEM it needs is
            missing (``dem_path``, or both ``dem_arr`` and ``dem_resolution``).
        AttributeError
            For a gray layer with a colormap read from ``image_path`` or computed with
            ``save_visualizations=True`` (see Notes).

        Warns
        -----
        UserWarning
            "Rendered image scale distorted" when an intermediate result has values outside [0, 1].

        Notes
        -----
        Side effects: check_data may modify layer attributes; each ``layer.image`` is clipped in place to its
        normalization limits (rvt.blend_func.normalize_lin).

        Computed layers: "slope gradient" uses DefaultValues.get_slope (unit ``default.slp_output_units``);
        "multiple directions hillshade" in memory is an RGB stack of three hillshades at azimuths 315°, 22.5° and
        90° with ``default.mhs_sun_el``, independent of ``default.mhs_nr_dir``; from file it is the 8-bit raster,
        normalized 0-255 first.

        Optimized path: the range test of the rendered image at the end of a layer is reused as the background
        test of the next layer, and _render_images gets range tests already evaluated on the same unchanged
        arrays; see the module docstring.

        Known issues (docs/KNOWN_ISSUES.md):

        - the blend mode and the opacity of the bottom layer are never used (it is the start image as it is);
        - overlay and soft_light write into the background, so the opacity of such a layer has no visible effect
          (``top`` and ``background`` hold the same values when the opacity is applied);
        - a colormap is applied only ``if len(image.shape) < 3`` with ``image`` the layer array, which is None
          for layers read from ``image_path`` or saved with ``save_visualizations=True``: AttributeError;
        - with ``save_visualizations=True`` the visualizations are saved into the directory of
          ``save_render_path`` but read back from the directory of ``dem_path``; if the two differ, reading fails;
        - the in-memory check ``self.dem_arr is None and self.dem_resolution is None`` uses ``and``, so a missing
          ``dem_arr`` with a given resolution is not reported and fails later in rvt.vis.
        """

        # Structure of this function:
        #   1. check the layers and the save arguments; derive the 8-bit render path; default DefaultValues;
        #   2. loop over the layers from the bottom (last) to the top (first):
        #      a. skip empty layers; check that the DEM needed to compute the layer is available;
        #      b. get the normalized image: from layer.image, from layer.image_path, or computed from the DEM
        #         (in memory with default.get_*, or saved with default.save_* and read back);
        #      c. apply the colormap to a gray image;
        #      d. bottom layer: becomes the rendered image (no blending, no opacity);
        #         other layers: rescale operands outside [0, 1], blend_images, _render_images with the opacity,
        #         range test of the result (warning if outside [0, 1]);
        #   3. save the float and / or 8-bit render; return the rendered image.

        # Perform checks
        self.check_data()

        if save_render_path is not None and self.dem_path is None:
            raise Exception(
                "rvt.blend.BlenderCombination.render_all_images: If you would like to save rendered image (blender), "
                "you have to define dem_path (BlenderCombination.add_dem_path())!")

        if not save_float and not save_8bit and save_render_path:
            raise Exception(
                "rvt.blend.BlenderCombination.render_all_images: If you would like to save rendered image (blender), "
                "you have to set save_float or save_8bit to True!")

        # Prepare directory for saving renders
        if save_render_path is not None:
            save_render_directory = os.path.abspath(os.path.dirname(save_render_path))
            save_render_8bit_path = os.path.join(
                save_render_directory,
                "{}_8bit.tif".format(os.path.splitext(os.path.basename(save_render_path))[0])
            )
        else:
            save_render_directory = None
            save_render_8bit_path = None

        # If default (rvt.default.DefaultValues class) is not defined, use predefined values
        if default is None:
            default = rvt.default.DefaultValues()

        # Rendering across all layers - form last to first layer
        rendered_image = None
        rendered_outside = None
        for i_img in range(len(self.layers) - 1, -1, -1):

            # Get visualization type (string)
            visualization = self.layers[i_img].vis
            if visualization is None:  # empty layer, skip
                continue

            # Get other parameters for processing this layer
            min_norm = self.layers[i_img].min
            max_norm = self.layers[i_img].max
            normalization = self.layers[i_img].normalization
            image = self.layers[i_img].image
            image_path = self.layers[i_img].image_path

            if save_visualizations and self.dem_path is None and image_path is None and image is None:
                raise Exception(
                    "rvt.blend.BlenderCombination.render_all_images: If you would like to save visualizations, "
                    "you have to define dem_path (BlenderCombination.add_dem_path())!")

            if not save_visualizations and self.dem_arr is None and self.dem_resolution is None and \
                    image_path is None and image is None:
                raise Exception(
                    "rvt.blend.BlenderCombination.render_all_images: If you would like to compute visualizations, "
                    "you have to define dem_arr and its resolution (BlenderCombination.add_dem_arr())!")

            # Normalize images
            norm_image = None
            if image is None and image_path is not None:
                # LOAD image from file if it doesn't exist (as an array) but we have image_path present
                norm_image = normalize_image(
                    visualization,
                    rvt.default.get_raster_arr(image_path)["array"],
                    min_norm,
                    max_norm,
                    normalization
                )
            elif image is not None:
                # if image exist (as an array)
                norm_image = normalize_image(
                    visualization,
                    image,
                    min_norm,
                    max_norm,
                    normalization
                )
            else:
                # calculate image from DEM
                if self.layers[i_img].vis.lower() == "slope gradient":
                    if save_visualizations:
                        default.save_slope(dem_path=self.dem_path, custom_dir=save_render_directory, save_float=True,
                                           save_8bit=False)
                        image_path = default.get_slope_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_slope(dem_arr=self.dem_arr, resolution_x=self.dem_resolution,
                                                  resolution_y=self.dem_resolution, no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "hillshade":
                    if save_visualizations:
                        default.save_hillshade(dem_path=self.dem_path, custom_dir=save_render_directory,
                                               save_float=True, save_8bit=False)
                        image_path = default.get_hillshade_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_hillshade(dem_arr=self.dem_arr, resolution_x=self.dem_resolution,
                                                      resolution_y=self.dem_resolution, no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "shadow":
                    if save_visualizations:
                        default.save_hillshade(dem_path=self.dem_path, custom_dir=save_render_directory,
                                               save_float=True, save_8bit=False, save_shadow=True)
                        image_path = default.get_shadow_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_shadow(dem_arr=self.dem_arr, resolution=self.dem_resolution,
                                                   no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)

                elif self.layers[i_img].vis.lower() == "multiple directions hillshade":
                    if save_visualizations:
                        default.save_multi_hillshade(dem_path=self.dem_path, custom_dir=save_render_directory,
                                                     save_float=False, save_8bit=True)
                        image_path = default.get_multi_hillshade_path(self.dem_path, bit8=True)
                        norm_image = normalize_image("", rvt.default.get_raster_arr(image_path)["array"],
                                                     0, 255, normalization)
                        norm_image = normalize_image(visualization, norm_image,
                                                     min_norm, max_norm, normalization)
                    else:
                        red_band_arr = rvt.vis.hillshade(dem=self.dem_arr, resolution_x=self.dem_resolution,
                                                         resolution_y=self.dem_resolution,
                                                         sun_elevation=default.mhs_sun_el, sun_azimuth=315,
                                                         no_data=no_data)
                        green_band_arr = rvt.vis.hillshade(dem=self.dem_arr, resolution_x=self.dem_resolution,
                                                           resolution_y=self.dem_resolution,
                                                           sun_elevation=default.mhs_sun_el, sun_azimuth=22.5,
                                                           no_data=no_data)
                        blue_band_arr = rvt.vis.hillshade(dem=self.dem_arr, resolution_x=self.dem_resolution,
                                                          resolution_y=self.dem_resolution,
                                                          sun_elevation=default.mhs_sun_el, sun_azimuth=90,
                                                          no_data=no_data)
                        image = np.array([red_band_arr, green_band_arr, blue_band_arr])
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "simple local relief model":
                    if save_visualizations:
                        default.save_slrm(dem_path=self.dem_path, custom_dir=save_render_directory, save_float=True,
                                          save_8bit=False)
                        image_path = default.get_slrm_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_slrm(dem_arr=self.dem_arr, no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "sky-view factor":
                    if save_visualizations:
                        default.save_sky_view_factor(dem_path=self.dem_path, save_svf=True, save_asvf=False,
                                                     save_opns=False, custom_dir=save_render_directory, save_float=True,
                                                     save_8bit=False)
                        image_path = default.get_svf_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_sky_view_factor(dem_arr=self.dem_arr, resolution=self.dem_resolution,
                                                            compute_svf=True, compute_asvf=False,
                                                            compute_opns=False, no_data=no_data)["svf"]
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "anisotropic sky-view factor":
                    if save_visualizations:
                        default.save_sky_view_factor(dem_path=self.dem_path, save_svf=False, save_asvf=True,
                                                     save_opns=False, custom_dir=save_render_directory, save_float=True,
                                                     save_8bit=False)
                        image_path = default.get_asvf_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_sky_view_factor(dem_arr=self.dem_arr, resolution=self.dem_resolution,
                                                            compute_svf=False, compute_asvf=True,
                                                            compute_opns=False, no_data=no_data)["asvf"]
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "openness - positive":
                    if save_visualizations:
                        default.save_sky_view_factor(dem_path=self.dem_path, save_svf=False, save_asvf=False,
                                                     save_opns=True, custom_dir=save_render_directory, save_float=True,
                                                     save_8bit=False)
                        image_path = default.get_opns_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_sky_view_factor(dem_arr=self.dem_arr, resolution=self.dem_resolution,
                                                            compute_svf=False, compute_asvf=False,
                                                            compute_opns=True, no_data=no_data)["opns"]
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "openness - negative":
                    if save_visualizations:
                        default.save_neg_opns(dem_path=self.dem_path, custom_dir=save_render_directory, save_float=True,
                                              save_8bit=False)
                        image_path = default.get_neg_opns_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_neg_opns(dem_arr=self.dem_arr, resolution=self.dem_resolution,
                                                     no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "sky illumination":
                    if save_visualizations:
                        default.save_sky_illumination(dem_path=self.dem_path, custom_dir=save_render_directory,
                                                      save_float=True, save_8bit=False)
                        image_path = default.get_sky_illumination_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_sky_illumination(dem_arr=self.dem_arr, resolution=self.dem_resolution,
                                                             no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "local dominance":
                    if save_visualizations:
                        default.save_local_dominance(dem_path=self.dem_path, custom_dir=save_render_directory,
                                                     save_float=True, save_8bit=False)
                        image_path = default.get_local_dominance_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_local_dominance(dem_arr=self.dem_arr, no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "multi-scale relief model":
                    if save_visualizations:
                        default.save_msrm(dem_path=self.dem_path, custom_dir=save_render_directory,
                                          save_float=True, save_8bit=False)
                        image_path = default.get_msrm_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_msrm(dem_arr=self.dem_arr, resolution=self.dem_resolution, no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)
                elif self.layers[i_img].vis.lower() == "multi-scale topographic position":
                    if save_visualizations:
                        default.save_mstp(
                            dem_path=self.dem_path, custom_dir=save_render_directory, save_float=True, save_8bit=False
                        )
                        image_path = default.get_mstp_path(self.dem_path)
                        norm_image = normalize_image(visualization, rvt.default.get_raster_arr(image_path)["array"],
                                                     min_norm, max_norm, normalization)
                    else:
                        image = default.get_mstp(dem_arr=self.dem_arr, no_data=no_data)
                        norm_image = normalize_image(visualization, image, min_norm, max_norm, normalization)

            # Apply colormap
            colormap = self.layers[i_img].colormap
            min_colormap_cut = self.layers[i_img].min_colormap_cut
            max_colormap_cut = self.layers[i_img].max_colormap_cut
            if colormap is not None and len(image.shape) < 3:
                norm_image = gray_scale_to_color_ramp(gray_scale=norm_image, colormap=colormap, output_8bit=False,
                                                      min_colormap_cut=min_colormap_cut,
                                                      max_colormap_cut=max_colormap_cut)

            # Blend current layer with background layer
            if rendered_image is None:
                # if current layer has visualization applied, but there has been no rendering
                # of images yet, than current layer will be the initial value of rendered_image
                rendered_image = norm_image
                rendered_outside = None  # range test of rendered_image not evaluated yet
                continue
            else:
                active = norm_image
                background = rendered_image

                # Clip images to [0, 1] if needed (new array; NaN and dtype kept)
                active_outside = _outside_0_1(active)
                if active_outside:
                    active = np.clip(active, 0, 1)
                # rendered_image is unchanged since its range test at the end of the previous layer
                background_outside = _outside_0_1(background) if rendered_outside is None else rendered_outside
                if background_outside:
                    background = np.clip(background, 0, 1)

                # Blend background with active layer
                blend_mode = self.layers[i_img].blend_mode
                top = blend_images(blend_mode, active, background)

                # Apply opacity. Range tests already evaluated on the same unchanged arrays are passed on: top is the
                # (unscaled) active itself for normal blending, and only overlay / soft_light modify the background.
                opacity = self.layers[i_img].opacity
                rendered_image = _render_images(
                    top, background, opacity,
                    active_outside=False if (top is active and not active_outside) else None,
                    background_outside=None if (background_outside or blend_mode.lower() in ("overlay", "soft_light"))
                    else False)

                rendered_outside = _outside_0_1(rendered_image)
                if rendered_outside:
                    warnings.warn("rvt.blend.BlenderCombination.render_all_images: Rendered image scale distorted")

        # Save image to file if path is present
        if save_render_path is not None:
            if save_float:
                rvt.default.save_raster(src_raster_path=self.dem_path, out_raster_path=save_render_path,
                                        out_raster_arr=rendered_image)
            if save_8bit:
                rendered_image_8bit = rvt.vis.byte_scale(rendered_image, c_min=0, c_max=1)
                rvt.default.save_raster(src_raster_path=self.dem_path, out_raster_path=save_render_8bit_path,
                                        out_raster_arr=rendered_image_8bit, e_type=1)

        return rendered_image  # returns float

    def create_log_file(self, dem_path, combination_name, render_path, default: rvt.default.DefaultValues,
                        terrain_sett_name=None, custom_dir=None, computation_time=None):
        """
        Write a text log of a blending run: DEM metadata, parameters and layers.

        The file is named ``<dem name>_blend_log_<YYYY-MM-DD_HH-MM-SS>.txt`` (local time of the call) and is
        written into ``custom_dir``, or into the directory of ``dem_path`` if ``custom_dir`` is None.

        Parameters
        ----------
        dem_path : str
            Path of the DEM raster; it is read completely to get its shape and resolution.
        combination_name : str
            Name of the combination written to the log.
        render_path : str
            Path of the render file written to the log.
        default : rvt.default.DefaultValues
            Parameters written to the log (overwrite, ve_factor, and the parameters of each layer's
            visualization).
        terrain_sett_name : str or None
            Name of the terrain settings written to the log; None omits the line.
        custom_dir : str or None
            Output directory; None uses the directory of ``dem_path``.
        computation_time : float or str or None
            Computation time written at the end; None omits the line.

        Notes
        -----
        A second call within the same second overwrites the log. Every layer must have a ``vis`` (not None).
        """
        # Structure of this function:
        #   1. read the DEM to get the number of bands, rows, columns and the resolution;
        #   2. build the log path from custom_dir (or the DEM directory), the DEM name and the current time;
        #   3. write the header, the DEM metadata, overwrite and ve_factor, the combination name, the terrain
        #      settings name and the render path;
        #   4. per layer: name, the DefaultValues parameters of its visualization, normalization, limits, opacity,
        #      colormap and cuts;
        #   5. write the computation time and close the file.
        dict_arr_res = rvt.default.get_raster_arr(raster_path=dem_path)
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
        log_name = "{}_blend_log_{}.txt".format(dem_name, log_file_time_str)
        log_path = os.path.join(log_dir, log_name)
        dat = open(log_path, "w")
        dat.write(
            "===============================================================================================\n"
            "Relief Visualization Toolbox (python), blender log\n"
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
        dat.write("\tOverwrite: {}\n".format(default.overwrite))
        dat.write("\tVertical exaggeration factor: {}\n".format(default.ve_factor))
        dat.write("\n")

        dat.write("# Combination:\n\n")
        dat.write("Combination name: {}\n".format(combination_name))
        if terrain_sett_name is not None:
            dat.write("Terrain settings: {}\n".format(terrain_sett_name))
        dat.write("\t>> Output render file:\n")
        dat.write("\t\t{}\n\n".format(render_path))
        dat.write("=== LAYERS ===\n\n")
        i_layer = 1
        for layer in self.layers:
            dat.write("Layer: {}\n".format(i_layer))
            dat.write("Visualization: {}\n".format(layer.vis))
            if layer.vis.lower() == "hillshade":
                dat.write("\ths_sun_el=\t\t{}\n".format(default.hs_sun_el))
                dat.write("\ths_sun_azi=\t\t{}\n".format(default.hs_sun_azi))
            elif layer.vis.lower() == "shadow":
                dat.write("\ths_sun_el=\t\t{}\n".format(default.hs_sun_el))
                dat.write("\ths_sun_azi=\t\t{}\n".format(default.hs_sun_azi))
            elif layer.vis.lower() == "multiple directions hillshade":
                dat.write("\tmhs_sun_el=\t\t{}\n".format(default.mhs_sun_el))
                dat.write("\tmhs_nr_dir=\t\t{}\n".format(default.mhs_nr_dir))
            elif layer.vis.lower() == "slope gradient":
                dat.write("\tslp_output_units=\t\t{}\n".format(default.slp_output_units))
            elif layer.vis.lower() == "simple local relief model":
                dat.write("\tslrm_rad_cell=\t\t{}\n".format(default.slrm_rad_cell))
            elif layer.vis.lower() == "sky-view factor":
                dat.write("\tsvf_n_dir=\t\t{}\n".format(default.svf_n_dir))
                dat.write("\tsvf_noise=\t\t{}\n".format(default.svf_noise))
                dat.write("\tsvf_r_max=\t\t{}\n".format(default.svf_r_max))
            elif layer.vis.lower() == "anisotropic sky-view factor":
                dat.write("\tsvf_n_dir=\t\t{}\n".format(default.svf_n_dir))
                dat.write("\tsvf_noise=\t\t{}\n".format(default.svf_noise))
                dat.write("\tsvf_r_max=\t\t{}\n".format(default.svf_r_max))
                dat.write("\tasvf_level=\t\t{}\n".format(default.asvf_level))
                dat.write("\tasvf_dir=\t\t{}\n".format(default.asvf_dir))
            elif layer.vis.lower() == "openness - positive":
                dat.write("\tsvf_n_dir=\t\t{}\n".format(default.svf_n_dir))
                dat.write("\tsvf_noise=\t\t{}\n".format(default.svf_noise))
                dat.write("\tsvf_r_max=\t\t{}\n".format(default.svf_r_max))
            elif layer.vis.lower() == "openness - negative":
                dat.write("\tsvf_n_dir=\t\t{}\n".format(default.svf_n_dir))
                dat.write("\tsvf_noise=\t\t{}\n".format(default.svf_noise))
                dat.write("\tsvf_r_max=\t\t{}\n".format(default.svf_r_max))
            elif layer.vis.lower() == "sky illumination":
                dat.write("\tsim_sky_mod=\t\t{}\n".format(default.sim_sky_mod))
                dat.write("\tsim_compute_shadow=\t\t{}\n".format(default.sim_compute_shadow))
                dat.write("\tsim_shadow_az=\t\t{}\n".format(default.sim_shadow_az))
                dat.write("\tsim_shadow_el=\t\t{}\n".format(default.sim_shadow_el))
                dat.write("\tsim_nr_dir=\t\t{}\n".format(default.sim_nr_dir))
                dat.write("\tsim_shadow_dist=\t\t{}\n".format(default.sim_shadow_dist))
            elif layer.vis.lower() == "local dominance":
                dat.write("\t\tld_rad_inc=\t\t{}\n".format(default.ld_rad_inc))
                dat.write("\t\tld_min_rad=\t\t{}\n".format(default.ld_min_rad))
                dat.write("\t\tld_max_rad=\t\t{}\n".format(default.ld_max_rad))
                dat.write("\t\tld_anglr_res=\t\t{}\n".format(default.ld_anglr_res))
                dat.write("\t\tld_observer_h=\t\t{}\n".format(default.ld_observer_h))
            elif layer.vis.lower() == "multi-scale relief model":
                dat.write("\t\tmsrm_feature_min=\t\t{}\n".format(default.msrm_feature_min))
                dat.write("\t\tmsrm_feature_max=\t\t{}\n".format(default.msrm_feature_max))
                dat.write("\t\tmsrm_scaling_factor=\t\t{}\n".format(default.msrm_scaling_factor))
            elif layer.vis.lower() == "multi-scale topographic position":
                dat.write("\t\tmstp_local_scale=\t\t({}, {}, {})\n".format(
                    default.mstp_local_scale[0], default.mstp_local_scale[1], default.mstp_local_scale[2]))
                dat.write("\t\tmstp_meso_scale=\t\t({}, {}, {})\n".format(
                    default.mstp_meso_scale[0], default.mstp_meso_scale[1], default.mstp_meso_scale[2]))
                dat.write("\t\tmstp_broad_scale=\t\t({}, {}, {})\n".format(
                    default.mstp_broad_scale[0], default.mstp_broad_scale[1], default.mstp_broad_scale[2]))
                dat.write("\t\tmstp_lightness=\t\t{}\n".format(default.mstp_lightness))
            dat.write("Norm: {}\n".format(layer.normalization))
            dat.write("Linear normalization, min: {}, max: {}\n".format(layer.min, layer.max))
            dat.write("Opacity: {}\n".format(layer.opacity))
            if layer.colormap is not None:
                dat.write("Colormap: {}\n".format(layer.colormap))
                if layer.min_colormap_cut is not None:
                    dat.write("Minimum Colormap cut: {}\n".format(layer.min_colormap_cut))
                if layer.max_colormap_cut is not None:
                    dat.write("Maximum Colormap cut: {}\n".format(layer.max_colormap_cut))
            dat.write("\n")

            i_layer += 1

        if computation_time is not None:
            dat.write("# Computation time: {}".format(computation_time))
        dat.close()


def compare_2_combinations(combination1: BlenderCombination, combination2: BlenderCombination):
    """
    Return True if two combinations have the same layers in the same order.

    Layers are compared by visualization name and blend mode (both case-insensitive) and by opacity.

    Parameters
    ----------
    combination1 : BlenderCombination
        First combination.
    combination2 : BlenderCombination
        Second combination.

    Returns
    -------
    bool
        True if the number of layers and, per layer, the three compared attributes are equal. Two empty
        combinations are equal.

    Notes
    -----
    Known issue (possibly intended): normalization, min and max (and colormap and cuts) are not compared, so
    combinations that differ only in their limits are reported equal. The comparisons of normalization, min and
    max were commented out in the upstream code; one reason may be that TerrainSettings.apply_terrain changes the
    limits of preset combinations, which then still match their preset. See docs/KNOWN_ISSUES.md.
    """
    if len(combination1.layers) != len(combination2.layers):
        return False
    for i_layer in range(len(combination1.layers)):
        if combination1.layers[i_layer].vis.lower() != combination2.layers[i_layer].vis.lower():
            return False
        # normalization, min and max are not compared (see Notes; the checks were commented out upstream)
        if combination1.layers[i_layer].blend_mode.lower() != combination2.layers[i_layer].blend_mode.lower():
            return False
        if combination1.layers[i_layer].opacity != combination2.layers[i_layer].opacity:
            return False
    return True


class BlenderCombinations:
    """
    A list of named combinations, e.g. the presets of ``settings/default_blender_combinations.json``.

    Attributes
    ----------
    combinations : list of BlenderCombination
        The combinations, in file / insertion order.
    """

    def __init__(self):
        self.combinations = []  # list of BlenderCombination

    def add_combination(self, combination: BlenderCombination, name=None):
        """
        Append a combination (by reference), optionally renaming it.

        Parameters
        ----------
        combination : BlenderCombination
            Combination to append.
        name : str or None
            New name written into ``combination.name``; None keeps the name.
        """
        if name is not None:
            combination.name = name
        self.combinations.append(combination)

    def remove_all_combinations(self):
        """Remove all combinations (``combinations`` becomes a new empty list)."""
        self.combinations = []

    def select_combination_by_name(self, name):
        """
        Return the first combination with the given name.

        Parameters
        ----------
        name : str
            Name to look for (exact, case-sensitive).

        Returns
        -------
        BlenderCombination or None
            The first match, or None if there is none.
        """
        for combination in self.combinations:
            if combination.name == name:
                return combination

    def remove_combination_by_name(self, name):
        """
        Remove every combination with the given name.

        Parameters
        ----------
        name : str
            Name to remove (exact, case-sensitive).

        Returns
        -------
        int
            0 if the list is empty afterwards, else 1. The value does not say whether anything was removed.
        """
        new_combinations = []
        for combination in self.combinations:
            if combination.name != name:
                new_combinations.append(combination)
        self.combinations = new_combinations
        if not new_combinations:
            return 0
        else:
            return 1

    def read_from_file(self, file_path):
        """
        Replace the combinations with those of a combinations JSON file.

        Parameters
        ----------
        file_path : str
            Path of a file ``{"combinations": [{"combination": {...}}, ...]}``, e.g.
            ``settings/default_blender_combinations.json``.

        Raises
        ------
        KeyError
            If "combinations" or a key required by BlenderCombination.read_from_json is missing.
        """
        self.combinations = []
        dat = open(file_path, "r")
        json_data = json.load(dat)
        combinations_data = json_data["combinations"]
        for combination_data in combinations_data:
            combination = BlenderCombination()
            combination.read_from_json(combination_data)
            self.combinations.append(combination)
        dat.close()

    def save_to_file(self, file_path):
        """
        Write all combinations as one JSON file ``{"combinations": [<to_json() of each>]}`` (indent 4).

        Parameters
        ----------
        file_path : str
            Output path; an existing file is overwritten.
        """
        json_data = {"combinations": []}
        for combination in self.combinations:
            json_data["combinations"].append(combination.to_json())
        dat = open(file_path, "w")
        dat.write(json.dumps(json_data, indent=4))
        dat.close()

    def combination_in_combinations(self, input_combination: BlenderCombination):
        """
        Return the name of the first stored combination equal to ``input_combination``.

        Equality is that of compare_2_combinations (visualization, blend mode and opacity per layer; limits are
        not compared).

        Parameters
        ----------
        input_combination : BlenderCombination
            Combination to look for.

        Returns
        -------
        str or None
            Name of the first equal combination, or None.
        """
        for combination in self.combinations:
            if compare_2_combinations(input_combination, combination):
                return combination.name
        return None

    def combinations_names(self):
        """
        Return the names of all combinations.

        Returns
        -------
        list of str
            Names in list order (duplicates kept).
        """
        names_list = []
        for combination in self.combinations:
            names_list.append(combination.name)
        return names_list


class TerrainSettings:
    """
    A terrain preset (e.g. "general", "flat", "steep"): visualization parameters and normalization stretches.

    Intended for a graphical interface that adapts DefaultValues and a combination to a terrain type. Every attribute
    is None until read; None means "keep the value of DefaultValues / of the layer" in apply_terrain.

    Attributes
    ----------
    name : str or None
        Name of the preset.
    <visualization parameters> : int, float, str, tuple or None
        slp_output_units, hs_sun_azi, hs_sun_el, mhs_nr_dir, mhs_sun_el, slrm_rad_cell, svf_n_dir, svf_r_max,
        svf_noise, asvf_dir, asvf_level, sim_sky_mod, sim_compute_shadow, sim_nr_dir, sim_shadow_dist,
        sim_shadow_az, sim_shadow_el, ld_min_rad, ld_max_rad, ld_rad_inc, ld_anglr_res, ld_observer_h,
        msrm_feature_min, msrm_feature_max, msrm_scaling_factor, mstp_local_scale, mstp_meso_scale,
        mstp_broad_scale, mstp_lightness: names, units and meaning of the DefaultValues attributes of the same
        name (radii in px, MSRM feature sizes in m, angles in degrees); the MSTP scales are (min, max, step)
        tuples of int.
    <stretches> : tuple of float or None
        hs_stretch, mhs_stretch, slp_stretch, slrm_stretch, svf_stretch, asvf_stretch, pos_opns_stretch,
        neg_opns_stretch, sim_stretch, ld_stretch, msrm_stretch, mstp_stretch: ``(min, max)`` written into the
        normalization limits of the layers of the matching visualization.
    """

    def __init__(self):
        self.name = None
        # slope gradient
        self.slp_output_units = None
        # hillshade
        self.hs_sun_azi = None
        self.hs_sun_el = None
        # multi hillshade
        self.mhs_nr_dir = None
        self.mhs_sun_el = None
        # simple local relief model
        self.slrm_rad_cell = None
        # sky view factor
        self.svf_n_dir = None
        self.svf_r_max = None
        self.svf_noise = None
        # anisotropic sky-view factor
        self.asvf_dir = None
        self.asvf_level = None
        # positive openness
        # negative openness
        # sky_illumination
        self.sim_sky_mod = None
        self.sim_compute_shadow = None
        self.sim_nr_dir = None
        self.sim_shadow_dist = None
        self.sim_shadow_az = None
        self.sim_shadow_el = None
        # local dominance
        self.ld_min_rad = None
        self.ld_max_rad = None
        self.ld_rad_inc = None
        self.ld_anglr_res = None
        self.ld_observer_h = None
        # multi-scale relief model
        self.msrm_feature_min = None
        self.msrm_feature_max = None
        self.msrm_scaling_factor = None
        # multi-scale topographic position
        self.mstp_local_scale = None
        self.mstp_meso_scale = None
        self.mstp_broad_scale = None
        self.mstp_lightness = None

        # linear histogram stretches tuple(min, max)
        self.hs_stretch = None
        self.mhs_stretch = None
        self.slp_stretch = None
        self.slrm_stretch = None
        self.svf_stretch = None
        self.asvf_stretch = None
        self.pos_opns_stretch = None
        self.neg_opns_stretch = None
        self.sim_stretch = None
        self.ld_stretch = None
        self.msrm_stretch = None
        self.mstp_stretch = None

    def read_from_file(self, file_path):
        """
        Read one terrain preset from a JSON file (see read_from_json).

        Parameters
        ----------
        file_path : str
            Path of a file ``{"terrain_settings": {...}}``.

        Raises
        ------
        KeyError
            If "terrain_settings" or "name" is missing.
        """
        dat = open(file_path, "r")
        json_data = json.load(dat)
        self.read_from_json(json_data)
        dat.close()

    def read_from_json(self, json_data):
        """
        Reset all attributes and fill them from a parsed terrain preset.

        The document is ``{"terrain_settings": {"name": str, "<visualization name>": {"<parameter>": {"value": v},
        "stretch": {"min": a, "max": b}}, ...}}`` with the visualization names "Slope gradient", "Hillshade",
        "Multiple directions hillshade", "Simple local relief model", "Sky-View Factor", "Anisotropic Sky-View
        Factor", "Openness - Positive", "Openness - Negative", "Sky illumination", "Local dominance", "Multi-scale
        relief model" and "Multi-scale topographic position" (exact spelling). MSTP scales are given as
        ``{"min", "max", "step"}``. A missing key leaves the attribute None.

        Parameters
        ----------
        json_data : dict
            Parsed terrain preset.

        Raises
        ------
        KeyError
            If "terrain_settings" or "name" is missing.
        ValueError
            If a value cannot be converted by ``int()`` / ``float()`` (TypeError for a JSON null). ``int()`` of a
            JSON float truncates it.

        Notes
        -----
        Known issue: the "Sky illumination" and "Local dominance" stretches are assigned to
        ``neg_opns_stretch`` (copy-paste), so ``neg_opns_stretch`` ends up with the last of the three sections
        present and ``sim_stretch`` / ``ld_stretch`` stay None. With the shipped
        ``settings/default_terrains_settings.json`` the negative openness layer gets the local dominance limits.
        See docs/KNOWN_ISSUES.md.
        """
        # Structure of this function:
        #   1. reset every attribute to None (self.__init__());
        #   2. read the preset name;
        #   3. read each visualization parameter in its own try / except KeyError (missing keys stay None),
        #      converted with str(), int() or float() as in DefaultValues; MSTP scales as (min, max, step);
        #   4. read the (min, max) stretch of each visualization in the same way.
        self.__init__()
        terrain_data = json_data["terrain_settings"]
        self.name = terrain_data["name"]
        try:
            self.slp_output_units = str(terrain_data["Slope gradient"]["slp_output_units"]["value"])
        except KeyError:
            pass
        try:
            self.hs_sun_azi = int(terrain_data["Hillshade"]["hs_sun_azi"]["value"])
        except KeyError:
            pass
        try:
            self.hs_sun_el = int(terrain_data["Hillshade"]["hs_sun_el"]["value"])
        except KeyError:
            pass
        try:
            self.mhs_nr_dir = int(terrain_data["Multiple directions hillshade"]["mhs_nr_dir"]["value"])
        except KeyError:
            pass
        try:
            self.mhs_sun_el = int(terrain_data["Multiple directions hillshade"]["mhs_sun_el"]["value"])
        except KeyError:
            pass
        try:
            self.slrm_rad_cell = int(terrain_data["Simple local relief model"]["slrm_rad_cell"]["value"])
        except KeyError:
            pass
        try:
            self.svf_n_dir = int(terrain_data["Sky-View Factor"]["svf_n_dir"]["value"])
        except KeyError:
            pass
        try:
            self.svf_r_max = int(terrain_data["Sky-View Factor"]["svf_r_max"]["value"])
        except KeyError:
            pass
        try:
            self.svf_noise = int(terrain_data["Sky-View Factor"]["svf_noise"]["value"])
        except KeyError:
            pass
        try:
            self.asvf_dir = int(terrain_data["Anisotropic Sky-View Factor"]["asvf_dir"]["value"])
        except KeyError:
            pass
        try:
            self.asvf_level = int(terrain_data["Anisotropic Sky-View Factor"]["asvf_level"]["value"])
        except KeyError:
            pass
        try:
            self.sim_sky_mod = str(terrain_data["Sky illumination"]["sim_sky_mod"]["value"])
        except KeyError:
            pass
        try:
            self.sim_compute_shadow = int(terrain_data["Sky illumination"]["sim_compute_shadow"]["value"])
        except KeyError:
            pass
        try:
            self.sim_nr_dir = int(terrain_data["Sky illumination"]["sim_nr_dir"]["value"])
        except KeyError:
            pass
        try:
            self.sim_shadow_dist = int(terrain_data["Sky illumination"]["sim_shadow_dist"]["value"])
        except KeyError:
            pass
        try:
            self.sim_shadow_az = int(terrain_data["Sky illumination"]["sim_shadow_az"]["value"])
        except KeyError:
            pass
        try:
            self.sim_shadow_el = int(terrain_data["Sky illumination"]["sim_shadow_el"]["value"])
        except KeyError:
            pass
        try:
            self.ld_min_rad = int(terrain_data["Local dominance"]["ld_min_rad"]["value"])
        except KeyError:
            pass
        try:
            self.ld_max_rad = int(terrain_data["Local dominance"]["ld_max_rad"]["value"])
        except KeyError:
            pass
        try:
            self.ld_rad_inc = int(terrain_data["Local dominance"]["ld_rad_inc"]["value"])
        except KeyError:
            pass
        try:
            self.ld_anglr_res = int(terrain_data["Local dominance"]["ld_anglr_res"]["value"])
        except KeyError:
            pass
        try:
            self.ld_observer_h = float(terrain_data["Local dominance"]["ld_observer_h"]["value"])
        except KeyError:
            pass
        try:
            self.msrm_feature_min = float(terrain_data["Multi-scale relief model"]["msrm_feature_min"]["value"])
        except KeyError:
            pass
        try:
            self.msrm_feature_max = float(terrain_data["Multi-scale relief model"]["msrm_feature_max"]["value"])
        except KeyError:
            pass
        try:
            self.msrm_scaling_factor = int(terrain_data["Multi-scale relief model"]["msrm_scaling_factor"]["value"])
        except KeyError:
            pass
        try:
            self.mstp_local_scale = (int(terrain_data["Multi-scale topographic position"]["mstp_local_scale"]["min"]),
                                     int(terrain_data["Multi-scale topographic position"]["mstp_local_scale"]["max"]),
                                     int(terrain_data["Multi-scale topographic position"]["mstp_local_scale"]["step"]))
        except KeyError:
            pass
        try:
            self.mstp_meso_scale = (int(terrain_data["Multi-scale topographic position"]["mstp_meso_scale"]["min"]),
                                    int(terrain_data["Multi-scale topographic position"]["mstp_meso_scale"]["max"]),
                                    int(terrain_data["Multi-scale topographic position"]["mstp_meso_scale"]["step"]))
        except KeyError:
            pass
        try:
            self.mstp_broad_scale = (int(terrain_data["Multi-scale topographic position"]["mstp_broad_scale"]["min"]),
                                     int(terrain_data["Multi-scale topographic position"]["mstp_broad_scale"]["max"]),
                                     int(terrain_data["Multi-scale topographic position"]["mstp_broad_scale"]["step"]))
        except KeyError:
            pass
        try:
            self.mstp_lightness = float(terrain_data["Multi-scale topographic position"]["mstp_lightness"]["value"])
        except KeyError:
            pass
        try:
            self.slp_stretch = (float(terrain_data["Slope gradient"]["stretch"]["min"]),
                                float(terrain_data["Slope gradient"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.hs_stretch = (float(terrain_data["Hillshade"]["stretch"]["min"]),
                               float(terrain_data["Hillshade"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.mhs_stretch = (float(terrain_data["Multiple directions hillshade"]["stretch"]["min"]),
                                float(terrain_data["Multiple directions hillshade"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.slrm_stretch = (float(terrain_data["Simple local relief model"]["stretch"]["min"]),
                                 float(terrain_data["Simple local relief model"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.svf_stretch = (float(terrain_data["Sky-View Factor"]["stretch"]["min"]),
                                float(terrain_data["Sky-View Factor"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.asvf_stretch = (float(terrain_data["Anisotropic Sky-View Factor"]["stretch"]["min"]),
                                 float(terrain_data["Anisotropic Sky-View Factor"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.pos_opns_stretch = (float(terrain_data["Openness - Positive"]["stretch"]["min"]),
                                     float(terrain_data["Openness - Positive"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.neg_opns_stretch = (float(terrain_data["Openness - Negative"]["stretch"]["min"]),
                                     float(terrain_data["Openness - Negative"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.neg_opns_stretch = (float(terrain_data["Sky illumination"]["stretch"]["min"]),
                                     float(terrain_data["Sky illumination"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.neg_opns_stretch = (float(terrain_data["Local dominance"]["stretch"]["min"]),
                                     float(terrain_data["Local dominance"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.msrm_stretch = (float(terrain_data["Multi-scale relief model"]["stretch"]["min"]),
                                 float(terrain_data["Multi-scale relief model"]["stretch"]["max"]))
        except KeyError:
            pass
        try:
            self.mstp_stretch = (float(terrain_data["Multi-scale topographic position"]["stretch"]["min"]),
                                 float(terrain_data["Multi-scale topographic position"]["stretch"]["max"]))
        except KeyError:
            pass

    def apply_terrain(self, default: rvt.default.DefaultValues, combination: BlenderCombination):
        """
        Copy the preset into a DefaultValues instance and into the limits of a combination, in place.

        Every visualization parameter that is not None overwrites the DefaultValues attribute of the same name.
        Every stretch that is not None overwrites ``min`` and ``max`` of each layer whose visualization name
        (case-insensitive) is the long name of that visualization, e.g. "sky-view factor" for ``svf_stretch``.

        Parameters
        ----------
        default : rvt.default.DefaultValues
            Parameters to overwrite (modified in place).
        combination : BlenderCombination
            Combination whose layer limits are overwritten (modified in place). Every layer must have a ``vis``.

        Notes
        -----
        Because of the read_from_json issue (docs/KNOWN_ISSUES.md), sky illumination and local dominance layers
        keep their limits, and negative openness layers get the limits of the last of the "Openness - Negative",
        "Sky illumination" and "Local dominance" sections of the preset.
        """
        if self.slp_output_units is not None:
            default.slp_output_units = self.slp_output_units
        if self.hs_sun_azi is not None:
            default.hs_sun_azi = self.hs_sun_azi
        if self.hs_sun_el is not None:
            default.hs_sun_el = self.hs_sun_el
        if self.mhs_nr_dir is not None:
            default.mhs_nr_dir = self.mhs_nr_dir
        if self.mhs_sun_el is not None:
            default.mhs_sun_el = self.mhs_sun_el
        if self.slrm_rad_cell is not None:
            default.slrm_rad_cell = self.slrm_rad_cell
        if self.svf_n_dir is not None:
            default.svf_n_dir = self.svf_n_dir
        if self.svf_r_max is not None:
            default.svf_r_max = self.svf_r_max
        if self.svf_noise is not None:
            default.svf_noise = self.svf_noise
        if self.asvf_dir is not None:
            default.asvf_dir = self.asvf_dir
        if self.asvf_level is not None:
            default.asvf_level = self.asvf_level
        if self.sim_sky_mod is not None:
            default.sim_sky_mod = self.sim_sky_mod
        if self.sim_compute_shadow is not None:
            default.sim_compute_shadow = self.sim_compute_shadow
        if self.sim_nr_dir is not None:
            default.sim_nr_dir = self.sim_nr_dir
        if self.sim_shadow_dist is not None:
            default.sim_shadow_dist = self.sim_shadow_dist
        if self.sim_shadow_az is not None:
            default.sim_shadow_az = self.sim_shadow_az
        if self.sim_shadow_el is not None:
            default.sim_shadow_el = self.sim_shadow_el
        if self.ld_min_rad is not None:
            default.ld_min_rad = self.ld_min_rad
        if self.ld_max_rad is not None:
            default.ld_max_rad = self.ld_max_rad
        if self.ld_rad_inc is not None:
            default.ld_rad_inc = self.ld_rad_inc
        if self.ld_anglr_res is not None:
            default.ld_anglr_res = self.ld_anglr_res
        if self.ld_observer_h is not None:
            default.ld_observer_h = self.ld_observer_h
        if self.msrm_feature_min is not None:
            default.msrm_feature_min = self.msrm_feature_min
        if self.msrm_feature_max is not None:
            default.msrm_feature_max = self.msrm_feature_max
        if self.msrm_scaling_factor is not None:
            default.msrm_scaling_factor = self.msrm_scaling_factor
        if self.mstp_local_scale is not None:
            default.mstp_local_scale = self.mstp_local_scale
        if self.mstp_meso_scale is not None:
            default.mstp_meso_scale = self.mstp_meso_scale
        if self.mstp_broad_scale is not None:
            default.mstp_broad_scale = self.mstp_broad_scale
        if self.mstp_lightness is not None:
            default.mstp_lightness = self.mstp_lightness

        # linear histogram stretches, combination values overwrite
        for layer in combination.layers:
            if layer.vis.lower() == "slope gradient" and self.slp_stretch is not None:
                layer.min = self.slp_stretch[0]
                layer.max = self.slp_stretch[1]
            elif layer.vis.lower() == "hillshade" and self.hs_stretch is not None:
                layer.min = self.hs_stretch[0]
                layer.max = self.hs_stretch[1]
            elif layer.vis.lower() == "multiple directions hillshade" and self.mhs_stretch is not None:
                layer.min = self.mhs_stretch[0]
                layer.max = self.mhs_stretch[1]
            elif layer.vis.lower() == "simple local relief model" and self.slrm_stretch is not None:
                layer.min = self.slrm_stretch[0]
                layer.max = self.slrm_stretch[1]
            elif layer.vis.lower() == "sky-view factor" and self.svf_stretch is not None:
                layer.min = self.svf_stretch[0]
                layer.max = self.svf_stretch[1]
            elif layer.vis.lower() == "anisotropic sky-view factor" and self.asvf_stretch is not None:
                layer.min = self.asvf_stretch[0]
                layer.max = self.asvf_stretch[1]
            elif layer.vis.lower() == "openness - positive" and self.pos_opns_stretch is not None:
                layer.min = self.pos_opns_stretch[0]
                layer.max = self.pos_opns_stretch[1]
            elif layer.vis.lower() == "openness - negative" and self.neg_opns_stretch is not None:
                layer.min = self.neg_opns_stretch[0]
                layer.max = self.neg_opns_stretch[1]
            elif layer.vis.lower() == "sky illumination" and self.sim_stretch is not None:
                layer.min = self.sim_stretch[0]
                layer.max = self.sim_stretch[1]
            elif layer.vis.lower() == "local dominance" and self.ld_stretch is not None:
                layer.min = self.ld_stretch[0]
                layer.max = self.ld_stretch[1]
            elif layer.vis.lower() == "multi-scale relief model" and self.msrm_stretch is not None:
                layer.min = self.msrm_stretch[0]
                layer.max = self.msrm_stretch[1]
            elif layer.vis.lower() == "multi-scale topographic position" and self.mstp_stretch is not None:
                layer.min = self.mstp_stretch[0]
                layer.max = self.mstp_stretch[1]


class TerrainsSettings:
    """
    A list of terrain presets, e.g. those of ``settings/default_terrains_settings.json``.

    Attributes
    ----------
    terrains_settings : list of TerrainSettings
        The presets, in file order.
    """

    def __init__(self):
        self.terrains_settings = []

    def read_from_file(self, file_path):
        """
        Append the terrain presets of a JSON file.

        Parameters
        ----------
        file_path : str
            Path of a file ``{"terrains_settings": [{"terrain_settings": {...}}, ...]}``.

        Raises
        ------
        KeyError
            If "terrains_settings" or a key required by TerrainSettings.read_from_json is missing.

        Notes
        -----
        Known issue: unlike BlenderCombinations.read_from_file, the list is not emptied first, so reading a file
        twice holds every preset twice. See docs/KNOWN_ISSUES.md.
        """
        dat = open(file_path, "r")
        json_data = json.load(dat)
        terrains_settings_json = json_data["terrains_settings"]
        for terrain_json in terrains_settings_json:
            terrain_settings = TerrainSettings()
            terrain_settings.read_from_json(terrain_json)
            self.terrains_settings.append(terrain_settings)
        dat.close()

    def select_terrain_settings_by_name(self, name):
        """
        Return the first terrain preset with the given name.

        Parameters
        ----------
        name : str
            Name to look for (exact, case-sensitive), e.g. "general", "flat", "steep".

        Returns
        -------
        TerrainSettings or None
            The first match, or None if there is none.
        """
        for terrain_setting in self.terrains_settings:
            if terrain_setting.name == name:
                return terrain_setting


# Advance blending combinations
def color_relief_image_map(dem, resolution, default: rvt.default.DefaultValues = rvt.default.DefaultValues(),
                           colormap="OrRd", min_colormap_cut=0, max_colormap_cut=1, no_data=None):
    """
    Compute the Color Relief Image Map (CRIM): openness difference blended onto a colored slope.

    Recipe as implemented (layers top first; the bottom layer is the start image):

    ==  ===========================================  =====================  ==========  =======
    #   Image                                        Normalization          Blend mode  Opacity
    ==  ===========================================  =====================  ==========  =======
    1   positive - negative openness (degrees)       value -28..28          overlay     50 %
    2   positive - negative openness (same array)    value -28..28          luminosity  50 %
    3   slope in radians, colored with ``colormap``  value 0..0.8 rad       (bottom)    --
    ==  ===========================================  =====================  ==========  =======

    0.8 rad is 45.8°. Openness is computed with DefaultValues.get_sky_view_factor (``default.svf_n_dir``,
    ``default.svf_noise``, ``default.ve_factor``) with ``svf_r_max`` forced to 10 px; negative openness is the
    positive openness of ``-dem``. The slope comes from rvt.vis.slope_aspect (radians, no ``ve_factor``) and is
    not inverted.

    Parameters
    ----------
    dem : numpy.ndarray
        2D DEM ``(H, W)`` in m, float. Pixels equal to ``no_data`` are set to NaN **in this array**.
    resolution : float
        Pixel size in m/px (x and y).
    default : rvt.default.DefaultValues
        Visualization parameters. **Its svf_r_max is set to 10 in place.** The default is one instance created at
        import time and shared by all calls.
    colormap : str
        matplotlib colormap name for the slope layer (default "OrRd").
    min_colormap_cut : float or None
        Lower end of the used part of the colormap, in [0, 1] (default 0).
    max_colormap_cut : float or None
        Upper end of the used part of the colormap, in [0, 1] (default 1).
    no_data : float or None
        No-data value of ``dem``; None means the DEM has no such value (NaN is no-data anyway).

    Returns
    -------
    numpy.ndarray
        float64 ``(3, H, W)`` RGB image, values in [0, 1], NaN where ``dem`` is NaN.

    Notes
    -----
    Known issues (docs/KNOWN_ISSUES.md):

    - ``default.svf_r_max = 10`` changes the caller's DefaultValues object (and the shared default instance), so
      a later use of the same object computes SVF / openness with r = 10 px;
    - the opacity of the overlay layer has no visible effect (the overlay result replaces the background before
      the opacity mix);
    - with a gray colormap (R = G = B) the luminosity layer can give NaN at pixels where the DEM is finite
      (rvt.blend_func.clip_color divides 0 by 0).
    """
    slope_norm = ("value", 0, 0.8)  # radians: 0..0.8 rad = 0..45.8 deg
    op_on_norm = ("value", -28, 28)
    default = copy.deepcopy(default)  # the caller's object (and the shared default instance) stays unchanged
    default.svf_r_max = 10

    if no_data is not None:
        dem[dem == no_data] = np.nan

    opns_pos_arr = default.get_sky_view_factor(dem_arr=dem, resolution=resolution,
                                               compute_svf=False, compute_asvf=False, compute_opns=True,
                                               no_data=None)["opns"]
    opns_neg_arr = default.get_sky_view_factor(dem_arr=-1 * dem, resolution=resolution,
                                               compute_svf=False, compute_asvf=False, compute_opns=True,
                                               no_data=None)["opns"]
    opns_pos_neg_arr = opns_pos_arr - opns_neg_arr

    slope_arr = rvt.vis.slope_aspect(
        dem=dem,
        resolution_x=resolution,
        resolution_y=resolution,
        output_units="radian"
    )["slope"]

    blend_combination = rvt.blend.BlenderCombination()
    blend_combination.create_layer(vis_method="Openness_Pos-Neg", normalization=op_on_norm[0], minimum=op_on_norm[1],
                                   maximum=op_on_norm[2], blend_mode="overlay", opacity=50,
                                   image=opns_pos_neg_arr)
    blend_combination.create_layer(vis_method="Openness_Pos-Neg", normalization=op_on_norm[0], minimum=op_on_norm[1],
                                   maximum=op_on_norm[2], blend_mode="luminosity", opacity=50,
                                   image=opns_pos_neg_arr)
    blend_combination.create_layer(vis_method="slope gradient rad", normalization=slope_norm[0], minimum=slope_norm[1],
                                   maximum=slope_norm[2], blend_mode="normal", opacity=100, colormap=colormap,
                                   min_colormap_cut=min_colormap_cut, max_colormap_cut=max_colormap_cut,
                                   image=slope_arr)
    crim_out = blend_combination.render_all_images()

    return crim_out


def e3mstp(dem, resolution, default: rvt.default.DefaultValues = rvt.default.DefaultValues(), no_data=None):
    """
    Compute the enhanced multi-scale topographic position visualization, version 3 (e3MSTP).

    Recipe as implemented (layers top first; the bottom layer is the start image):

    ==  =================================================  ===============  ==========  =======
    #   Image                                              Normalization    Blend mode  Opacity
    ==  =================================================  ===============  ==========  =======
    1   SLRM (DefaultValues.get_slrm), m                   value -0.5..0.5  screen      25 %
    2   CRIM, color_relief_image_map with "OrRd", 0..1     value 0..1       soft_light  70 %
    3   MSTP (DefaultValues.get_mstp), RGB                 value 0..1       (bottom)    --
    ==  =================================================  ===============  ==========  =======

    The SLRM uses ``default.slrm_rad_cell`` (px) and ``default.ve_factor``; the CRIM is computed with ``default``
    (see color_relief_image_map); the MSTP uses ``default.mstp_local_scale``, ``mstp_meso_scale``,
    ``mstp_broad_scale`` (px) and ``mstp_lightness``.

    Parameters
    ----------
    dem : numpy.ndarray
        2D DEM ``(H, W)`` in m, float. Pixels equal to ``no_data`` are set to NaN **in this array**.
    resolution : float
        Pixel size in m/px (x and y).
    default : rvt.default.DefaultValues
        Visualization parameters. **Its svf_r_max is set to 10 in place** (through color_relief_image_map). The
        default is one instance created at import time and shared by all calls.
    no_data : float or None
        No-data value of ``dem``; None means the DEM has no such value (NaN is no-data anyway).

    Returns
    -------
    numpy.ndarray
        float64 ``(3, H, W)`` RGB image, values in [0, 1], NaN where ``dem`` is NaN (also where it was
        ``no_data``).

    Notes
    -----
    Discrepancy: the earlier docstring described layer 2 as a CRIM colored with "Reds_r(0.5-1)"; the code uses
    "OrRd" with cuts 0..1 (this docstring describes the code). See docs/KNOWN_ISSUES.md.

    Known issues (docs/KNOWN_ISSUES.md):

    - the opacity of the soft_light layer has no visible effect (70 % and 65 % give results that differ by
      less than 1e-6): the soft light result replaces the background before the opacity mix;
    - ``default.ve_factor`` is not applied to the MSTP layer nor to the slope inside the CRIM;
    - ``default.svf_r_max`` is set to 10 on the caller's object (see color_relief_image_map).
    """
    if no_data is not None:
        dem[dem == no_data] = np.nan
    slrm_arr = default.get_slrm(dem_arr=dem)
    crim_red_arr = color_relief_image_map(dem=dem, resolution=resolution, default=default,
                                          colormap="OrRd", min_colormap_cut=0, max_colormap_cut=1)
    mstp_arr = default.get_mstp(dem_arr=dem)

    blend_combination = rvt.blend.BlenderCombination()
    blend_combination.create_layer(vis_method="slrm", normalization="value",
                                   minimum=-0.5,
                                   maximum=0.5, blend_mode="screen", opacity=25,
                                   image=slrm_arr)
    blend_combination.create_layer(vis_method="crim_red", normalization="value",
                                   minimum=0,
                                   maximum=1, blend_mode="soft_light", opacity=70,
                                   image=crim_red_arr)
    blend_combination.create_layer(vis_method="mstp",
                                   normalization="value", minimum=0, maximum=1,
                                   blend_mode="normal", opacity=100,
                                   image=mstp_arr)
    e3mstp_out = blend_combination.render_all_images()
    return e3mstp_out


def e4mstp(dem, resolution, default: rvt.default.DefaultValues=rvt.default.DefaultValues(), no_data=None):
    """
    Compute the enhanced multi-scale topographic position visualization, version 4 (e4MSTP).

    Recipe as implemented (layers top first; the bottom layer is the start image):

    ==  ===========================================  ===============  ==========  =======
    #   Image                                        Normalization    Blend mode  Opacity
    ==  ===========================================  ===============  ==========  =======
    1   MSTP (DefaultValues.get_mstp), RGB           value 0..1       overlay     90 %
    2   combined SVF (below), float32                value -0.5..0.5  multiply    25 %
    3   combined openness / LD (below), float32      value 0..1       multiply    100 %
    4   slope in radians, inverted, "Reds_r" 0..1    value 0..55      (bottom)    --
    ==  ===========================================  ===============  ==========  =======

    Combined SVF, a two-layer combination rendered first: SVF with ``default`` (value 0.7..1, normal 50 %) over
    the "flat terrain" SVF (value 0.9..1, bottom). The flat terrain SVF uses a new DefaultValues with
    ``svf_r_max = int(10 / resolution)`` px (10 m, e.g. 20 px at 0.5 m/px) and ``svf_noise = 3``; it cannot be
    set by the caller and ignores ``default``.

    Combined openness / LD: positive minus negative openness (value -15..15, normal 50 %) over local dominance
    (value 0.5..1.8, bottom), both with ``default``.

    Layer 2 maps the combined SVF (values in [0, 1]) with the limits -0.5..0.5: values are clipped to 0.5 and
    shifted by 0.5, so [0, 0.5] becomes [0.5, 1] and every value >= 0.5 becomes 1. Layer 4 is inverted because its
    name is "Slope gradient" (rvt.blend_func.normalize_image).

    Parameters
    ----------
    dem : numpy.ndarray
        2D DEM ``(H, W)`` in m, float. Pixels equal to ``no_data`` are set to NaN **in this array**.
    resolution : float
        Pixel size in m/px (x and y).
    default : rvt.default.DefaultValues
        Visualization parameters for MSTP, SVF, openness and local dominance; not modified. The default is one
        instance created at import time and shared by all calls.
    no_data : float or None
        No-data value of ``dem``; None means the DEM has no such value (NaN is no-data anyway).

    Returns
    -------
    numpy.ndarray
        float64 ``(3, H, W)`` RGB image, values in [0, 1]. Where ``dem`` is NaN (after the ``no_data``
        replacement) the output is NaN if ``no_data`` is None, else ``no_data`` in all three bands.

    Notes
    -----
    Known issues (docs/KNOWN_ISSUES.md):

    - the slope comes from rvt.vis.slope_aspect with its default unit, radians (about 0..1.5), but is normalized
      over 0..55 as if in degrees, so the inverted slope layer stays within about 0.97..1;
    - the opacity of the MSTP overlay layer has no visible effect (the overlay result replaces the background
      before the opacity mix);
    - ``default.ve_factor`` is not applied to the MSTP layer, the slope layer and the flat terrain SVF.
    """

    # Nodata check
    if no_data is not None:
        dem[dem == no_data] = np.nan

    # Calculate intermediate visualisations:
    # ------------------------------------------------------------------------------------------------------------------
    ld_arr = default.get_local_dominance(dem).squeeze()
    svf_temp = default.get_sky_view_factor(dem, resolution, compute_svf=True, compute_opns=True)
    opns_arr = svf_temp["opns"].squeeze()
    svf_arr = svf_temp["svf"].squeeze()
    neg_opns_arr = default.get_neg_opns(dem, resolution).squeeze()

    # SVF for flat terrain (settings hardcoded, instead of importing second defaults.json)
    default_2 = rvt.default.DefaultValues()
    default_2.svf_r_max = int(10 / resolution)  # i.e. 10 meters for flat terrain
    default_2.svf_noise = 3
    svf_flat = default_2.get_sky_view_factor(dem, resolution, compute_svf=True, compute_opns=False)["svf"].squeeze()

    # 1) MSTP
    mstp_arr = default.get_mstp(dem_arr=dem)

    # 2) Comb svf
    comb_svf = rvt.blend.BlenderCombination()
    comb_svf.create_layer(
        vis_method="Sky-view factor",
        normalization="Value", minimum=0.7, maximum=1,
        blend_mode="normal", opacity=50,
        image=svf_arr
    )
    comb_svf.create_layer(
        vis_method="Sky-view factor",
        normalization="Value", minimum=0.9, maximum=1,
        blend_mode="normal", opacity=100,
        image=svf_flat
    )
    cs_svf = comb_svf.render_all_images(
        save_visualizations=False,
        save_render_path=None,
        no_data=np.nan
    )
    comb_svf_arr = cs_svf.astype("float32")

    # 3) Comb openness LD
    comb_ol = rvt.blend.BlenderCombination()
    comb_ol.create_layer(
        vis_method="Openness difference",
        normalization="Value", minimum=-15, maximum=15,
        blend_mode="normal", opacity=50,
        image=(opns_arr - neg_opns_arr)
    )
    comb_ol.create_layer(
        vis_method="Local dominance",
        normalization="Value", minimum=0.5, maximum=1.8,
        blend_mode="normal", opacity=100,
        image=ld_arr
    )
    comb_opns_ld_arr = comb_ol.render_all_images(
        save_visualizations=False,
        save_render_path=None,
        no_data=np.nan
    )
    comb_opns_ld_arr = comb_opns_ld_arr.astype("float32")

    # 4) Coloured slope
    slope_arr = rvt.vis.slope_aspect(dem=dem, resolution_x=resolution, resolution_y=resolution,
                                     output_units="degree")["slope"]
    # ------------------------------------------------------------------------------------------------------------------

    # Prepare layers for blending
    blend_combination = rvt.blend.BlenderCombination()
    blend_combination.create_layer(
        vis_method="mstp",
        normalization="value", minimum=0, maximum=1,
        blend_mode="overlay", opacity=90,
        image=mstp_arr
    )
    blend_combination.create_layer(
        vis_method="Comb svf",
        normalization="value", minimum=-0.5, maximum=0.5,
        blend_mode="multiply", opacity=25,
        image=comb_svf_arr
    )
    blend_combination.create_layer(
        vis_method="Comb openness LD",
        normalization="value", minimum=0, maximum=1,
        blend_mode="multiply", opacity=100,
        image=comb_opns_ld_arr
    )
    blend_combination.create_layer(
        vis_method="Slope gradient",
        normalization="value", minimum=0, maximum=55,
        blend_mode="normal", opacity=100,
        colormap="Reds_r", min_colormap_cut=0, max_colormap_cut=1,
        image=slope_arr
    )

    # Run blend
    e4mstp_out = blend_combination.render_all_images(
        save_visualizations=False,
        save_render_path=None,
        no_data=np.nan
    )

    # Postprocessing
    if no_data is not None:
        e4mstp_out[:, np.isnan(dem)] = no_data

    return e4mstp_out
