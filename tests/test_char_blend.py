"""
Characterization tests for rvt.blend: the classes, functions and methods that tests/test_blend.py does not assert.

tests/test_blend.py compares the layer loop of BlenderCombination.render_all_images with the code of tag
baseline-fixed. This file adds:

- BlenderLayer (attributes, every branch of check_data), BlenderCombination (layer handling, JSON read / save /
  round trip, layers_info, compare_2_combinations, render_all_images input checks, rendering from a DEM, saving,
  create_log_file), BlenderCombinations, TerrainSettings, TerrainsSettings (on the settings files shipped in the
  repository), create_blender_file_example;
- color_relief_image_map, e3mstp and e4mstp: shape, dtype, range, NaN mask, no_data handling, side effects.

Expected values come from the JSON files themselves, from analytic results (multiply of 1 and a is a), from
properties, or from the visualization functions of rvt.default / rvt.vis combined by plain code in this file. Files
are written only under tmp_path. Behaviour that looks wrong is pinned with the comment
"# pins current behaviour, KNOWN_ISSUES candidate".
"""

import copy
import json
import os
import re
import warnings

import numpy as np
import pytest

import rvt.blend as rb
import rvt.blend_func as bf
import rvt.default
import rvt.vis
from conftest import REPO_ROOT, assert_bitwise, assert_same, synthetic_dem

SETTINGS = os.path.join(REPO_ROOT, "settings")

VIS_NAMES = ["Slope gradient", "Hillshade", "Shadow", "Multiple directions hillshade", "Simple local relief model",
             "Sky-view factor", "Anisotropic sky-view factor", "Openness - Positive", "Openness - Negative",
             "Sky illumination", "Local dominance", "Multi-scale relief model", "Multi-scale topographic position"]


def _write_tif(path, arr, res=0.5):
    """Write a float32 GeoTIFF (one band, north-up, `res` m/px) with GDAL, as rvt.default reads them."""
    from osgeo import gdal
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform((100.0, res, 0.0, 200.0, 0.0, -res))
    ds.GetRasterBand(1).WriteArray(np.asarray(arr, dtype=np.float32))
    ds.FlushCache()
    ds = None
    return str(path)


# ----------------------------------------------------------------------------------------------------------------------
# create_blender_file_example
# ----------------------------------------------------------------------------------------------------------------------

def test_create_blender_file_example_content(tmp_path):
    path = tmp_path / "example.json"
    assert rb.create_blender_file_example(str(path)) is None
    data = json.loads(path.read_text())
    assert list(data) == ["combination"]
    assert data["combination"]["name"] == "VAT - Archaeological"
    layers = data["combination"]["layers"]
    assert [lay["layer"] for lay in layers] == ["1", "2", "3", "4", "5"]
    assert [lay["visualization_method"] for lay in layers] == [
        "Sky-View Factor", "Openness - Positive", "Slope gradient", "Hillshade", "None"]
    assert [lay["blend_mode"] for lay in layers[:4]] == ["Multiply", "Overlay", "Luminosity", "Normal"]
    assert [lay["opacity"] for lay in layers[:4]] == [25, 50, 50, 100]
    assert [(lay["min"], lay["max"]) for lay in layers[:4]] == [(0.7, 1.0), (68, 93), (0, 50), (0, 1)]
    assert set(layers[4]) == {"layer", "visualization_method"}  # the "None" layer carries nothing else
    # the shipped example file holds the same data
    with open(os.path.join(SETTINGS, "blender_file_example.json")) as fh:
        assert json.load(fh) == data
    # the file is readable as a combination; the layer with visualization "None" is skipped
    comb = rb.BlenderCombination()
    comb.read_from_file(str(path))
    assert [lay.vis for lay in comb.layers] == ["Sky-View Factor", "Openness - Positive", "Slope gradient", "Hillshade"]


def test_create_blender_file_example_overwrites(tmp_path):
    path = tmp_path / "example.json"
    path.write_text("old content")
    rb.create_blender_file_example(str(path))
    assert json.loads(path.read_text())["combination"]["name"] == "VAT - Archaeological"
    # indent = 4
    assert path.read_text().startswith('{\n    "combination": {')


@pytest.mark.skipif(os.name != "nt", reason="the default path 'settings\\blender_file_example.json' uses a backslash")
def test_create_blender_file_example_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rb.create_blender_file_example()  # creates ./settings/ when it does not exist
    assert (tmp_path / "settings" / "blender_file_example.json").is_file()
    # pins current behaviour, KNOWN_ISSUES candidate: the default path is hard-coded with a backslash; on POSIX
    # os.path.dirname() of it is "" and os.makedirs("") raises FileNotFoundError (not testable on Windows).


# ----------------------------------------------------------------------------------------------------------------------
# BlenderLayer
# ----------------------------------------------------------------------------------------------------------------------

def test_blender_layer_defaults_and_attributes():
    layer = rb.BlenderLayer()
    assert (layer.vis, layer.normalization, layer.min, layer.max) == (None, "value", None, None)
    assert (layer.blend_mode, layer.opacity, layer.colormap) == ("normal", 100, None)
    assert (layer.min_colormap_cut, layer.max_colormap_cut, layer.image, layer.image_path) == (None, None, None, None)
    img = np.zeros((2, 2))
    layer = rb.BlenderLayer("Hillshade", "perc", 2, 3, "multiply", 40, "gray", 0.1, 0.9, img, "x.tif")
    assert vars(layer) == {
        "vis": "Hillshade", "normalization": "perc", "min": 2, "max": 3, "blend_mode": "multiply", "opacity": 40,
        "colormap": "gray", "min_colormap_cut": 0.1, "max_colormap_cut": 0.9, "image_path": "x.tif", "image": img}
    assert layer.image is img  # stored, not copied


def _valid_layer(**kwargs):
    args = dict(vis_method="Hillshade", normalization="value", minimum=0, maximum=1, blend_mode="normal", opacity=100)
    args.update(kwargs)
    return rb.BlenderLayer(**args)


@pytest.mark.parametrize("vis", VIS_NAMES + [name.upper() for name in VIS_NAMES])
def test_check_data_accepts_known_visualizations_without_image(vis):
    _valid_layer(vis_method=vis).check_data()


def test_check_data_unknown_visualization_needs_an_image_or_path():
    with pytest.raises(Exception, match="Incorrect vis"):
        _valid_layer(vis_method="mstp").check_data()
    _valid_layer(vis_method="mstp", image=np.zeros((2, 2))).check_data()
    _valid_layer(vis_method="mstp", image_path="some.tif").check_data()  # the path is not opened here


@pytest.mark.parametrize("mode", ["normal", "Multiply", "OVERLAY", "luminosity", "Screen", "soft_light"])
def test_check_data_blend_modes(mode):
    _valid_layer(blend_mode=mode).check_data()


def test_check_data_errors():
    with pytest.raises(Exception, match="normalization value incorrect"):
        _valid_layer(normalization="linear").check_data()
    with pytest.raises(Exception, match="min \\+ max has to smaller then 100"):
        _valid_layer(normalization="perc", minimum=60, maximum=40).check_data()
    with pytest.raises(Exception, match="min bigger than max"):
        _valid_layer(minimum=2, maximum=1).check_data()
    with pytest.raises(Exception, match="blend_mode incorrect"):
        _valid_layer(blend_mode="dissolve").check_data()
    with pytest.raises(Exception, match="colormap not defined"):
        _valid_layer(min_colormap_cut=0.2).check_data()
    with pytest.raises(Exception, match="colormap not defined"):
        _valid_layer(max_colormap_cut=0.8).check_data()
    _valid_layer(colormap="gray", min_colormap_cut=0.2, max_colormap_cut=0.8).check_data()
    # percent: min + max must stay below 100, 99.9 is accepted
    _valid_layer(normalization="perc", minimum=60, maximum=39.9).check_data()


def test_check_data_accepted_but_questionable_values():
    # pins current behaviour, KNOWN_ISSUES candidate: `0 > opacity > 100` can never be true, so opacity outside
    # 0..100 passes the check (and render_images extrapolates).
    _valid_layer(opacity=150).check_data()
    _valid_layer(opacity=-5).check_data()
    # pins current behaviour, KNOWN_ISSUES candidate: min == max is accepted for "value" (only min > max raises);
    # the division by zero then happens in normalize_lin.
    _valid_layer(minimum=1, maximum=1).check_data()
    # without limits the comparisons raise TypeError (None > None, None + None)
    with pytest.raises(TypeError):
        rb.BlenderLayer("Hillshade", "value").check_data()
    with pytest.raises(TypeError):
        rb.BlenderLayer("Hillshade", "perc").check_data()


def test_check_data_percent_is_renamed_to_perc():
    layer = _valid_layer(normalization="percent", minimum=2, maximum=3)
    layer.check_data()
    assert layer.normalization == "perc"
    layer = rb.BlenderLayer(None, "percent")  # the rename happens before the vis None reset, which then clears it
    layer.check_data()
    assert layer.normalization is None


def test_check_data_without_visualization_clears_everything():
    layer = rb.BlenderLayer(None, "perc", 1, 2, "overlay", 50, "gray", 0.1, 0.9, np.zeros((2, 2)), "x.tif")
    layer.check_data()
    assert vars(layer) == {"vis": None, "normalization": None, "min": None, "max": None, "blend_mode": None,
                           "opacity": None, "colormap": None, "min_colormap_cut": None, "max_colormap_cut": None,
                           "image_path": None, "image": None}


# ----------------------------------------------------------------------------------------------------------------------
# BlenderCombination: layers, info, JSON
# ----------------------------------------------------------------------------------------------------------------------

def test_combination_defaults_and_dem_setters():
    comb = rb.BlenderCombination()
    assert (comb.dem_arr, comb.dem_resolution, comb.dem_path, comb.name, comb.layers) == (None, None, None, "", [])
    dem = np.zeros((3, 3))
    comb2 = rb.BlenderCombination(dem_arr=dem, dem_resolution=0.5, dem_path="a.tif")
    assert comb2.dem_arr is dem and comb2.dem_resolution == 0.5 and comb2.dem_path == "a.tif"
    comb.add_dem_arr(dem, 2.0)
    assert comb.dem_arr is dem and comb.dem_resolution == 2.0
    comb.add_dem_path("b.tif")
    assert comb.dem_path == "b.tif"
    comb.add_dem_arr(None, None)  # changing is allowed
    assert comb.dem_arr is None


def test_combination_create_add_remove_layers():
    comb = rb.BlenderCombination()
    comb.create_layer(vis_method=None)  # no visualization: nothing is added
    assert comb.layers == []
    comb.create_layer("Hillshade", "perc", 2, 3, "overlay", 40, "gray", 0.1, 0.9, np.zeros((2, 2)), "x.tif")
    assert len(comb.layers) == 1
    layer = comb.layers[0]
    assert isinstance(layer, rb.BlenderLayer)
    assert (layer.vis, layer.normalization, layer.min, layer.max, layer.blend_mode, layer.opacity, layer.colormap,
            layer.min_colormap_cut, layer.max_colormap_cut, layer.image_path) == (
        "Hillshade", "perc", 2, 3, "overlay", 40, "gray", 0.1, 0.9, "x.tif")
    comb.create_layer("Slope gradient")  # defaults of create_layer
    assert (comb.layers[1].normalization, comb.layers[1].blend_mode, comb.layers[1].opacity) == ("value", "normal", 100)
    extra = rb.BlenderLayer("Shadow")
    comb.add_layer(extra)
    assert comb.layers[-1] is extra  # the instance itself is appended
    comb.add_layer(rb.BlenderLayer(None))  # empty layers are ignored
    assert len(comb.layers) == 3
    comb.remove_all_layers()
    assert comb.layers == []


def test_combination_layers_info():
    comb = rb.BlenderCombination()
    assert comb.layers_info() == []
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
    comb.add_layer(rb.BlenderLayer("Slope gradient", "perc", 2, 3, "multiply", 40, "gray", 0.1, 0.9))
    info = comb.layers_info()
    assert info == [
        "layer 1, visualization = Hillshade, normalization = value, min = 0, max = 1, blend_mode = normal, "
        "opacity = 100, colormap = None, min_colormap_cut = None, max_colormap_cut",
        # pins current behaviour, KNOWN_ISSUES candidate: the format string ends with "max_colormap_cut" and has no
        # "= {}" for it, so the value of max_colormap_cut (0.9 here) is never printed.
        "layer 2, visualization = Slope gradient, normalization = perc, min = 2, max = 3, blend_mode = multiply, "
        "opacity = 40, colormap = gray, min_colormap_cut = 0.1, max_colormap_cut"]


def _json_doc(layers, name="test"):
    return {"combination": {"name": name, "layers": layers}}


def test_combination_read_from_json_types_and_skips():
    doc = _json_doc([
        {"layer": "1", "visualization_method": "Hillshade", "norm": "Value", "min": "0.5", "max": 2, "blend_mode": "Overlay",
         "opacity": 50.9},
        {"layer": "2", "visualization_method": None},
        {"layer": "3", "visualization_method": "None"},
        {"layer": "4", "visualization_method": "NULL"},
        {"layer": "5", "visualization_method": "Slope gradient", "norm": "perc", "min": 1, "max": 2.5,
         "blend_mode": "normal", "opacity": 100, "colormap": "gray"},
    ])
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("old"))  # replaced by the read
    comb.read_from_json(doc)
    assert comb.name == "test"
    assert [lay.vis for lay in comb.layers] == ["Hillshade", "Slope gradient"]
    first, second = comb.layers
    assert (first.normalization, first.min, first.max, first.blend_mode, first.opacity) == ("Value", 0.5, 2.0, "Overlay", 50)
    assert isinstance(first.min, float) and isinstance(first.max, float) and isinstance(first.opacity, int)
    assert first.colormap is None  # no "colormap" key
    assert (second.colormap, second.min_colormap_cut, second.max_colormap_cut) == ("gray", None, None)
    assert first.image is None and first.image_path is None


def test_combination_read_from_json_colormap_variants():
    def read(extra):
        layer = {"layer": "1", "visualization_method": "Hillshade", "norm": "value", "min": 0, "max": 1,
                 "blend_mode": "normal", "opacity": 100}
        layer.update(extra)
        comb = rb.BlenderCombination()
        comb.read_from_json(_json_doc([layer]))
        return comb.layers[0]

    assert read({"colormap": "None"}).colormap is None
    assert read({"colormap": "null"}).colormap is None
    assert read({"colormap": "NULL"}).colormap is None
    lay = read({"colormap": "gray", "min_colormap_cut": "None", "max_colormap_cut": "null"})
    assert (lay.colormap, lay.min_colormap_cut, lay.max_colormap_cut) == ("gray", None, None)
    # the cuts are read only when a colormap is given
    lay = read({"min_colormap_cut": 0.2, "max_colormap_cut": 0.8})
    assert (lay.colormap, lay.min_colormap_cut, lay.max_colormap_cut) == (None, None, None)
    # pins current behaviour, KNOWN_ISSUES candidate: the cuts are converted with str(), so numbers read from JSON
    # become the strings "0.2" / "0.8"; gray_scale_to_color_ramp then fails on str > int comparisons.
    lay = read({"colormap": "gray", "min_colormap_cut": 0.2, "max_colormap_cut": 0.8})
    assert (lay.min_colormap_cut, lay.max_colormap_cut) == ("0.2", "0.8")
    with pytest.raises(TypeError):
        bf.gray_scale_to_color_ramp(np.zeros((2, 2)), lay.colormap, lay.min_colormap_cut, lay.max_colormap_cut)


def test_combination_read_from_json_missing_keys_raise():
    comb = rb.BlenderCombination()
    with pytest.raises(KeyError):
        comb.read_from_json({"layers": []})
    with pytest.raises(KeyError):
        comb.read_from_json({"combination": {"name": "x"}})
    with pytest.raises(KeyError):  # "norm" is required
        comb.read_from_json(_json_doc([{"layer": "1", "visualization_method": "Hillshade", "min": 0, "max": 1,
                                        "blend_mode": "normal", "opacity": 100}]))


def test_combination_to_json_variants():
    comb = rb.BlenderCombination()
    comb.name = "mine"
    comb.add_layer(rb.BlenderLayer("Hillshade", "Value", 0, 1, "Normal", 100))
    comb.add_layer(rb.BlenderLayer("Slope gradient", "perc", 2, 3, "multiply", 40, colormap="gray"))
    comb.add_layer(rb.BlenderLayer("Shadow", "value", 0, 1, "normal", 10, "gray", min_colormap_cut=0.1))
    comb.add_layer(rb.BlenderLayer("Sky-view factor", "value", 0.5, 1, "screen", 20, "gray", max_colormap_cut=0.9))
    comb.add_layer(rb.BlenderLayer("Local dominance", "value", 0.5, 1.8, "overlay", 30, "gray", 0.1, 0.9))
    base = {"norm": "value", "min": 0, "max": 1, "blend_mode": "normal", "opacity": 10}
    assert comb.to_json() == {"combination": {"name": "mine", "layers": [
        {"layer": "1", "visualization_method": "Hillshade", "norm": "Value", "min": 0, "max": 1, "blend_mode": "Normal",
         "opacity": 100},
        {"layer": "2", "visualization_method": "Slope gradient", "norm": "perc", "min": 2, "max": 3,
         "blend_mode": "multiply", "opacity": 40, "colormap": "gray"},
        dict(base, layer="3", visualization_method="Shadow", colormap="gray", min_colormap_cut=0.1),
        {"layer": "4", "visualization_method": "Sky-view factor", "norm": "value", "min": 0.5, "max": 1,
         "blend_mode": "screen", "opacity": 20, "colormap": "gray", "max_colormap_cut": 0.9},
        {"layer": "5", "visualization_method": "Local dominance", "norm": "value", "min": 0.5, "max": 1.8,
         "blend_mode": "overlay", "opacity": 30, "colormap": "gray", "min_colormap_cut": 0.1, "max_colormap_cut": 0.9},
    ]}}
    assert rb.BlenderCombination().to_json() == {"combination": {"name": "", "layers": []}}


def test_combination_file_round_trip(tmp_path):
    comb = rb.BlenderCombination()
    comb.name = "round trip"
    comb.add_layer(rb.BlenderLayer("Sky-View Factor", "Value", 0.7, 1.0, "Multiply", 25))
    comb.add_layer(rb.BlenderLayer("Openness - Positive", "Value", 68.0, 93.0, "Overlay", 50))
    comb.add_layer(rb.BlenderLayer("Slope gradient", "perc", 1.5, 2.5, "Luminosity", 50, colormap="gray"))
    path = tmp_path / "comb.json"
    comb.save_to_file(str(path))
    # the file holds the json structure, indent = 4
    assert json.loads(path.read_text()) == comb.to_json()
    assert path.read_text() == json.dumps(comb.to_json(), indent=4)
    other = rb.BlenderCombination()
    other.read_from_file(str(path))
    assert other.name == "round trip"
    assert [vars(lay) for lay in other.layers] == [vars(lay) for lay in comb.layers]
    assert rb.compare_2_combinations(comb, other)
    # a second save of the re-read combination is byte-identical
    path2 = tmp_path / "comb2.json"
    other.save_to_file(str(path2))
    assert path.read_text() == path2.read_text()


def test_combination_round_trip_turns_colormap_cuts_into_strings(tmp_path):
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1, "normal", 100, "gray", 0.2, 0.8))
    path = tmp_path / "cuts.json"
    comb.save_to_file(str(path))
    other = rb.BlenderCombination()
    other.read_from_file(str(path))
    # pins current behaviour, KNOWN_ISSUES candidate: see test_combination_read_from_json_colormap_variants
    assert (other.layers[0].min_colormap_cut, other.layers[0].max_colormap_cut) == ("0.2", "0.8")


@pytest.mark.parametrize("file_name", ["blender_VAT.json", "blender_custom_layers.json", "blender_file_example.json"])
def test_combination_read_shipped_files(file_name):
    path = os.path.join(SETTINGS, file_name)
    with open(path) as fh:
        raw = json.load(fh)["combination"]
    expected = [lay for lay in raw["layers"] if str(lay["visualization_method"]).lower() not in ("none", "null")]
    comb = rb.BlenderCombination()
    comb.read_from_file(path)
    assert comb.name == raw["name"]
    assert len(comb.layers) == len(expected) == 4
    for layer, lay in zip(comb.layers, expected):
        assert (layer.vis, layer.normalization, layer.min, layer.max, layer.blend_mode, layer.opacity) == (
            lay["visualization_method"], lay["norm"], float(lay["min"]), float(lay["max"]), lay["blend_mode"],
            lay["opacity"])
        layer.check_data()  # the shipped combinations are valid


def test_combination_check_data_checks_every_layer():
    comb = rb.BlenderCombination()
    comb.add_layer(_valid_layer(normalization="percent", minimum=1, maximum=2))
    comb.add_layer(_valid_layer(vis_method="Shadow"))
    comb.check_data()
    assert comb.layers[0].normalization == "perc"  # the layers are changed in place
    comb.add_layer(_valid_layer(blend_mode="dissolve"))
    with pytest.raises(Exception, match="blend_mode incorrect"):
        comb.check_data()


def _comb_with(*specs):
    comb = rb.BlenderCombination()
    for vis, mode, opacity in specs:
        comb.add_layer(rb.BlenderLayer(vis, "value", 0, 1, mode, opacity))
    return comb


def test_compare_2_combinations():
    c1 = _comb_with(("Hillshade", "normal", 100), ("Slope gradient", "multiply", 50))
    c2 = _comb_with(("HILLSHADE", "Normal", 100), ("slope gradient", "MULTIPLY", 50))
    assert rb.compare_2_combinations(c1, c2)  # vis and blend_mode are compared case-insensitively
    assert rb.compare_2_combinations(c1, c1)
    assert not rb.compare_2_combinations(c1, _comb_with(("Hillshade", "normal", 100)))  # other length
    assert not rb.compare_2_combinations(c1, _comb_with(("Hillshade", "normal", 100), ("Shadow", "multiply", 50)))
    assert not rb.compare_2_combinations(c1, _comb_with(("Hillshade", "normal", 100), ("Slope gradient", "screen", 50)))
    assert not rb.compare_2_combinations(c1, _comb_with(("Hillshade", "normal", 100), ("Slope gradient", "multiply", 49)))
    assert rb.compare_2_combinations(rb.BlenderCombination(), rb.BlenderCombination())  # both empty
    # pins current behaviour, KNOWN_ISSUES candidate: normalization, min and max are not compared (the checks are
    # commented out in the source); combinations that differ only in their limits are "equal".
    c3 = rb.BlenderCombination()
    c3.add_layer(rb.BlenderLayer("Hillshade", "perc", 5, 6, "normal", 100))
    c3.add_layer(rb.BlenderLayer("Slope gradient", "value", 10, 20, "multiply", 50))
    assert rb.compare_2_combinations(c1, c3)


# ----------------------------------------------------------------------------------------------------------------------
# BlenderCombinations
# ----------------------------------------------------------------------------------------------------------------------

def test_combinations_list_handling():
    combs = rb.BlenderCombinations()
    assert combs.combinations == [] and combs.combinations_names() == []
    c1 = _comb_with(("Hillshade", "normal", 100))
    c2 = _comb_with(("Slope gradient", "normal", 100))
    c3 = _comb_with(("Shadow", "normal", 100))
    combs.add_combination(c1, name="first")
    combs.add_combination(c2, "second")
    combs.add_combination(c3)  # no name: the combination keeps its own (empty) name
    assert c1.name == "first" and c2.name == "second" and c3.name == ""
    assert combs.combinations == [c1, c2, c3]
    assert combs.combinations_names() == ["first", "second", ""]
    assert combs.select_combination_by_name("second") is c2
    assert combs.select_combination_by_name("missing") is None
    dup = _comb_with(("Shadow", "multiply", 10))
    combs.add_combination(dup, "first")
    assert combs.select_combination_by_name("first") is c1  # first match wins
    # remove by name removes every match and returns 1 while combinations remain
    assert combs.remove_combination_by_name("first") == 1
    assert combs.combinations == [c2, c3]
    assert combs.remove_combination_by_name("nothing") == 1  # nothing removed, list not empty
    assert combs.remove_combination_by_name("second") == 1
    assert combs.remove_combination_by_name("") == 0  # the list is empty afterwards
    assert combs.remove_combination_by_name("anything") == 0  # empty list stays empty
    combs.add_combination(c1)
    combs.remove_all_combinations()
    assert combs.combinations == []


def test_combinations_combination_in_combinations():
    combs = rb.BlenderCombinations()
    combs.add_combination(_comb_with(("Hillshade", "normal", 100)), "plain")
    combs.add_combination(_comb_with(("Hillshade", "normal", 100)), "same as plain")
    combs.add_combination(_comb_with(("Slope gradient", "overlay", 20), ("Hillshade", "normal", 100)), "two layers")
    assert combs.combination_in_combinations(_comb_with(("hillshade", "NORMAL", 100))) == "plain"  # first match
    assert combs.combination_in_combinations(_comb_with(("Slope gradient", "overlay", 20), ("Hillshade", "normal", 100))) == \
        "two layers"
    assert combs.combination_in_combinations(_comb_with(("Shadow", "normal", 100))) is None
    assert rb.BlenderCombinations().combination_in_combinations(_comb_with(("Shadow", "normal", 100))) is None


def test_combinations_read_shipped_file_and_round_trip(tmp_path):
    path = os.path.join(SETTINGS, "default_blender_combinations.json")
    with open(path) as fh:
        raw = json.load(fh)["combinations"]
    combs = rb.BlenderCombinations()
    combs.add_combination(rb.BlenderCombination(), "to be replaced")
    combs.read_from_file(path)  # the list is replaced
    assert combs.combinations_names() == [c["combination"]["name"] for c in raw] == [
        "Archaeological (VAT)", "Archaeological combined (VAT combined)", "Prismatic openness", "City"]
    assert [len(c.layers) for c in combs.combinations] == [4, 0, 3, 3]
    city = combs.select_combination_by_name("City")
    assert [(lay.vis, lay.normalization, lay.min, lay.max, lay.blend_mode, lay.opacity) for lay in city.layers] == [
        ("Shadow", "Value", 0.0, 1.0, "Normal", 70), ("Sky-View Factor", "Value", 0.1, 1.0, "Normal", 10),
        ("Sky illumination", "Perc", 1.0, 0.0, "Normal", 100)]
    for comb in combs.combinations:
        comb.check_data()
    # save, read again: same combinations, and the file is the json structure with indent = 4
    out = tmp_path / "combinations.json"
    combs.save_to_file(str(out))
    assert json.loads(out.read_text()) == {"combinations": [c.to_json() for c in combs.combinations]}
    again = rb.BlenderCombinations()
    again.read_from_file(str(out))
    assert again.combinations_names() == combs.combinations_names()
    for c_a, c_b in zip(again.combinations, combs.combinations):
        assert [vars(lay) for lay in c_a.layers] == [vars(lay) for lay in c_b.layers]
    # the shipped file is reproduced value for value
    assert json.loads(out.read_text()) == {"combinations": raw}


# ----------------------------------------------------------------------------------------------------------------------
# TerrainSettings, TerrainsSettings
# ----------------------------------------------------------------------------------------------------------------------

PARAM_ATTRS = ["slp_output_units", "hs_sun_azi", "hs_sun_el", "mhs_nr_dir", "mhs_sun_el", "slrm_rad_cell", "svf_n_dir",
               "svf_r_max", "svf_noise", "asvf_dir", "asvf_level", "sim_sky_mod", "sim_compute_shadow", "sim_nr_dir",
               "sim_shadow_dist", "sim_shadow_az", "sim_shadow_el", "ld_min_rad", "ld_max_rad", "ld_rad_inc",
               "ld_anglr_res", "ld_observer_h", "msrm_feature_min", "msrm_feature_max", "msrm_scaling_factor",
               "mstp_local_scale", "mstp_meso_scale", "mstp_broad_scale", "mstp_lightness"]
STRETCH_ATTRS = ["hs_stretch", "mhs_stretch", "slp_stretch", "slrm_stretch", "svf_stretch", "asvf_stretch",
                 "pos_opns_stretch", "neg_opns_stretch", "sim_stretch", "ld_stretch", "msrm_stretch", "mstp_stretch"]
# stretch attribute -> visualization name used by apply_terrain
STRETCH_VIS = {"slp_stretch": "Slope gradient", "hs_stretch": "Hillshade", "mhs_stretch": "Multiple directions hillshade",
               "slrm_stretch": "Simple local relief model", "svf_stretch": "Sky-view factor",
               "asvf_stretch": "Anisotropic sky-view factor", "pos_opns_stretch": "Openness - Positive",
               "neg_opns_stretch": "Openness - Negative", "sim_stretch": "Sky illumination",
               "ld_stretch": "Local dominance", "msrm_stretch": "Multi-scale relief model",
               "mstp_stretch": "Multi-scale topographic position"}


def _terrain_doc():
    """A terrain_settings document with every key read_from_json knows, with distinct values; numbers given as
    floats / strings where the reader converts them."""
    def sc(lo, hi):
        return {"stretch": {"min": lo, "max": hi}}
    return {"terrain_settings": {
        "name": "full",
        "Slope gradient": dict(sc(0, 50), slp_output_units={"value": "degree"}),
        "Hillshade": dict(sc(0, 1), hs_sun_azi={"value": 123.9}, hs_sun_el={"value": "35"}),
        "Multiple directions hillshade": dict(sc(0.1, 0.9), mhs_nr_dir={"value": 8}, mhs_sun_el={"value": 41}),
        "Simple local relief model": dict(sc(-1, 1), slrm_rad_cell={"value": 20}),
        "Sky-View Factor": dict(sc(0.7, 1), svf_n_dir={"value": 12}, svf_r_max={"value": 11}, svf_noise={"value": 2}),
        "Anisotropic Sky-View Factor": dict(sc(0.65, 1), asvf_dir={"value": 90}, asvf_level={"value": 2}),
        "Openness - Positive": sc(68, 93),
        "Openness - Negative": sc(60, 95),
        "Sky illumination": dict(sc(0.25, 0.3), sim_sky_mod={"value": "overcast"}, sim_compute_shadow={"value": 1},
                                 sim_nr_dir={"value": 40}, sim_shadow_dist={"value": 100},
                                 sim_shadow_az={"value": 315}, sim_shadow_el={"value": 35}),
        "Local dominance": dict(sc(0.5, 1.8), ld_min_rad={"value": 10}, ld_max_rad={"value": 20},
                                ld_rad_inc={"value": 1}, ld_anglr_res={"value": 15}, ld_observer_h={"value": 1.7}),
        "Multi-scale relief model": dict(sc(-2, 2), msrm_feature_min={"value": 1.0}, msrm_feature_max={"value": 5.5},
                                         msrm_scaling_factor={"value": 3}),
        "Multi-scale topographic position": dict(
            sc(0, 1), mstp_local_scale={"min": 3, "max": 21, "step": 2},
            mstp_meso_scale={"min": 23, "max": 203, "step": 18}, mstp_broad_scale={"min": 205, "max": 2025, "step": 180},
            mstp_lightness={"value": 1.2}),
    }}


def test_terrain_settings_init_all_none():
    ts = rb.TerrainSettings()
    assert sorted(vars(ts)) == sorted(["name"] + PARAM_ATTRS + STRETCH_ATTRS)
    assert all(value is None for value in vars(ts).values())


def test_terrain_settings_read_from_json_full_document():
    ts = rb.TerrainSettings()
    ts.read_from_json(_terrain_doc())
    assert ts.name == "full"
    assert (ts.slp_output_units, ts.hs_sun_azi, ts.hs_sun_el) == ("degree", 123, 35)  # int(123.9) truncates
    assert (ts.mhs_nr_dir, ts.mhs_sun_el, ts.slrm_rad_cell) == (8, 41, 20)
    assert (ts.svf_n_dir, ts.svf_r_max, ts.svf_noise, ts.asvf_dir, ts.asvf_level) == (12, 11, 2, 90, 2)
    assert (ts.sim_sky_mod, ts.sim_compute_shadow, ts.sim_nr_dir, ts.sim_shadow_dist, ts.sim_shadow_az,
            ts.sim_shadow_el) == ("overcast", 1, 40, 100, 315, 35)
    assert (ts.ld_min_rad, ts.ld_max_rad, ts.ld_rad_inc, ts.ld_anglr_res, ts.ld_observer_h) == (10, 20, 1, 15, 1.7)
    assert (ts.msrm_feature_min, ts.msrm_feature_max, ts.msrm_scaling_factor) == (1.0, 5.5, 3)
    assert ts.mstp_local_scale == (3, 21, 2) and ts.mstp_meso_scale == (23, 203, 18)
    assert ts.mstp_broad_scale == (205, 2025, 180) and ts.mstp_lightness == 1.2
    assert isinstance(ts.hs_sun_azi, int) and isinstance(ts.ld_observer_h, float) and isinstance(ts.slp_output_units, str)
    assert ts.slp_stretch == (0.0, 50.0) and ts.hs_stretch == (0.0, 1.0) and ts.mhs_stretch == (0.1, 0.9)
    assert ts.slrm_stretch == (-1.0, 1.0) and ts.svf_stretch == (0.7, 1.0) and ts.asvf_stretch == (0.65, 1.0)
    assert ts.pos_opns_stretch == (68.0, 93.0)
    assert ts.msrm_stretch == (-2.0, 2.0) and ts.mstp_stretch == (0.0, 1.0)
    assert all(isinstance(v, float) for v in ts.slp_stretch)
    # pins current behaviour, KNOWN_ISSUES candidate: the "Sky illumination" and "Local dominance" stretches are
    # assigned to neg_opns_stretch (copy-paste in read_from_json), so it ends up with the local dominance values
    # instead of (60, 95), and sim_stretch / ld_stretch are never set.
    assert ts.neg_opns_stretch == (0.5, 1.8)
    assert ts.sim_stretch is None and ts.ld_stretch is None


def test_terrain_settings_read_from_json_partial_and_reset():
    ts = rb.TerrainSettings()
    ts.read_from_json(_terrain_doc())
    ts.read_from_json({"terrain_settings": {"name": "small", "Hillshade": {"hs_sun_el": {"value": 15}}}})
    # read_from_json starts with __init__(): nothing of the first document is left
    assert ts.name == "small" and ts.hs_sun_el == 15
    assert all(getattr(ts, attr) is None for attr in PARAM_ATTRS + STRETCH_ATTRS if attr != "hs_sun_el")
    # a section with one missing key: the others are read
    ts.read_from_json({"terrain_settings": {"name": "x", "Sky-View Factor": {"svf_r_max": {"value": 7}}}})
    assert (ts.svf_r_max, ts.svf_n_dir, ts.svf_noise) == (7, None, None)
    # a stretch needs both min and max
    ts.read_from_json({"terrain_settings": {"name": "x", "Hillshade": {"stretch": {"min": 0}}}})
    assert ts.hs_stretch is None
    with pytest.raises(KeyError):
        ts.read_from_json({"terrain_settings": {"Hillshade": {}}})  # name is required
    with pytest.raises(KeyError):
        ts.read_from_json({"name": "x"})


def test_terrain_settings_read_from_file_and_shipped_terrains(tmp_path):
    path = tmp_path / "terrain.json"
    path.write_text(json.dumps(_terrain_doc()))
    ts = rb.TerrainSettings()
    ts.read_from_file(str(path))
    assert ts.name == "full" and ts.hs_sun_azi == 123

    shipped = os.path.join(SETTINGS, "default_terrains_settings.json")
    terrains = rb.TerrainsSettings()
    assert terrains.terrains_settings == []
    terrains.read_from_file(shipped)
    names = [t.name for t in terrains.terrains_settings]
    assert names == ["general", "flat", "steep"]
    with open(shipped) as fh:
        raw = json.load(fh)["terrains_settings"]
    assert names == [r["terrain_settings"]["name"] for r in raw]
    general = terrains.select_terrain_settings_by_name("general")
    assert general is terrains.terrains_settings[0]
    assert general.slp_stretch == (0.0, 50.0) and general.hs_stretch == (0.0, 1.0) and general.hs_sun_el == 35
    assert general.mhs_stretch == (0.0, 1.0) and general.mhs_sun_el == 35 and general.slrm_stretch == (-1.0, 1.0)
    assert general.slrm_rad_cell == 20 and general.svf_stretch == (0.7, 1.0)
    assert (general.svf_r_max, general.svf_noise) == (10, 0) and general.asvf_stretch == (0.65, 1.0)
    assert general.pos_opns_stretch == (68.0, 93.0)
    assert (general.ld_min_rad, general.ld_max_rad, general.ld_observer_h) == (10, 20, 1.7)
    assert general.svf_n_dir is None and general.hs_sun_azi is None and general.msrm_stretch is None
    # the shipped file lists "Openness - Negative" (60, 95), "Sky illumination" (0.25, 0.3) and "Local dominance"
    # (0.5, 1.8); each later section overwrites neg_opns_stretch
    # pins current behaviour, KNOWN_ISSUES candidate
    assert general.neg_opns_stretch == (0.5, 1.8)
    assert general.sim_stretch is None and general.ld_stretch is None
    flat = terrains.select_terrain_settings_by_name("flat")
    assert flat.slp_stretch == (0.0, 15.0) and flat.hs_sun_el == 15 and flat.mhs_sun_el == 15 and flat.slrm_rad_cell == 10
    assert flat.svf_stretch == (0.9, 1.0) and (flat.svf_r_max, flat.svf_noise) == (20, 3)
    assert flat.pos_opns_stretch == (85.0, 93.0) and flat.neg_opns_stretch == (0.5, 3.0)  # local dominance values
    steep = terrains.select_terrain_settings_by_name("steep")
    assert steep.slp_stretch == (0.0, 60.0) and steep.hs_sun_el == 55 and steep.svf_stretch == (0.55, 1.0)
    assert (steep.ld_min_rad, steep.ld_max_rad, steep.ld_observer_h) == (10, 10, 16.0)
    assert steep.neg_opns_stretch == (0.55, 0.95)
    assert terrains.select_terrain_settings_by_name("missing") is None


def test_terrains_settings_read_appends_and_selects_first():
    path = os.path.join(SETTINGS, "default_terrains_settings.json")
    terrains = rb.TerrainsSettings()
    terrains.read_from_file(path)
    terrains.read_from_file(path)
    # pins current behaviour, KNOWN_ISSUES candidate: unlike BlenderCombinations.read_from_file, the list is not
    # emptied before reading, so a second read appends the same terrains again.
    assert [t.name for t in terrains.terrains_settings] == ["general", "flat", "steep"] * 2
    assert terrains.select_terrain_settings_by_name("flat") is terrains.terrains_settings[1]


def _full_terrain_settings():
    ts = rb.TerrainSettings()
    ts.read_from_json(_terrain_doc())
    return ts


def test_apply_terrain_overwrites_defaults_with_values_that_are_set():
    ts = _full_terrain_settings()
    default = rvt.default.DefaultValues()
    comb = rb.BlenderCombination()
    ts.apply_terrain(default, comb)  # empty combination: only the defaults change
    for attr in PARAM_ATTRS:
        assert getattr(default, attr) == getattr(ts, attr), attr
    assert default.mstp_local_scale == (3, 21, 2)


def test_apply_terrain_leaves_unset_values_alone():
    default = rvt.default.DefaultValues()
    before = copy.deepcopy(vars(default))
    ts = rb.TerrainSettings()
    ts.apply_terrain(default, rb.BlenderCombination())  # nothing set: nothing changes
    assert vars(default) == before
    ts.hs_sun_el = 77
    ts.mstp_lightness = 0.5
    ts.apply_terrain(default, rb.BlenderCombination())
    after = vars(default)
    assert after["hs_sun_el"] == 77 and after["mstp_lightness"] == 0.5
    assert {k: v for k, v in after.items() if k not in ("hs_sun_el", "mstp_lightness")} == \
        {k: v for k, v in before.items() if k not in ("hs_sun_el", "mstp_lightness")}
    # a value of 0 is "set" (the test is `is not None`)
    ts.svf_noise = 0
    default.svf_noise = 3
    ts.apply_terrain(default, rb.BlenderCombination())
    assert default.svf_noise == 0


def test_apply_terrain_stretches():
    ts = rb.TerrainSettings()
    for i, attr in enumerate(STRETCH_ATTRS):
        setattr(ts, attr, (float(i), float(i) + 0.5))  # assigned by hand: includes sim_stretch and ld_stretch
    comb = rb.BlenderCombination()
    for attr in STRETCH_ATTRS:
        comb.add_layer(rb.BlenderLayer(STRETCH_VIS[attr].upper(), "value", -99, 99))  # matched case-insensitively
    comb.add_layer(rb.BlenderLayer("Something else", "value", -99, 99))
    ts.apply_terrain(rvt.default.DefaultValues(), comb)
    for i, attr in enumerate(STRETCH_ATTRS):
        layer = comb.layers[i]
        assert (layer.min, layer.max) == (float(i), float(i) + 0.5), attr
        assert layer.normalization == "value" and layer.blend_mode == "normal"  # nothing else is touched
    assert (comb.layers[-1].min, comb.layers[-1].max) == (-99, 99)  # no stretch for an unknown visualization


def test_apply_terrain_without_stretch_keeps_the_layer_limits():
    ts = rb.TerrainSettings()
    ts.hs_stretch = (0.25, 0.75)
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
    comb.add_layer(rb.BlenderLayer("Slope gradient", "value", 0, 50))  # slp_stretch is None
    ts.apply_terrain(rvt.default.DefaultValues(), comb)
    assert (comb.layers[0].min, comb.layers[0].max) == (0.25, 0.75)
    assert (comb.layers[1].min, comb.layers[1].max) == (0, 50)


def test_apply_terrain_shipped_general_changes_default_and_combination():
    terrains = rb.TerrainsSettings()
    terrains.read_from_file(os.path.join(SETTINGS, "default_terrains_settings.json"))
    comb = rb.BlenderCombination()
    comb.read_from_file(os.path.join(SETTINGS, "blender_VAT.json"))
    default = rvt.default.DefaultValues()
    terrains.select_terrain_settings_by_name("steep").apply_terrain(default, comb)
    assert default.hs_sun_el == 55 and default.svf_r_max == 10
    limits = {lay.vis: (lay.min, lay.max) for lay in comb.layers}
    assert limits == {"Sky-View Factor": (0.55, 1.0), "Openness - Positive": (55.0, 95.0),
                      "Slope gradient": (0.0, 60.0), "Hillshade": (0.0, 1.0)}  # Hillshade: no stretch in "steep"


# ----------------------------------------------------------------------------------------------------------------------
# BlenderCombination.render_all_images: inputs, errors, rendering from a DEM, saving
# ----------------------------------------------------------------------------------------------------------------------

def _gray(h=6, w=7, seed=0, lo=0.0, hi=1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=(h, w))


def test_render_without_layers_returns_none():
    assert rb.BlenderCombination().render_all_images() is None


def test_render_single_layer_is_the_normalized_image_and_layer_order():
    img = np.array([[0.0, 5.0], [10.0, 20.0]])
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 10, image=img.copy()))
    assert_same(comb.render_all_images(), np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float32))
    # layers[0] is the top layer, layers[-1] the background; the rendering starts with the last layer
    top = np.array([[0.5, 0.5], [0.25, 1.0]])
    bottom = np.array([[1.0, 0.5], [0.5, 0.0]])
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("top", "value", 0, 1, "multiply", 100, image=top.copy()))
    comb.add_layer(rb.BlenderLayer("bottom", "value", 0, 1, "normal", 100, image=bottom.copy()))
    out = comb.render_all_images()
    # the normalized layer images are float32 (normalize_lin), so the render stays float32; the dyadic values are exact
    assert_same(out, (top * bottom).astype(np.float32))
    # opacity 50 %: half of the multiplied image plus half of the background
    comb.layers[0].opacity = 50
    comb.layers[0].image = top.copy()
    comb.layers[1].image = bottom.copy()
    assert_same(comb.render_all_images(), (0.5 * (top * bottom) + 0.5 * bottom).astype(np.float32))
    # the same stack with opacity 0 gives the background
    comb.layers[0].opacity = 0
    comb.layers[0].image = top.copy()
    comb.layers[1].image = bottom.copy()
    assert_same(comb.render_all_images(), bottom.astype(np.float32))


def test_render_screen_and_normal_blend_modes_are_analytic():
    top = np.array([[0.0, 0.25], [0.5, 1.0]])
    bottom = np.array([[0.5, 0.5], [0.5, 0.5]])
    for mode, expected in (("screen", 1 - (1 - top) * (1 - bottom)), ("normal", top), ("Multiply", top * bottom)):
        comb = rb.BlenderCombination()
        comb.add_layer(rb.BlenderLayer("top", "value", 0, 1, mode, 100, image=top.copy()))
        comb.add_layer(rb.BlenderLayer("bottom", "value", 0, 1, "normal", 100, image=bottom.copy()))
        assert_same(comb.render_all_images(), expected.astype(np.float32))


def test_render_checks_the_layers_first():
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1, "dissolve", 100, image=np.zeros((2, 2))))
    with pytest.raises(Exception, match="blend_mode incorrect"):
        comb.render_all_images()
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "percent", 1, 2, image=np.zeros((2, 2)) + np.arange(4).reshape(2, 2)))
    comb.render_all_images()
    assert comb.layers[0].normalization == "perc"  # check_data() runs inside render_all_images


def test_render_input_errors():
    layer_img = rb.BlenderLayer("Hillshade", "value", 0, 1, image=np.zeros((2, 2)))
    comb = rb.BlenderCombination()
    comb.add_layer(layer_img)
    with pytest.raises(Exception, match="define dem_path"):
        comb.render_all_images(save_render_path="out.tif")
    comb.add_dem_path("dem.tif")
    with pytest.raises(Exception, match="save_float or save_8bit to True"):
        comb.render_all_images(save_render_path="out.tif", save_float=False, save_8bit=False)
    # layers without an image need a DEM: neither dem_arr / resolution nor (for save_visualizations) a dem_path
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
    with pytest.raises(Exception, match="define dem_arr and its resolution"):
        comb.render_all_images()
    with pytest.raises(Exception, match="define dem_path"):
        comb.render_all_images(save_visualizations=True)


def test_render_layer_from_image_path(tmp_path):
    arr = np.array([[0.0, 5.0, 10.0], [2.5, 7.5, 20.0]], dtype=np.float32)
    path = _write_tif(tmp_path / "vis.tif", arr)
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 10, image_path=path))
    out = comb.render_all_images()
    assert_same(out, np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 1.0]], dtype=np.float32))
    # pins current behaviour, KNOWN_ISSUES candidate: with a colormap and an image_path the code reads `image.shape`
    # of the (None) layer image, so the combination raises AttributeError instead of coloring the raster.
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 10, colormap="gray", image_path=path))
    with pytest.raises(AttributeError):
        comb.render_all_images()


def test_render_colormap_layer_gives_three_bands():
    img = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, np.nan]])
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1, colormap="gray", min_colormap_cut=0, max_colormap_cut=1,
                                   image=img.copy()))
    out = comb.render_all_images()
    assert out.shape == (3, 2, 3)
    valid = ~np.isnan(img)
    for band in range(3):
        np.testing.assert_allclose(out[band][valid], img[valid], atol=1 / 128, rtol=0)  # gray ramp, LUT step
    assert np.all(out[:, ~valid] == 0)  # NaN pixel: matplotlib "bad" color


def _render_dem(nan_patch=False):
    # relief 25 m on 1 m/px: enough relief for shadows; without NaN because sky illumination and the shadow of this
    # DEM are all NaN / all lit when the DEM has NaN pixels
    return synthetic_dem((40, 44), seed=31, relief=25.0, nan_patch=nan_patch)


def _raw_visualization(vis, dem, default, res):
    """The visualization as render_all_images computes it, taken from rvt.default / rvt.vis with the same
    settings (no_data=None)."""
    if vis == "Slope gradient":
        return default.get_slope(dem, res, res)
    if vis == "Hillshade":
        return default.get_hillshade(dem, res, res)
    if vis == "Shadow":
        return default.get_shadow(dem, res)
    if vis == "Multiple directions hillshade":
        return np.array([rvt.vis.hillshade(dem=dem, resolution_x=res, resolution_y=res,
                                           sun_elevation=default.mhs_sun_el, sun_azimuth=az) for az in (315, 22.5, 90)])
    if vis == "Simple local relief model":
        return default.get_slrm(dem)
    if vis == "Sky-view factor":
        return default.get_sky_view_factor(dem, res, compute_svf=True)["svf"]
    if vis == "Anisotropic sky-view factor":
        return default.get_sky_view_factor(dem, res, compute_svf=False, compute_asvf=True)["asvf"]
    if vis == "Openness - Positive":
        return default.get_sky_view_factor(dem, res, compute_svf=False, compute_opns=True)["opns"]
    if vis == "Openness - Negative":
        return default.get_neg_opns(dem, res)
    if vis == "Sky illumination":
        return default.get_sky_illumination(dem, res)
    if vis == "Local dominance":
        return default.get_local_dominance(dem)
    if vis == "Multi-scale relief model":
        return default.get_msrm(dem, res)
    if vis == "Multi-scale topographic position":
        return default.get_mstp(dem)
    raise AssertionError(vis)


@pytest.mark.parametrize("vis", VIS_NAMES)
def test_render_single_layer_computed_from_dem(vis):
    dem = _render_dem()
    default = rvt.default.DefaultValues()
    res = 1.0
    raw = _raw_visualization(vis, dem.copy(), default, res)
    lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
    assert lo < hi
    expected = bf.normalize_image(vis, raw.copy(), lo, hi, "value")
    comb = rb.BlenderCombination(dem_arr=dem.copy(), dem_resolution=res)
    comb.add_layer(rb.BlenderLayer(vis, "value", lo, hi))
    out = comb.render_all_images(default=rvt.default.DefaultValues())
    assert_bitwise(out, expected)
    assert out.dtype == np.float32
    assert np.nanmin(out) >= 0 and np.nanmax(out) <= 1
    # multiple directions hillshade and the multi-scale topographic position are 3-band images
    if vis in ("Multiple directions hillshade", "Multi-scale topographic position"):
        assert out.shape == (3,) + dem.shape
    else:
        assert out.shape == dem.shape
    assert np.isfinite(out).all()  # the DEM has no NaN
    assert out.min() < out.max()  # the stretch limits are the data range: 0 and 1 are both reached
    assert out.min() == 0.0 or out.max() == 1.0


@pytest.mark.parametrize("vis", ["Slope gradient", "Hillshade", "Simple local relief model"])
def test_render_nan_dem_pixels_stay_nan(vis):
    dem = _render_dem(nan_patch=True)
    default = rvt.default.DefaultValues()
    raw = _raw_visualization(vis, dem.copy(), default, 1.0)
    comb = rb.BlenderCombination(dem_arr=dem.copy(), dem_resolution=1.0)
    comb.add_layer(rb.BlenderLayer(vis, "value", float(np.nanmin(raw)), float(np.nanmax(raw))))
    out = comb.render_all_images(default=rvt.default.DefaultValues())
    assert np.isnan(out[np.isnan(dem)]).all()  # NaN where the DEM is NaN
    assert np.isfinite(out).any()
    assert np.nanmin(out) >= 0 and np.nanmax(out) <= 1


def test_render_default_argument_matches_explicit_default():
    dem = _render_dem()
    comb = rb.BlenderCombination(dem_arr=dem.copy(), dem_resolution=1.0)
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
    out_none = comb.render_all_images()
    comb.dem_arr = dem.copy()
    out_default = comb.render_all_images(default=rvt.default.DefaultValues())
    assert_bitwise(out_none, out_default)


def test_render_no_data_is_replaced_by_nan():
    dem = synthetic_dem((30, 34), seed=32, relief=8.0)
    dem_nd = dem.copy()
    dem_nd[10:12, 5:7] = -9999.0
    dem_nan = dem.copy()
    dem_nan[10:12, 5:7] = np.nan
    outs = []
    for arr, kwargs in ((dem_nd, {"no_data": -9999.0}), (dem_nan, {})):
        comb = rb.BlenderCombination(dem_arr=arr, dem_resolution=1.0)
        comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
        outs.append(comb.render_all_images(**kwargs))
    assert_bitwise(outs[0], outs[1])
    assert np.isnan(outs[0][9:13, 4:8]).any()


def test_render_dem_layers_blended_analytic_order():
    # two layers computed from the DEM: hillshade (top, multiply, 100 %) over slope gradient (background, normal)
    dem = _render_dem()
    default = rvt.default.DefaultValues()
    hs = bf.normalize_image("Hillshade", default.get_hillshade(dem.copy(), 1.0, 1.0), 0, 1, "value")
    slp = bf.normalize_image("Slope gradient", default.get_slope(dem.copy(), 1.0, 1.0), 0, 50, "value")
    comb = rb.BlenderCombination(dem_arr=dem.copy(), dem_resolution=1.0)
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1, "multiply", 100))
    comb.add_layer(rb.BlenderLayer("Slope gradient", "value", 0, 50, "normal", 100))
    out = comb.render_all_images(default=rvt.default.DefaultValues())
    # float32 normalized images multiplied in float32 (same dtype and operation as the blend)
    assert hs.dtype == slp.dtype == np.float32
    assert_same(out, hs * slp)


def test_render_saves_float_and_8bit_rasters(tmp_path):
    dem_path = _write_tif(tmp_path / "dem.tif", synthetic_dem((20, 24), seed=33, relief=5.0), res=0.5)
    top = _gray(20, 24, seed=1)
    bottom = _gray(20, 24, seed=2)
    comb = rb.BlenderCombination(dem_path=dem_path)
    comb.add_layer(rb.BlenderLayer("top", "value", 0, 1, "multiply", 100, image=top.copy()))
    comb.add_layer(rb.BlenderLayer("bottom", "value", 0, 1, "normal", 100, image=bottom.copy()))
    out_path = str(tmp_path / "render.tif")
    out = comb.render_all_images(save_render_path=out_path, save_float=True, save_8bit=True)
    saved = rvt.default.get_raster_arr(out_path)
    assert saved["array"].dtype == np.float32 and saved["array"].shape == (20, 24)
    assert saved["resolution"] == (0.5, 0.5)  # georeference copied from dem_path
    assert_same(saved["array"], out.astype(np.float32))
    saved8 = rvt.default.get_raster_arr(str(tmp_path / "render_8bit.tif"))
    assert saved8["array"].dtype == np.uint8
    assert np.all(np.abs(saved8["array"].astype(float) - 255.0 * out) <= 1.0 + 1e-6)
    # only the float raster
    (tmp_path / "render_8bit.tif").unlink()
    comb.layers[0].image = top.copy()
    comb.layers[1].image = bottom.copy()
    comb.render_all_images(save_render_path=out_path, save_float=True, save_8bit=False)
    assert not (tmp_path / "render_8bit.tif").exists()
    # only the 8-bit raster
    os.remove(out_path)
    comb.layers[0].image = top.copy()
    comb.layers[1].image = bottom.copy()
    comb.render_all_images(save_render_path=out_path, save_float=False, save_8bit=True)
    assert not os.path.exists(out_path) and (tmp_path / "render_8bit.tif").exists()


def test_render_save_visualizations_writes_the_layer_raster_and_reads_it_back(tmp_path):
    dem = synthetic_dem((20, 24), seed=34, relief=5.0)
    dem_path = _write_tif(tmp_path / "dem.tif", dem, res=1.0)
    comb = rb.BlenderCombination(dem_path=dem_path)
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
    out = comb.render_all_images(default=rvt.default.DefaultValues(), save_visualizations=True,
                                 save_render_path=str(tmp_path / "r.tif"), save_float=True, save_8bit=False)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "r.tif" in files and "dem_HS_A315_H35.tif" in files  # the hillshade raster of the default settings is kept
    assert out.shape == dem.shape and np.nanmin(out) >= 0 and np.nanmax(out) <= 1
    # the same as computing in memory from the saved (float32) DEM array: the visualization is written as float32
    # and read back, which does not change float32 values
    expected = bf.normalize_image("Hillshade", rvt.default.DefaultValues().get_hillshade(
        rvt.default.get_raster_arr(dem_path)["array"], 1.0, 1.0), 0, 1, "value")
    assert_same(out, expected)


def test_render_save_visualizations_with_render_in_another_directory_fails(tmp_path):
    dem_path = _write_tif(tmp_path / "dem.tif", synthetic_dem((20, 24), seed=35, relief=5.0), res=1.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    comb = rb.BlenderCombination(dem_path=dem_path)
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
    # pins current behaviour, KNOWN_ISSUES candidate: the visualization raster is saved to the directory of
    # save_render_path (custom_dir), but its path is then taken from the DEM directory (get_hillshade_path), so
    # reading it fails: gdal.Open returns None -> AttributeError (RuntimeError if GDAL exceptions are enabled).
    with pytest.raises((AttributeError, RuntimeError)):
        comb.render_all_images(default=rvt.default.DefaultValues(), save_visualizations=True,
                               save_render_path=str(out_dir / "r.tif"), save_float=True, save_8bit=False)
    assert any("HS_A315_H35" in p.name for p in out_dir.iterdir())  # it was written to out_dir


# ----------------------------------------------------------------------------------------------------------------------
# create_log_file
# ----------------------------------------------------------------------------------------------------------------------

def _log_text(directory):
    logs = list(directory.glob("*_blend_log_*.txt"))
    assert len(logs) == 1
    return logs[0].name, logs[0].read_text()


def test_create_log_file_header_and_layers(tmp_path):
    dem_path = _write_tif(tmp_path / "my_dem.tif", np.zeros((7, 9)), res=0.25)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    default = rvt.default.DefaultValues()
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1, "normal", 100))
    comb.add_layer(rb.BlenderLayer("Slope gradient", "perc", 2, 3, "multiply", 40, "gray", 0.1, 0.9))
    comb.create_log_file(dem_path, "my combination", "render.tif", default, custom_dir=str(log_dir))
    name, text = _log_text(log_dir)
    assert re.fullmatch(r"my_dem_blend_log_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.txt", name)
    lines = text.splitlines()
    assert lines[0] == "=" * 95
    assert "Relief Visualization Toolbox (python), blender log" in lines
    assert "\tInput filename:\t\t{}".format(dem_path) in lines
    assert "\tNumber of rows:\t\t7" in lines
    assert "\tNumber of columns:\t9" in lines
    assert "\tNumber of bands:\t1" in lines
    assert "\tResolution (x, y):\t0.25, 0.25" in lines
    assert "\tOverwrite: {}".format(default.overwrite) in lines
    assert "\tVertical exaggeration factor: {}".format(default.ve_factor) in lines
    assert "Combination name: my combination" in lines
    assert not any(line.startswith("Terrain settings:") for line in lines)  # only with terrain_sett_name
    assert "\t>> Output render file:" in lines and "\t\trender.tif" in lines
    assert "=== LAYERS ===" in lines
    # layer blocks
    assert "Layer: 1" in lines and "Visualization: Hillshade" in lines
    assert "\ths_sun_el=\t\t{}".format(default.hs_sun_el) in lines
    assert "\ths_sun_azi=\t\t{}".format(default.hs_sun_azi) in lines
    assert "Norm: value" in lines and "Linear normalization, min: 0, max: 1" in lines and "Opacity: 100" in lines
    assert "Layer: 2" in lines and "Visualization: Slope gradient" in lines
    assert "\tslp_output_units=\t\t{}".format(default.slp_output_units) in lines
    assert "Norm: perc" in lines and "Linear normalization, min: 2, max: 3" in lines and "Opacity: 40" in lines
    assert "Colormap: gray" in lines and "Minimum Colormap cut: 0.1" in lines and "Maximum Colormap cut: 0.9" in lines
    # layer 1 has no colormap lines: exactly one colormap entry in the file
    assert sum(line.startswith("Colormap:") for line in lines) == 1
    assert "# Computation time" not in text
    assert text.index("Layer: 1") < text.index("Layer: 2")


def test_create_log_file_options(tmp_path):
    dem_path = _write_tif(tmp_path / "dem.tif", np.zeros((4, 5)), res=2.0)
    default = rvt.default.DefaultValues()
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1, "normal", 100, "gray", min_colormap_cut=0.3))
    # custom_dir None: the log goes to the directory of the DEM; terrain settings name and computation time
    comb.create_log_file(dem_path, "c", "r.tif", default, terrain_sett_name="steep", computation_time="12.5 s")
    name, text = _log_text(tmp_path)
    assert name.startswith("dem_blend_log_")
    lines = text.splitlines()
    assert "Terrain settings: steep" in lines
    assert text.endswith("# Computation time: 12.5 s")  # no newline after the last line
    assert "Colormap: gray" in lines and "Minimum Colormap cut: 0.3" in lines
    assert not any(line.startswith("Maximum Colormap cut") for line in lines)
    assert "\tResolution (x, y):\t2.0, 2.0" in lines


@pytest.mark.parametrize("vis, keys", [
    ("Hillshade", ["hs_sun_el", "hs_sun_azi"]),
    ("Shadow", ["hs_sun_el", "hs_sun_azi"]),
    ("Multiple directions hillshade", ["mhs_sun_el", "mhs_nr_dir"]),
    ("Slope gradient", ["slp_output_units"]),
    ("Simple local relief model", ["slrm_rad_cell"]),
    ("Sky-view factor", ["svf_n_dir", "svf_noise", "svf_r_max"]),
    ("Anisotropic sky-view factor", ["svf_n_dir", "svf_noise", "svf_r_max", "asvf_level", "asvf_dir"]),
    ("Openness - Positive", ["svf_n_dir", "svf_noise", "svf_r_max"]),
    ("Openness - Negative", ["svf_n_dir", "svf_noise", "svf_r_max"]),
    ("Sky illumination", ["sim_sky_mod", "sim_compute_shadow", "sim_shadow_az", "sim_shadow_el", "sim_nr_dir",
                          "sim_shadow_dist"]),
    ("Local dominance", ["ld_rad_inc", "ld_min_rad", "ld_max_rad", "ld_anglr_res", "ld_observer_h"]),
    ("Multi-scale relief model", ["msrm_feature_min", "msrm_feature_max", "msrm_scaling_factor"]),
    ("Multi-scale topographic position", ["mstp_local_scale", "mstp_meso_scale", "mstp_broad_scale", "mstp_lightness"]),
    ("Something else", []),
])
def test_create_log_file_visualization_parameters(tmp_path, vis, keys):
    dem_path = _write_tif(tmp_path / "dem.tif", np.zeros((4, 5)))
    default = rvt.default.DefaultValues()
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer(vis, "value", 0, 1))
    comb.create_log_file(dem_path, "c", "r.tif", default, custom_dir=str(tmp_path))
    _, text = _log_text(tmp_path)
    # every parameter line has the form "<tabs>name=<tabs>value" between "Visualization:" and "Norm:"
    block = text.split("Visualization: {}\n".format(vis))[1].split("Norm: value")[0]
    found = re.findall(r"^\t+(\w+)=\t+(.*)$", block, flags=re.M)
    assert [k for k, _ in found] == keys
    for key, value in found:
        shown = getattr(default, key)
        if key.startswith("mstp_") and key.endswith("_scale"):
            shown = "({}, {}, {})".format(*shown)
        assert value == str(shown), key


def test_create_log_file_three_band_dem(tmp_path):
    from osgeo import gdal
    path = str(tmp_path / "rgb.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 6, 5, 3, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
    for band in range(3):
        ds.GetRasterBand(band + 1).WriteArray(np.zeros((5, 6), dtype=np.float32))
    ds.FlushCache()
    ds = None
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("Hillshade", "value", 0, 1))
    comb.create_log_file(path, "c", "r.tif", rvt.default.DefaultValues(), custom_dir=str(tmp_path))
    lines = _log_text(tmp_path)[1].splitlines()
    assert "\tNumber of bands:\t3" in lines and "\tNumber of rows:\t\t5" in lines and "\tNumber of columns:\t6" in lines


# ----------------------------------------------------------------------------------------------------------------------
# color_relief_image_map, e3mstp, e4mstp
# ----------------------------------------------------------------------------------------------------------------------

def _pipeline_dem(seed=41):
    return synthetic_dem((40, 44), seed=seed, relief=6.0, nan_patch=True)


def _check_rgb_output(out, dem):
    assert out.shape == (3,) + dem.shape
    assert out.dtype == np.float64
    nan_mask = np.isnan(dem)
    for band in range(3):  # NaN exactly where the DEM has NaN, in all bands
        assert np.array_equal(np.isnan(out[band]), nan_mask)
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0
    assert np.isfinite(out[:, ~nan_mask]).all()


def test_color_relief_image_map_output():
    dem = _pipeline_dem()
    out = rb.color_relief_image_map(dem.copy(), 1.0, default=rvt.default.DefaultValues())
    _check_rgb_output(out, dem)
    # the default colormap "OrRd" is colored: the bands differ
    assert not np.array_equal(out[0][~np.isnan(dem)], out[1][~np.isnan(dem)])
    # deterministic
    assert_bitwise(out, rb.color_relief_image_map(dem.copy(), 1.0, default=rvt.default.DefaultValues()))


def test_color_relief_image_map_gray_colormap_gives_equal_bands():
    dem = _pipeline_dem()
    out = rb.color_relief_image_map(dem.copy(), 1.0, default=rvt.default.DefaultValues(), colormap="gray",
                                    min_colormap_cut=None, max_colormap_cut=None)
    assert out.shape == (3,) + dem.shape
    # pins current behaviour, KNOWN_ISSUES candidate: where the luminosity blend of a *gray* background goes below 0,
    # clip_color divides 0 by 0 (lum == min, see test_clip_color_nan_and_gray_negative_pixel), so a gray colormap
    # gives NaN at pixels where the DEM is finite; the NaN mask is then larger than that of the DEM.
    nan_mask = np.isnan(dem)
    assert np.isnan(out[:, nan_mask]).all()
    finite = np.isfinite(out).all(axis=0)
    assert finite.sum() > 0.5 * dem.size
    # gray background, luminosity and overlay of gray images: R == G == B (float32 casts of equal values)
    np.testing.assert_allclose(out[0][finite], out[1][finite], atol=1e-6, rtol=0)
    np.testing.assert_allclose(out[0][finite], out[2][finite], atol=1e-6, rtol=0)
    assert np.nanmin(out) >= 0 and np.nanmax(out) <= 1


def test_color_relief_image_map_colormap_cuts_change_the_result():
    dem = _pipeline_dem()
    full = rb.color_relief_image_map(dem.copy(), 1.0, default=rvt.default.DefaultValues(), colormap="gray")
    cut = rb.color_relief_image_map(dem.copy(), 1.0, default=rvt.default.DefaultValues(), colormap="gray",
                                    min_colormap_cut=0.5, max_colormap_cut=1)
    assert not np.array_equal(np.nan_to_num(full), np.nan_to_num(cut))
    # with the lower half of the ramp removed the (gray) background is brighter on average
    assert np.nanmean(cut) > np.nanmean(full)
    with pytest.raises(Exception, match="gray_scale_to_color_ramp"):
        rb.color_relief_image_map(dem.copy(), 1.0, default=rvt.default.DefaultValues(), min_colormap_cut=0.9,
                                  max_colormap_cut=0.1)


def test_color_relief_image_map_side_effects():
    dem = _pipeline_dem()
    dem_before = dem.copy()
    default = rvt.default.DefaultValues()
    default.svf_r_max = 25
    rb.color_relief_image_map(dem, 1.0, default=default)
    assert_same(dem, dem_before)  # without no_data the DEM is not modified
    # pins current behaviour, KNOWN_ISSUES candidate: the function sets svf_r_max = 10 on the DefaultValues instance
    # it receives (and the default argument is one shared instance created at import time)
    assert default.svf_r_max == 10


def test_color_relief_image_map_no_data():
    dem = synthetic_dem((40, 44), seed=42, relief=6.0)
    dem_nd = dem.copy()
    dem_nd[5:8, 6:9] = -9999.0
    dem_nan = dem.copy()
    dem_nan[5:8, 6:9] = np.nan
    out_nd = rb.color_relief_image_map(dem_nd, 1.0, default=rvt.default.DefaultValues(), no_data=-9999.0)
    out_nan = rb.color_relief_image_map(dem_nan.copy(), 1.0, default=rvt.default.DefaultValues())
    assert_bitwise(out_nd, out_nan)
    # the no_data pixels are replaced by NaN in the caller's array
    assert np.array_equal(np.isnan(dem_nd), np.isnan(dem_nan))
    assert np.isnan(out_nd[:, 5:8, 6:9]).all()


@pytest.mark.parametrize("func", [rb.e3mstp, rb.e4mstp])
def test_e3_e4_mstp_output_shape_dtype_range_nan(func):
    dem = _pipeline_dem(seed=43)
    out = func(dem.copy(), 1.0, default=rvt.default.DefaultValues())
    _check_rgb_output(out, dem)
    # three distinct bands (colored result)
    valid = ~np.isnan(dem)
    assert not np.array_equal(out[0][valid], out[1][valid]) or not np.array_equal(out[1][valid], out[2][valid])
    # the DEM is left unchanged when no_data is None
    dem2 = dem.copy()
    func(dem2, 1.0, default=rvt.default.DefaultValues())
    assert_same(dem2, dem)


@pytest.mark.parametrize("func", [rb.e3mstp, rb.e4mstp])
def test_e3_e4_mstp_resolution_argument_changes_the_result(func):
    dem = synthetic_dem((40, 44), seed=44, relief=6.0)
    out_1 = func(dem.copy(), 1.0, default=rvt.default.DefaultValues())
    out_05 = func(dem.copy(), 0.5, default=rvt.default.DefaultValues())
    assert out_1.shape == out_05.shape == (3, 40, 44)
    assert not np.array_equal(out_1, out_05)  # slope / sky-view factor depend on the pixel size


def test_e3mstp_no_data():
    dem = synthetic_dem((40, 44), seed=45, relief=6.0)
    dem_nd = dem.copy()
    dem_nd[10:12, 20:23] = -9999.0
    dem_nan = dem.copy()
    dem_nan[10:12, 20:23] = np.nan
    out_nd = rb.e3mstp(dem_nd, 1.0, default=rvt.default.DefaultValues(), no_data=-9999.0)
    out_nan = rb.e3mstp(dem_nan.copy(), 1.0, default=rvt.default.DefaultValues())
    assert_bitwise(out_nd, out_nan)
    assert np.isnan(dem_nd[10:12, 20:23]).all()  # replaced in the caller's array
    assert np.isnan(out_nd[:, 10:12, 20:23]).all()


def test_e4mstp_no_data_pixels_are_filled_with_the_no_data_value():
    dem = synthetic_dem((40, 44), seed=46, relief=6.0)
    dem_nd = dem.copy()
    dem_nd[10:12, 20:23] = -9999.0
    dem_nd[0, 43] = -9999.0
    dem_nan = dem_nd.copy()
    dem_nan[dem_nan == -9999.0] = np.nan
    out_nd = rb.e4mstp(dem_nd, 1.0, default=rvt.default.DefaultValues(), no_data=-9999.0)
    out_nan = rb.e4mstp(dem_nan.copy(), 1.0, default=rvt.default.DefaultValues())
    mask = np.isnan(dem_nan)
    assert np.isnan(dem_nd[mask]).all()  # replaced by NaN in the caller's array
    assert np.all(out_nd[:, mask] == -9999.0)  # unlike e3mstp, the output carries the no_data value
    assert not np.isnan(out_nd).any()
    expected = out_nan.copy()
    expected[:, mask] = -9999.0
    assert_bitwise(out_nd, expected)


def test_e4mstp_does_not_change_the_default_object_and_ignores_it_for_flat_terrain():
    dem = synthetic_dem((40, 44), seed=47, relief=6.0)
    default = rvt.default.DefaultValues()
    before = copy.deepcopy(vars(default))
    rb.e4mstp(dem.copy(), 1.0, default=default)
    assert vars(default) == before
    # e3mstp goes through color_relief_image_map, which sets svf_r_max = 10 on the object
    rb.e3mstp(dem.copy(), 1.0, default=default)
    assert default.svf_r_max == 10


def test_pipelines_equal_manual_layer_stack_for_crim():
    # color_relief_image_map is the stack: openness difference (overlay 50 %), the same image (luminosity 50 %), and
    # the colored slope in radians as background; rebuilt here with BlenderCombination and the rvt.default getters
    dem = _pipeline_dem(seed=48)
    default = rvt.default.DefaultValues()
    default.svf_r_max = 10
    pos = default.get_sky_view_factor(dem.copy(), 1.0, compute_svf=False, compute_asvf=False, compute_opns=True)["opns"]
    neg = default.get_sky_view_factor(-1 * dem, 1.0, compute_svf=False, compute_asvf=False, compute_opns=True)["opns"]
    diff = pos - neg
    slope = rvt.vis.slope_aspect(dem=dem.copy(), resolution_x=1.0, resolution_y=1.0, output_units="radian")["slope"]
    comb = rb.BlenderCombination()
    comb.add_layer(rb.BlenderLayer("a", "value", -28, 28, "overlay", 50, image=diff))
    comb.add_layer(rb.BlenderLayer("b", "value", -28, 28, "luminosity", 50, image=diff.copy()))
    comb.add_layer(rb.BlenderLayer("c", "value", 0, 0.8, "normal", 100, colormap="OrRd", min_colormap_cut=0,
                                   max_colormap_cut=1, image=slope))
    expected = comb.render_all_images()
    out = rb.color_relief_image_map(dem.copy(), 1.0, default=rvt.default.DefaultValues())
    assert_bitwise(out, expected)
