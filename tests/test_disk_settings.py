"""
0.2.0 disk windows: ``DefaultValues`` attributes, settings files, output file names, tile overlap.

- ``slrm_kernel_shape``, ``msrm_kernel_shape``, ``mstp_kernel_shape``: default ``"square"``.
- json: ``save_default_to_file`` writes each into its group ("Simple local relief model", "Multi-scale relief
  model", "Multi-scale topographic position") as ``{"value": ..., "description": ...}``; the groups hold exactly
  their 0.1.1 keys (written out by hand below from ``save_default_to_file`` at v0.1.1) plus the new one; no other
  group gets one. Round trip of the three values; a json without the new keys (every file written by 0.1.1) still
  loads and gives ``"square"``.
- txt: the names are the attribute names (case-insensitive like every txt name).
- file names: ``_disk`` / ``_diskea`` before ``.tif`` / ``_8bit.tif`` when the shape is not square; square names
  unchanged (hand-written from the formatting rules of ``get_*_file_name``).
- ``rvt.tile._get_rvt_visualization_overlap``: floor(r_w) = floor((2r + 1) / sqrt(pi)) for "disk_equal_area"
  (the window reaches farther than r), unchanged for "square" and "disk" (STATUS J5). MSRM with
  "disk_equal_area" is not pinned here: its overlap is a feature size in m used as px (KNOWN_ISSUES), not a radius.
"""

import json
import math
from pathlib import Path

import pytest

import rvt.default
import rvt.tile
from rvt.default import DefaultValues, RVTVisualization

V = RVTVisualization
NEW = {"Simple local relief model": "slrm_kernel_shape", "Multi-scale relief model": "msrm_kernel_shape",
       "Multi-scale topographic position": "mstp_kernel_shape"}

# keys of the three groups as v0.1.1 writes them (DefaultValues.save_default_to_file), in file order
KEYS_0_1_1 = {
    "Simple local relief model": ["slrm_compute", "slrm_rad_cell", "slrm_save_float", "slrm_save_8bit",
                                  "slrm_bytscl"],
    "Multi-scale relief model": ["msrm_compute", "msrm_feature_min", "msrm_feature_max", "msrm_scaling_factor",
                                 "msrm_save_float", "msrm_save_8bit", "msrm_bytscl"],
    "Multi-scale topographic position": ["mstp_compute", "mstp_local_scale", "mstp_meso_scale", "mstp_broad_scale",
                                         "mstp_lightness", "mstp_save_float", "mstp_save_8bit", "mstp_bytscl"],
}


def _with_shapes(dv, **shapes):
    """Set ``<vis>_kernel_shape`` attributes after checking that each exists with its default "square"."""
    for vis, shape in shapes.items():
        name = "{}_kernel_shape".format(vis)
        assert getattr(dv, name) == "square"
        setattr(dv, name, shape)
    return dv


# ----------------------------------------------------------------------------------------------------------------------
# attributes and settings files
# ----------------------------------------------------------------------------------------------------------------------

def test_kernel_shape_attributes_default_to_square():
    dv = DefaultValues()
    for name in NEW.values():
        value = getattr(dv, name)
        assert value == "square" and type(value) is str


def test_json_groups_hold_the_0_1_1_keys_and_the_kernel_shape(tmp_path):
    path = tmp_path / "settings.json"
    DefaultValues().save_default_to_file(str(path))
    groups = json.loads(path.read_text())["default_settings"]
    for group, keys in KEYS_0_1_1.items():
        assert sorted(groups[group]) == sorted(keys + [NEW[group]]), group
        entry = groups[group][NEW[group]]
        assert sorted(entry) == ["description", "value"]
        assert entry["value"] == "square"
        assert isinstance(entry["description"], str) and entry["description"]
    others = [(g, k) for g, v in groups.items() if isinstance(v, dict) for k in v if k.endswith("_kernel_shape")
              and g not in NEW]
    assert others == []


def test_json_round_trip_of_the_kernel_shapes(tmp_path):
    path = tmp_path / "settings.json"
    src = _with_shapes(DefaultValues(), slrm="disk", msrm="disk_equal_area", mstp="disk")
    src.save_default_to_file(str(path))
    groups = json.loads(path.read_text())["default_settings"]
    assert groups["Simple local relief model"]["slrm_kernel_shape"]["value"] == "disk"
    assert groups["Multi-scale relief model"]["msrm_kernel_shape"]["value"] == "disk_equal_area"
    assert groups["Multi-scale topographic position"]["mstp_kernel_shape"]["value"] == "disk"
    dst = DefaultValues()
    dst.read_default_from_file(str(path))
    assert (dst.slrm_kernel_shape, dst.msrm_kernel_shape, dst.mstp_kernel_shape) == ("disk", "disk_equal_area", "disk")
    assert vars(dst) == vars(src)  # every other attribute unchanged too (0 == 0.0 after json, KNOWN_ISSUES)


def test_json_without_the_kernel_shape_keys_loads_as_square(tmp_path):
    # a settings file of 0.1.1: the same file without the three keys
    path = tmp_path / "old.json"
    src = DefaultValues()
    src.slrm_rad_cell = 30
    src.mstp_lightness = 1.3
    src.save_default_to_file(str(path))
    data = json.loads(path.read_text())
    for group, key in NEW.items():
        data["default_settings"][group].pop(key, None)
    path.write_text(json.dumps(data, indent=4))
    dst = DefaultValues()
    dst.read_default_from_file(str(path))
    assert dst.slrm_rad_cell == 30 and dst.mstp_lightness == 1.3
    assert (dst.slrm_kernel_shape, dst.msrm_kernel_shape, dst.mstp_kernel_shape) == ("square", "square", "square")


def test_txt_kernel_shape_names_are_the_attribute_names(tmp_path):
    path = tmp_path / "settings.txt"
    path.write_text("trend_radius = 12\n"
                    "slrm_kernel_shape = disk\n"
                    "MSRM_kernel_shape = disk_equal_area   # names are case-insensitive\n"
                    "mstp_kernel_shape = disk\n")
    dv = DefaultValues()
    dv.read_default_from_file(str(path))
    assert dv.slrm_rad_cell == 12
    assert (dv.slrm_kernel_shape, dv.msrm_kernel_shape, dv.mstp_kernel_shape) == ("disk", "disk_equal_area", "disk")
    other = tmp_path / "other.txt"
    other.write_text("mstp_kernel_shape = disk_equal_area\n")
    dv2 = DefaultValues()
    dv2.read_default_from_file(str(other))
    assert (dv2.slrm_kernel_shape, dv2.msrm_kernel_shape, dv2.mstp_kernel_shape) == ("square", "square",
                                                                                     "disk_equal_area")


# ----------------------------------------------------------------------------------------------------------------------
# file names
# ----------------------------------------------------------------------------------------------------------------------

DEM_PATH = str(Path("data") / "dem_01.tif")
SQUARE_NAMES = {  # written by hand from the formatting rules of get_*_file_name (defaults)
    "slrm": "dem_01_SLRM_R20.tif",
    "msrm": "dem_01_MSRM_F_M0-20_S2.tif",
    "mstp": "dem_01_MSTP_5_50_500_L0.9.tif",
}
SUFFIX = {"disk": "_disk", "disk_equal_area": "_diskea"}
MEMBER = {"slrm": V.SIMPLE_LOCAL_RELIEF_MODEL, "msrm": V.MULTI_SCALE_RELIEF_MODEL,
          "mstp": V.MULTI_SCALE_TOPOGRAPHIC_POSITION}


def _name(dv, vis, bit8):
    return getattr(dv, "get_{}_file_name".format(vis))(DEM_PATH, bit8=bit8)


@pytest.mark.parametrize("vis", sorted(SQUARE_NAMES))
def test_square_file_names_are_unchanged(vis):
    dv = DefaultValues()
    assert _name(dv, vis, False) == SQUARE_NAMES[vis]
    assert _name(dv, vis, True) == SQUARE_NAMES[vis].replace(".tif", "_8bit.tif")


@pytest.mark.parametrize("shape", sorted(SUFFIX))
@pytest.mark.parametrize("vis", sorted(SQUARE_NAMES))
def test_disk_file_names_have_the_suffix(vis, shape):
    dv = _with_shapes(DefaultValues(), **{vis: shape})
    stem = SQUARE_NAMES[vis][:-len(".tif")] + SUFFIX[shape]
    assert _name(dv, vis, False) == stem + ".tif"
    assert _name(dv, vis, True) == stem + "_8bit.tif"
    assert getattr(dv, "get_{}_path".format(vis))(DEM_PATH) == str(Path("data") / (stem + ".tif"))
    assert dv.get_visualization_file_name(MEMBER[vis], DEM_PATH, path_8bit=False) == stem + ".tif"
    assert dv.get_visualization_file_name(MEMBER[vis], DEM_PATH, path_8bit=True) == stem + "_8bit.tif"
    for other in SQUARE_NAMES:  # the attribute of one visualization does not rename another
        if other != vis:
            assert _name(dv, other, False) == SQUARE_NAMES[other]


# ----------------------------------------------------------------------------------------------------------------------
# tile overlap
# ----------------------------------------------------------------------------------------------------------------------

def _overlap(member, dv):
    return rvt.tile._get_rvt_visualization_overlap(rvt_visualization=member, rvt_default=dv)


def test_overlap_square_is_unchanged():
    dv = DefaultValues()
    assert _overlap(V.SIMPLE_LOCAL_RELIEF_MODEL, dv) == 20
    assert _overlap(V.MULTI_SCALE_RELIEF_MODEL, dv) == 20
    assert _overlap(V.MULTI_SCALE_TOPOGRAPHIC_POSITION, dv) == 500


def test_overlap_disk_is_the_square_overlap():
    dv = _with_shapes(DefaultValues(), slrm="disk", msrm="disk", mstp="disk")
    assert _overlap(V.SIMPLE_LOCAL_RELIEF_MODEL, dv) == 20
    assert _overlap(V.MULTI_SCALE_RELIEF_MODEL, dv) == 20
    assert _overlap(V.MULTI_SCALE_TOPOGRAPHIC_POSITION, dv) == 500


@pytest.mark.parametrize("slrm_r, slrm_px, broad, mstp_px", [
    (20, 23, (50, 500, 50), 564),  # floor(41 / sqrt(pi)) = floor(23.13); floor(1001 / sqrt(pi)) = floor(564.76)
    (10, 11, (50, 200, 50), 226),  # floor(21 / sqrt(pi)) = floor(11.85); floor(401 / sqrt(pi)) = floor(226.24)
])
def test_overlap_disk_equal_area_is_floor_of_the_window_radius(slrm_r, slrm_px, broad, mstp_px):
    for r, px in ((slrm_r, slrm_px), (broad[1], mstp_px)):
        assert px == math.floor((2 * r + 1) / math.sqrt(math.pi))  # the hand-written values follow the definition
    dv = _with_shapes(DefaultValues(), slrm="disk_equal_area", mstp="disk_equal_area")
    dv.slrm_rad_cell = slrm_r
    dv.mstp_broad_scale = broad
    assert _overlap(V.SIMPLE_LOCAL_RELIEF_MODEL, dv) == slrm_px  # e.g. 23 px = 5.75 m at 0.25 m/px
    assert _overlap(V.MULTI_SCALE_TOPOGRAPHIC_POSITION, dv) == mstp_px  # e.g. 564 px = 141 m at 0.25 m/px
