"""
Characterization tests for ``rvt/default.py`` at tag ``start``.

They document what ``DefaultValues`` does today, without changing it:

- the default value of every attribute (a table written by hand from the documented defaults);
- the settings files: JSON round trip of every saved attribute, the ``.txt`` parser, the default path;
- the output file and path names (strings written by hand from the formatting rules in the code);
- every ``get_*`` compute method is a thin wrapper of the matching ``rvt.vis`` function: shape, dtype, NaN mask and
  bitwise agreement with the direct call, with parameters that differ from the defaults so that a wrong wiring shows;
- ``float_to_8bit`` against a plain reference implementation of the byte scale kept in this file;
- ``calculate_visualization`` for every ``RVTVisualization`` member;
- every ``save_*`` method on a small GeoTIFF: files, dtype, geotransform, projection, nodata, return codes;
- ``get_raster_arr``, ``get_raster_size``, ``save_raster`` round trip.

Expected values are written by hand, follow from a property, or come from the reference implementation below; none is
pasted from the output of the code under test. Behaviour that looks wrong is pinned with the comment
``# pins current behaviour, KNOWN_ISSUES candidate`` and is not changed.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

import rvt.default
import rvt.vis
from conftest import assert_bitwise, synthetic_dem
from rvt.default import DefaultValues, RVTVisualization

V = RVTVisualization

# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------

ORIGIN_X = 500000.0
ORIGIN_Y = 4500000.0
GEOTRANSFORM = (ORIGIN_X, 2.0, 0.0, ORIGIN_Y, 0.0, -2.0)  # gdal order, 2 m/px
NODATA = -9999.0
SHAPE = (48, 56)  # rows, columns


def write_geotiff(path, arr, res_x=2.0, res_y=2.0, nodata=None, dtype="float32"):
    """Write a GeoTIFF (EPSG:32633, north-up, origin ORIGIN_X/ORIGIN_Y) with rasterio; arr is 2D or (bands, H, W)."""
    arr = np.asarray(arr)
    bands = arr[np.newaxis] if arr.ndim == 2 else arr
    with rasterio.open(str(path), "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
                       count=bands.shape[0], dtype=dtype, crs=CRS.from_epsg(32633),
                       transform=Affine(res_x, 0.0, ORIGIN_X, 0.0, -res_y, ORIGIN_Y), nodata=nodata) as ds:
        ds.write(bands.astype(dtype))
    return Path(path)


def read_raster(path):
    """Return (array, profile info dict) read with rasterio."""
    with rasterio.open(str(path)) as ds:
        arr = ds.read()
        info = {"transform": ds.transform.to_gdal(), "epsg": ds.crs.to_epsg(), "nodata": ds.nodata,
                "nodatavals": ds.nodatavals, "dtypes": ds.dtypes, "count": ds.count}
    return arr, info


def small_defaults():
    """DefaultValues with a small, non-default parameter set (fast on 48 x 56 px, and wrong wiring shows)."""
    dv = DefaultValues()
    dv.ve_factor = 2
    dv.slp_output_units = "percent"
    dv.hs_sun_azi, dv.hs_sun_el = 100, 40
    dv.mhs_nr_dir, dv.mhs_sun_el = 8, 50
    dv.slrm_rad_cell = 10
    dv.svf_n_dir, dv.svf_r_max, dv.svf_noise = 8, 5, 1
    dv.asvf_dir, dv.asvf_level = 45, 2
    dv.sim_sky_mod, dv.sim_compute_shadow = "uniform", 1
    dv.sim_nr_dir, dv.sim_shadow_dist, dv.sim_shadow_az, dv.sim_shadow_el = 8, 10, 200, 20
    dv.ld_min_rad, dv.ld_max_rad, dv.ld_rad_inc, dv.ld_anglr_res, dv.ld_observer_h = 3, 6, 2, 20, 2.0
    dv.msrm_feature_min, dv.msrm_feature_max, dv.msrm_scaling_factor = 2, 10, 2
    dv.mstp_local_scale, dv.mstp_meso_scale, dv.mstp_broad_scale = (1, 3, 1), (3, 9, 3), (9, 15, 3)
    dv.mstp_lightness = 0.8
    return dv


@pytest.fixture(scope="module")
def dem():
    """Terrain with NaN patches and slopes steeper than 51 degrees (the default 8-bit slope range ends there)."""
    return synthetic_dem(SHAPE, seed=2, relief=30.0, nan_patch=True)


@pytest.fixture(scope="module")
def dem_clean():
    return synthetic_dem(SHAPE, seed=2, relief=30.0)


# ----------------------------------------------------------------------------------------------------------------------
# attribute defaults
# ----------------------------------------------------------------------------------------------------------------------

# Written by hand from the documented defaults (class docstring of DefaultValues and the settings in
# docs of the tool): flags are 0/1 ints, scales are tuples.
EXPECTED_DEFAULTS = {
    "overwrite": 0, "ve_factor": 1,
    "slp_compute": 0, "slp_output_units": "degree",
    "hs_compute": 1, "hs_sun_azi": 315, "hs_sun_el": 35, "hs_shadow": 0,
    "mhs_compute": 0, "mhs_nr_dir": 16, "mhs_sun_el": 35,
    "slrm_compute": 0, "slrm_rad_cell": 20,
    "svf_compute": 0, "svf_n_dir": 16, "svf_r_max": 10, "svf_noise": 0,
    "asvf_compute": 0, "asvf_dir": 315, "asvf_level": 1,
    "pos_opns_compute": 0, "neg_opns_compute": 0,
    "sim_compute": 0, "sim_sky_mod": "overcast", "sim_compute_shadow": 0, "sim_shadow_dist": 100,
    "sim_nr_dir": 32, "sim_shadow_az": 315, "sim_shadow_el": 35,
    "ld_compute": 0, "ld_min_rad": 10, "ld_max_rad": 20, "ld_rad_inc": 1, "ld_anglr_res": 15, "ld_observer_h": 1.7,
    "msrm_compute": 0, "msrm_feature_min": 0, "msrm_feature_max": 20, "msrm_scaling_factor": 2,
    "mstp_compute": 0, "mstp_local_scale": (1, 5, 1), "mstp_meso_scale": (5, 50, 5),
    "mstp_broad_scale": (50, 500, 50), "mstp_lightness": 0.9,
    "slp_save_float": 1, "hs_save_float": 1, "mhs_save_float": 1, "slrm_save_float": 1, "svf_save_float": 1,
    "neg_opns_save_float": 1, "sim_save_float": 1, "ld_save_float": 1, "msrm_save_float": 1, "mstp_save_float": 1,
    "slp_save_8bit": 0, "hs_save_8bit": 0, "mhs_save_8bit": 0, "slrm_save_8bit": 0, "svf_save_8bit": 0,
    "neg_opns_save_8bit": 0, "sim_save_8bit": 0, "ld_save_8bit": 0, "msrm_save_8bit": 0, "mstp_save_8bit": 0,
    "slp_bytscl": ("value", 0.0, 51.0), "hs_bytscl": ("value", 0.0, 1.0), "mhs_bytscl": ("value", 0.0, 1.0),
    "slrm_bytscl": ("value", -2.0, 2.0), "svf_bytscl": ("value", 0.6375, 1.0),
    "asvf_bytscl": ("value", 0.70, 0.90), "pos_opns_bytscl": ("value", 60.0, 95.0),
    "neg_opns_bytscl": ("value", 60.0, 95.0), "sim_bytscl": ("percent", 0.25, 0.0),
    "ld_bytscl": ("value", 0.5, 1.8), "msrm_bytscl": ("value", -2.5, 2.5), "mstp_bytscl": ("value", 0.0, 1.0),
    "tile_size_limit": 100000000, "tile_size": (4000, 4000),
}


def test_default_attribute_names_are_exactly_the_documented_table():
    assert sorted(vars(DefaultValues())) == sorted(EXPECTED_DEFAULTS)


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULTS))
def test_default_attribute_value_and_type(name):
    value = getattr(DefaultValues(), name)
    expected = EXPECTED_DEFAULTS[name]
    assert value == expected
    assert type(value) is type(expected)
    if isinstance(expected, tuple):
        assert [type(x) for x in value] == [type(x) for x in expected]


def test_instances_do_not_share_state():
    a, b = DefaultValues(), DefaultValues()
    a.hs_sun_azi = 1
    a.mstp_local_scale = (9, 9, 9)
    assert b.hs_sun_azi == 315
    assert b.mstp_local_scale == (1, 5, 1)


def test_class_docstring_attributes_all_exist():
    names = re.findall(r"^    (\w+) : ", DefaultValues.__doc__, flags=re.M)
    assert len(names) > 60
    missing = [n for n in names if n not in vars(DefaultValues())]
    assert missing == []


def test_rvt_visualization_members():
    assert {m.name: m.value for m in V} == {
        "SLOPE": "slp", "HILLSHADE": "hs", "SHADOW": "shd", "MULTI_HILLSHADE": "mhs",
        "SIMPLE_LOCAL_RELIEF_MODEL": "slrm", "SKY_VIEW_FACTOR": "svf", "ANISOTROPIC_SKY_VIEW_FACTOR": "asvf",
        "POSITIVE_OPENNESS": "pos_opns", "NEGATIVE_OPENNESS": "neg_opns", "SKY_ILLUMINATION": "sim",
        "LOCAL_DOMINANCE": "ld", "MULTI_SCALE_RELIEF_MODEL": "msrm", "MULTI_SCALE_TOPOGRAPHIC_POSITION": "mstp"}


# ----------------------------------------------------------------------------------------------------------------------
# settings files
# ----------------------------------------------------------------------------------------------------------------------

def _changed_copy():
    """DefaultValues in which every attribute differs from its default (rule: +1, +0.5, other string, ...)."""
    dv = DefaultValues()
    for name, value in sorted(vars(dv).items()):
        if isinstance(value, str):
            setattr(dv, name, {"degree": "percent", "overcast": "uniform"}[value])
        elif isinstance(value, int):
            setattr(dv, name, value + 1)
        elif isinstance(value, float):
            setattr(dv, name, value + 0.5)
        elif isinstance(value[0], str):  # bytscl: (mode, min, max)
            setattr(dv, name, ("percent", value[1] + 1.0, value[2] + 2.0))
        else:  # scale tuples and tile_size
            setattr(dv, name, tuple(x + 1 for x in value))
    return dv


# attributes that save_default_to_file does not write
NOT_SAVED = {"tile_size", "tile_size_limit"}
# attributes that the JSON reader converts with float() whatever the type that was saved
READ_AS_FLOAT = {"ve_factor", "msrm_feature_min", "msrm_feature_max"}


def test_json_round_trip_of_every_saved_attribute(tmp_path):
    src = _changed_copy()
    path = tmp_path / "settings.json"
    src.save_default_to_file(str(path))
    dst = DefaultValues()
    dst.read_default_from_file(str(path))
    for name, value in vars(src).items():
        got = getattr(dst, name)
        if name in NOT_SAVED:
            # pins current behaviour, KNOWN_ISSUES candidate: tile settings are neither saved nor read
            assert got == EXPECTED_DEFAULTS[name], name
            continue
        assert got == value, name
        expected_type = float if name in READ_AS_FLOAT else type(value)
        assert type(got) is expected_type, name


def test_json_file_structure(tmp_path):
    path = tmp_path / "settings.json"
    DefaultValues().save_default_to_file(str(path))
    data = json.loads(path.read_text())
    assert list(data) == ["default_settings"]
    groups = data["default_settings"]
    assert groups["overwrite"]["value"] == 0
    assert groups["ve_factor"]["value"] == 1
    assert groups["Hillshade"]["hs_sun_azi"]["value"] == 315
    assert groups["Hillshade"]["hs_sun_el"]["value"] == 35
    assert groups["Hillshade"]["hs_bytscl"]["mode"] == "value"
    assert groups["Hillshade"]["hs_bytscl"]["min"] == 0.0
    assert groups["Hillshade"]["hs_bytscl"]["max"] == 1.0
    assert groups["Slope gradient"]["slp_output_units"]["value"] == "degree"
    assert "description" in groups["Hillshade"]["hs_sun_azi"]


def test_json_default_path_is_settings_default_settings_json_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    DefaultValues().save_default_to_file()
    target = tmp_path / "settings" / "default_settings.json"
    assert target.is_file()
    dv = DefaultValues()
    dv.hs_sun_azi = 1
    dv.read_default_from_file(str(target))
    assert dv.hs_sun_azi == 315


TXT_SETTINGS = """\
# a comment line

Overwrite = 1
exaggeration_factor = 2.5   # trailing comment
slope_gradient = 1
hillshading = 0
sun_azimuth = 270
sun_elevation = 20
multiple_hillshading = 1
hillshade_directions = 8
simple_local_relief = 1
trend_radius = 12
sky_view_factor = 1
svf_directions = 32
search_radius = 7
remove_noise = 1
noise_removal = high
anisotropic_svf = 1
anisotropy_direction = 45
anisotropy_level = high
positive_openness = 1
negative_openness = 1
sky_illumination = 1
sky_model = uniform
max_shadow_dist = 50
local_dominance = 1
min_radius = 5
max_radius = 15
some_unknown_key = 5
"""


def test_txt_settings_are_parsed(tmp_path):
    path = tmp_path / "settings.txt"
    path.write_text(TXT_SETTINGS)
    dv = DefaultValues()
    dv.read_default_from_file(str(path))
    expected = dict(EXPECTED_DEFAULTS)
    expected.update(
        overwrite=1, ve_factor=2.5, slp_compute=1, hs_compute=0, hs_sun_azi=270, hs_sun_el=20, mhs_compute=1,
        mhs_nr_dir=8, slrm_compute=1, slrm_rad_cell=12, svf_compute=1, svf_n_dir=32, svf_r_max=7, svf_noise=3,
        asvf_compute=1, asvf_dir=45, asvf_level=2, pos_opns_compute=1, neg_opns_compute=1, sim_compute=1,
        sim_sky_mod="uniform", sim_shadow_dist=50, ld_compute=1, ld_min_rad=5, ld_max_rad=15)
    assert vars(dv) == expected
    # pins current behaviour, KNOWN_ISSUES candidate: "sun_elevation" is tested twice in the parser (hillshade
    # first, multi hillshade second), so the second branch is dead and mhs_sun_el keeps its default 35 here.
    assert dv.mhs_sun_el == 35


def test_txt_noise_removal_depends_on_the_order_of_the_lines(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text("noise_removal = high\nremove_noise = 1\n")  # remove_noise not seen yet -> value ignored
    second = tmp_path / "b.txt"
    second.write_text("remove_noise = 1\nnoise_removal = medium\n")
    third = tmp_path / "c.txt"
    third.write_text("remove_noise = 1\nnoise_removal = low\nremove_noise = 0\n")  # 0 resets to 0
    got = []
    for p in (first, second, third):
        dv = DefaultValues()
        dv.read_default_from_file(str(p))
        got.append(dv.svf_noise)
    assert got == [0, 2, 0]


def test_txt_wrong_lines_warn_and_unknown_extension_is_ignored(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("hillshading\na = b = c\nsun_azimuth = 11\n")
    dv = DefaultValues()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dv.read_default_from_file(str(path))
    messages = [str(w.message) for w in caught if "Wrong line" in str(w.message)]
    assert len(messages) == 2
    assert dv.hs_sun_azi == 11
    other = tmp_path / "settings.cfg"
    other.write_text("sun_azimuth = 22\n")
    dv.read_default_from_file(str(other))  # neither .txt nor .json: nothing is read, nothing raised
    assert dv.hs_sun_azi == 11


# ----------------------------------------------------------------------------------------------------------------------
# file and path names
# ----------------------------------------------------------------------------------------------------------------------

# (method suffix, name for dem "dem_01.tif" with the default attributes); 8-bit names insert "_8bit" before ".tif"
NAMES = [
    ("slope", "dem_01_SLOPE.tif"),
    ("hillshade", "dem_01_HS_A315_H35.tif"),
    ("multi_hillshade", "dem_01_MULTI-HS_D16_H35.tif"),
    ("slrm", "dem_01_SLRM_R20.tif"),
    ("svf", "dem_01_SVF_R10_D16.tif"),
    ("asvf", "dem_01_SVF-A_R10_D16_A315_ALlow.tif"),
    ("opns", "dem_01_OPEN-POS_R10_D16.tif"),
    ("neg_opns", "dem_01_OPEN-NEG_R10_D16.tif"),
    ("sky_illumination", "dem_01_SIM_overcast_D32_100px.tif"),
    ("local_dominance", "dem_01_LD_R_M10-20_DI1_A15_OH1.7.tif"),
    ("msrm", "dem_01_MSRM_F_M0-20_S2.tif"),
    ("mstp", "dem_01_MSTP_5_50_500_L0.9.tif"),
]
DEM_PATH = str(Path("data") / "dem_01.tif")


def _name8(name):
    return name[:-len(".tif")] + "_8bit.tif"


@pytest.mark.parametrize("suffix, name", NAMES)
def test_file_names_and_paths_with_default_attributes(suffix, name):
    dv = DefaultValues()
    file_name = getattr(dv, "get_{}_file_name".format(suffix))
    path = getattr(dv, "get_{}_path".format(suffix))
    assert file_name(DEM_PATH) == name
    assert file_name(DEM_PATH, bit8=True) == _name8(name)
    assert path(DEM_PATH) == str(Path("data") / name)
    assert path(DEM_PATH, bit8=True) == str(Path("data") / _name8(name))


def test_shadow_name_has_no_8bit_variant():
    dv = DefaultValues()
    assert dv.get_shadow_file_name(DEM_PATH) == "dem_01_shadow_A315_H35.tif"
    assert dv.get_shadow_path(DEM_PATH) == str(Path("data") / "dem_01_shadow_A315_H35.tif")


def test_names_follow_the_attributes():
    dv = small_defaults()
    dv.hs_sun_azi, dv.hs_sun_el = 100, 40
    assert dv.get_hillshade_file_name(DEM_PATH) == "dem_01_HS_A100_H40.tif"
    assert dv.get_shadow_file_name(DEM_PATH) == "dem_01_shadow_A100_H40.tif"
    assert dv.get_multi_hillshade_file_name(DEM_PATH) == "dem_01_MULTI-HS_D8_H50.tif"
    assert dv.get_slrm_file_name(DEM_PATH) == "dem_01_SLRM_R10.tif"
    assert dv.get_svf_file_name(DEM_PATH) == "dem_01_SVF_R5_D8_NRlow.tif"
    assert dv.get_asvf_file_name(DEM_PATH) == "dem_01_SVF-A_R5_D8_A45_ALhigh_NRlow.tif"
    assert dv.get_opns_file_name(DEM_PATH, bit8=True) == "dem_01_OPEN-POS_R5_D8_NRlow_8bit.tif"
    assert dv.get_neg_opns_file_name(DEM_PATH) == "dem_01_OPEN-NEG_R5_D8_NRlow.tif"
    assert dv.get_sky_illumination_file_name(DEM_PATH) == "dem_01_SIM_uniform_D8_10px.tif"
    assert dv.get_local_dominance_file_name(DEM_PATH) == "dem_01_LD_R_M3-6_DI2_A20_OH2.0.tif"
    assert dv.get_msrm_file_name(DEM_PATH) == "dem_01_MSRM_F_M2-10_S2.tif"
    assert dv.get_mstp_file_name(DEM_PATH, bit8=True) == "dem_01_MSTP_3_9_15_L0.8_8bit.tif"


@pytest.mark.parametrize("noise, svf_part, asvf_part, opns_part, neg_part", [
    (0, "", "", "", ""),
    (1, "_NRlow", "_NRlow", "_NRlow", "_NRlow"),
    (2, "_NRmedium", "_NRmedium", "_NRmedium", "_NRmedium"),
    # pins current behaviour, KNOWN_ISSUES candidate: negative openness names level 3 "NRstrong", the other three
    # visualizations name it "NRhigh"
    (3, "_NRhigh", "_NRhigh", "_NRhigh", "_NRstrong"),
    (4, "", "", "", ""),  # levels outside 0..3 add nothing
])
def test_noise_removal_suffix(noise, svf_part, asvf_part, opns_part, neg_part):
    dv = DefaultValues()
    dv.svf_noise = noise
    assert dv.get_svf_file_name("d.tif") == "d_SVF_R10_D16{}.tif".format(svf_part)
    assert dv.get_asvf_file_name("d.tif") == "d_SVF-A_R10_D16_A315_ALlow{}.tif".format(asvf_part)
    assert dv.get_opns_file_name("d.tif") == "d_OPEN-POS_R10_D16{}.tif".format(opns_part)
    assert dv.get_neg_opns_file_name("d.tif") == "d_OPEN-NEG_R10_D16{}.tif".format(neg_part)


@pytest.mark.parametrize("level, part", [(1, "_ALlow"), (2, "_ALhigh"), (0, ""), (3, "")])
def test_asvf_level_suffix(level, part):
    dv = DefaultValues()
    dv.asvf_level = level
    assert dv.get_asvf_file_name("d.tif") == "d_SVF-A_R10_D16_A315{}.tif".format(part)


def test_dem_name_is_cut_at_the_first_dot():
    # pins current behaviour, KNOWN_ISSUES candidate: dem_name = basename.split(".")[0], so every dot of the base
    # name and not only the extension is dropped ("my.site_v2.tif" gives "my")
    dv = DefaultValues()
    assert dv.get_slope_file_name("my.site_v2.tif") == "my_SLOPE.tif"
    assert dv.get_slope_file_name(str(Path("a") / "b" / "plain.tiff")) == "plain_SLOPE.tif"


def test_path_is_normalised_and_joined_to_the_dem_directory():
    dv = DefaultValues()
    messy = "data/./sub/../dem_01.tif"
    assert dv.get_slope_path(messy) == str(Path("data") / "dem_01_SLOPE.tif")
    assert dv.get_slope_path("dem_01.tif") == "dem_01_SLOPE.tif"  # no directory: bare name


@pytest.mark.parametrize("member, name", [
    (V.SLOPE, "dem_01_SLOPE.tif"),
    (V.HILLSHADE, "dem_01_HS_A315_H35.tif"),
    (V.SHADOW, "dem_01_shadow_A315_H35.tif"),
    (V.MULTI_HILLSHADE, "dem_01_MULTI-HS_D16_H35.tif"),
    (V.SIMPLE_LOCAL_RELIEF_MODEL, "dem_01_SLRM_R20.tif"),
    (V.SKY_VIEW_FACTOR, "dem_01_SVF_R10_D16.tif"),
    (V.ANISOTROPIC_SKY_VIEW_FACTOR, "dem_01_SVF-A_R10_D16_A315_ALlow.tif"),
    (V.POSITIVE_OPENNESS, "dem_01_OPEN-POS_R10_D16.tif"),
    (V.NEGATIVE_OPENNESS, "dem_01_OPEN-NEG_R10_D16.tif"),
    (V.SKY_ILLUMINATION, "dem_01_SIM_overcast_D32_100px.tif"),
    (V.LOCAL_DOMINANCE, "dem_01_LD_R_M10-20_DI1_A15_OH1.7.tif"),
    (V.MULTI_SCALE_RELIEF_MODEL, "dem_01_MSRM_F_M0-20_S2.tif"),
    (V.MULTI_SCALE_TOPOGRAPHIC_POSITION, "dem_01_MSTP_5_50_500_L0.9.tif"),
])
def test_visualization_file_name_and_path_by_member(member, name):
    dv = DefaultValues()
    out_dir = Path("out") / "vis"
    name8 = name if member == V.SHADOW else _name8(name)  # shadow: no 8-bit name
    assert dv.get_visualization_file_name(member, Path(DEM_PATH), False) == name
    assert dv.get_visualization_file_name(member, Path(DEM_PATH), True) == name8
    assert dv.get_visualization_path(member, Path(DEM_PATH), out_dir, False) == out_dir / name
    assert dv.get_visualization_path(member, Path(DEM_PATH), out_dir, True) == out_dir / name8


def test_visualization_name_of_unknown_member_is_none():
    dv = DefaultValues()
    assert dv.get_visualization_file_name("slp", Path(DEM_PATH), False) is None
    assert dv.get_visualization_path("slp", Path(DEM_PATH), Path("o"), False) is None


# ----------------------------------------------------------------------------------------------------------------------
# get_* compute methods are thin wrappers of rvt.vis
# ----------------------------------------------------------------------------------------------------------------------

RES_X, RES_Y = 2.0, 3.0  # unequal on purpose: wrong x/y wiring shows

# Each case: (id, wrapper call, direct call of rvt.vis with the literal parameters of small_defaults(),
#             dtype, shape rule, NaN rule). Both calls receive their own copy of the DEM.
# shape rules: "full" = (H, W); "crop1" = (n, H-2, W-2); "bands3" = (3, H, W)
CASES = [
    ("slope",
     lambda dv, d: dv.get_slope(d, RES_X, RES_Y),
     lambda d: rvt.vis.slope_aspect(d, resolution_x=RES_X, resolution_y=RES_Y, output_units="percent",
                                    ve_factor=2)["slope"],
     np.float32, "full", "same"),
    ("shadow",
     lambda dv, d: dv.get_shadow(d, RES_X),
     lambda d: rvt.vis.shadow_horizon(d, RES_X, shadow_az=100, shadow_el=40, ve_factor=2)["shadow"],
     None, "full", "none"),
    ("hillshade",
     lambda dv, d: dv.get_hillshade(d, RES_X, RES_Y),
     lambda d: rvt.vis.hillshade(d, RES_X, RES_Y, sun_azimuth=100, sun_elevation=40, ve_factor=2),
     np.float64, "full", "same"),
    ("multi_hillshade",
     lambda dv, d: dv.get_multi_hillshade(d, RES_X, RES_Y),
     lambda d: rvt.vis.multi_hillshade(d, RES_X, RES_Y, nr_directions=8, sun_elevation=50, ve_factor=2),
     np.float64, "crop1", "any"),
    ("slrm",
     lambda dv, d: dv.get_slrm(d),
     lambda d: rvt.vis.slrm(d, radius_cell=10, ve_factor=2),
     np.float32, "full", "same"),
    ("svf",
     lambda dv, d: dv.get_sky_view_factor(d, RES_X)["svf"],
     lambda d: rvt.vis.sky_view_factor(d, RES_X, compute_svf=True, compute_opns=False, compute_asvf=False,
                                       svf_n_dir=8, svf_r_max=5, svf_noise=1, asvf_dir=45, asvf_level=2,
                                       ve_factor=2)["svf"],
     np.float64, "full", "same"),
    ("asvf",
     lambda dv, d: dv.get_sky_view_factor(d, RES_X, compute_svf=False, compute_asvf=True)["asvf"],
     lambda d: rvt.vis.sky_view_factor(d, RES_X, compute_svf=False, compute_opns=False, compute_asvf=True,
                                       svf_n_dir=8, svf_r_max=5, svf_noise=1, asvf_dir=45, asvf_level=2,
                                       ve_factor=2)["asvf"],
     np.float64, "full", "same"),
    ("opns",
     lambda dv, d: dv.get_sky_view_factor(d, RES_X, compute_svf=False, compute_opns=True)["opns"],
     lambda d: rvt.vis.sky_view_factor(d, RES_X, compute_svf=False, compute_opns=True, compute_asvf=False,
                                       svf_n_dir=8, svf_r_max=5, svf_noise=1, asvf_dir=45, asvf_level=2,
                                       ve_factor=2)["opns"],
     np.float64, "full", "same"),
    # negative openness = positive openness of the negated DEM
    ("neg_opns",
     lambda dv, d: dv.get_neg_opns(d, RES_X),
     lambda d: rvt.vis.sky_view_factor(-d, RES_X, compute_svf=False, compute_opns=True, compute_asvf=False,
                                       svf_n_dir=8, svf_r_max=5, svf_noise=1, ve_factor=2)["opns"],
     np.float64, "full", "same"),
    ("local_dominance",
     lambda dv, d: dv.get_local_dominance(d),
     lambda d: rvt.vis.local_dominance(d, min_rad=3, max_rad=6, rad_inc=2, angular_res=20, observer_height=2.0,
                                       ve_factor=2),
     np.float64, "full", "same"),
    ("msrm",
     lambda dv, d: dv.get_msrm(d, RES_X),
     lambda d: rvt.vis.msrm(d, RES_X, feature_min=2, feature_max=10, scaling_factor=2, ve_factor=2),
     np.float64, "full", "same"),
    # pins current behaviour, KNOWN_ISSUES candidate: get_mstp does not pass ve_factor to rvt.vis.mstp (it is
    # set to 2 in small_defaults() and the direct call leaves it at its default 1)
    ("mstp",
     lambda dv, d: dv.get_mstp(d),
     lambda d: rvt.vis.mstp(d, local_scale=(1, 3, 1), meso_scale=(3, 9, 3), broad_scale=(9, 15, 3), lightness=0.8),
     np.float32, "bands3", "same"),
]
SIM_CASE = (
    lambda dv, d: dv.get_sky_illumination(d, RES_X),
    lambda d: rvt.vis.sky_illumination(d, RES_X, sky_model="uniform", compute_shadow=True, max_fine_radius=10,
                                       num_directions=8, shadow_az=200, shadow_el=20, ve_factor=2),
)


def _check_wrapper(case, dem_arr):
    _, wrapper, direct, dtype, shape_rule, nan_rule = case
    dv = small_defaults()
    got = wrapper(dv, dem_arr.copy())
    want = direct(dem_arr.copy())
    assert_bitwise(got, want)  # same shape, dtype and bytes: DefaultValues only forwards its attributes
    n_rows, n_cols = dem_arr.shape
    if shape_rule == "full":
        assert got.shape == (n_rows, n_cols)
    elif shape_rule == "crop1":
        assert got.shape == (8, n_rows - 2, n_cols - 2)  # mhs_nr_dir = 8, one pixel lost on each edge
    else:
        assert got.shape == (3, n_rows, n_cols)
    if dtype is not None:
        assert got.dtype == dtype
    else:
        assert got.dtype.kind in "iuf"
    if nan_rule == "same":
        mask = np.isnan(dem_arr)
        assert mask.any()
        got2 = got if got.ndim == 2 else got
        assert np.array_equal(np.isnan(got2), np.broadcast_to(mask, got2.shape))
    elif nan_rule == "any":
        assert np.isnan(got).any()
    else:  # "none": integer result, no NaN
        assert not np.isnan(got.astype(np.float64)).any()


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_get_method_is_a_wrapper_of_the_vis_function(case, dem):
    _check_wrapper(case, dem)


def test_get_sky_illumination_is_a_wrapper(dem_clean):
    dv = small_defaults()
    got = dv.get_sky_illumination(dem_clean.copy(), RES_X)
    want = SIM_CASE[1](dem_clean.copy())
    assert_bitwise(got, want)
    assert got.shape == dem_clean.shape and got.dtype == np.float64
    assert not np.isnan(got).any()


def test_sky_illumination_spreads_a_single_nan_over_most_of_the_output(dem_clean):
    # pins current behaviour, KNOWN_ISSUES candidate: one NaN pixel in the DEM turns far more than that pixel into
    # NaN (here 1 input pixel of 2688 gives more than half of the output NaN; the exact count is not pinned)
    dv = small_defaults()
    arr = dem_clean.copy()
    arr[20, 20] = np.nan
    got = dv.get_sky_illumination(arr, RES_X)
    assert np.isnan(got[20, 20])
    assert np.isnan(got).mean() > 0.5


def test_get_sky_view_factor_returns_only_the_requested_keys(dem_clean):
    dv = small_defaults()
    assert sorted(dv.get_sky_view_factor(dem_clean.copy(), RES_X)) == ["svf"]
    got = dv.get_sky_view_factor(dem_clean.copy(), RES_X, compute_svf=True, compute_asvf=True, compute_opns=True)
    assert sorted(got) == ["asvf", "opns", "svf"]
    for key in got:
        assert got[key].shape == dem_clean.shape


def test_no_data_value_becomes_nan_in_the_output(dem_clean):
    arr = dem_clean.copy()
    arr[5, 7] = NODATA
    dv = small_defaults()
    for out in (dv.get_slope(arr.copy(), RES_X, RES_Y, no_data=NODATA),
                dv.get_hillshade(arr.copy(), RES_X, RES_Y, no_data=NODATA),
                dv.get_slrm(arr.copy(), no_data=NODATA),
                dv.get_msrm(arr.copy(), RES_X, no_data=NODATA),
                dv.get_local_dominance(arr.copy(), no_data=NODATA)):
        assert np.isnan(out[5, 7])
        assert np.count_nonzero(np.isnan(out)) == 1


def test_neg_opns_does_not_recognise_no_data(dem_clean):
    # pins current behaviour, KNOWN_ISSUES candidate: get_neg_opns negates the DEM before calling rvt.vis, so a
    # no_data value of -9999 becomes +9999 and no longer matches no_data: the pixel is not turned into NaN (it is
    # processed as a 9999 m peak). The positive openness of the same DEM does mask it.
    arr = dem_clean.copy()
    arr[5, 7] = NODATA
    dv = small_defaults()
    neg = dv.get_neg_opns(arr.copy(), RES_X, no_data=NODATA)
    pos = dv.get_sky_view_factor(arr.copy(), RES_X, compute_svf=False, compute_opns=True, no_data=NODATA)["opns"]
    assert np.isnan(pos[5, 7])
    assert not np.isnan(neg[5, 7])


# ----------------------------------------------------------------------------------------------------------------------
# float_to_8bit against a plain reference of the byte scale
# ----------------------------------------------------------------------------------------------------------------------

def ref_8bit(values, lo, hi, invert=False):
    """Reference of float_to_8bit for 'value' mode, written from the definition: clip to [lo, hi], scale to 0..1,
    optionally invert (slope, negative openness: steep = dark), scale by 255.9999 and truncate (IDL BYTSCL rule);
    NaN becomes 255."""
    v = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
    n = (v - lo) / (hi - lo)
    if invert:
        n = 1.0 - n
    b = np.clip(255.9999 * n, 0.0, 255.0)
    b[np.isnan(b)] = 255.0
    return np.floor(b).astype(np.uint8)


# fractions of the stretch range; < 0 and > 1 test the cut-off; fractional parts of 255.9999 * f stay 0.2..0.8 away
# from an integer so float32 rounding cannot flip the truncation
FRACTIONS = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.95, 1.0, 1.4, -0.4, np.nan])


@pytest.mark.parametrize("member, attr, invert", [
    (V.SLOPE, "slp_bytscl", True),
    (V.HILLSHADE, "hs_bytscl", False),
    (V.SIMPLE_LOCAL_RELIEF_MODEL, "slrm_bytscl", False),
    (V.SKY_VIEW_FACTOR, "svf_bytscl", False),
    (V.ANISOTROPIC_SKY_VIEW_FACTOR, "asvf_bytscl", False),
    (V.POSITIVE_OPENNESS, "pos_opns_bytscl", False),
    (V.NEGATIVE_OPENNESS, "neg_opns_bytscl", True),
    (V.LOCAL_DOMINANCE, "ld_bytscl", False),
    (V.MULTI_SCALE_RELIEF_MODEL, "msrm_bytscl", False),
    (V.MULTI_SCALE_TOPOGRAPHIC_POSITION, "mstp_bytscl", False),
])
def test_float_to_8bit_value_mode(member, attr, invert):
    dv = DefaultValues()
    mode, lo, hi = getattr(dv, attr)
    assert mode == "value"
    arr = (lo + FRACTIONS * (hi - lo)).astype(np.float32).reshape(3, 4)
    got = dv.float_to_8bit(arr.copy(), member)
    assert got.dtype == np.uint8 and got.shape == (3, 4)
    assert np.array_equal(got, ref_8bit(arr, lo, hi, invert))
    assert got.reshape(-1)[-1] == 255  # NaN


def test_float_to_8bit_uses_the_current_stretch_attributes():
    dv = DefaultValues()
    dv.slp_bytscl = ("value", 10.0, 40.0)
    arr = (10.0 + FRACTIONS * 30.0).astype(np.float32).reshape(3, 4)
    assert np.array_equal(dv.float_to_8bit(arr.copy(), V.SLOPE), ref_8bit(arr, 10.0, 40.0, invert=True))


def test_float_to_8bit_percent_mode():
    dv = DefaultValues()  # sim_bytscl = ("percent", 0.25, 0.0): cut 0.25 % at the low end, 0 % at the high end
    arr = (np.arange(1, 201, dtype=np.float32) / 10.0).reshape(10, 20)
    lo, hi = np.nanpercentile(arr, [0.25, 100.0])
    got = dv.float_to_8bit(arr.copy(), V.SKY_ILLUMINATION)
    want = ref_8bit(arr, lo, hi)
    assert got.dtype == np.uint8
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1  # float32 vs float64 percentile arithmetic
    assert got.min() == 0 and got.max() == 255


def test_float_to_8bit_shadow_is_returned_unchanged():
    dv = DefaultValues()
    arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert dv.float_to_8bit(arr, V.SHADOW) is arr


def test_float_to_8bit_modifies_its_argument():
    # pins current behaviour, KNOWN_ISSUES candidate: the cut-off of the stretch is applied to the caller's array
    dv = DefaultValues()
    arr = np.array([[0.0, 30.0, 60.0, 90.0]], dtype=np.float32)
    dv.float_to_8bit(arr, V.SLOPE)
    assert arr.tolist() == [[0.0, 30.0, 51.0, 51.0]]


def test_float_to_8bit_rejects_unknown_visualization():
    with pytest.raises(Exception, match="Wrong visualization"):
        DefaultValues().float_to_8bit(np.zeros((2, 2), np.float32), "slp")


def test_float_to_8bit_multi_hillshade_from_the_dem(dem_clean):
    dv = DefaultValues()
    dv.ve_factor = 2  # pins current behaviour, KNOWN_ISSUES candidate: ignored by the 8-bit multi hillshade
    got = dv.float_to_8bit(dem_clean.copy(), V.MULTI_HILLSHADE, x_res=RES_X, y_res=RES_Y, no_data=None)
    bands = [rvt.vis.hillshade(dem_clean.copy(), RES_X, RES_Y, sun_azimuth=az, sun_elevation=35)
             for az in (315, 22.5, 90)]
    want = np.array([ref_8bit(b, 0.0, 1.0) for b in bands])
    assert got.shape == (3,) + dem_clean.shape
    assert got.dtype == np.uint8
    assert np.array_equal(got, want)


# ----------------------------------------------------------------------------------------------------------------------
# calculate_visualization
# ----------------------------------------------------------------------------------------------------------------------

# member, wrapper returning the float visualization (small_defaults, no_data None), 8-bit shape rule
CALC = {
    V.SLOPE: lambda dv, d: dv.get_slope(d, RES_X, RES_Y),
    V.SHADOW: lambda dv, d: dv.get_shadow(d, RES_X),
    V.HILLSHADE: lambda dv, d: dv.get_hillshade(d, RES_X, RES_Y),
    V.MULTI_HILLSHADE: lambda dv, d: dv.get_multi_hillshade(d, RES_X, RES_Y),
    V.SIMPLE_LOCAL_RELIEF_MODEL: lambda dv, d: dv.get_slrm(d),
    V.SKY_VIEW_FACTOR: lambda dv, d: dv.get_sky_view_factor(d, RES_X, True, False, False)["svf"],
    V.ANISOTROPIC_SKY_VIEW_FACTOR: lambda dv, d: dv.get_sky_view_factor(d, RES_X, False, True, False)["asvf"],
    V.POSITIVE_OPENNESS: lambda dv, d: dv.get_sky_view_factor(d, RES_X, False, False, True)["opns"],
    V.NEGATIVE_OPENNESS: lambda dv, d: dv.get_neg_opns(d, RES_X),
    V.SKY_ILLUMINATION: lambda dv, d: dv.get_sky_illumination(d, RES_X),
    V.LOCAL_DOMINANCE: lambda dv, d: dv.get_local_dominance(d),
    V.MULTI_SCALE_RELIEF_MODEL: lambda dv, d: dv.get_msrm(d, RES_X),
    V.MULTI_SCALE_TOPOGRAPHIC_POSITION: lambda dv, d: dv.get_mstp(d),
}


def test_calc_table_covers_every_member():
    assert set(CALC) == set(V)


@pytest.mark.parametrize("member", list(V), ids=[m.name for m in V])
def test_calculate_visualization_float_only(member, dem_clean):
    dv = small_defaults()
    arr = dem_clean  # NaN free: sky illumination needs it
    vis_float, vis_8bit = dv.calculate_visualization(member, arr.copy(), RES_X, RES_Y, None, True, False)
    assert vis_8bit is None
    assert_bitwise(vis_float, CALC[member](dv, arr.copy()))


@pytest.mark.parametrize("member", list(V), ids=[m.name for m in V])
def test_calculate_visualization_8bit_only(member, dem_clean):
    dv = small_defaults()
    arr = dem_clean
    vis_float, vis_8bit = dv.calculate_visualization(member, arr.copy(), RES_X, RES_Y, None, False, True)
    assert vis_float is None
    float_vis = CALC[member](dv, arr.copy())
    if member == V.SHADOW:
        assert_bitwise(vis_8bit, float_vis)  # shadow has no 8-bit conversion
        return
    if member == V.MULTI_HILLSHADE:
        want = dv.float_to_8bit(arr.copy(), member, x_res=RES_X, y_res=RES_Y, no_data=None)
    else:
        want = dv.float_to_8bit(float_vis.copy(), member)
    assert vis_8bit.dtype == np.uint8
    assert_bitwise(vis_8bit, want)
    expected_shape = {V.MULTI_HILLSHADE: (3,) + tuple(np.subtract(arr.shape, 2)),
                      V.MULTI_SCALE_TOPOGRAPHIC_POSITION: (3,) + arr.shape}.get(member, arr.shape)
    if member == V.MULTI_HILLSHADE:
        # the 8-bit multi hillshade is three hillshades (315, 22.5, 90 deg) of the full DEM, not the cropped result
        expected_shape = (3,) + arr.shape
    assert vis_8bit.shape == expected_shape


def test_calculate_visualization_neither_flag_gives_two_nones(dem_clean):
    dv = small_defaults()
    assert dv.calculate_visualization(V.SLOPE, dem_clean.copy(), RES_X, RES_Y, None, False, False) == (None, None)


def test_calculate_visualization_unknown_member(dem_clean):
    dv = small_defaults()
    assert dv.calculate_visualization("slp", dem_clean.copy(), RES_X, RES_Y, None, True, False) == (None, None)
    with pytest.raises(Exception, match="Wrong visualization"):
        dv.calculate_visualization("slp", dem_clean.copy(), RES_X, RES_Y, None, True, True)


def test_calculate_visualization_with_both_flags_clips_the_float_result(dem_clean):
    # pins current behaviour, KNOWN_ISSUES candidate: the float result is the same array object that float_to_8bit
    # cuts off in place, so with save_float and save_8bit both true the returned float slope is limited to the
    # 8-bit stretch range (0..51 degrees) and differs from the float-only result wherever the slope is steeper
    dv = small_defaults()
    dv.slp_output_units = "degree"
    only_float, _ = dv.calculate_visualization(V.SLOPE, dem_clean.copy(), RES_X, RES_Y, None, True, False)
    both_float, both_8bit = dv.calculate_visualization(V.SLOPE, dem_clean.copy(), RES_X, RES_Y, None, True, True)
    assert only_float.max() > 51.0  # the test DEM has slopes above the stretch range
    assert both_float.max() <= 51.0
    assert np.array_equal(both_float, np.minimum(only_float, np.float32(51.0)))
    assert both_8bit.dtype == np.uint8


# ----------------------------------------------------------------------------------------------------------------------
# save_* on a small GeoTIFF
# ----------------------------------------------------------------------------------------------------------------------

def _expected_float(case_key, dv, arr):
    """Float visualization of the stored DEM (2 m/px in x and y, no-data -9999) through the get_* methods, as the
    save_* method should write it."""
    d = arr.copy()
    nd = NODATA
    table = {
        "slope": lambda: dv.get_slope(d, 2.0, 2.0, no_data=nd),
        "hillshade": lambda: dv.get_hillshade(d, 2.0, 2.0, no_data=nd),
        "multi_hillshade": lambda: dv.get_multi_hillshade(d, 2.0, 2.0, no_data=nd),
        "slrm": lambda: dv.get_slrm(d, no_data=nd),
        "neg_opns": lambda: dv.get_neg_opns(d, 2.0, no_data=nd),
        "sky_illumination": lambda: dv.get_sky_illumination(d, 2.0, no_data=nd),
        "local_dominance": lambda: dv.get_local_dominance(d, no_data=nd),
        "msrm": lambda: dv.get_msrm(d, 2.0, no_data=nd),
        "mstp": lambda: dv.get_mstp(d, no_data=nd),
    }
    return table[case_key]().astype(np.float32)


# (method, key, float file name, 8-bit file name, float bands, 8-bit bands) for small_defaults()
SAVE_CASES = [
    ("save_slope", "slope", "dem_SLOPE.tif", "dem_SLOPE_8bit.tif", 1, 1),
    ("save_hillshade", "hillshade", "dem_HS_A100_H40.tif", "dem_HS_A100_H40_8bit.tif", 1, 1),
    ("save_multi_hillshade", "multi_hillshade", "dem_MULTI-HS_D8_H50.tif", "dem_MULTI-HS_D8_H50_8bit.tif", 8, 3),
    ("save_slrm", "slrm", "dem_SLRM_R10.tif", "dem_SLRM_R10_8bit.tif", 1, 1),
    ("save_neg_opns", "neg_opns", "dem_OPEN-NEG_R5_D8_NRlow.tif", "dem_OPEN-NEG_R5_D8_NRlow_8bit.tif", 1, 1),
    ("save_sky_illumination", "sky_illumination", "dem_SIM_uniform_D8_10px.tif",
     "dem_SIM_uniform_D8_10px_8bit.tif", 1, 1),
    ("save_local_dominance", "local_dominance", "dem_LD_R_M3-6_DI2_A20_OH2.0.tif",
     "dem_LD_R_M3-6_DI2_A20_OH2.0_8bit.tif", 1, 1),
    ("save_msrm", "msrm", "dem_MSRM_F_M2-10_S2.tif", "dem_MSRM_F_M2-10_S2_8bit.tif", 1, 1),
    ("save_mstp", "mstp", "dem_MSTP_3_9_15_L0.8.tif", "dem_MSTP_3_9_15_L0.8_8bit.tif", 3, 3),
]


def _make_dem_file(directory, dem_clean):
    directory.mkdir(parents=True, exist_ok=True)
    arr = dem_clean.copy()
    arr[5, 7] = NODATA
    return write_geotiff(directory / "dem.tif", arr, nodata=NODATA), arr


@pytest.mark.parametrize("method, key, float_name, name_8bit, n_float, n_8bit", SAVE_CASES,
                         ids=[c[0] for c in SAVE_CASES])
def test_save_method_writes_float_and_8bit_files(method, key, float_name, name_8bit, n_float, n_8bit,
                                                 tmp_path, dem_clean):
    dem_path, arr = _make_dem_file(tmp_path / "in", dem_clean)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    dv = small_defaults()
    assert getattr(dv, method)(str(dem_path), custom_dir=str(out_dir), save_float=True, save_8bit=True) == 1
    assert sorted(p.name for p in out_dir.iterdir()) == sorted([float_name, name_8bit])

    data, info = read_raster(out_dir / float_name)
    assert info["count"] == n_float
    assert set(info["dtypes"]) == {"float32"}
    assert info["transform"] == GEOTRANSFORM  # also for the multi hillshade: cropped array, same origin
    assert info["epsg"] == 32633
    assert np.isnan(info["nodata"])
    want = _expected_float(key, dv, arr)
    assert data.shape == (want.shape if want.ndim == 3 else (1,) + want.shape)
    assert np.array_equal(data.reshape(want.shape), want, equal_nan=True)
    masked = key not in ("multi_hillshade", "neg_opns")  # mhs: pixel lost in the crop; neg_opns: see its own test
    if masked:
        assert np.isnan(data[0, 5, 7])  # the no-data pixel is NaN in the float output

    data8, info8 = read_raster(out_dir / name_8bit)
    assert info8["count"] == n_8bit
    assert set(info8["dtypes"]) == {"uint8"}
    assert info8["transform"] == GEOTRANSFORM
    assert info8["epsg"] == 32633
    if masked:
        assert data8[0, 5, 7] == 255  # NaN becomes 255 in the byte scale
    assert data8.max() <= 255 and data8.min() >= 0


@pytest.mark.parametrize("method, key, float_name, name_8bit, n_float, n_8bit", SAVE_CASES[:3],
                         ids=[c[0] for c in SAVE_CASES[:3]])
def test_save_method_without_custom_dir_writes_next_to_the_dem(method, key, float_name, name_8bit, n_float, n_8bit,
                                                               tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    dv = small_defaults()
    assert getattr(dv, method)(str(dem_path), save_float=True, save_8bit=False) == 1
    assert sorted(p.name for p in (tmp_path / "in").iterdir()) == sorted(["dem.tif", float_name])


def test_save_defaults_come_from_the_attributes(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    dv = small_defaults()  # slp_save_float = 1, slp_save_8bit = 0
    dv.save_slope(str(dem_path), custom_dir=str(tmp_path))
    assert sorted(p.name for p in tmp_path.glob("*.tif")) == ["dem_SLOPE.tif"]
    dv.slp_save_float, dv.slp_save_8bit = 0, 1
    dv.save_slope(str(dem_path), custom_dir=str(tmp_path))
    assert sorted(p.name for p in tmp_path.glob("*.tif")) == ["dem_SLOPE.tif", "dem_SLOPE_8bit.tif"]


def test_save_existing_files_are_kept_unless_overwrite(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    out = tmp_path / "out"
    out.mkdir()
    dv = small_defaults()
    assert dv.save_slope(str(dem_path), custom_dir=str(out), save_float=True, save_8bit=True) == 1
    target = out / "dem_SLOPE.tif"
    first = target.read_bytes()
    target.write_bytes(first + b"\0")  # marker: a rewrite would remove it
    assert dv.save_slope(str(dem_path), custom_dir=str(out), save_float=True, save_8bit=True) == 0
    assert target.read_bytes() == first + b"\0"
    dv.overwrite = 1
    assert dv.save_slope(str(dem_path), custom_dir=str(out), save_float=True, save_8bit=True) == 1
    assert target.read_bytes() != first + b"\0"


def test_save_errors(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    dv = small_defaults()
    with pytest.raises(Exception, match="Both save_float and save_8bit are False"):
        dv.save_slope(str(dem_path), save_float=False, save_8bit=False)
    with pytest.raises(Exception, match="dem_path doesn't exist"):
        dv.save_slope(str(tmp_path / "missing.tif"), save_float=True, save_8bit=False)
    with pytest.raises(Exception, match="dem_path doesn't exist"):
        dv.save_hillshade(str(tmp_path / "missing.tif"))


def test_save_sky_view_factor_writes_the_requested_files(tmp_path, dem_clean):
    dem_path, arr = _make_dem_file(tmp_path / "in", dem_clean)
    out = tmp_path / "out"
    out.mkdir()
    dv = small_defaults()
    assert dv.save_sky_view_factor(str(dem_path), save_svf=True, save_asvf=True, save_opns=True,
                                   custom_dir=str(out), save_float=True, save_8bit=True) == 1
    names = {"dem_SVF_R5_D8_NRlow.tif", "dem_SVF-A_R5_D8_A45_ALhigh_NRlow.tif", "dem_OPEN-POS_R5_D8_NRlow.tif"}
    assert sorted(p.name for p in out.iterdir()) == sorted(names | {_name8(n) for n in names})
    want = dv.get_sky_view_factor(arr.copy(), 2.0, compute_svf=True, compute_asvf=True, compute_opns=True,
                                  no_data=NODATA)
    for name, key in (("dem_SVF_R5_D8_NRlow.tif", "svf"), ("dem_SVF-A_R5_D8_A45_ALhigh_NRlow.tif", "asvf"),
                      ("dem_OPEN-POS_R5_D8_NRlow.tif", "opns")):
        data, info = read_raster(out / name)
        assert info["dtypes"] == ("float32",) and info["transform"] == GEOTRANSFORM and np.isnan(info["nodata"])
        assert np.array_equal(data[0], want[key].astype(np.float32), equal_nan=True)
        data8, info8 = read_raster(out / _name8(name))
        assert info8["dtypes"] == ("uint8",) and info8["transform"] == GEOTRANSFORM


def test_save_sky_view_factor_default_writes_only_svf(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    dv = small_defaults()
    dv.save_sky_view_factor(str(dem_path), custom_dir=str(tmp_path))
    assert sorted(p.name for p in tmp_path.glob("*.tif")) == ["dem_SVF_R5_D8_NRlow.tif"]


def test_save_hillshade_with_shadow(tmp_path, dem_clean):
    dem_path = write_geotiff(tmp_path / "dem.tif", dem_clean, nodata=NODATA)  # no pixel carries the no-data value
    dv = small_defaults()
    assert dv.save_hillshade(str(dem_path), save_float=True, save_8bit=False, save_shadow=True) == 1
    data, info = read_raster(tmp_path / "dem_shadow_A100_H40.tif")
    assert info["dtypes"] == ("float32",) and info["transform"] == GEOTRANSFORM and np.isnan(info["nodata"])
    want = dv.get_shadow(dem_clean.copy(), 2.0).astype(np.float32)
    assert want.any()
    assert np.array_equal(data[0], want)


def test_shadow_of_a_dem_with_nan_is_all_zero(dem):
    # pins current behaviour, KNOWN_ISSUES candidate: one NaN anywhere in the DEM makes rvt.vis.shadow_horizon
    # return a shadow raster of zeros only (int64), also in DefaultValues.get_shadow
    got = small_defaults().get_shadow(dem.copy(), RES_X)
    assert not got.any()


def test_save_hillshade_shadow_is_all_zero_when_the_dem_has_no_data(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)  # one pixel holds the no-data value -9999
    dv = small_defaults()
    dv.save_hillshade(str(dem_path), save_float=True, save_8bit=False, save_shadow=True)
    data, _ = read_raster(tmp_path / "in" / "dem_shadow_A100_H40.tif")
    # pins current behaviour, KNOWN_ISSUES candidate: get_hillshade replaces the no-data value by NaN in the array
    # that save_hillshade then passes to get_shadow, and a NaN gives an all-zero shadow (see the test above)
    assert not data.any()


def test_get_hillshade_replaces_no_data_in_the_callers_array(dem_clean):
    arr = dem_clean.copy()
    arr[5, 7] = NODATA
    small_defaults().get_hillshade(arr, RES_X, RES_Y, no_data=NODATA)
    # pins current behaviour (documented in docs/CONVENTIONS.md): the caller's array is modified
    assert np.isnan(arr[5, 7])


def test_save_hillshade_shadow_ignores_custom_dir(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    out = tmp_path / "out"
    out.mkdir()
    dv = small_defaults()
    dv.save_hillshade(str(dem_path), custom_dir=str(out), save_float=True, save_8bit=False, save_shadow=True)
    # pins current behaviour, KNOWN_ISSUES candidate: the shadow path is os.path.join(custom_dir, get_shadow_path(..))
    # and get_shadow_path already holds the DEM directory, so (absolute second argument wins) the shadow is written
    # next to the DEM and not into custom_dir
    assert sorted(p.name for p in out.iterdir()) == ["dem_HS_A100_H40.tif"]
    assert (tmp_path / "in" / "dem_shadow_A100_H40.tif").is_file()


def test_save_neg_opns_does_not_mask_no_data(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    dv = small_defaults()
    dv.save_neg_opns(str(dem_path), custom_dir=str(tmp_path), save_float=True, save_8bit=False)
    data, _ = read_raster(tmp_path / "dem_OPEN-NEG_R5_D8_NRlow.tif")
    # pins current behaviour, KNOWN_ISSUES candidate (see test_neg_opns_does_not_recognise_no_data)
    assert not np.isnan(data[0, 5, 7])


def test_save_mstp_8bit_nodata_marker(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    dv = small_defaults()
    dv.save_mstp(str(dem_path), custom_dir=str(tmp_path), save_float=False, save_8bit=True)
    _, info = read_raster(tmp_path / "dem_MSTP_3_9_15_L0.8_8bit.tif")
    assert info["dtypes"] == ("uint8",) * 3
    # pins current behaviour, KNOWN_ISSUES candidate: save_mstp passes no_data=NaN for the 8-bit raster, which GDAL
    # cannot store in a byte band; the file records no no-data value
    assert info["nodata"] is None


def test_save_visualizations_runs_the_selected_ones_and_writes_a_log(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    dv = small_defaults()
    dv.slp_compute, dv.hs_compute = 1, 1  # all other *_compute flags are 0 by default
    dv.save_visualizations(str(dem_path))
    names = sorted(p.name for p in (tmp_path / "in").iterdir())
    logs = [n for n in names if re.fullmatch(r"dem_vis_log_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.txt", n)]
    assert len(logs) == 1
    assert [n for n in names if n not in logs] == ["dem.tif", "dem_HS_A100_H40.tif", "dem_SLOPE.tif"]
    text = (tmp_path / "in" / logs[0]).read_text()
    assert "visualizations log" in text and "Input filename:" in text
    assert re.search(r"Number of rows:\s+48", text) and re.search(r"Number of columns:\s+56", text)
    assert re.search(r"# Computation time: \d+\.\d{3}s$", text)


def test_create_log_file_compute_time_line(tmp_path, dem_clean):
    dem_path, _ = _make_dem_file(tmp_path / "in", dem_clean)
    DefaultValues().create_log_file(str(dem_path), custom_dir=str(tmp_path), compute_time=1.5)
    (log,) = tmp_path.glob("dem_vis_log_*.txt")
    assert log.read_text().endswith("# Computation time: 1.500s")


# ----------------------------------------------------------------------------------------------------------------------
# get_raster_arr, get_raster_size, save_raster
# ----------------------------------------------------------------------------------------------------------------------

def test_get_raster_arr_single_band(tmp_path):
    arr = np.arange(30, dtype=np.float32).reshape(5, 6)
    path = write_geotiff(tmp_path / "a.tif", arr, res_x=2.0, res_y=3.0, nodata=-1.0)
    got = rvt.default.get_raster_arr(str(path))
    assert sorted(got) == ["array", "no_data", "resolution"]
    assert np.array_equal(got["array"], arr) and got["array"].dtype == np.float32
    assert got["resolution"] == (2.0, 3.0)  # (x_res, y_res), both positive
    assert got["no_data"] == -1.0


def test_get_raster_arr_without_no_data_and_multiband(tmp_path):
    arr = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
    path = write_geotiff(tmp_path / "b.tif", arr)
    got = rvt.default.get_raster_arr(str(path))
    assert got["no_data"] is None
    assert got["array"].shape == (2, 5, 6) and np.array_equal(got["array"], arr)


def test_get_raster_size_is_columns_then_rows(tmp_path):
    path = write_geotiff(tmp_path / "c.tif", np.zeros((5, 6), np.float32))
    assert rvt.default.get_raster_size(str(path)) == (6, 5)
    path3 = write_geotiff(tmp_path / "d.tif", np.zeros((3, 4, 7), np.float32))
    assert rvt.default.get_raster_size(str(path3), band=2) == (7, 4)


def test_save_raster_round_trip_2d(tmp_path):
    src = write_geotiff(tmp_path / "src.tif", np.zeros((5, 6), np.float32), nodata=NODATA)
    arr = np.linspace(-1, 1, 30, dtype=np.float32).reshape(5, 6)
    arr[1, 2] = np.nan
    out = tmp_path / "out.tif"
    rvt.default.save_raster(str(src), str(out), arr, no_data=np.nan)
    data, info = read_raster(out)
    assert info["dtypes"] == ("float32",)  # e_type = 6 (GDT_Float32) by default
    assert info["transform"] == GEOTRANSFORM and info["epsg"] == 32633 and np.isnan(info["nodata"])
    assert np.array_equal(data[0], arr, equal_nan=True)
    with rasterio.open(str(out)) as ds:
        assert ds.compression is not None and ds.compression.value == "LZW"


def test_save_raster_without_no_data_and_with_byte_type(tmp_path):
    src = write_geotiff(tmp_path / "src.tif", np.zeros((5, 6), np.float32), nodata=NODATA)
    arr = (np.arange(30) * 8).astype(np.uint8).reshape(5, 6)
    out = tmp_path / "out.tif"
    rvt.default.save_raster(str(src), str(out), arr, e_type=1)
    data, info = read_raster(out)
    assert info["dtypes"] == ("uint8",) and info["nodata"] is None
    assert np.array_equal(data[0], arr)


def test_save_raster_3d(tmp_path):
    src = write_geotiff(tmp_path / "src.tif", np.zeros((5, 6), np.float32))
    arr = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6)
    out = tmp_path / "out.tif"
    rvt.default.save_raster(str(src), str(out), arr, no_data=np.nan)
    data, info = read_raster(out)
    assert info["count"] == 3 and np.array_equal(data, arr)
    # save_raster calls SetNoDataValue on band 1; a GeoTIFF stores one no-data value for the whole file
    assert all(np.isnan(v) for v in info["nodatavals"])


@pytest.mark.parametrize("shape", [(6,), (2, 2, 2, 2)])
def test_save_raster_rejects_other_dimensions(tmp_path, shape):
    src = write_geotiff(tmp_path / "src.tif", np.zeros((5, 6), np.float32))
    with pytest.raises(Exception, match="2D or 3D"):
        rvt.default.save_raster(str(src), str(tmp_path / "o.tif"), np.zeros(shape, np.float32))
