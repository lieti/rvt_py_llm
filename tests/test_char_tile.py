"""
Characterization tests for ``rvt/tile.py`` at tag ``start``.

Both public functions are run on a small GeoTIFF (120 rows x 130 columns, 2 m/px, written with rasterio) with tiles
of 50 x 50 px, so that the raster is cut into 3 x 3 tiles of which the last row and column are narrower.

- ``save_visualization_tile_by_tile`` is tested with functions whose result is known analytically (a scaled copy, a
  neighbour sum with edge padding, a 3-band stack): the tiled result must equal the whole-array result exactly.
- ``save_rvt_visualization_tile_by_tile`` is tested against the untiled ``rvt.vis`` function with the same parameters
  (spelled out in this file) for the visualizations whose value at a pixel depends only on a neighbourhood that the
  overlap covers; the raster must exist and match the input in shape and georeferencing.
- Behaviour that looks wrong is pinned with the comment ``# pins current behaviour, KNOWN_ISSUES candidate`` and is
  not changed: the overlap rule of MSRM (metres used as pixels), the 1-pixel overlap of the shadow, the multi
  hillshade offset, the float MSTP raster with one band, the unreached ``Exception(...)`` statements.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

import rvt.default
import rvt.tile
import rvt.vis
from conftest import synthetic_dem
from rvt.default import DefaultValues, RVTVisualization

V = RVTVisualization

ORIGIN_X = 500000.0
ORIGIN_Y = 4500000.0
NODATA = -9999.0
SHAPE = (120, 130)  # rows, columns: tiles of 50 give 50 + 50 + 20 rows and 50 + 50 + 30 columns


def write_geotiff(path, arr, res=2.0, nodata=None):
    """Write a one-band float32 GeoTIFF, EPSG:32633, north-up, origin (ORIGIN_X, ORIGIN_Y), square pixels."""
    with rasterio.open(str(path), "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=CRS.from_epsg(32633),
                       transform=Affine(res, 0.0, ORIGIN_X, 0.0, -res, ORIGIN_Y), nodata=nodata) as ds:
        ds.write(arr.astype("float32"), 1)
    return Path(path)


def read_raster(path):
    with rasterio.open(str(path)) as ds:
        info = {"transform": ds.transform.to_gdal(), "epsg": ds.crs.to_epsg(), "nodata": ds.nodata,
                "dtypes": ds.dtypes, "count": ds.count, "shape": (ds.height, ds.width)}
        return ds.read(), info


def ref_8bit(values, lo, hi, invert=False):
    """Reference byte scale ('value' mode): clip to [lo, hi], scale to 0..1, optionally invert, times 255.9999,
    truncate; NaN becomes 255."""
    v = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
    n = (v - lo) / (hi - lo)
    if invert:
        n = 1.0 - n
    b = np.clip(255.9999 * n, 0.0, 255.0)
    b[np.isnan(b)] = 255.0
    return np.floor(b).astype(np.uint8)


def tile_defaults():
    """DefaultValues with 50 x 50 px tiles and radii small enough for the test raster."""
    dv = DefaultValues()
    dv.tile_size = (50, 50)
    dv.slrm_rad_cell = 10
    dv.svf_r_max = 5
    dv.sim_shadow_dist = 10
    dv.sim_nr_dir = 8
    dv.ld_min_rad, dv.ld_max_rad = 3, 6
    dv.msrm_feature_min, dv.msrm_feature_max = 0, 10
    dv.mstp_local_scale, dv.mstp_meso_scale, dv.mstp_broad_scale = (1, 3, 1), (3, 9, 3), (9, 15, 3)
    return dv


@pytest.fixture(scope="module")
def dem():
    return synthetic_dem(SHAPE, seed=3, relief=30.0)


@pytest.fixture
def dem_path(tmp_path, dem):
    path = write_geotiff(tmp_path / "dem.tif", dem, nodata=NODATA)
    (tmp_path / "out").mkdir()
    return path


def run_tiled(visualization, dv, path, save_float=True, save_8bit=False):
    out_dir = path.parent / "out"
    rvt.tile.save_rvt_visualization_tile_by_tile(
        rvt_visualization=visualization, rvt_default=dv, dem_path=path, output_dir_path=out_dir,
        save_float=save_float, save_8bit=save_8bit)
    return out_dir


# ----------------------------------------------------------------------------------------------------------------------
# save_visualization_tile_by_tile: analytic functions
# ----------------------------------------------------------------------------------------------------------------------

def _scaled(dem, factor=2.0):
    return dem * factor


def _neighbour_sum(dem):
    """Sum of the pixels above and below, edge-padded: needs an overlap of 1 pixel."""
    padded = np.pad(dem, 1, mode="edge")
    return padded[:-2, 1:-1] + padded[2:, 1:-1]


def _three_bands(dem):
    return np.stack([dem, dem * 2.0, dem * 3.0])


def test_tile_by_tile_without_overlap_equals_whole_array(tmp_path, dem):
    path = write_geotiff(tmp_path / "dem.tif", dem)
    out = tmp_path / "scaled.tif"
    rvt.tile.save_visualization_tile_by_tile(_scaled, {}, path, overlap=0, tile_size_x=50, tile_size_y=50,
                                             out_raster_path=out)
    data, info = read_raster(out)
    assert info["shape"] == SHAPE and info["count"] == 1 and info["dtypes"] == ("float32",)
    assert info["transform"] == (ORIGIN_X, 2.0, 0.0, ORIGIN_Y, 0.0, -2.0) and info["epsg"] == 32633
    assert np.isnan(info["nodata"])  # the blank raster is created with NaN as no-data value
    assert np.array_equal(data[0], dem * np.float32(2.0))


@pytest.mark.parametrize("overlap, tile", [(1, 50), (1, 64), (5, 50), (30, 50)])
def test_tile_by_tile_with_overlap_equals_whole_array(tmp_path, dem, overlap, tile):
    # also an overlap of 30 px > half a tile and a tile that does not divide the raster (64: 2 x 64 + 2 columns)
    path = write_geotiff(tmp_path / "dem.tif", dem)
    out = tmp_path / "sum.tif"
    rvt.tile.save_visualization_tile_by_tile(_neighbour_sum, {}, path, overlap=overlap, tile_size_x=tile,
                                             tile_size_y=tile, out_raster_path=out)
    data, _ = read_raster(out)
    assert np.array_equal(data[0], _neighbour_sum(dem))


def test_tile_by_tile_multi_band_output(tmp_path, dem):
    path = write_geotiff(tmp_path / "dem.tif", dem)
    out = tmp_path / "bands.tif"
    rvt.tile.save_visualization_tile_by_tile(_three_bands, {}, path, overlap=0, tile_size_x=50, tile_size_y=50,
                                             out_raster_path=out, out_raster_nr_of_bands=3)
    data, info = read_raster(out)
    assert info["count"] == 3 and info["shape"] == SHAPE
    assert np.array_equal(data, _three_bands(dem))


def test_tile_by_tile_dict_result_and_byte_type(tmp_path, dem):
    path = write_geotiff(tmp_path / "dem.tif", dem)
    out = tmp_path / "dict.tif"

    def as_dict(dem):  # a function that returns a dictionary, as rvt.vis.slope_aspect does
        return {"low": (dem < 0).astype(np.uint8), "other": dem}

    rvt.tile.save_visualization_tile_by_tile(as_dict, {}, path, overlap=0, tile_size_x=50, tile_size_y=50,
                                             out_raster_path=out, out_raster_e_type=1,
                                             out_visualization_dict_key="low")
    data, info = read_raster(out)
    assert info["dtypes"] == ("uint8",)
    assert np.array_equal(data[0], (dem < 0).astype(np.uint8))


def test_tile_by_tile_fills_none_parameters_from_the_dem_and_mutates_the_dict(tmp_path, dem):
    path = write_geotiff(tmp_path / "dem.tif", dem, res=2.0, nodata=NODATA)
    out = tmp_path / "slope.tif"
    params = {"resolution_x": None, "resolution_y": None, "output_units": "degree", "ve_factor": 1, "no_data": None}
    rvt.tile.save_visualization_tile_by_tile(rvt.vis.slope_aspect, params, path, overlap=1, tile_size_x=50,
                                             tile_size_y=50, out_raster_path=out, out_visualization_dict_key="slope")
    # pins current behaviour, KNOWN_ISSUES candidate: the None entries of the caller's dict are replaced in place
    assert params["resolution_x"] == 2.0 and params["resolution_y"] == 2.0 and params["no_data"] == NODATA
    data, _ = read_raster(out)
    want = rvt.vis.slope_aspect(dem.copy(), 2.0, 2.0, output_units="degree", ve_factor=1)["slope"]
    assert np.allclose(data[0], want, rtol=0, atol=1e-5, equal_nan=True)


def test_tile_by_tile_problems_that_are_not_raised(tmp_path, dem):
    path = write_geotiff(tmp_path / "dem.tif", dem)
    # pins current behaviour, KNOWN_ISSUES candidate: tile sizes below 50 px are meant to be rejected, but the
    # Exception(...) at rvt/tile.py:105 is built and never raised; the call works
    out = tmp_path / "small_tiles.tif"
    rvt.tile.save_visualization_tile_by_tile(_scaled, {}, path, overlap=0, tile_size_x=30, tile_size_y=30,
                                             out_raster_path=out)
    data, _ = read_raster(out)
    assert np.array_equal(data[0], dem * np.float32(2.0))
    # same for a missing DEM (rvt/tile.py:103): the failure comes later, from gdal.Open returning None
    with pytest.raises((AttributeError, RuntimeError)):
        rvt.tile.save_visualization_tile_by_tile(_scaled, {}, tmp_path / "missing.tif", overlap=0, tile_size_x=50,
                                                 tile_size_y=50, out_raster_path=tmp_path / "x.tif")
    # function_parameters is typed Optional but None fails at the first "in" test (rvt/tile.py:118)
    with pytest.raises(TypeError):
        rvt.tile.save_visualization_tile_by_tile(_scaled, None, path, overlap=0, tile_size_x=50, tile_size_y=50,
                                                 out_raster_path=tmp_path / "y.tif")


# ----------------------------------------------------------------------------------------------------------------------
# save_rvt_visualization_tile_by_tile
# ----------------------------------------------------------------------------------------------------------------------

# Untiled references with the parameters of tile_defaults() written out (res 2 m/px, no no-data pixel in the DEM).
REFS = {
    V.SLOPE: (lambda d: rvt.vis.slope_aspect(d, 2.0, 2.0, output_units="degree")["slope"], 1e-5),
    V.HILLSHADE: (lambda d: rvt.vis.hillshade(d, 2.0, 2.0, sun_azimuth=315, sun_elevation=35), 1e-5),
    V.SIMPLE_LOCAL_RELIEF_MODEL: (lambda d: rvt.vis.slrm(d, radius_cell=10), 1e-5),
    V.SKY_VIEW_FACTOR: (lambda d: rvt.vis.sky_view_factor(d, 2.0, compute_svf=True, svf_n_dir=16, svf_r_max=5,
                                                          svf_noise=0)["svf"], 1e-5),
    V.ANISOTROPIC_SKY_VIEW_FACTOR: (lambda d: rvt.vis.sky_view_factor(
        d, 2.0, compute_svf=False, compute_asvf=True, svf_n_dir=16, svf_r_max=5, svf_noise=0, asvf_dir=315,
        asvf_level=1)["asvf"], 1e-5),
    # openness values are of the order of 60..90 degrees: float32 spacing there is 4e-6..8e-6
    V.POSITIVE_OPENNESS: (lambda d: rvt.vis.sky_view_factor(
        d, 2.0, compute_svf=False, compute_opns=True, svf_n_dir=16, svf_r_max=5, svf_noise=0)["opns"], 1e-4),
    V.NEGATIVE_OPENNESS: (lambda d: rvt.vis.sky_view_factor(
        -d, 2.0, compute_svf=False, compute_opns=True, svf_n_dir=16, svf_r_max=5, svf_noise=0)["opns"], 1e-4),
    V.LOCAL_DOMINANCE: (lambda d: rvt.vis.local_dominance(d, min_rad=3, max_rad=6, rad_inc=1, angular_res=15,
                                                          observer_height=1.7), 1e-5),
    V.MULTI_SCALE_RELIEF_MODEL: (lambda d: rvt.vis.msrm(d, 2.0, feature_min=0, feature_max=10, scaling_factor=2),
                                 1e-5),
}


@pytest.mark.parametrize("member", list(REFS), ids=[m.name for m in REFS])
def test_tiled_result_equals_untiled_for_local_visualizations(member, dem_path, dem):
    dv = tile_defaults()
    out_dir = run_tiled(member, dv, dem_path)
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == [Path(dv.get_visualization_path(member, dem_path, out_dir, False)).name]  # no 8-bit file
    data, info = read_raster(out_dir / files[0])
    assert info["shape"] == SHAPE and info["count"] == 1 and info["dtypes"] == ("float32",)
    assert info["transform"] == (ORIGIN_X, 2.0, 0.0, ORIGIN_Y, 0.0, -2.0) and info["epsg"] == 32633
    assert np.isnan(info["nodata"])
    reference, atol = REFS[member]
    want = reference(dem.copy())
    assert np.allclose(data[0], want, rtol=0, atol=atol, equal_nan=True)
    assert np.array_equal(np.isnan(data[0]), np.isnan(want))


def test_tiled_slope_with_a_no_data_pixel(tmp_path, dem):
    arr = dem.copy()
    arr[60, 70] = NODATA  # inside one tile, far from the seams
    path = write_geotiff(tmp_path / "dem.tif", arr, nodata=NODATA)
    (tmp_path / "out").mkdir()
    out_dir = run_tiled(V.SLOPE, tile_defaults(), path)
    data, _ = read_raster(out_dir / "dem_SLOPE.tif")
    want = rvt.vis.slope_aspect(arr.copy(), 2.0, 2.0, output_units="degree", no_data=NODATA)["slope"]
    assert np.isnan(data[0, 60, 70]) and np.count_nonzero(np.isnan(data)) == 1
    assert np.allclose(data[0], want, rtol=0, atol=1e-5, equal_nan=True)


def test_tiled_8bit_only(dem_path, dem):
    dv = tile_defaults()
    out_dir = run_tiled(V.SLOPE, dv, dem_path, save_float=False, save_8bit=True)
    assert [p.name for p in out_dir.iterdir()] == ["dem_SLOPE_8bit.tif"]
    data, info = read_raster(out_dir / "dem_SLOPE_8bit.tif")
    assert info["dtypes"] == ("uint8",) and info["count"] == 1 and info["shape"] == SHAPE
    assert info["transform"] == (ORIGIN_X, 2.0, 0.0, ORIGIN_Y, 0.0, -2.0)
    slope = rvt.vis.slope_aspect(dem.copy(), 2.0, 2.0, output_units="degree")["slope"]
    assert slope.max() > 51.0  # the test DEM reaches the end of the stretch range, so the clip is exercised
    assert np.array_equal(data[0], ref_8bit(slope, 0.0, 51.0, invert=True))


def test_tiled_float_and_8bit_together_clip_the_float_raster(dem_path, dem):
    dv = tile_defaults()
    out_dir = run_tiled(V.SLOPE, dv, dem_path, save_float=True, save_8bit=True)
    assert sorted(p.name for p in out_dir.iterdir()) == ["dem_SLOPE.tif", "dem_SLOPE_8bit.tif"]
    data, _ = read_raster(out_dir / "dem_SLOPE.tif")
    slope = rvt.vis.slope_aspect(dem.copy(), 2.0, 2.0, output_units="degree")["slope"]
    # pins current behaviour, KNOWN_ISSUES candidate: DefaultValues.calculate_visualization hands the float array to
    # float_to_8bit, which clips it in place; with both outputs requested the float raster is limited to the 8-bit
    # stretch range (0..51 degrees) and differs from the float-only raster wherever the slope is steeper
    assert slope.max() > 51.0
    assert data.max() <= 51.0
    assert np.allclose(data[0], np.minimum(slope, 51.0), rtol=0, atol=1e-5)


def test_tiled_sky_illumination_runs_and_matches_in_shape(dem_path, dem):
    # not compared value by value: the result is not a function of a fixed neighbourhood (rvt/vis.py sky_illumination
    # uses pyramids of coarser DEMs)
    dv = tile_defaults()
    out_dir = run_tiled(V.SKY_ILLUMINATION, dv, dem_path)
    data, info = read_raster(out_dir / "dem_SIM_overcast_D8_10px.tif")
    assert info["shape"] == SHAPE and info["dtypes"] == ("float32",)
    assert np.isfinite(data).all()


def test_tiled_shadow_overlap_is_one_pixel(dem_path, dem):
    dv = tile_defaults()
    out_dir = run_tiled(V.SHADOW, dv, dem_path)
    assert [p.name for p in out_dir.iterdir()] == ["dem_shadow_A315_H35.tif"]
    data, info = read_raster(out_dir / "dem_shadow_A315_H35.tif")
    assert info["shape"] == SHAPE and info["dtypes"] == ("float32",)
    want = rvt.vis.shadow_horizon(dem.copy(), 2.0, shadow_az=315, shadow_el=35)["shadow"]
    assert want.any()
    # pins current behaviour, KNOWN_ISSUES candidate: _get_rvt_visualization_overlap returns 1 px for the shadow,
    # but a shadow reaches as far as a relief of h metres casts it (here 30 m of relief at el = 35 deg = tens of
    # pixels), so the tiled shadow differs from the untiled one
    assert np.count_nonzero(data[0] != want) > 0


def test_tiled_multi_hillshade_geometry(dem_path, dem):
    dv = tile_defaults()
    out_dir = run_tiled(V.MULTI_HILLSHADE, dv, dem_path)
    data, info = read_raster(out_dir / "dem_MULTI-HS_D16_H35.tif")
    assert info["count"] == 16 and info["shape"] == SHAPE and set(info["dtypes"]) == {"float32"}
    untiled = rvt.vis.multi_hillshade(dem.copy(), 2.0, 2.0, nr_directions=16, sun_elevation=35)
    assert untiled.shape == (16, SHAPE[0] - 2, SHAPE[1] - 2)  # rvt.vis loses one pixel on every edge
    # pins current behaviour, KNOWN_ISSUES candidate: the tile code assumes a result as large as the input tile.
    # (1) the cropped result of a tile is written at the tile origin, so inside tile (0, 0) pixel [i, j] holds the
    #     value that belongs to input pixel [i + 1, j + 1] (the untiled array has the same index shift);
    assert np.allclose(data[:, 5:40, 5:40], untiled[:, 5:40, 5:40], rtol=0, atol=1e-5)
    # (2) two rows and two columns at the end of every tile row / column stay NaN (the blank-raster value)
    nan_rows = [i for i in range(SHAPE[0]) if np.isnan(data[:, i, :]).all()]
    nan_cols = [j for j in range(SHAPE[1]) if np.isnan(data[:, :, j]).all()]
    assert nan_rows == [48, 49, 98, 99, 118, 119]
    assert nan_cols == [48, 49, 98, 99, 128, 129]


def test_tiled_mstp_float_raster_has_one_band_for_a_three_band_result(dem_path):
    dv = tile_defaults()
    # pins current behaviour, KNOWN_ISSUES candidate: _create_rvt_visualization_blank_raster makes a one-band
    # float raster for MSTP, but rvt.vis.mstp returns 3 bands; writing band 2 fails (GDAL returns None, or raises
    # when exceptions are enabled) and the incomplete file stays in the output directory
    with pytest.raises((AttributeError, RuntimeError)):
        run_tiled(V.MULTI_SCALE_TOPOGRAPHIC_POSITION, dv, dem_path, save_float=True, save_8bit=False)
    assert (dem_path.parent / "out" / "dem_MSTP_3_9_15_L0.9.tif").is_file()


def test_tiled_mstp_8bit_equals_untiled(dem_path, dem):
    dv = tile_defaults()
    out_dir = run_tiled(V.MULTI_SCALE_TOPOGRAPHIC_POSITION, dv, dem_path, save_float=False, save_8bit=True)
    data, info = read_raster(out_dir / "dem_MSTP_3_9_15_L0.9_8bit.tif")
    assert info["count"] == 3 and info["dtypes"] == ("uint8",) * 3 and info["shape"] == SHAPE
    mstp = rvt.vis.mstp(dem.copy(), local_scale=(1, 3, 1), meso_scale=(3, 9, 3), broad_scale=(9, 15, 3),
                        lightness=0.9)
    assert mstp.shape == (3,) + SHAPE
    assert np.array_equal(data, ref_8bit(mstp, 0.0, 1.0))


def test_tiled_multi_hillshade_8bit_has_three_bands(dem_path):
    dv = tile_defaults()
    out_dir = run_tiled(V.MULTI_HILLSHADE, dv, dem_path, save_float=False, save_8bit=True)
    data, info = read_raster(out_dir / "dem_MULTI-HS_D16_H35_8bit.tif")
    assert info["count"] == 3 and info["dtypes"] == ("uint8",) * 3 and info["shape"] == SHAPE


def test_output_dir_defaults_to_the_dem_directory(tmp_path, dem):
    path = write_geotiff(tmp_path / "dem.tif", dem, nodata=NODATA)
    rvt.tile.save_rvt_visualization_tile_by_tile(V.HILLSHADE, tile_defaults(), path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dem.tif", "dem_HS_A315_H35.tif"]


def test_tiled_arguments_that_are_not_checked(dem_path):
    dv = tile_defaults()
    # pins current behaviour, KNOWN_ISSUES candidate: neither "no output requested" (rvt/tile.py:327) nor a missing
    # DEM nor tile sizes below 50 px raise: the Exception(...) statements are not preceded by raise. Nothing is
    # written and the call returns None when both outputs are off.
    assert rvt.tile.save_rvt_visualization_tile_by_tile(
        V.SLOPE, dv, dem_path, dem_path.parent / "out", save_float=False, save_8bit=False) is None
    assert list((dem_path.parent / "out").iterdir()) == []
    dv.tile_size = (30, 30)
    run_tiled(V.SLOPE, dv, dem_path)
    assert (dem_path.parent / "out" / "dem_SLOPE.tif").is_file()
    with pytest.raises((AttributeError, RuntimeError)):  # failure comes from gdal.Open returning None
        rvt.tile.save_rvt_visualization_tile_by_tile(V.SLOPE, dv, dem_path.parent / "missing.tif")


# ----------------------------------------------------------------------------------------------------------------------
# overlap rule
# ----------------------------------------------------------------------------------------------------------------------

def _overlap(member, dv):
    return rvt.tile._get_rvt_visualization_overlap(rvt_visualization=member, rvt_default=dv)


@pytest.mark.parametrize("member, expected", [
    (V.SLOPE, 1), (V.HILLSHADE, 1), (V.SHADOW, 1), (V.MULTI_HILLSHADE, 1),
    (V.SIMPLE_LOCAL_RELIEF_MODEL, 20),  # slrm_rad_cell
    (V.SKY_VIEW_FACTOR, 10), (V.ANISOTROPIC_SKY_VIEW_FACTOR, 10), (V.POSITIVE_OPENNESS, 10),
    (V.NEGATIVE_OPENNESS, 10),  # svf_r_max
    (V.SKY_ILLUMINATION, 100),  # sim_shadow_dist
    (V.LOCAL_DOMINANCE, 20),  # ld_max_rad
    (V.MULTI_SCALE_RELIEF_MODEL, 20),  # msrm_feature_max: 20 m, used as 20 px (see below)
    (V.MULTI_SCALE_TOPOGRAPHIC_POSITION, 500),  # largest radius of the broad scale
])
def test_overlap_per_visualization_with_default_attributes(member, expected):
    assert _overlap(member, DefaultValues()) == expected


def test_overlap_follows_the_attributes_and_is_truncated_to_int():
    dv = DefaultValues()
    dv.svf_r_max, dv.ld_max_rad, dv.sim_shadow_dist, dv.slrm_rad_cell = 7, 9, 30, 12
    dv.mstp_broad_scale = (50, 300, 50)
    dv.msrm_feature_max = 37.9
    assert [_overlap(m, dv) for m in (V.SKY_VIEW_FACTOR, V.LOCAL_DOMINANCE, V.SKY_ILLUMINATION,
                                      V.SIMPLE_LOCAL_RELIEF_MODEL, V.MULTI_SCALE_TOPOGRAPHIC_POSITION,
                                      V.MULTI_SCALE_RELIEF_MODEL)] == [7, 9, 30, 12, 300, 37]


def test_msrm_overlap_is_metres_used_as_pixels(tmp_path, dem):
    # MSRM filters with radii up to about (feature_max - res) / (2 * res) px (rvt/vis.py:1943: n = ceil(sqrt(19.5))
    # = 5, so r = 25 px = 12.5 m at 0.5 m/px for feature_max = 20 m). The overlap is int(feature_max) = 20 "px"
    # (rvt/tile.py:289), 5 px too little at 0.5 m/px, so pixels within 5 px of a tile seam differ from the untiled
    # result. At 2 m/px the radius is 9 px, the overlap of 20 px is enough and the tiled result is equal.
    dv = tile_defaults()
    dv.msrm_feature_min, dv.msrm_feature_max, dv.msrm_scaling_factor = 0, 20, 2
    rows = np.arange(SHAPE[0])[:, None]
    cols = np.arange(SHAPE[1])[None, :]
    near_seam = (((rows >= 44) & (rows <= 55)) | ((rows >= 94) & (rows <= 105))
                 | ((cols >= 44) & (cols <= 55)) | ((cols >= 94) & (cols <= 105)))
    results = {}
    for res, tag in ((0.5, "fine"), (2.0, "coarse")):  # no dots in the file names: the DEM name is cut at the first dot
        path = write_geotiff(tmp_path / "{}.tif".format(tag), dem, res=res, nodata=NODATA)
        out_dir = tmp_path / "out_{}".format(tag)
        out_dir.mkdir()
        rvt.tile.save_rvt_visualization_tile_by_tile(V.MULTI_SCALE_RELIEF_MODEL, dv, path, out_dir)
        data, _ = read_raster(out_dir / "{}_MSRM_F_M0-20_S2.tif".format(tag))
        want = rvt.vis.msrm(dem.copy(), res, feature_min=0, feature_max=20, scaling_factor=2)
        results[res] = np.abs(data[0] - want)
    # pins current behaviour, KNOWN_ISSUES candidate
    assert results[0.5][near_seam].max() > 1e-3
    assert results[0.5][~near_seam].max() < 1e-5
    assert results[2.0].max() < 1e-5
