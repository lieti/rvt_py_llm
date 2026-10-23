"""
Characterization tests for rvt.vis: every public function, behaviour as it is at tag ``start``.

The tests in this file complement the reference-comparison tests of the other test files
(test_mean_filter.py, test_mstp.py, test_sky_view_factor.py, test_local_dominance.py,
test_sky_illumination.py). They cover what those do not assert: output dtype and shape, parameter
validation (exception type), no_data handling (including whether the caller's array is modified in
place), NaN-mask propagation, edge behaviour, and the small helper functions.

Expected values are independent of the code under test:

- analytic results (a plane has slope atan(|grad|); a flat DEM has hillshade sin(elevation) and sky view
  factor 1; the integral image of ones is (i + 1) (j + 1); a wall of known height has horizon atan(h / d));
- properties (shape, dtype, NaN mask, transpose and flip symmetry, +2,500 m offset invariance, input array
  not modified unless the function documents it);
- plain references written in this file (brute-force loops).

Tolerances are stated where they are used. A comment ``# pins current behaviour, KNOWN_ISSUES candidate``
marks an assertion that fixes behaviour that looks unintended; it is documented, not endorsed.

Coverage map (public function: tests)
-------------------------------------
byte_scale: test_byte_scale_*
slope_aspect: test_slope_aspect_*
roll_fill_nans: test_roll_fill_nans_*
hillshade: test_hillshade_*
multi_hillshade: test_multi_hillshade_*
mean_filter: test_mean_filter_char_*
slrm: test_slrm_char_*
horizon_shift_vector: test_horizon_shift_vector_*
sky_view_factor_compute: test_svf_compute_char_*
sky_view_factor: test_sky_view_factor_*
local_dominance: test_local_dominance_char_*
horizon_generate_coarse_dem: test_coarse_dem_*
horizon_generate_pyramids: test_pyramids_*
sky_illumination: test_sky_illumination_char_*
shadow_horizon: test_shadow_horizon_*
msrm: test_msrm_char_*
integral_image: test_integral_image_*
topographic_dev: test_topographic_dev_*
max_elevation_deviation: test_max_elevation_deviation_char_*
mstp: test_mstp_char_*
fill_where_nan: test_fill_where_nan_*
"""

import numpy as np
import pytest

import rvt.vis
from conftest import assert_same, synthetic_dem

NO_DATA = -9999.0


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------

def plane(shape, d_east, d_south, dtype=np.float32):
    """Plane z = d_east * column + d_south * row (m per pixel). Dyadic gradients are exact in float32."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    return (d_east * cols + d_south * rows).astype(dtype)


def lambert(dz_de, dz_dn, azimuth, elevation):
    """Hillshade of a plane by the dot product of its unit normal with the unit vector to the sun.
    dz_de, dz_dn: slope of the surface towards east and north (m per m); angles in degrees."""
    normal = np.array([-dz_de, -dz_dn, 1.0])
    normal /= np.linalg.norm(normal)
    az, el = np.radians(azimuth), np.radians(elevation)
    sun = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    return max(0.0, float(normal @ sun))


def with_value(dem, value, rows=slice(3, 5), cols=slice(4, 7)):
    out = dem.copy()
    out[rows, cols] = value
    return out


def circular_diff(a, b):
    """Difference of two angles in radians, wrapped to (-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def assert_raises_exact(exc_type, func, *args, **kwargs):
    """The call raises exactly exc_type (not a subclass). The message text is not asserted."""
    with pytest.raises(Exception) as info:
        func(*args, **kwargs)
    assert type(info.value) is exc_type, repr(info.value)


# ----------------------------------------------------------------------------------------------------------------------
# byte_scale
# ----------------------------------------------------------------------------------------------------------------------

def test_byte_scale_2d_default_range_and_nan():
    data = np.array([[0, 0.5, 1.0], [np.nan, 0.25, 0.75]])
    out = rvt.vis.byte_scale(data)
    assert out.dtype == np.uint8 and out.shape == data.shape
    # (255 + 0.9999) * t truncated to uint8, clipped to 255; NaN becomes 255 (min and max ignore NaN)
    assert np.array_equal(out, [[0, 127, 255], [255, 63, 191]])


def test_byte_scale_low_high():
    data = np.array([[0, 0.5, 1.0], [np.nan, 0.25, 0.75]])
    out = rvt.vis.byte_scale(data, low=10, high=200)
    # pins current behaviour, KNOWN_ISSUES candidate: low is added after clipping to high, so the maximum is
    # high + low (210), and the NaN code 255 wraps around in uint8 (255 + 10 = 9)
    assert np.array_equal(out, [[10, 110, 210], [9, 60, 160]])


def test_byte_scale_explicit_limits_clip():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = rvt.vis.byte_scale(data, c_min=2, c_max=3)
    assert np.array_equal(out, [[0, 0], [255, 255]])


def test_byte_scale_integer_input():
    out = rvt.vis.byte_scale(np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert out.dtype == np.uint8
    assert np.array_equal(out, [[0, 85], [170, 255]])


def test_byte_scale_constant_input():
    # pins current behaviour, KNOWN_ISSUES candidate: c_scale is set to 1 for a constant array but the division
    # uses c_max - c_min = 0, so 0 / 0 is NaN and every pixel becomes the NaN code 255
    with pytest.warns(RuntimeWarning):
        out = rvt.vis.byte_scale(np.full((2, 3), 3.0))
    assert np.array_equal(out, np.full((2, 3), 255))


def test_byte_scale_bands_scaled_independently():
    band = np.array([[0, 0.5, 1.0], [0.25, 0.75, 0.5]])
    bands = np.stack([band, 10 * band + 5])
    out = rvt.vis.byte_scale(bands)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    expected = rvt.vis.byte_scale(band)
    assert np.array_equal(out[0], expected) and np.array_equal(out[1], expected)


def test_byte_scale_no_data_2d_does_not_modify_input():
    data = np.array([[1.0, NO_DATA], [3.0, 4.0]])
    out = rvt.vis.byte_scale(data, no_data=NO_DATA)
    assert np.array_equal(out, [[0, 255], [170, 255]])  # no-data pixel gets 255, min/max ignore it
    assert data[0, 1] == NO_DATA  # a 2D input is copied into a 1-band stack first


def test_byte_scale_no_data_multiband_modifies_input():
    bands = np.array([[[1.0, NO_DATA], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]])
    rvt.vis.byte_scale(bands, no_data=NO_DATA)
    # pins current behaviour, KNOWN_ISSUES candidate: for a 3D input the bands are views of the caller's array
    assert np.isnan(bands[0, 0, 1]) and bands[1, 0, 1] == 2.0


def test_byte_scale_uint8_input():
    # pins current behaviour, KNOWN_ISSUES candidate: with the default high = 255, (high + 1) = 256 does not fit
    # uint8 under NEP 50 (numpy 2), so uint8 input raises OverflowError
    with pytest.raises(OverflowError):
        rvt.vis.byte_scale(np.array([[0, 10], [20, 255]], dtype=np.uint8))


def test_byte_scale_invalid_limits():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_raises_exact(ValueError, rvt.vis.byte_scale, data, high=10, low=20)
    assert_raises_exact(ValueError, rvt.vis.byte_scale, data, c_min=5, c_max=3)


# ----------------------------------------------------------------------------------------------------------------------
# roll_fill_nans
# ----------------------------------------------------------------------------------------------------------------------

def test_roll_fill_nans_hand_example():
    dem = np.array([[1.0, 2.0, np.nan, 4.0]])
    out = rvt.vis.roll_fill_nans(dem, 1, axis=1)
    # roll by 1: [4, 1, 2, nan]; pixel 3 keeps its own value 4 (the rolled one is NaN); pixel 2 is NaN itself
    assert_same(out, np.array([[4.0, 1.0, np.nan, 4.0]]))
    assert np.isnan(dem[0, 2]) and dem[0, 3] == 4.0  # input unchanged


@pytest.mark.parametrize("shift,axis", [(1, 0), (-1, 0), (1, 1), (-1, 1), (2, 1), (-3, 0)])
def test_roll_fill_nans_matches_loop_reference(shift, axis):
    rng = np.random.default_rng(7)
    dem = rng.random((5, 6)).astype(np.float32)
    dem[rng.random(dem.shape) < 0.25] = np.nan
    expected = np.empty_like(dem)
    n_rows, n_cols = dem.shape
    for i in range(n_rows):
        for j in range(n_cols):
            # neighbour that np.roll moves onto (i, j): index i - shift (mod size) along axis
            ni, nj = ((i - shift) % n_rows, j) if axis == 0 else (i, (j - shift) % n_cols)
            if np.isnan(dem[i, j]):
                expected[i, j] = np.nan
            elif np.isnan(dem[ni, nj]):
                expected[i, j] = dem[i, j]  # NaN neighbour: edge-padding-like fill with the pixel itself
            else:
                expected[i, j] = dem[ni, nj]
    assert_same(rvt.vis.roll_fill_nans(dem, shift, axis), expected)


def test_roll_fill_nans_without_nan_is_np_roll():
    dem = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert_same(rvt.vis.roll_fill_nans(dem, -1, axis=0), np.roll(dem, -1, axis=0))


# ----------------------------------------------------------------------------------------------------------------------
# slope_aspect
# ----------------------------------------------------------------------------------------------------------------------

PLANES = [(0.5, 0.25), (0.25, -0.5), (0.0, 0.5), (-0.75, 0.0), (-0.25, -0.25)]
RESOLUTIONS = [(1, 1), (2, 4), (0.5, 0.25)]


@pytest.mark.parametrize("res_x,res_y", RESOLUTIONS)
@pytest.mark.parametrize("d_east,d_south", PLANES)
def test_slope_aspect_plane_analytic(d_east, d_south, res_x, res_y):
    dem = plane((11, 13), d_east, d_south)
    out = rvt.vis.slope_aspect(dem, res_x, res_y)
    assert sorted(out) == ["aspect", "slope"]
    assert out["slope"].dtype == np.float32 and out["slope"].shape == dem.shape
    assert out["aspect"].dtype == np.float32 and out["aspect"].shape == dem.shape
    # surface gradient towards east / north (rows grow to the south), per metre
    dz_de, dz_dn = d_east / res_x, -d_south / res_y
    inner = (slice(1, -1), slice(1, -1))
    # slope = atan(|grad|); tolerance 1e-6 rad: the result is float32 (eps 6e-8 at 1)
    np.testing.assert_allclose(out["slope"][inner], np.arctan(np.hypot(dz_de, dz_dn)), rtol=0, atol=1e-6)
    # aspect = geographic azimuth of the down-slope direction, 0 = north, clockwise, in (-pi, pi]
    expected_aspect = np.arctan2(-dz_de, -dz_dn)
    assert np.max(np.abs(circular_diff(out["aspect"][inner], expected_aspect))) < 1e-6


def test_slope_aspect_aspect_range_is_signed():
    # pins current behaviour, KNOWN_ISSUES candidate: the docstring says "0 or 2pi for the North direction", the
    # code returns arctan2, so aspect lies in (-pi, pi] and the down-slope direction west is -pi/2, not 3pi/2
    dem = plane((9, 9), 0.5, 0.0)  # rises to the east, so the down-slope direction is west
    aspect = rvt.vis.slope_aspect(dem)["aspect"]
    assert abs(aspect[4, 4] - (-np.pi / 2)) < 1e-6


@pytest.mark.parametrize("d_east,d_south", [(0.5, 0.25), (-0.25, 0.5)])
def test_slope_aspect_output_units(d_east, d_south):
    dem = plane((9, 10), d_east, d_south)
    rad = rvt.vis.slope_aspect(dem, 1, 1, "radian")
    deg = rvt.vis.slope_aspect(dem, 1, 1, "degree")
    pct = rvt.vis.slope_aspect(dem, 1, 1, "percent")
    grad = np.hypot(d_east, d_south)
    inner = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(deg["slope"][inner], np.degrees(np.arctan(grad)), rtol=1e-6)
    np.testing.assert_allclose(pct["slope"][inner], 100 * grad, rtol=1e-6)  # 100 * tan(slope)
    np.testing.assert_allclose(deg["aspect"][inner], np.degrees(rad["aspect"][inner]), rtol=1e-6)
    # pins current behaviour, KNOWN_ISSUES candidate: 'percent' converts only the slope; the aspect stays in radian
    assert_same(pct["aspect"], rad["aspect"])
    for out in (rad, deg, pct):
        assert out["slope"].dtype == np.float32 and out["aspect"].dtype == np.float32


def test_slope_aspect_edges_use_half_gradient():
    # edge padding: at the border the central difference spans (pixel, neighbour) only, i.e. half the gradient
    d_east, d_south = 0.5, 0.25
    out = rvt.vis.slope_aspect(plane((8, 9), d_east, d_south))["slope"]
    fx = np.ones(9)
    fx[[0, -1]] = 0.5
    fy = np.ones(8)
    fy[[0, -1]] = 0.5
    expected = np.arctan(np.hypot(d_east * fx[None, :], d_south * fy[:, None]))
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)


def test_slope_aspect_flat():
    out = rvt.vis.slope_aspect(np.full((6, 7), 123.5, dtype=np.float32))
    assert np.all(out["slope"] == 0)
    # pins current behaviour: dz/dy = 0 is replaced by 1e-8 for numeric stability, so arctan2(0, 1e-8) = 0
    assert np.all(out["aspect"] == 0)


@pytest.mark.parametrize("ve", [2, -2, 0.5])
def test_slope_aspect_ve_factor(ve):
    dem = plane((9, 9), 0.5, 0.25)
    slope = rvt.vis.slope_aspect(dem, ve_factor=ve)["slope"]
    expected = np.arctan(abs(ve) * np.hypot(0.5, 0.25))
    np.testing.assert_allclose(slope[1:-1, 1:-1], expected, rtol=0, atol=1e-6)


def test_slope_aspect_ve_factor_limits_accepted():
    dem = plane((5, 5), 0.5, 0.25)
    for ve in (10000, -10000, 0):
        out = rvt.vis.slope_aspect(dem, ve_factor=ve)
        assert out["slope"].shape == dem.shape


def test_slope_aspect_transpose_symmetry():
    dem = synthetic_dem((13, 17), seed=61, nan_patch=True)
    a = rvt.vis.slope_aspect(dem, 2.0, 0.5)
    b = rvt.vis.slope_aspect(np.ascontiguousarray(dem.T), 0.5, 2.0)
    assert_same(b["slope"], np.ascontiguousarray(a["slope"].T))  # sqrt(a^2 + b^2) is symmetric: exact


def test_slope_aspect_offset_invariance():
    dem = synthetic_dem((15, 16), seed=62, grid=True)
    a = rvt.vis.slope_aspect(dem, 0.5, 0.5)
    b = rvt.vis.slope_aspect(dem + np.float32(2500.0), 0.5, 0.5)  # differences on a 1/256 m grid are exact
    assert_same(b, a)


def test_slope_aspect_nan_propagation_and_neighbours():
    dem = plane((9, 11), 0.5, 0.25)
    dem[4, 5] = np.nan
    out = rvt.vis.slope_aspect(dem)
    assert np.array_equal(np.isnan(out["slope"]), np.isnan(dem))
    assert np.array_equal(np.isnan(out["aspect"]), np.isnan(dem))
    # a pixel next to the NaN uses its own value in place of the NaN neighbour: half the gradient on that axis
    np.testing.assert_allclose(out["slope"][4, 4], np.arctan(np.hypot(0.25, 0.25)), atol=1e-6)  # x halved
    np.testing.assert_allclose(out["slope"][4, 6], np.arctan(np.hypot(0.25, 0.25)), atol=1e-6)
    np.testing.assert_allclose(out["slope"][3, 5], np.arctan(np.hypot(0.5, 0.125)), atol=1e-6)  # y halved
    np.testing.assert_allclose(out["slope"][5, 5], np.arctan(np.hypot(0.5, 0.125)), atol=1e-6)


def test_slope_aspect_no_data():
    dem = plane((9, 11), 0.5, 0.25)
    with_nan = with_value(dem, np.nan, slice(4, 6), slice(5, 7))
    with_nd = with_value(dem, NO_DATA, slice(4, 6), slice(5, 7))
    kept = with_nd.copy()
    out = rvt.vis.slope_aspect(with_nd, no_data=NO_DATA)
    assert_same(out, rvt.vis.slope_aspect(with_nan))
    assert np.array_equal(with_nd, kept)  # the dtype conversion copies first: the caller's array is not modified


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.float32])
def test_slope_aspect_input_dtypes(dtype):
    dem = (plane((7, 8), 2, 1, dtype=np.float64)).astype(dtype)
    out = rvt.vis.slope_aspect(dem)
    assert out["slope"].dtype == np.float32 and out["slope"].shape == (7, 8)


def test_slope_aspect_does_not_modify_input():
    dem = synthetic_dem((10, 11), seed=63, nan_patch=True)
    kept = dem.copy()
    rvt.vis.slope_aspect(dem, 1, 1, "degree")
    assert_same(dem, kept)


@pytest.mark.parametrize("kwargs", [dict(ve_factor=10001), dict(ve_factor=-10001), dict(resolution_x=-1),
                                    dict(resolution_y=-0.5), dict(output_units="grad"), dict(output_units="")])
def test_slope_aspect_invalid_parameters(kwargs):
    assert_raises_exact(Exception, rvt.vis.slope_aspect, plane((6, 6), 1, 1), **kwargs)


def test_slope_aspect_invalid_dimensions():
    assert_raises_exact(Exception, rvt.vis.slope_aspect, np.zeros((3, 4, 4), dtype=np.float32))
    assert_raises_exact(Exception, rvt.vis.slope_aspect, np.zeros(5, dtype=np.float32))


# ----------------------------------------------------------------------------------------------------------------------
# hillshade, multi_hillshade
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("res_x,res_y", [(1, 1), (2, 4)])
@pytest.mark.parametrize("az,el", [(315, 35), (90, 10), (270, 5), (0, 60), (180, 20), (45, 89)])
def test_hillshade_plane_lambert(az, el, res_x, res_y):
    d_east, d_south = 0.5, 0.25
    dem = plane((14, 15), d_east, d_south)
    out = rvt.vis.hillshade(dem, res_x, res_y, sun_azimuth=az, sun_elevation=el)
    assert out.shape == dem.shape
    expected = lambert(d_east / res_x, -d_south / res_y, az, el)
    # the double padding of the implementation makes the outermost two rows and columns border-affected; the
    # slope is float32, so the tolerance is 1e-6
    np.testing.assert_allclose(out[2:-2, 2:-2], expected, rtol=0, atol=1e-6)


def test_hillshade_negative_cosine_is_zero():
    dem = plane((12, 12), 0.5, 0.0)  # rises to the east; the sun at the east, 10 degrees above the horizon, is behind it
    assert lambert(0.5, 0.0, 90, 10) == 0.0
    out = rvt.vis.hillshade(dem, 1, 1, sun_azimuth=90, sun_elevation=10)
    assert np.all(out == 0)


@pytest.mark.parametrize("el", [0, 35, 90])
@pytest.mark.parametrize("az", [0, 123, 360])
def test_hillshade_flat_is_sine_of_elevation(az, el):
    out = rvt.vis.hillshade(np.full((8, 9), 1500.0, dtype=np.float32), 1, 1, sun_azimuth=az, sun_elevation=el)
    np.testing.assert_allclose(out, np.sin(np.radians(el)), rtol=0, atol=1e-12)


def test_hillshade_dtype_and_default_angles():
    dem = plane((10, 10), 0.5, 0.25)
    out = rvt.vis.hillshade(dem, 1, 1)
    # the float64 scalars cos / sin(zenith) promote the float32 slope to float64 (numpy 2, NEP 50)
    assert out.dtype == np.float64
    assert_same(out, rvt.vis.hillshade(dem, 1, 1, sun_azimuth=315, sun_elevation=35))


def test_hillshade_ve_factor():
    dem = plane((14, 15), 0.5, 0.25)
    out = rvt.vis.hillshade(dem, 1, 1, ve_factor=2)
    np.testing.assert_allclose(out[2:-2, 2:-2], lambert(1.0, -0.5, 315, 35), rtol=0, atol=1e-6)
    out = rvt.vis.hillshade(dem, 1, 1, ve_factor=-1)  # a mirrored relief: the plane falls to the east
    np.testing.assert_allclose(out[2:-2, 2:-2], lambert(-0.5, 0.25, 315, 35), rtol=0, atol=1e-6)


def test_hillshade_with_given_slope_and_aspect_loses_a_border():
    dem = plane((9, 10), 0.5, 0.25)
    slope = np.full(dem.shape, np.pi / 6)
    aspect = np.full(dem.shape, 0.3)
    out = rvt.vis.hillshade(dem, 1, 1, sun_azimuth=100, sun_elevation=40, slope=slope, aspect=aspect)
    # pins current behaviour, KNOWN_ISSUES candidate: the result is always cropped by one pixel on each side, which
    # fits the internally padded slope but not a given (H, W) slope, so the output is (H - 2, W - 2)
    assert out.shape == (7, 8)
    zen = np.radians(50)
    expected = np.cos(zen) * np.cos(np.pi / 6) + np.sin(zen) * np.sin(np.pi / 6) * np.cos(0.3 - np.radians(100))
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_hillshade_given_slope_only_is_ignored():
    dem = synthetic_dem((12, 13), seed=64)
    garbage = np.full(dem.shape, 1.0)
    assert_same(rvt.vis.hillshade(dem, 1, 1, slope=garbage), rvt.vis.hillshade(dem, 1, 1))
    assert_same(rvt.vis.hillshade(dem, 1, 1, aspect=garbage), rvt.vis.hillshade(dem, 1, 1))


def test_hillshade_nan_mask():
    dem = plane((12, 12), 0.5, 0.25)
    dem[6, 7] = np.nan
    out = rvt.vis.hillshade(dem, 1, 1)
    assert np.array_equal(np.isnan(out), np.isnan(dem))


def test_hillshade_no_data_modifies_float_input_in_place():
    dem = with_value(plane((12, 12), 0.5, 0.25), NO_DATA, slice(6, 8), slice(5, 7))
    expected = with_value(plane((12, 12), 0.5, 0.25), np.nan, slice(6, 8), slice(5, 7))
    out = rvt.vis.hillshade(dem, 1, 1, no_data=NO_DATA)
    # pins current behaviour, KNOWN_ISSUES candidate: no_data is replaced by NaN in the caller's array
    assert_same(dem, expected)
    assert_same(out, rvt.vis.hillshade(expected.copy(), 1, 1))


def test_hillshade_does_not_modify_input_without_no_data():
    dem = synthetic_dem((10, 11), seed=65, nan_patch=True)
    kept = dem.copy()
    rvt.vis.hillshade(dem, 1, 1)
    assert_same(dem, kept)


@pytest.mark.parametrize("kwargs", [dict(sun_azimuth=361), dict(sun_azimuth=-1), dict(sun_elevation=91),
                                    dict(sun_elevation=-1), dict(ve_factor=10001), dict(ve_factor=-10001)])
def test_hillshade_invalid_parameters(kwargs):
    assert_raises_exact(Exception, rvt.vis.hillshade, plane((6, 6), 1, 1), 1, 1, **kwargs)


def test_hillshade_invalid_resolution_and_dimensions():
    assert_raises_exact(Exception, rvt.vis.hillshade, plane((6, 6), 1, 1), -1, 1)
    assert_raises_exact(Exception, rvt.vis.hillshade, plane((6, 6), 1, 1), 1, -1)
    assert_raises_exact(Exception, rvt.vis.hillshade, np.zeros((2, 6, 6), dtype=np.float32), 1, 1)


def test_multi_hillshade_shape_is_cropped():
    dem = plane((20, 23), 0.5, 0.25)
    out = rvt.vis.multi_hillshade(dem, 1, 1)
    # pins current behaviour, KNOWN_ISSUES candidate: (nr_directions, H - 2, W - 2), not (nr_directions, H, W)
    assert out.shape == (16, 18, 21)
    assert out.dtype == np.float64


@pytest.mark.parametrize("n_dir", [1, 4, 8])
def test_multi_hillshade_plane_lambert_per_direction(n_dir):
    dem = plane((14, 15), 0.5, 0.25)
    out = rvt.vis.multi_hillshade(dem, 1, 1, nr_directions=n_dir, sun_elevation=35)
    assert out.shape == (n_dir, 12, 13)
    for i in range(n_dir):
        # azimuth i * 360 / n_dir; the cropped output holds only pixels with a full 3 x 3 neighbourhood
        np.testing.assert_allclose(out[i], lambert(0.5, -0.25, i * 360 / n_dir, 35), rtol=0, atol=1e-6)


def test_multi_hillshade_flat_and_resolution_and_ve():
    flat = rvt.vis.multi_hillshade(np.full((9, 10), 40.0, dtype=np.float32), 1, 1, sun_elevation=50)
    np.testing.assert_allclose(flat, np.sin(np.radians(50)), rtol=0, atol=1e-12)
    dem = plane((14, 15), 0.5, 0.25)
    out = rvt.vis.multi_hillshade(dem, 2, 4, nr_directions=4, ve_factor=2)
    for i in range(4):
        np.testing.assert_allclose(out[i], lambert(2 * 0.5 / 2, -2 * 0.25 / 4, 90 * i, 35), rtol=0, atol=1e-6)


def test_multi_hillshade_nan_mask_and_no_data():
    dem = plane((14, 15), 0.5, 0.25)
    with_nan = with_value(dem, np.nan, slice(6, 8), slice(5, 7))
    with_nd = with_value(dem, NO_DATA, slice(6, 8), slice(5, 7))
    out = rvt.vis.multi_hillshade(with_nan.copy(), 1, 1, nr_directions=4)
    assert np.array_equal(np.isnan(out), np.broadcast_to(np.isnan(with_nan)[1:-1, 1:-1], out.shape))
    out_nd = rvt.vis.multi_hillshade(with_nd, 1, 1, nr_directions=4, no_data=NO_DATA)
    assert_same(out_nd, out)
    assert_same(with_nd, with_nan)  # pins current behaviour, KNOWN_ISSUES candidate: no_data replaced in the input


@pytest.mark.parametrize("kwargs", [dict(sun_elevation=91), dict(sun_elevation=-1), dict(nr_directions=0),
                                    dict(nr_directions=-4), dict(ve_factor=10001), dict(ve_factor=-10001)])
def test_multi_hillshade_invalid_parameters(kwargs):
    assert_raises_exact(Exception, rvt.vis.multi_hillshade, plane((8, 8), 1, 1), 1, 1, **kwargs)


def test_multi_hillshade_invalid_resolution_and_dimensions():
    assert_raises_exact(Exception, rvt.vis.multi_hillshade, plane((8, 8), 1, 1), -1, 1)
    assert_raises_exact(Exception, rvt.vis.multi_hillshade, plane((8, 8), 1, 1), 1, -1)
    assert_raises_exact(Exception, rvt.vis.multi_hillshade, np.zeros((2, 8, 8), dtype=np.float32), 1, 1)


# ----------------------------------------------------------------------------------------------------------------------
# mean_filter, slrm, msrm
# ----------------------------------------------------------------------------------------------------------------------

def test_mean_filter_char_plane_interior_unchanged_and_dtype():
    dem = plane((30, 31), 0.5, 0.25)
    for radius in (1, 4):
        out = rvt.vis.mean_filter(dem, radius)
        assert out.dtype == np.float32 and out.shape == dem.shape
        # the mean of a plane over a symmetric window is its centre value; tolerance 1e-5: float32 result
        np.testing.assert_allclose(out[radius:-radius, radius:-radius], dem[radius:-radius, radius:-radius],
                                   rtol=0, atol=1e-5)
    out64 = rvt.vis.mean_filter(dem.astype(np.float64), 2)
    assert out64.dtype == np.float32  # float64 input also returns float32


def test_mean_filter_char_constant_dem_and_nan_mask():
    dem = np.full((15, 16), 7.25, dtype=np.float32)
    assert np.all(rvt.vis.mean_filter(dem, 3) == 7.25)
    dem[5:7, 6:9] = np.nan
    out = rvt.vis.mean_filter(dem, 3)
    assert np.array_equal(np.isnan(out), np.isnan(dem))  # NaN stays NaN, neighbours get the mean of the valid ones
    assert np.all(out[~np.isnan(dem)] == 7.25)


def test_mean_filter_char_fractional_radius_is_truncated():
    dem = synthetic_dem((20, 22), seed=66)
    assert_same(rvt.vis.mean_filter(dem, 2.7), rvt.vis.mean_filter(dem, 2))
    # pins current behaviour: 0 < radius < 1 is not the "radius 0" shortcut but gives a 1 x 1 window, i.e. a copy
    out = rvt.vis.mean_filter(dem, 0.5)
    assert out is not dem
    assert_same(out, dem)


def test_mean_filter_char_does_not_modify_input():
    dem = synthetic_dem((18, 19), seed=67, nan_patch=True)
    kept = dem.copy()
    rvt.vis.mean_filter(dem, 3)
    assert_same(dem, kept)


def test_slrm_char_plane_and_dtype():
    dem = plane((70, 71), 0.5, 0.25)
    out = rvt.vis.slrm(dem, radius_cell=10)
    assert out.dtype == np.float32 and out.shape == dem.shape
    np.testing.assert_allclose(out[10:-10, 10:-10], 0, rtol=0, atol=1e-4)  # trend of a plane is the plane
    flat = rvt.vis.slrm(np.full((30, 30), 12.5, dtype=np.float32), radius_cell=10)
    assert np.all(flat == 0)


def test_slrm_char_ve_factor_scales_result():
    dem = synthetic_dem((40, 41), seed=68, grid=True)
    # grid values times 4 are exact; the mean filter is linear, so the result scales by 4 (tolerance: float32 rounding)
    np.testing.assert_allclose(rvt.vis.slrm(dem, 10, ve_factor=4), 4 * rvt.vis.slrm(dem, 10), rtol=1e-5, atol=1e-5)


def test_slrm_char_offset_and_nan_mask():
    dem = synthetic_dem((40, 41), seed=69, nan_patch=True)
    a = rvt.vis.slrm(dem, 10)
    assert np.array_equal(np.isnan(a), np.isnan(dem))
    b = rvt.vis.slrm(dem + np.float32(2500.0), 10)
    assert np.array_equal(np.isnan(b), np.isnan(dem))
    # float32 spacing at 2,500 m is 2.4e-4 m: the relief differs by rounding of that size, so the tolerance is 2e-3
    np.testing.assert_allclose(b, a, rtol=0, atol=2e-3, equal_nan=True)


def test_slrm_char_no_data_modifies_input():
    dem = with_value(synthetic_dem((30, 31), seed=70), NO_DATA)
    expected = with_value(synthetic_dem((30, 31), seed=70), np.nan)
    out = rvt.vis.slrm(dem, 10, no_data=NO_DATA)
    assert_same(dem, expected)  # pins current behaviour, KNOWN_ISSUES candidate: no_data replaced in the input
    assert_same(out, rvt.vis.slrm(expected.copy(), 10))


@pytest.mark.parametrize("radius", [10, 50])
def test_slrm_char_radius_limits_accepted(radius):
    assert rvt.vis.slrm(np.zeros((20, 20), dtype=np.float32), radius_cell=radius).shape == (20, 20)


@pytest.mark.parametrize("kwargs", [dict(radius_cell=9), dict(radius_cell=51), dict(radius_cell=-1),
                                    dict(ve_factor=10001), dict(ve_factor=-10001)])
def test_slrm_char_invalid_parameters(kwargs):
    assert_raises_exact(Exception, rvt.vis.slrm, np.zeros((20, 20), dtype=np.float32), **kwargs)


def test_slrm_char_invalid_dimensions():
    assert_raises_exact(Exception, rvt.vis.slrm, np.zeros((2, 20, 20), dtype=np.float32))


def test_msrm_char_plane_nan_dtype_shape():
    dem = plane((40, 41), 0.5, 0.25)
    # feature_min = resolution = 1 m -> i = 0; feature_max = 7 m -> n = ceil((7 - 1) / 2) = 3: radii 0, 1, 2, 3 px
    out = rvt.vis.msrm(dem, 1, 1, 7, 1)
    assert out.dtype == np.float64 and out.shape == dem.shape
    np.testing.assert_allclose(out[3:-3, 3:-3], 0, rtol=0, atol=1e-5)  # differences of means of a plane vanish
    nan_dem = with_value(synthetic_dem((40, 41), seed=71), np.nan)
    assert np.array_equal(np.isnan(rvt.vis.msrm(nan_dem, 1, 1, 7, 2)), np.isnan(nan_dem))


def test_msrm_char_scaling_factor_is_truncated_and_feature_min_clamped():
    dem = synthetic_dem((40, 41), seed=72)
    assert_same(rvt.vis.msrm(dem, 1, 1, 7, 2.9), rvt.vis.msrm(dem, 1, 1, 7, 2))
    assert_same(rvt.vis.msrm(dem, 1, 0.1, 7, 1), rvt.vis.msrm(dem, 1, 1, 7, 1))  # feature_min < resolution -> resolution


def test_msrm_char_single_surface_gives_nan():
    # pins current behaviour, KNOWN_ISSUES candidate: feature_max = feature_min leaves one filtered surface, no
    # difference of two surfaces, and the sum is divided by zero: every pixel is NaN (RuntimeWarning)
    with pytest.warns(RuntimeWarning):
        out = rvt.vis.msrm(synthetic_dem((20, 21), seed=73), 1, 1, 1, 1)
    assert np.all(np.isnan(out))


def test_msrm_char_no_data_modifies_input():
    dem = with_value(synthetic_dem((30, 31), seed=74), NO_DATA)
    expected = with_value(synthetic_dem((30, 31), seed=74), np.nan)
    out = rvt.vis.msrm(dem, 1, 1, 7, 1, no_data=NO_DATA)
    assert_same(dem, expected)  # pins current behaviour, KNOWN_ISSUES candidate: no_data replaced in the input
    assert_same(out, rvt.vis.msrm(expected.copy(), 1, 1, 7, 1))


@pytest.mark.parametrize("kwargs", [dict(ve_factor=10001), dict(ve_factor=-10001), dict(resolution=-1)])
def test_msrm_char_invalid_parameters(kwargs):
    args = dict(dem=np.zeros((20, 20), dtype=np.float32), resolution=1, feature_min=1, feature_max=7,
                scaling_factor=1)
    args.update(kwargs)
    assert_raises_exact(Exception, rvt.vis.msrm, **args)


# ----------------------------------------------------------------------------------------------------------------------
# horizon_shift_vector
# ----------------------------------------------------------------------------------------------------------------------

def test_horizon_shift_vector_four_directions_hand_example():
    shift = rvt.vis.horizon_shift_vector(num_directions=4, radius_pixels=3, min_radius=1)
    assert list(shift) == [0.0, 90.0, 180.0, 270.0]
    # the shift pairs are (x, y) = (cos, sin) * radius rounded to integers, ordered by distance
    expected = {0.0: [(1, 0), (2, 0), (3, 0)], 90.0: [(0, 1), (0, 2), (0, 3)],
                180.0: [(-1, 0), (-2, 0), (-3, 0)], 270.0: [(0, -1), (0, -2), (0, -3)]}
    for angle, pairs in expected.items():
        assert sorted(shift[angle]) == ["distance", "shift"]
        assert [(int(a), int(b)) for a, b in shift[angle]["shift"]] == pairs
        assert np.array_equal(shift[angle]["distance"], [1.0, 2.0, 3.0])


def test_horizon_shift_vector_defaults_and_keys():
    shift = rvt.vis.horizon_shift_vector()
    assert list(shift) == list(np.round(np.arange(16) * 22.5, 1))  # azimuth keys rounded to 0.1 degree
    for entry in shift.values():
        assert len(entry["shift"]) == len(entry["distance"]) > 0
        assert np.all(np.diff(entry["distance"]) >= 0)  # sorted by increasing distance
        assert len(set(entry["shift"])) == len(entry["shift"])  # each position once
        for dy, dx in entry["shift"]:
            assert max(abs(dy), abs(dx)) <= 10  # within the radius of 10 px


@pytest.mark.parametrize("n_dir,radius,min_radius", [(8, 10, 1), (16, 6, 3), (4, 5, 5), (32, 8, 2)])
def test_horizon_shift_vector_geometry(n_dir, radius, min_radius):
    shift = rvt.vis.horizon_shift_vector(n_dir, radius, min_radius)
    assert len(shift) == n_dir
    n_radii = int((radius - min_radius) * 3 + 1)  # radii every 1/3 px
    for angle, entry in shift.items():
        assert len(entry["shift"]) <= n_radii
        dist = np.hypot(*np.array(entry["shift"], dtype=np.float64).T)
        assert np.array_equal(dist, entry["distance"])
        assert dist.max() <= radius + 0.5 * np.sqrt(2)  # rounding moves a point by at most half a pixel per axis


def test_horizon_shift_vector_quarter_turn_symmetry():
    shift = rvt.vis.horizon_shift_vector(8, 10, 1)
    angles = list(shift)
    for k in range(8):
        turned = {(-int(dx), int(dy)) for dy, dx in shift[angles[k]]["shift"]}  # rotate (x, y) by +90 degrees
        assert turned == {(int(a), int(b)) for a, b in shift[angles[(k + 2) % 8]]["shift"]}


# ----------------------------------------------------------------------------------------------------------------------
# sky_view_factor_compute, sky_view_factor
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("flags", [(True, False, False), (False, True, False), (False, False, True),
                                   (True, True, True), (False, False, False)])
def test_svf_compute_char_keys_dtype_shape(flags):
    compute_svf, compute_opns, compute_asvf = flags
    height = synthetic_dem((20, 23), seed=75)
    out = rvt.vis.sky_view_factor_compute(height, radius_max=5, compute_svf=compute_svf, compute_opns=compute_opns,
                                          compute_asvf=compute_asvf)
    expected_keys = [k for k, on in (("svf", compute_svf), ("asvf", compute_asvf), ("opns", compute_opns)) if on]
    assert sorted(out) == sorted(expected_keys)  # with no flag set the dict is empty
    for arr in out.values():
        assert arr.shape == height.shape and arr.dtype == np.float64


def test_svf_compute_char_does_not_modify_input():
    height = synthetic_dem((20, 23), seed=76, nan_patch=True)
    kept = height.copy()
    rvt.vis.sky_view_factor_compute(height, radius_max=5, compute_opns=True, compute_asvf=True)
    assert_same(height, kept)


def test_sky_view_factor_flat_analytic():
    flat = np.full((20, 22), 37.0, dtype=np.float32)
    out = rvt.vis.sky_view_factor(flat, 1, compute_svf=True, compute_opns=True, compute_asvf=True, svf_r_max=5)
    assert sorted(out) == ["asvf", "opns", "svf"]
    np.testing.assert_allclose(out["svf"], 1.0, rtol=0, atol=1e-12)  # open hemisphere
    np.testing.assert_allclose(out["asvf"], 1.0, rtol=0, atol=1e-12)  # weights normalised by their sum
    np.testing.assert_allclose(out["opns"], 90.0, rtol=0, atol=1e-10)  # horizon at 0 degrees: openness 90 degrees


def test_sky_view_factor_value_ranges():
    dem = synthetic_dem((30, 33), seed=77, relief=40.0)
    out = rvt.vis.sky_view_factor(dem, 0.5, compute_svf=True, compute_opns=True, compute_asvf=True, svf_r_max=8)
    for key in ("svf", "asvf"):
        assert out[key].min() >= 0 and out[key].max() <= 1 + 1e-12
    assert out["opns"].min() >= 0 and out["opns"].max() <= 180
    # a 10 m spike on flat terrain sees the full sky; everything else sees less of it
    spike = np.zeros((21, 21), dtype=np.float32)
    spike[10, 10] = 10
    svf = rvt.vis.sky_view_factor(spike, 1, svf_r_max=5)["svf"]
    assert svf[10, 10] == 1.0 and svf.min() < 1.0
    opns = rvt.vis.sky_view_factor(spike, 1, compute_svf=False, compute_opns=True, svf_r_max=5)["opns"]
    assert opns[10, 10] > 90  # the horizon is below the spike: openness above 90 degrees


@pytest.mark.parametrize("name,func", [("fliplr", lambda a: a[:, ::-1]), ("flipud", lambda a: a[::-1]),
                                       ("transpose", lambda a: a.T), ("rot90", np.rot90)])
def test_sky_view_factor_symmetry(name, func):
    # the 16 search directions are symmetric under these transforms; the 16-term sums are added in a different
    # order, so the results agree to float64 rounding (tolerance 1e-10)
    dem = synthetic_dem((24, 26), seed=78, grid=True)
    a = rvt.vis.sky_view_factor(dem, 1, compute_svf=True, compute_opns=True, svf_r_max=6)
    b = rvt.vis.sky_view_factor(np.ascontiguousarray(func(dem)), 1, compute_svf=True, compute_opns=True, svf_r_max=6)
    for key in ("svf", "opns"):
        np.testing.assert_allclose(b[key], func(a[key]), rtol=0, atol=1e-10)


def test_sky_view_factor_resolution_ve_factor_and_offset():
    dem = synthetic_dem((24, 26), seed=79, grid=True)
    base = rvt.vis.sky_view_factor(dem, 1, compute_opns=True, svf_r_max=6)
    # dem * 2 and dem * 2 / 2 are exact in float32: same input to the computation
    assert_same(rvt.vis.sky_view_factor(dem * np.float32(2), 2, compute_opns=True, svf_r_max=6), base)
    assert_same(rvt.vis.sky_view_factor(dem, 1, compute_opns=True, svf_r_max=6, ve_factor=2),
                rvt.vis.sky_view_factor(dem * np.float32(2), 1, compute_opns=True, svf_r_max=6))
    assert_same(rvt.vis.sky_view_factor(dem + np.float32(2500.0), 1, compute_opns=True, svf_r_max=6), base)


@pytest.mark.parametrize("flags,keys", [(dict(compute_svf=True), ["svf"]),
                                        (dict(compute_svf=False, compute_opns=True), ["opns"]),
                                        (dict(compute_svf=False, compute_asvf=True), ["asvf"]),
                                        (dict(compute_svf=True, compute_asvf=True), ["asvf", "svf"])])
def test_sky_view_factor_selected_outputs(flags, keys):
    out = rvt.vis.sky_view_factor(synthetic_dem((20, 24), seed=80), 1, svf_r_max=4, svf_n_dir=8, **flags)
    assert sorted(out) == keys
    for arr in out.values():
        assert arr.shape == (20, 24) and arr.dtype == np.float64


def test_sky_view_factor_noise_levels_and_anisotropy_levels_run():
    dem = synthetic_dem((24, 26), seed=81)
    for noise in (0, 1, 2, 3):
        out = rvt.vis.sky_view_factor(dem, 1, svf_r_max=10, svf_noise=noise)
        assert out["svf"].shape == dem.shape
    for level in (1, 2):
        out = rvt.vis.sky_view_factor(dem, 1, compute_svf=False, compute_asvf=True, svf_r_max=5, asvf_level=level,
                                      asvf_dir=90)
        assert out["asvf"].shape == dem.shape


def test_sky_view_factor_nan_mask_and_no_data():
    dem = synthetic_dem((26, 28), seed=82, nan_patch=True)
    out = rvt.vis.sky_view_factor(dem, 1, compute_opns=True, compute_asvf=True, svf_r_max=5)
    for arr in out.values():
        assert np.array_equal(np.isnan(arr), np.isnan(dem))  # the NaN mask is re-applied to every output
    with_nd = dem.copy()
    with_nd[np.isnan(with_nd)] = NO_DATA
    kept = with_nd.copy()
    out_nd = rvt.vis.sky_view_factor(with_nd, 1, compute_opns=True, compute_asvf=True, svf_r_max=5, no_data=NO_DATA)
    assert_same(out_nd, out)
    assert_same(with_nd, kept)  # the float32 conversion copies first: the caller's array is not modified


@pytest.mark.parametrize("kwargs", [dict(ve_factor=10001), dict(ve_factor=-10001), dict(svf_noise=4),
                                    dict(svf_noise=-1), dict(asvf_level=0), dict(asvf_level=3),
                                    dict(compute_svf=False), dict(resolution=-1)])
def test_sky_view_factor_invalid_parameters(kwargs):
    args = dict(dem=synthetic_dem((12, 12), seed=83), resolution=1, svf_r_max=3)
    args.update(kwargs)
    assert_raises_exact(Exception, rvt.vis.sky_view_factor, **args)


def test_sky_view_factor_invalid_dimensions():
    assert_raises_exact(Exception, rvt.vis.sky_view_factor, np.zeros((2, 12, 12), dtype=np.float32), 1)


# ----------------------------------------------------------------------------------------------------------------------
# local_dominance
# ----------------------------------------------------------------------------------------------------------------------

def test_local_dominance_char_flat_peak_pit_analytic():
    params = dict(min_rad=2, max_rad=6)
    flat = rvt.vis.local_dominance(np.zeros((30, 31), dtype=np.float32), **params)
    assert flat.dtype == np.float64 and flat.shape == (30, 31)
    # every pixel is above every shifted pixel by the observer height: the sum equals its normalisation (tolerance
    # 1e-6: the float32 sum 0 + 1.7 is 1.7 within 1.2e-7 relative)
    np.testing.assert_allclose(flat, 1.0, rtol=0, atol=1e-6)
    peak = np.zeros((41, 41), dtype=np.float32)
    peak[20, 20] = 10.0
    out = rvt.vis.local_dominance(peak, **params)
    # (h + observer_height) / observer_height with h = 10 m and observer height 1.7 m
    np.testing.assert_allclose(out[20, 20], 11.7 / 1.7, rtol=1e-5)
    pit = -peak
    assert rvt.vis.local_dominance(pit, **params)[20, 20] == 0  # every neighbour is higher than the pit + 1.7 m


def test_local_dominance_char_offset_invariance_flat():
    flat = np.full((25, 26), 2500.0, dtype=np.float32)
    # float32 spacing at 2,500 m is 2.4e-4 m: (2500 + 1.7) - 2500 differs from 1.7 by up to 1.2e-4 m (7e-5 relative)
    np.testing.assert_allclose(rvt.vis.local_dominance(flat, min_rad=2, max_rad=6), 1.0, rtol=0, atol=5e-4)


def test_local_dominance_char_nan_only_at_nan_pixels():
    dem = np.zeros((25, 25), dtype=np.float32)
    dem[12, 12] = np.nan
    out = rvt.vis.local_dominance(dem, min_rad=2, max_rad=5)
    assert np.array_equal(np.isnan(out), np.isnan(dem))  # neighbours stay finite: comparisons with NaN are False


def test_local_dominance_char_non_negative_and_input_untouched():
    dem = synthetic_dem((28, 33), seed=84, nan_patch=True)
    kept = dem.copy()
    out = rvt.vis.local_dominance(dem, min_rad=2, max_rad=7)
    assert_same(dem, kept)
    assert np.nanmin(out) >= 0
    assert out.shape == dem.shape


def test_local_dominance_char_no_data_modifies_input():
    dem = with_value(synthetic_dem((26, 27), seed=85), NO_DATA)
    expected = with_value(synthetic_dem((26, 27), seed=85), np.nan)
    out = rvt.vis.local_dominance(dem, min_rad=2, max_rad=5, no_data=NO_DATA)
    assert_same(dem, expected)  # pins current behaviour, KNOWN_ISSUES candidate: no_data replaced in the input
    assert_same(out, rvt.vis.local_dominance(expected.copy(), min_rad=2, max_rad=5))


def test_local_dominance_char_ve_factor_sign():
    dem = synthetic_dem((26, 27), seed=86)
    # a negative factor mirrors the relief (peaks become pits)
    assert_same(rvt.vis.local_dominance(dem, 2, 5, ve_factor=-1), rvt.vis.local_dominance(-dem, 2, 5))


@pytest.mark.parametrize("kwargs", [dict(ve_factor=10001), dict(ve_factor=-10001)])
def test_local_dominance_char_invalid_parameters(kwargs):
    assert_raises_exact(Exception, rvt.vis.local_dominance, np.zeros((12, 12), dtype=np.float32), 2, 4, **kwargs)


def test_local_dominance_char_invalid_dimensions():
    assert_raises_exact(Exception, rvt.vis.local_dominance, np.zeros((2, 12, 12), dtype=np.float32), 2, 4)


# ----------------------------------------------------------------------------------------------------------------------
# horizon_generate_coarse_dem, horizon_generate_pyramids
# ----------------------------------------------------------------------------------------------------------------------

def _symmetric_index(i, n):
    """Index into an array of length n extended by reflection including the edge value (numpy 'symmetric')."""
    i = i % (2 * n)
    return i if i < n else 2 * n - 1 - i


def ref_coarse_dem(dem_fine, scale, conv_from, conv_to, max_radius):
    """Plain reference: coarse[I, J] = max over the scale x scale window of the fine grid / scale, never below 0.
    The fine grid is cropped by max_radius, extended by edge values to (scale * k + 1) pixels at the right and below,
    extended by reflection for the window, and the window offset wraps around modulo the extended size (np.roll)."""
    fine = dem_fine[max_radius:-max_radius, max_radius:-max_radius]
    n_rows, n_cols = fine.shape
    n_coarse = (n_rows // scale + 1, n_cols // scale + 1)

    def edge_pad(n):
        return 0 if n % scale == 1 else abs(1 - n % scale)

    fine = np.pad(fine, ((0, edge_pad(n_rows)), (0, edge_pad(n_cols))), mode="edge")
    n_rows, n_cols = fine.shape
    ext_rows, ext_cols = n_rows - conv_from + conv_to, n_cols - conv_from + conv_to  # length of the extended grid
    out = np.zeros(n_coarse)
    for i_row in range(n_coarse[0]):
        for i_col in range(n_coarse[1]):
            best = 0.0  # the maximum starts at 0: heights below 0 m are lost
            for a in range(conv_from, conv_from + scale):
                for b in range(conv_from, conv_from + scale):
                    r = (i_row * scale - conv_from - a) % ext_rows + conv_from
                    c = (i_col * scale - conv_from - b) % ext_cols + conv_from
                    best = max(best, float(fine[_symmetric_index(r, n_rows), _symmetric_index(c, n_cols)]))
            out[i_row, i_col] = best / scale
    return np.pad(out, max_radius, mode="constant", constant_values=out.min())


def _conv_window(scale):
    conv_to = scale // 2
    return (1 - conv_to if scale % 2 == 0 else -conv_to), conv_to


@pytest.mark.parametrize("n_rows,n_cols,scale,radius", [(12, 13, 2, 3), (14, 11, 3, 2), (10, 10, 2, 2),
                                                       (9, 16, 4, 3), (7, 7, 3, 1), (11, 12, 5, 2), (6, 9, 2, 1)])
def test_coarse_dem_matches_reference(n_rows, n_cols, scale, radius):
    rng = np.random.default_rng(scale * 100 + n_rows)
    fine = rng.random((n_rows + 2 * radius, n_cols + 2 * radius)) * 50 + 5
    conv_from, conv_to = _conv_window(scale)
    out = rvt.vis.horizon_generate_coarse_dem(fine, scale, conv_from, conv_to, radius)
    expected = ref_coarse_dem(fine, scale, conv_from, conv_to, radius)
    assert out.dtype == np.float64
    # shape: floor(n / scale) + 1 coarse pixels plus the border of max_radius pixels on each side
    assert out.shape == (n_rows // scale + 1 + 2 * radius, n_cols // scale + 1 + 2 * radius)
    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=0)  # same maxima; division by scale is exact to 1 ulp


def test_coarse_dem_constant_and_border_value():
    fine = np.full((16, 18), 12.0)
    out = rvt.vis.horizon_generate_coarse_dem(fine, 2, 0, 1, 3)
    np.testing.assert_allclose(out, 6.0, rtol=0, atol=1e-14)  # constant height / scale, border = minimum = same


def test_coarse_dem_border_is_minimum_of_coarse_values():
    rng = np.random.default_rng(5)
    fine = rng.random((20, 20)) * 10 + 1
    out = rvt.vis.horizon_generate_coarse_dem(fine, 3, -1, 1, 4)
    inner = out[4:-4, 4:-4]
    assert np.all(out[:4] == inner.min()) and np.all(out[-4:] == inner.min())
    assert np.all(out[:, :4] == inner.min()) and np.all(out[:, -4:] == inner.min())


def test_coarse_dem_negative_heights_are_clamped_to_zero():
    fine = -5.0 - np.random.default_rng(6).random((16, 16))
    out = rvt.vis.horizon_generate_coarse_dem(fine, 2, 0, 1, 3)
    # pins current behaviour, KNOWN_ISSUES candidate: the running maximum starts at np.zeros, so a DEM below 0 m
    # gives a coarse DEM of 0
    assert np.all(out == 0)


def test_coarse_dem_even_scale_first_row_wraps_to_the_far_edge():
    fine = np.zeros((8 + 6, 8 + 6))
    fine[3 + 7, 3 + 0] = 8.0  # last row, first column of the cropped 8 x 8 fine grid
    out = rvt.vis.horizon_generate_coarse_dem(fine, 2, 0, 1, 3)
    # pins current behaviour, KNOWN_ISSUES candidate: for an even scale the window reaches one pixel above the first
    # row; the padding supplies none there, np.roll wraps to the last row, so the first coarse row sees the far edge
    assert out[3 + 0, 3 + 0] == 8.0 / 2


def test_coarse_dem_does_not_modify_input():
    rng = np.random.default_rng(8)
    fine = rng.random((18, 18))
    kept = fine.copy()
    rvt.vis.horizon_generate_coarse_dem(fine, 2, 0, 1, 3)
    assert np.array_equal(fine, kept)


def test_pyramids_structure_three_levels():
    rng = np.random.default_rng(9)
    dem = (rng.random((12, 13)) * 20 + 5).astype(np.float32)
    kept = dem.copy()
    pyramid = rvt.vis.horizon_generate_pyramids(dem, num_directions=4, max_fine_radius=30, max_pyramid_radius=7,
                                                pyramid_scale=3)
    assert np.array_equal(dem, kept)
    # levels: 30 / 3^L <= 7 first holds for L = 2 (30, 10, 3.3); the last level searches round(30 / 9) = 3 px
    assert sorted(pyramid) == [0, 1, 2]
    for level, entry in pyramid.items():
        assert sorted(entry) == ["dem", "i_col", "i_lin", "min_radius", "num_directions", "radius_pixels", "shift"]
        assert entry["num_directions"] == 4
        assert np.array_equal(entry["i_lin"], np.arange(entry["dem"].shape[0]))
        assert np.array_equal(entry["i_col"], np.arange(entry["dem"].shape[1]))
    assert [pyramid[i]["radius_pixels"] for i in range(3)] == [7, 7, 3]
    # min radius: 1 on level 0, floor(7 / 3) + 1 - 1 = 2 above
    assert [pyramid[i]["min_radius"] for i in range(3)] == [1, 2, 2]
    for i in range(3):
        expected_shift = rvt.vis.horizon_shift_vector(4, pyramid[i]["radius_pixels"], pyramid[i]["min_radius"])
        assert list(pyramid[i]["shift"]) == list(expected_shift)
        for angle in expected_shift:
            assert pyramid[i]["shift"][angle]["shift"] == expected_shift[angle]["shift"]
            assert np.array_equal(pyramid[i]["shift"][angle]["distance"], expected_shift[angle]["distance"])
    # level 0: the DEM padded by max_pyramid_radius with its minimum; the other levels: coarse DEMs, float64
    level0 = pyramid[0]["dem"]
    assert level0.dtype == np.float32 and level0.shape == (12 + 14, 13 + 14)
    assert np.array_equal(level0[7:-7, 7:-7], dem)
    assert np.all(level0[:7] == dem.min()) and np.all(level0[:, -7:] == dem.min())
    assert pyramid[1]["dem"].dtype == np.float64 and pyramid[1]["dem"].shape == (12 // 3 + 1 + 14, 13 // 3 + 1 + 14)
    assert pyramid[2]["dem"].shape == ((12 // 3 + 1) // 3 + 1 + 14, (13 // 3 + 1) // 3 + 1 + 14)
    conv_from, conv_to = _conv_window(3)
    assert np.array_equal(pyramid[1]["dem"], rvt.vis.horizon_generate_coarse_dem(level0, 3, conv_from, conv_to, 7))


def test_pyramids_single_level_and_defaults():
    dem = (np.random.default_rng(10).random((10, 11)) * 9).astype(np.float32)
    pyramid = rvt.vis.horizon_generate_pyramids(dem, num_directions=4, max_fine_radius=5, max_pyramid_radius=7,
                                                pyramid_scale=3)
    assert sorted(pyramid) == [0]  # 5 <= 7: the fine level only
    assert pyramid[0]["radius_pixels"] == 5.0 and pyramid[0]["min_radius"] == 1
    assert pyramid[0]["dem"].shape == (10 + 14, 11 + 14)
    default = rvt.vis.horizon_generate_pyramids(dem)  # num_directions=4, max_fine_radius=100, radius 7, scale 3
    assert sorted(default) == [0, 1, 2, 3]  # 100, 33.3, 11.1, 3.7: the first level at or below 7 is 3
    assert default[3]["radius_pixels"] == 4.0  # round(100 / 27) = 4
    assert default[0]["num_directions"] == 4


# ----------------------------------------------------------------------------------------------------------------------
# sky_illumination, shadow_horizon
# ----------------------------------------------------------------------------------------------------------------------

SI = dict(max_fine_radius=30, num_directions=8)


@pytest.mark.parametrize("level", [0.0, 100.0, 2500.0])
def test_sky_illumination_char_flat_is_one(level):
    flat = np.full((30, 31), level, dtype=np.float32)
    for model in ("uniform", "overcast"):
        out = rvt.vis.sky_illumination(flat, 1, sky_model=model, **SI)
        assert out.dtype == np.float64 and out.shape == flat.shape
        # tolerance 1e-6: the float32 value of pi enters the uniform model (1.0000000278 in practice)
        np.testing.assert_allclose(out, 1.0, rtol=0, atol=1e-6)
        shaded = rvt.vis.sky_illumination(flat, 1, sky_model=model, compute_shadow=True, **SI)
        if model == "overcast":
            np.testing.assert_allclose(shaded, 1.0, rtol=0, atol=1e-6)  # 0.8 * 1 + 0.2 * (lit = 1)
        else:
            # pins current behaviour, KNOWN_ISSUES candidate: the uniform model is divided by pi only after the
            # shadow term is added, so the shaded uniform result is 0.8 * pi + 0.2 = 2.713 on flat terrain
            np.testing.assert_allclose(shaded, 0.8 * np.pi + 0.2, rtol=0, atol=1e-6)


def test_sky_illumination_char_sky_model_case_insensitive_and_invalid():
    dem = synthetic_dem((24, 26), seed=87)
    assert_same(rvt.vis.sky_illumination(dem.copy(), 1, sky_model="UNIFORM", **SI),
                rvt.vis.sky_illumination(dem.copy(), 1, sky_model="uniform", **SI))
    assert_raises_exact(Exception, rvt.vis.sky_illumination, dem.copy(), 1, sky_model="clear")


def test_sky_illumination_char_shape_dtype_and_input_untouched():
    dem = synthetic_dem((22, 29), seed=88, nan_patch=True)
    kept = dem.copy()
    for kwargs in (dict(), dict(sky_model="uniform"), dict(compute_shadow=True)):
        out = rvt.vis.sky_illumination(dem, 1, **SI, **kwargs)
        assert out.shape == dem.shape and out.dtype == np.float64
    both = rvt.vis.sky_illumination(dem, 1, compute_shadow=True, shadow_horizon_only=True, **SI)
    assert sorted(both) == ["horizon", "shadow"]
    assert_same(dem, kept)


def test_sky_illumination_char_nan_pixels():
    dem = synthetic_dem((24, 26), seed=89, nan_patch=True)
    out = rvt.vis.sky_illumination(dem, 1, **SI)
    nan_in = np.isnan(dem)
    assert np.isnan(out[nan_in]).all()  # the slope is NaN there, so the result is NaN at least at the input NaN


def test_sky_illumination_char_no_data_modifies_input():
    dem = with_value(synthetic_dem((24, 26), seed=90), NO_DATA)
    expected = with_value(synthetic_dem((24, 26), seed=90), np.nan)
    out = rvt.vis.sky_illumination(dem, 1, no_data=NO_DATA, **SI)
    assert_same(dem, expected)  # pins current behaviour, KNOWN_ISSUES candidate: no_data replaced in the input
    assert_same(out, rvt.vis.sky_illumination(expected.copy(), 1, **SI))


@pytest.mark.parametrize("kwargs", [dict(ve_factor=10001), dict(ve_factor=-10001), dict(shadow_az=361),
                                    dict(shadow_az=-1), dict(shadow_el=91), dict(shadow_el=-1)])
def test_sky_illumination_char_invalid_parameters(kwargs):
    assert_raises_exact(Exception, rvt.vis.sky_illumination, np.zeros((12, 12), dtype=np.float32), 1, **kwargs)
    assert_raises_exact(Exception, rvt.vis.shadow_horizon, np.zeros((12, 12), dtype=np.float32), 1, **kwargs)


def test_sky_illumination_char_invalid_resolution():
    assert_raises_exact(Exception, rvt.vis.sky_illumination, np.zeros((12, 12), dtype=np.float32), -1)
    assert_raises_exact(Exception, rvt.vis.shadow_horizon, np.zeros((12, 12), dtype=np.float32), -1)


def test_sky_illumination_char_boundary_angles_accepted():
    dem = synthetic_dem((20, 22), seed=91)
    for az, el in ((0, 0), (360, 90)):
        out = rvt.vis.shadow_horizon(dem.copy(), 1, shadow_az=az, shadow_el=el)
        assert out["shadow"].shape == dem.shape


def _wall(west_height=5.0, shape=(20, 40), wall_cols=10):
    dem = np.zeros(shape, dtype=np.float32)
    dem[:, :wall_cols] = west_height
    return dem


def test_shadow_horizon_flat():
    out = rvt.vis.shadow_horizon(np.full((30, 31), 10.0, dtype=np.float32), 1)
    assert sorted(out) == ["horizon", "shadow"]
    assert out["horizon"].dtype == np.float64 and out["shadow"].shape == (30, 31)
    assert np.all(out["horizon"] == 0)
    # the "shadow" array is 1 where the horizon is below the sun (lit) and 0 in shade
    assert np.all(out["shadow"] == 1)


def test_shadow_horizon_wall_from_the_west_and_east():
    # a 5 m high block on the west side; pixels east of it see its edge at the horizon: atan(5 / d), d = distance in px
    dem = _wall()
    out = rvt.vis.shadow_horizon(dem, 1, shadow_az=270, shadow_el=35)  # sun in the west: horizon searched to the west
    dist = np.arange(1, 13)
    expected_horizon = np.degrees(np.arctan(5.0 / dist))
    # tolerance 1e-3 degrees: float32 slope; distances <= 12 px are searched on the fine level
    np.testing.assert_allclose(out["horizon"][10, 10:22], expected_horizon, rtol=0, atol=1e-3)
    assert np.all(out["horizon"][10, :10] == 0)  # on top of the block nothing is higher
    expected_shadow = (expected_horizon < 35).astype(int)  # 5 / d < tan(35 deg) from d = 8
    assert np.array_equal(out["shadow"][10, 10:22], expected_shadow)
    assert np.array_equal(out["shadow"][10, 10:22], [0] * 7 + [1] * 5)
    # the mirrored DEM with the sun in the east gives the mirrored result
    mirrored = rvt.vis.shadow_horizon(np.ascontiguousarray(dem[:, ::-1]), 1, shadow_az=90, shadow_el=35)
    np.testing.assert_allclose(mirrored["horizon"][10, 39 - np.arange(10, 22)], expected_horizon, rtol=0, atol=1e-3)


def test_shadow_horizon_shadow_dtype_is_integer():
    out = rvt.vis.shadow_horizon(_wall(), 1)
    assert out["shadow"].dtype.kind == "i" and set(np.unique(out["shadow"])) <= {0, 1}


def test_shadow_horizon_resolution_and_ve_factor_scale_the_height():
    # the vertical angle depends on height / (distance * resolution): 5 m at 0.5 m/px equals 10 m at 1 m/px
    a = rvt.vis.shadow_horizon(_wall(10.0), 1, shadow_az=270)["horizon"][10, 10:16]
    b = rvt.vis.shadow_horizon(_wall(5.0), 1, shadow_az=270, ve_factor=2)["horizon"][10, 10:16]
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a, np.degrees(np.arctan(10.0 / np.arange(1, 7))), rtol=0, atol=1e-3)


def test_shadow_horizon_no_data_modifies_input():
    dem = with_value(_wall(), NO_DATA, slice(15, 17), slice(25, 28))
    rvt.vis.shadow_horizon(dem, 1, no_data=NO_DATA)
    assert np.isnan(dem[15:17, 25:28]).all()  # pins current behaviour, KNOWN_ISSUES candidate: replaced in the input


# ----------------------------------------------------------------------------------------------------------------------
# integral_image, topographic_dev
# ----------------------------------------------------------------------------------------------------------------------

def test_integral_image_of_ones_and_docstring_example():
    out = rvt.vis.integral_image(np.ones((4, 6)))
    rows, cols = np.mgrid[1:5, 1:7]
    assert out.dtype == np.float64
    assert np.array_equal(out, rows * cols)  # (i + 1) (j + 1)
    example = np.array([[7, 4, 7, 2], [6, 9, 9, 5], [6, 6, 7, 6]])
    assert np.array_equal(rvt.vis.integral_image(example), [[7, 11, 18, 20], [13, 26, 42, 49], [19, 38, 61, 74]])


@pytest.mark.parametrize("data_type", [np.float64, np.float32, np.int64, np.int32])
def test_integral_image_data_type_and_box_sum(data_type):
    rng = np.random.default_rng(11)
    dem = rng.integers(0, 50, size=(9, 10)).astype(np.float64)
    out = rvt.vis.integral_image(dem, data_type)
    # cumsum keeps float and int64 types and widens smaller integers to the platform integer
    assert out.dtype == np.cumsum(np.zeros(2, dtype=data_type)).dtype and out.shape == dem.shape
    if data_type is not np.int32:
        assert out.dtype == np.dtype(data_type)
    assert out[-1, -1] == dem.sum()
    # sum of the box rows 2..5, columns 3..7 from four table entries (exact for integer-valued data)
    assert out[5, 7] - out[1, 7] - out[5, 2] + out[1, 2] == dem[2:6, 3:8].sum()


def test_integral_image_bool_to_int_and_input_untouched():
    mask = np.array([[True, False, True], [True, True, False]])
    out = rvt.vis.integral_image(mask, np.int64)
    assert out.dtype == np.int64
    assert np.array_equal(out, [[1, 1, 2], [2, 3, 4]])
    dem = np.arange(6.0).reshape(2, 3)
    kept = dem.copy()
    rvt.vis.integral_image(dem)
    assert np.array_equal(dem, kept)


def test_integral_image_requires_two_dimensions():
    with pytest.raises((ValueError, IndexError)):
        rvt.vis.integral_image(np.arange(5.0))  # cumsum along axis 1 of a 1D array


def ref_topographic_dev(dem, radius):
    """Brute-force DEV on the pixels whose window does not wrap: output (i, j) belongs to the pixel (i + 1, j + 1),
    window radius radius, NaN ignored in mean and standard deviation."""
    n_rows, n_cols = dem.shape
    out = np.full(dem.shape, np.nan)
    for i in range(radius, n_rows - radius - 1):
        for j in range(radius, n_cols - radius - 1):
            win = dem[i + 1 - radius:i + 2 + radius, j + 1 - radius:j + 2 + radius].astype(np.float64)
            valid = win[~np.isnan(win)]
            if valid.size == 0:
                continue
            mean = valid.mean()
            std = np.sqrt(abs((valid ** 2).mean() - mean ** 2))
            out[i, j] = (dem[i + 1, j + 1] - mean) / (std + 1e-6)
    return out


@pytest.mark.parametrize("radius", [1, 2, 4])
@pytest.mark.parametrize("with_nan", [False, True])
def test_topographic_dev_matches_brute_force(radius, with_nan):
    rng = np.random.default_rng(12 + radius)
    dem = rng.random((16, 17)) * 10
    if with_nan:
        dem[6:9, 7:10] = np.nan
        dem[2, 3] = np.nan
    valid = ~np.isnan(dem)
    dem0 = np.where(valid, dem, 0.0)
    out = rvt.vis.topographic_dev(dem, rvt.vis.integral_image(valid, np.int64), rvt.vis.integral_image(dem0),
                                  rvt.vis.integral_image(dem0 ** 2), radius)
    assert out.shape == dem.shape and out.dtype == np.float64
    expected = ref_topographic_dev(dem, radius)
    inner = (slice(radius, 16 - radius - 1), slice(radius, 17 - radius - 1))
    # tolerance: the sums come from float64 summed-area tables (differences of values up to 1e4 * 10: 1e-12 relative)
    np.testing.assert_allclose(out[inner], expected[inner], rtol=1e-8, atol=1e-8, equal_nan=True)


def test_topographic_dev_non_positive_radius_returns_the_dem():
    dem = np.arange(20.0).reshape(4, 5)
    tables = [rvt.vis.integral_image(np.ones(dem.shape), np.int64), rvt.vis.integral_image(dem),
              rvt.vis.integral_image(dem ** 2)]
    assert rvt.vis.topographic_dev(dem, *tables, 0) is dem
    assert rvt.vis.topographic_dev(dem, *tables, -3) is dem
    assert rvt.vis.topographic_dev(dem, *tables, 0.9) is dem  # int(0.9) = 0


# ----------------------------------------------------------------------------------------------------------------------
# max_elevation_deviation, mstp
# ----------------------------------------------------------------------------------------------------------------------

def test_max_elevation_deviation_char_shape_dtype_nan_and_input():
    dem = synthetic_dem((22, 25), seed=92, nan_patch=True)
    kept = dem.copy()
    out = rvt.vis.max_elevation_deviation(dem, 1, 5, 2)
    assert out.shape == dem.shape and out.dtype == np.float32
    assert np.array_equal(np.isnan(out), np.isnan(dem))
    assert_same(dem, kept)  # the zero-fill of NaN is done on a padded copy


def test_max_elevation_deviation_char_float_radii_and_step():
    dem = synthetic_dem((22, 25), seed=93)
    assert_same(rvt.vis.max_elevation_deviation(dem, 1.9, 5.2, 2.7), rvt.vis.max_elevation_deviation(dem, 1, 5, 2))


def test_max_elevation_deviation_char_single_spike_is_deviant_at_every_scale():
    dem = np.zeros((31, 31), dtype=np.float32)
    dem[15, 15] = 10.0
    out = rvt.vis.max_elevation_deviation(dem, 1, 4, 1)
    # window of n = (2r + 1)^2 pixels holding one spike: dev = (h - h / n) / (h sqrt(1 / n - 1 / n^2)) = sqrt(n - 1);
    # the maximum |dev| is at the largest radius (n = 81): sqrt(80); tolerance 1e-5: float32 result
    np.testing.assert_allclose(out[15, 15], np.sqrt(80.0), rtol=1e-5)


MSTP_SCALES = dict(local_scale=(1, 2, 1), meso_scale=(2, 4, 1), broad_scale=(4, 6, 2))


def test_mstp_char_shape_dtype_range_and_nan():
    dem = synthetic_dem((30, 32), seed=94, nan_patch=True)
    out = rvt.vis.mstp(dem, **MSTP_SCALES)
    assert out.shape == (3, 30, 32) and out.dtype == np.float32
    nan_in = np.isnan(dem)
    for band in out:
        assert np.array_equal(np.isnan(band), nan_in)
        assert band[~nan_in].min() >= 0 and band[~nan_in].max() <= 1


def test_mstp_char_lightness_is_monotone_and_zero_is_black():
    dem = synthetic_dem((30, 32), seed=95)
    dark = rvt.vis.mstp(dem, lightness=0.5, **MSTP_SCALES)
    light = rvt.vis.mstp(dem, lightness=2.0, **MSTP_SCALES)
    assert np.all(light >= dark)  # 1 - exp(-lightness * |dev|) increases with lightness
    assert np.all(rvt.vis.mstp(dem, lightness=0, **MSTP_SCALES) == 0)


def test_mstp_char_no_data_modifies_input():
    dem = with_value(synthetic_dem((30, 32), seed=96), NO_DATA)
    expected = with_value(synthetic_dem((30, 32), seed=96), np.nan)
    out = rvt.vis.mstp(dem, no_data=NO_DATA, **MSTP_SCALES)
    assert_same(dem, expected)  # pins current behaviour, KNOWN_ISSUES candidate: no_data replaced in the input
    assert_same(out, rvt.vis.mstp(expected.copy(), **MSTP_SCALES))


def test_mstp_char_does_not_modify_input_without_no_data():
    dem = synthetic_dem((30, 32), seed=97, nan_patch=True)
    kept = dem.copy()
    rvt.vis.mstp(dem, **MSTP_SCALES)
    assert_same(dem, kept)


@pytest.mark.parametrize("name,value", [("local_scale", (3, 2, 1)), ("meso_scale", (6, 4, 1)),
                                        ("broad_scale", (9, 6, 1)),  # minimum above maximum
                                        ("local_scale", (1, 2, 2)), ("meso_scale", (2, 4, 3)),
                                        ("broad_scale", (4, 6, 5)),  # step larger than max - min
                                        ("local_scale", (3, 3, 1))])  # max - min = 0 < step
def test_mstp_char_invalid_scales(name, value):
    kwargs = dict(MSTP_SCALES)
    kwargs[name] = value
    assert_raises_exact(Exception, rvt.vis.mstp, np.zeros((12, 12), dtype=np.float32), **kwargs)


def test_mstp_char_step_equal_to_range_is_accepted():
    out = rvt.vis.mstp(synthetic_dem((20, 20), seed=98), local_scale=(1, 2, 1), meso_scale=(2, 4, 2),
                       broad_scale=(4, 6, 2))
    assert out.shape == (3, 20, 20)


def test_mstp_char_ve_factor_limits():
    dem = synthetic_dem((20, 20), seed=99)
    assert_raises_exact(Exception, rvt.vis.mstp, dem, ve_factor=10001, **MSTP_SCALES)
    # pins current behaviour, KNOWN_ISSUES candidate: the lower limit is -1000 in the code, -10000 in the message
    assert_raises_exact(Exception, rvt.vis.mstp, dem, ve_factor=-1001, **MSTP_SCALES)
    assert rvt.vis.mstp(dem, ve_factor=-1000, **MSTP_SCALES).shape == (3, 20, 20)
    assert rvt.vis.mstp(dem, ve_factor=10000, **MSTP_SCALES).shape == (3, 20, 20)


# ----------------------------------------------------------------------------------------------------------------------
# fill_where_nan
# ----------------------------------------------------------------------------------------------------------------------

def ref_idw(dem, radius, power):
    """Inverse distance weighting from the valid pixels of the original DEM inside a (2r + 1)^2 window clipped at the
    array border; NaN when the window holds no valid pixel."""
    out = dem.astype(np.float64)
    n_rows, n_cols = dem.shape
    for i, j in zip(*np.where(np.isnan(dem))):
        r0, r1 = max(i - radius, 0), min(i + radius + 1, n_rows)
        c0, c1 = max(j - radius, 0), min(j + radius + 1, n_cols)
        win = dem[r0:r1, c0:c1].astype(np.float64)
        ii, jj = np.mgrid[r0:r1, c0:c1]
        ok = ~np.isnan(win)
        if not ok.any():
            out[i, j] = np.nan
            continue
        weight = 1.0 / np.hypot(ii - i, jj - j)[ok] ** power
        out[i, j] = np.sum(win[ok] * weight) / np.sum(weight)
    return out


def test_fill_where_nan_returns_input_when_complete():
    dem = synthetic_dem((8, 9), seed=100)
    for method in ("idw", "linear_row", "kd_tree", "nearest_neighbour", "not a method"):
        assert rvt.vis.fill_where_nan(dem, method) is dem  # no NaN: returned before the method is looked at


def test_fill_where_nan_linear_row_hand_example():
    dem = np.array([[1, np.nan, 3], [np.nan, np.nan, 6]], dtype=np.float32)
    kept = dem.copy()
    out = rvt.vis.fill_where_nan(dem, "linear_row")
    # the flattened array 1, nan, 3, nan, nan, 6 is interpolated linearly: 1, 2, 3, 4, 5, 6
    assert_same(out, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    assert_same(dem, kept)  # input untouched, output is a copy with the input dtype


@pytest.mark.parametrize("method", ["kd_tree", "nearest_neighbour", "nearest_neighbor"])
def test_fill_where_nan_nearest_hand_example(method):
    dem = np.array([[7, np.nan, np.nan, np.nan, np.nan, 9]], dtype=np.float32)
    # columns 1, 2 are nearer to the 7 (distances 1, 2 against 4, 3); columns 3, 4 are nearer to the 9
    assert_same(rvt.vis.fill_where_nan(dem, method), np.array([[7, 7, 7, 9, 9, 9]], dtype=np.float32))


@pytest.mark.parametrize("method,radius,power", [("idw", 20, 2), ("idw_5_2", 5, 2.0), ("idw_3_1.5", 3, 1.5),
                                                  ("idw_5", 20, 2)])  # two parts: radius and power stay default
@pytest.mark.parametrize("shape", [(14, 14), (16, 9)])  # square and tall
def test_fill_where_nan_idw_matches_reference(method, radius, power, shape):
    dem = synthetic_dem(shape, seed=101, nan_patch=True)
    dem[shape[0] - 2:, shape[1] - 2:] = np.nan  # a corner block at the bottom and right edge
    out = rvt.vis.fill_where_nan(dem, method)
    assert out.dtype == np.float32 and out.shape == dem.shape
    # tolerance 1e-6 relative: float32 data weighted in float64, stored as float32
    np.testing.assert_allclose(out, ref_idw(dem, radius, power), rtol=1e-6, atol=1e-6, equal_nan=True)
    valid = ~np.isnan(dem)
    assert np.array_equal(out[valid], dem[valid])  # valid pixels are kept


def test_fill_where_nan_idw_window_without_valid_pixel_stays_nan():
    dem = synthetic_dem((20, 20), seed=102)
    dem[5:15, 5:15] = np.nan
    out = rvt.vis.fill_where_nan(dem, "idw_2_2")
    assert np.isnan(out[9:11, 9:11]).all()  # the 5 x 5 window around the centre holds no valid pixel
    assert not np.isnan(out[5, 5])


def test_fill_where_nan_idw_wide_array_truncates_the_window():
    rng = np.random.default_rng(13)
    dem = rng.random((8, 14)).astype(np.float32)
    dem[:, 8:] += 1000
    dem[3, 7] = np.nan
    out = rvt.vis.fill_where_nan(dem, "idw_8_2")
    # pins current behaviour, KNOWN_ISSUES candidate (vis.py: `i_column_end = dem.shape[0]` where shape[1] is meant):
    # when the window passes the right edge, its end column is the number of ROWS (8), so on this wide 8 x 14 array
    # the columns 8..13 are dropped and the result ignores the 1000 m pixels in them
    truncated = ref_idw(dem[:, :8].copy(), 8, 2)[3, 7]
    correct = ref_idw(dem, 8, 2)[3, 7]
    np.testing.assert_allclose(out[3, 7], truncated, rtol=1e-6)
    assert abs(out[3, 7] - correct) > 100


def test_fill_where_nan_idw_wide_array_can_raise_index_error():
    rng = np.random.default_rng(14)
    dem = rng.random((8, 14)).astype(np.float32)
    dem[3, 12] = np.nan
    # pins current behaviour, KNOWN_ISSUES candidate: the same truncation leaves a window narrower than the centre
    # column, and indexing it fails
    with pytest.raises(IndexError):
        rvt.vis.fill_where_nan(dem, "idw_5_2")


def test_fill_where_nan_unknown_method_raises():
    dem = np.array([[1, np.nan, 3]], dtype=np.float32)
    assert_raises_exact(Exception, rvt.vis.fill_where_nan, dem, "cubic")
