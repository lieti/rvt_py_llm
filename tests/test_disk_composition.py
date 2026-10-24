"""
0.2.0 disk windows: wiring of ``kernel_shape`` through the higher-level functions, determinism, invalid values.

- ``mstp`` and ``slrm`` with a disk shape equal (bitwise) the composition of the lower-level functions with the same
  shape, written out here as the code composes them (float32 DEM times ``ve_factor``; per scale
  ``1 - exp(-lightness * |DEV_max|)`` clipped to [0, 1], bands broad, meso, local; SLRM ``z - mean_filter(z)``).
  ``msrm`` is checked against the brute-force reference in test_disk_oracle.py (it computes its spectra once, so it
  is not bitwise a composition).
- ``DefaultValues.get_slrm``, ``get_msrm``, ``get_mstp`` pass their own ``*_kernel_shape`` attribute and no other.
- ``rvt.blend.e3mstp`` and ``e4mstp`` take the shape from ``DefaultValues``: with the three attributes set they
  equal the recipe of tests/test_char_e3e4_recipe.py (its helpers, imported) with the SLRM and MSTP layers computed
  with the shape, and differ from the square result.
- (f) two calls give the same bits; ``RVT_NO_ACCEL=1`` gives the same bits on the disk path (it has no numba
  kernel).
- (g) an unknown shape raises ``ValueError`` in ``mean_filter`` and ``max_elevation_deviation`` and ``Exception``
  (exactly that type, as for their other bad arguments) in ``slrm``, ``msrm``, ``mstp`` (STATUS J7).
"""

import numpy as np
import pytest

import rvt.blend as rb
import rvt.default
import rvt.vis
import test_char_e3e4_recipe as recipe
from conftest import assert_bitwise, assert_same, synthetic_dem

SHAPES = ("disk", "disk_equal_area")
SCALES = dict(local_scale=(1, 5, 1), meso_scale=(5, 20, 5), broad_scale=(10, 40, 10))


def _dem(nan=True):
    return synthetic_dem((64, 72), seed=401, relief=20.0, nan_patch=nan)


def _with_shapes(dv, **shapes):
    """Set ``<vis>_kernel_shape`` attributes after checking that each exists with its default "square"."""
    for vis, shape in shapes.items():
        name = "{}_kernel_shape".format(vis)
        assert getattr(dv, name) == "square"
        setattr(dv, name, shape)
    return dv


# ----------------------------------------------------------------------------------------------------------------------
# (e) composition
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nan", [False, True], ids=["plain", "nan"])
@pytest.mark.parametrize("shape", SHAPES)
def test_mstp_with_disk_is_the_composition(shape, nan):
    dem = _dem(nan)
    out = rvt.vis.mstp(dem.copy(), lightness=1.1, ve_factor=2, kernel_shape=shape, **SCALES)
    z = dem.astype(np.float32) * 2
    bands = []
    for scale in (SCALES["broad_scale"], SCALES["meso_scale"], SCALES["local_scale"]):
        dev = rvt.vis.max_elevation_deviation(dem=z, minimum_radius=scale[0], maximum_radius=scale[1],
                                              step=scale[2], kernel_shape=shape)
        band = 1 - np.exp(-1.1 * np.abs(dev))
        band[band < 0] = 0
        band[band > 1] = 1
        bands.append(band)
    assert_bitwise(out, np.asarray(bands))
    assert not np.array_equal(out, rvt.vis.mstp(dem.copy(), lightness=1.1, ve_factor=2, **SCALES), equal_nan=True)


@pytest.mark.parametrize("nan", [False, True], ids=["plain", "nan"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("radius", [10, 20, 50])
def test_slrm_with_disk_is_the_composition(radius, shape, nan):
    dem = _dem(nan)
    out = rvt.vis.slrm(dem.copy(), radius, 2, kernel_shape=shape)
    z = dem.astype(np.float32) * 2
    assert_bitwise(out, z - rvt.vis.mean_filter(z, radius, kernel_shape=shape))
    assert not np.array_equal(out, rvt.vis.slrm(dem.copy(), radius, 2), equal_nan=True)


def test_slrm_with_disk_no_data():
    dem = synthetic_dem((48, 52), seed=402, relief=20.0)
    dem[10:14, 20:25] = -9999.0
    expected_input = dem.copy()
    expected_input[expected_input == -9999.0] = np.nan
    z = expected_input.astype(np.float32)
    out = rvt.vis.slrm(dem, 12, no_data=-9999.0, kernel_shape="disk")
    assert np.array_equal(dem, expected_input, equal_nan=True)  # no_data -> NaN in the caller's array, as before
    assert_bitwise(out, z - rvt.vis.mean_filter(z, 12, kernel_shape="disk"))


# ----------------------------------------------------------------------------------------------------------------------
# DefaultValues.get_*
# ----------------------------------------------------------------------------------------------------------------------

def _small_default():
    dv = rvt.default.DefaultValues()
    dv.slrm_rad_cell = 12
    dv.msrm_feature_min = 0.0
    dv.msrm_feature_max = 16.0  # at 0.5 m/px: i = 0, n = ceil(15.5 ** 0.5) = 4, radii 0, 1, 4, 9, 16 px
    dv.msrm_scaling_factor = 2
    dv.mstp_local_scale, dv.mstp_meso_scale, dv.mstp_broad_scale = (1, 5, 1), (5, 20, 5), (10, 40, 10)
    dv.mstp_lightness = 1.1
    return dv


def _vis_calls(dem, dv, shapes):
    """The rvt.vis calls that get_slrm, get_msrm, get_mstp make, with the shape given per visualization."""
    return {
        "slrm": rvt.vis.slrm(dem=dem.copy(), radius_cell=dv.slrm_rad_cell, ve_factor=dv.ve_factor,
                             kernel_shape=shapes["slrm"]),
        "msrm": rvt.vis.msrm(dem=dem.copy(), resolution=0.5, feature_min=dv.msrm_feature_min,
                             feature_max=dv.msrm_feature_max, scaling_factor=dv.msrm_scaling_factor,
                             ve_factor=dv.ve_factor, kernel_shape=shapes["msrm"]),
        # get_mstp does not pass ve_factor (pinned by the frozen tests)
        "mstp": rvt.vis.mstp(dem=dem.copy(), local_scale=dv.mstp_local_scale, meso_scale=dv.mstp_meso_scale,
                             broad_scale=dv.mstp_broad_scale, lightness=dv.mstp_lightness,
                             kernel_shape=shapes["mstp"]),
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("vis", ["slrm", "msrm", "mstp"])
def test_get_method_passes_its_own_kernel_shape(vis, shape):
    dem = _dem()
    dv = _with_shapes(_small_default(), **{vis: shape})
    shapes = {name: (shape if name == vis else "square") for name in ("slrm", "msrm", "mstp")}
    expected = _vis_calls(dem, dv, shapes)
    assert_bitwise(dv.get_slrm(dem.copy()), expected["slrm"])
    assert_bitwise(dv.get_msrm(dem.copy(), 0.5), expected["msrm"])
    assert_bitwise(dv.get_mstp(dem.copy()), expected["mstp"])
    square = _vis_calls(dem, dv, {"slrm": "square", "msrm": "square", "mstp": "square"})
    assert not np.array_equal(expected[vis], square[vis], equal_nan=True)


# ----------------------------------------------------------------------------------------------------------------------
# e3MSTP and e4MSTP
# ----------------------------------------------------------------------------------------------------------------------

def _disk_default(shape):
    # radii of the custom default: SLRM 12 px, MSTP meso 3..12 px, broad 12..30 px, so the shape changes them
    return _with_shapes(recipe._custom_default(), slrm=shape, msrm=shape, mstp=shape)


def _disk_layers(images, dem, d, shape):
    """Replace the SLRM and MSTP layers of the recipe images by the calls with the shape (as get_slrm, get_mstp)."""
    images = dict(images)
    if "slrm" in images:
        images["slrm"] = rvt.vis.slrm(dem=dem.copy(), radius_cell=d.slrm_rad_cell, ve_factor=d.ve_factor,
                                      kernel_shape=shape)
    images["mstp"] = rvt.vis.mstp(dem=dem.copy(), local_scale=d.mstp_local_scale, meso_scale=d.mstp_meso_scale,
                                  broad_scale=d.mstp_broad_scale, lightness=d.mstp_lightness, kernel_shape=shape)
    return images


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("case", ["custom_default", "nan_patch_custom"])
def test_e3mstp_with_disk_attributes_equals_the_recipe(case, shape):
    dem = synthetic_dem((40, 44), seed=64 if case == "custom_default" else 62, relief=20.0,
                        nan_patch=case != "custom_default")
    resolution = 1.0
    d = _disk_default(shape)
    images = _disk_layers(recipe._e3_images(dem.copy(), resolution, d), dem, d, shape)
    expected = recipe._e3_recipe(images)
    out = rb.e3mstp(dem.copy(), resolution, default=d)
    assert_same(out, expected)
    square = rb.e3mstp(dem.copy(), resolution, default=recipe._custom_default())
    assert not np.array_equal(out, square, equal_nan=True)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("case", ["custom_default", "nan_patch_custom"])
def test_e4mstp_with_disk_attributes_equals_the_recipe(case, shape):
    dem = synthetic_dem((40, 44), seed=64 if case == "custom_default" else 62, relief=20.0,
                        nan_patch=case != "custom_default")
    resolution = 1.0
    d = _disk_default(shape)
    images = _disk_layers(recipe._e4_images(dem.copy(), resolution, d), dem, d, shape)
    expected = recipe._e4_recipe(images)
    out = rb.e4mstp(dem.copy(), resolution, default=d)
    assert_same(out, expected)
    square = rb.e4mstp(dem.copy(), resolution, default=recipe._custom_default())
    assert not np.array_equal(out, square, equal_nan=True)


# ----------------------------------------------------------------------------------------------------------------------
# (f) determinism, RVT_NO_ACCEL
# ----------------------------------------------------------------------------------------------------------------------

def _disk_outputs(dem, shape):
    return {
        "mean_filter": rvt.vis.mean_filter(dem.copy(), 13, kernel_shape=shape),
        "max_elevation_deviation": rvt.vis.max_elevation_deviation(dem.copy(), 5, 20, 5, kernel_shape=shape),
        "mstp": rvt.vis.mstp(dem.copy(), kernel_shape=shape, **SCALES),
        "slrm": rvt.vis.slrm(dem.copy(), 20, kernel_shape=shape),
        "msrm": rvt.vis.msrm(dem.copy(), 1.0, 1.0, 33.0, 2, kernel_shape=shape),
    }


@pytest.mark.parametrize("shape", SHAPES)
def test_disk_path_is_deterministic_and_does_not_depend_on_numba(shape, monkeypatch):
    dem = _dem()
    monkeypatch.delenv("RVT_NO_ACCEL", raising=False)
    first = _disk_outputs(dem, shape)
    second = _disk_outputs(dem, shape)
    assert_bitwise(second, first)
    monkeypatch.setenv("RVT_NO_ACCEL", "1")
    assert_bitwise(_disk_outputs(dem, shape), first)


# ----------------------------------------------------------------------------------------------------------------------
# (g) invalid shape
# ----------------------------------------------------------------------------------------------------------------------

BAD = ["circle", "octagon", ""]


@pytest.mark.parametrize("bad", BAD)
def test_unknown_shape_raises_value_error_in_window_functions(bad):
    dem = _dem()
    for radius in (3, 10):
        with pytest.raises(ValueError):
            rvt.vis.mean_filter(dem.copy(), radius, kernel_shape=bad)
    for band in ((1, 5, 1), (5, 20, 5)):
        with pytest.raises(ValueError):
            rvt.vis.max_elevation_deviation(dem.copy(), *band, kernel_shape=bad)


@pytest.mark.parametrize("bad", BAD)
def test_unknown_shape_raises_exception_in_slrm_msrm_mstp(bad):
    dem = _dem()
    calls = [
        lambda: rvt.vis.slrm(dem.copy(), 20, kernel_shape=bad),
        lambda: rvt.vis.msrm(dem.copy(), 1.0, 1.0, 33.0, 2, kernel_shape=bad),
        lambda: rvt.vis.msrm(dem.copy(), 1.0, 1.0, 9.0, 2, kernel_shape=bad),  # largest radius 4 px
        lambda: rvt.vis.mstp(dem.copy(), kernel_shape=bad, **SCALES),
    ]
    for call in calls:
        with pytest.raises(Exception) as info:
            call()
        assert info.type is Exception  # the type these functions raise for bad arguments, not a subclass
