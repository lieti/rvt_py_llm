"""
Sky illumination: rvt.vis.sky_illumination (level 0 searched on the unpadded extent, blocked level search and
directional terms, fit-free linear spline, numba kernels `_sky_illum_search_kernel` and
`_sky_illum_search_literal_kernel`) against the np.roll / RectBivariateSpline formulation of tag baseline-fixed.
"""

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import rvt.vis
from conftest import assert_bitwise, assert_same, synthetic_dem


# ----------------------------------------------------------------------------------------------------------------------
# reference implementation (baseline-fixed, transcribed; slope_aspect and horizon_generate_pyramids are unchanged and
# taken from rvt)
# ----------------------------------------------------------------------------------------------------------------------

def ref_sky_illumination(dem, resolution, sky_model="overcast", compute_shadow=False, shadow_horizon_only=False,
                         max_fine_radius=100, num_directions=32, shadow_az=315, shadow_el=35, ve_factor=1,
                         no_data=None):
    pyramid_scale = 2
    max_pyramid_radius = 20
    if no_data is not None:
        dem[dem == no_data] = np.nan
    dem = dem.astype(np.float32)
    dem = dem * ve_factor
    if sky_model.lower() == "overcast":
        compute_overcast, compute_uniform = True, False
    elif sky_model.lower() == "uniform":
        compute_overcast, compute_uniform = False, True
    else:
        raise ValueError(sky_model)

    _ = rvt.vis.slope_aspect(np.pad(dem, max_pyramid_radius, mode="symmetric"), resolution, resolution)
    slope = _["slope"]
    aspect = _["aspect"]
    pyramid = rvt.vis.horizon_generate_pyramids(dem, num_directions=num_directions, max_fine_radius=max_fine_radius,
                                                max_pyramid_radius=max_pyramid_radius, pyramid_scale=pyramid_scale)
    n_levels = np.max([i for i in pyramid])
    conv_to = int(np.floor(pyramid_scale / 2.))
    conv_from = 1 - conv_to if (pyramid_scale % 2) == 0 else -conv_to
    da = np.pi / num_directions

    uniform_a = np.zeros((dem.shape[0] + 2 * max_pyramid_radius, dem.shape[1] + 2 * max_pyramid_radius),
                         dtype=np.float32)
    uniform_b = np.copy(uniform_a)
    if compute_overcast:
        overcast_c = np.zeros((dem.shape[0] + 2 * max_pyramid_radius, dem.shape[1] + 2 * max_pyramid_radius),
                              dtype=np.float32)
        overcast_d = np.copy(overcast_c)
    if compute_shadow:
        _ = np.array([d for d in pyramid[0]["shift"]])
        i = np.argmin(np.abs(_ - (360 - shadow_az)))
        shadow_az = _[i]
        shadow_out = np.zeros(dem.shape, dtype=np.float32)

    for i_dir, direction in enumerate(pyramid[0]["shift"]):
        dir_rad = np.radians(direction)
        max_slope = np.zeros(pyramid[n_levels]["dem"].shape, dtype=np.float32) - 1000
        for i_level in reversed(range(n_levels + 1)):
            height = pyramid[i_level]["dem"]
            move = pyramid[i_level]["shift"]
            for i_rad, radius in enumerate(move[direction]["distance"]):
                shift_indx = move[direction]["shift"][i_rad]
                _ = np.maximum((np.roll(height, shift_indx, axis=(0, 1)) - height) / radius, 0.)
                max_slope = np.maximum(max_slope, _)
            if i_level > 0:
                lin_fine = pyramid[i_level - 1]["i_lin"] + (
                        conv_from + max_pyramid_radius * pyramid_scale - max_pyramid_radius)
                col_fine = pyramid[i_level - 1]["i_col"] + (
                        conv_from + max_pyramid_radius * pyramid_scale - max_pyramid_radius)
                lin_coarse = pyramid[i_level]["i_lin"] * pyramid_scale
                col_coarse = pyramid[i_level]["i_col"] * pyramid_scale
                interp_spline = RectBivariateSpline(lin_coarse, col_coarse, max_slope, kx=1, ky=1)
                max_slope = interp_spline(lin_fine, col_fine)

        _ = np.arctan(max_slope)
        uniform_a = uniform_a + (np.cos(_)) ** 2
        _d_aspect = -2 * np.sin(da) * np.cos(dir_rad - aspect)
        uniform_b = uniform_b + np.maximum(_d_aspect * (np.pi / 4. - _ / 2. - np.sin(2. * _) / 4.), 0)
        if compute_overcast:
            _cos3 = (np.cos(_)) ** 3
            overcast_c = overcast_c + np.maximum(_cos3, 0)
            overcast_d = overcast_d + np.maximum(_d_aspect * (2. / 3. - np.cos(_) + _cos3 / 3.), 0)
        if compute_shadow and (direction == shadow_az):
            horizon_out = np.degrees(_[max_pyramid_radius:-max_pyramid_radius, max_pyramid_radius:-max_pyramid_radius])
            shadow_out = (horizon_out < shadow_el) * 1
            if shadow_horizon_only:
                return {"shadow": shadow_out, "horizon": horizon_out}

    uniform_out = da * np.cos(slope) * uniform_a + np.sin(slope) * np.minimum(uniform_b, np.pi)
    uniform_out = uniform_out[max_pyramid_radius:-max_pyramid_radius, max_pyramid_radius:-max_pyramid_radius]
    if compute_overcast:
        overcast_out = (2. * da / 3.) * np.cos(slope) * overcast_c + np.sin(slope) * overcast_d
        overcast_out = overcast_out[max_pyramid_radius:-max_pyramid_radius, max_pyramid_radius:-max_pyramid_radius]
        overcast_out = 0.33 * uniform_out + 0.67 * overcast_out
        overcast_out = overcast_out / overcast_out.max()
    if compute_shadow:
        uniform_sh_out = (0.8 * uniform_out + 0.2 * shadow_out)
        if compute_overcast:
            overcast_sh_out = (0.8 * overcast_out + 0.2 * shadow_out)
    uniform_out = uniform_out / np.pi

    if compute_uniform and not compute_shadow:
        return uniform_out
    elif compute_uniform and compute_shadow:
        return uniform_sh_out
    elif compute_overcast and not compute_shadow:
        return overcast_out
    else:
        return overcast_sh_out


# ----------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------

CASES = [
    dict(sky_model="overcast"),
    dict(sky_model="uniform"),
    dict(sky_model="overcast", compute_shadow=True, shadow_az=315, shadow_el=35),
    dict(sky_model="uniform", compute_shadow=True, shadow_az=90, shadow_el=10),
    dict(sky_model="overcast", max_fine_radius=20),   # one pyramid level only
    dict(sky_model="overcast", max_fine_radius=45),   # two levels, last radius not a power of the scale
    dict(sky_model="uniform", compute_shadow=True, shadow_az=0, shadow_el=20, max_fine_radius=60, ve_factor=3),
]


@pytest.mark.parametrize("case", CASES)
def test_sky_illumination_matches_reference(case):
    dem = synthetic_dem((27, 34), seed=31, relief=15.0)
    kwargs = dict(dict(resolution=0.5, num_directions=16), **case)
    assert_same(rvt.vis.sky_illumination(dem.copy(), **kwargs), ref_sky_illumination(dem.copy(), **kwargs))


@pytest.mark.parametrize("case", [dict(sky_model="overcast"), dict(sky_model="uniform"),
                                  dict(sky_model="uniform", compute_shadow=True, shadow_horizon_only=True)])
def test_sky_illumination_default_directions_and_nan(case):
    # 32 directions (default), NaN cells (NaN propagates through the pyramid padding and the spline: the
    # RectBivariateSpline fallback runs and the level-0 search starts from NaN)
    dem = synthetic_dem((22, 25), seed=32, nan_patch=True)
    kwargs = dict(dict(resolution=1.0, max_fine_radius=40), **case)
    assert_same(rvt.vis.sky_illumination(dem.copy(), **kwargs), ref_sky_illumination(dem.copy(), **kwargs))


def test_sky_illumination_nan_reaches_level0_search():
    # one NaN corner cell: the pyramid padding (constant dem.min() = NaN) and the spline make the level-0 search start
    # from NaN, while cells farther than the search radius from the padding see only finite heights there
    dem = synthetic_dem((66, 70), seed=39, relief=12.0)
    dem[0, 0] = np.nan
    kwargs = dict(resolution=0.5, sky_model="uniform", num_directions=8, max_fine_radius=40)
    expected = ref_sky_illumination(dem.copy(), **kwargs)
    assert_same(rvt.vis.sky_illumination(dem.copy(), **kwargs), expected)


@pytest.mark.parametrize("shadow_az", [315, 180, 7])
def test_shadow_horizon_matches_reference(shadow_az):
    dem = synthetic_dem((24, 30), seed=33, relief=30.0)
    kwargs = dict(resolution=0.25, compute_shadow=True, shadow_horizon_only=True, shadow_az=shadow_az, shadow_el=35,
                  num_directions=16, max_fine_radius=50)
    expected = ref_sky_illumination(dem.copy(), **kwargs)
    assert_same(rvt.vis.sky_illumination(dem.copy(), **kwargs), expected)
    assert_same(rvt.vis.shadow_horizon(dem.copy(), resolution=0.25, shadow_az=shadow_az, shadow_el=35),
                ref_sky_illumination(dem.copy(), resolution=0.25, compute_shadow=True, shadow_horizon_only=True,
                                     shadow_az=shadow_az, shadow_el=35))


def test_sky_illumination_multi_block():
    # wide tile: level 0 and the directional terms run in several row blocks (multiples of 16 rows)
    dem = synthetic_dem((40, 1400), seed=34, relief=10.0)
    kwargs = dict(resolution=1.0, sky_model="overcast", num_directions=8, max_fine_radius=30)
    assert_same(rvt.vis.sky_illumination(dem.copy(), **kwargs), ref_sky_illumination(dem.copy(), **kwargs))


def test_sky_illumination_deterministic():
    dem = synthetic_dem((26, 29), seed=35)
    kwargs = dict(resolution=0.5, sky_model="overcast", compute_shadow=True, num_directions=16)
    assert_bitwise(rvt.vis.sky_illumination(dem.copy(), **kwargs), rvt.vis.sky_illumination(dem.copy(), **kwargs))


@pytest.mark.parametrize("case", [dict(sky_model="overcast", compute_shadow=True),
                                  dict(sky_model="uniform", max_fine_radius=30)])
def test_sky_illumination_accel_equals_numpy(run_both, case):
    dem = synthetic_dem((28, 31), seed=36, relief=12.0)
    kwargs = dict(dict(resolution=0.5, num_directions=16), **case)
    kernels = [rvt.vis._sky_illum_search_kernel]
    if case.get("compute_shadow"):
        kernels.append(rvt.vis._sky_illum_search_literal_kernel)
    out_accel, out_numpy = run_both(lambda: rvt.vis.sky_illumination(dem.copy(), **kwargs), kernels=kernels)
    assert_bitwise(out_accel, out_numpy)
    assert_same(out_numpy, ref_sky_illumination(dem.copy(), **kwargs))


def test_sky_illumination_numpy_path(numpy_path):
    dem = synthetic_dem((23, 27), seed=37, relief=12.0)
    kwargs = dict(resolution=0.5, sky_model="overcast", compute_shadow=True, num_directions=16)
    assert_same(rvt.vis.sky_illumination(dem.copy(), **kwargs), ref_sky_illumination(dem.copy(), **kwargs))


def test_resample_matches_rect_bivariate_spline():
    # the fit-free evaluation of the linear spline against scipy's fit, on a pyramid-like grid and an irregular one
    rng = np.random.default_rng(38)
    cases = [(np.arange(15) * 2, np.arange(19) * 2, np.arange(28) + 1, np.arange(36) + 1),
             (np.array([0., 0.5, 2., 3., 7.]), np.array([0., 1., 1.1, 4.]), np.linspace(-1, 8, 13), np.linspace(0, 4, 9))]
    for x, y, xf, yf in cases:
        z = rng.normal(size=(x.size, y.size))
        assert_same(rvt.vis._sky_illum_resample(x, y, z, xf, yf), RectBivariateSpline(x, y, z, kx=1, ky=1)(xf, yf))
        z[1, 2] = np.nan
        assert_same(rvt.vis._sky_illum_resample(x, y, z, xf, yf), RectBivariateSpline(x, y, z, kx=1, ky=1)(xf, yf))
