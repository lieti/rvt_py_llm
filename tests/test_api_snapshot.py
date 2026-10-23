"""
Snapshot of the public API of rvt (signatures, DefaultValues attributes, RVTVisualization members).

The expected values in ``SNAPSHOT`` were generated once from the code at tag ``start`` by calling ``collect_api()``
and pasting the result; this is the only generated expectation among the characterization tests (every other
expected value is written by hand or derived from a plain reference implementation). The test fails when a public
name is added, removed or renamed, or when a signature, a default argument or a ``DefaultValues`` default changes.
It is meant to be read as a list: if the change is intended, edit ``SNAPSHOT`` in the same commit.

What is recorded, for the modules rvt.vis, rvt.blend, rvt.blend_func, rvt.default and rvt.tile:

- every public module-level function defined in the module: ``str(inspect.signature(f))``;
- every public class defined in the module: the signature of its constructor, and the signature of every public
  method (name without a leading underscore) defined in the class body;
- the members of ``rvt.default.RVTVisualization`` (name and value);
- the attribute names and ``repr`` of the values of a fresh ``rvt.default.DefaultValues()``.

Memory addresses (default argument that is an instance) and the ``pathlib`` module path of ``Path`` are normalised
so that the snapshot does not depend on the platform or the Python minor version.
"""

import enum
import importlib
import inspect
import re

import pytest

import rvt.default

MODULES = ("rvt.vis", "rvt.blend", "rvt.blend_func", "rvt.default", "rvt.tile")

_ADDRESS = re.compile(r" object at 0x[0-9A-Fa-f]+>")


def _sig(obj):
    """str(inspect.signature(obj)) without memory addresses and without the private pathlib module name."""
    text = str(inspect.signature(obj))
    text = _ADDRESS.sub(" object>", text)
    return text.replace("pathlib._local.Path", "pathlib.Path")


def _is_public(name):
    return not name.startswith("_")


def collect_api():
    """Return the API as a flat dict of strings: key = dotted name, value = signature or repr."""
    api = {}
    for mod_name in MODULES:
        module = importlib.import_module(mod_name)
        for name, obj in sorted(vars(module).items()):
            if not _is_public(name) or getattr(obj, "__module__", None) != mod_name:
                continue
            if inspect.isfunction(obj):
                api["{}.{}".format(mod_name, name)] = _sig(obj)
            elif inspect.isclass(obj):
                key = "{}.{}".format(mod_name, name)
                if issubclass(obj, enum.Enum):
                    # record the members instead of the constructor
                    for member in obj:
                        api["{}.{}".format(key, member.name)] = repr(member.value)
                    continue
                api[key] = _sig(obj)
                for meth_name, meth in sorted(vars(obj).items()):
                    if _is_public(meth_name) and inspect.isfunction(meth):
                        api["{}.{}".format(key, meth_name)] = _sig(meth)
    defaults = rvt.default.DefaultValues()
    for attr, value in sorted(vars(defaults).items()):
        api["rvt.default.DefaultValues().{}".format(attr)] = repr(value)
    return api


SNAPSHOT = {
    # rvt.blend.BlenderCombination
    'rvt.blend.BlenderCombination': '(dem_arr=None, dem_resolution=None, dem_path=None)',
    'rvt.blend.BlenderCombination.add_dem_arr': '(self, dem_arr, dem_resolution)',
    'rvt.blend.BlenderCombination.add_dem_path': '(self, dem_path)',
    'rvt.blend.BlenderCombination.add_layer': '(self, layer: rvt.blend.BlenderLayer)',
    'rvt.blend.BlenderCombination.check_data': '(self)',
    'rvt.blend.BlenderCombination.create_layer': "(self, vis_method=None, normalization='value', minimum=None, maximum=None, blend_mode='normal', opacity=100, colormap=None, min_colormap_cut=None, max_colormap_cut=None, image=None, image_path=None)",
    'rvt.blend.BlenderCombination.create_log_file': '(self, dem_path, combination_name, render_path, default: rvt.default.DefaultValues, terrain_sett_name=None, custom_dir=None, computation_time=None)',
    'rvt.blend.BlenderCombination.layers_info': '(self)',
    'rvt.blend.BlenderCombination.read_from_file': '(self, file_path)',
    'rvt.blend.BlenderCombination.read_from_json': '(self, json_data)',
    'rvt.blend.BlenderCombination.remove_all_layers': '(self)',
    'rvt.blend.BlenderCombination.render_all_images': '(self, default=None, save_visualizations=False, save_render_path=None, save_float=True, save_8bit=False, no_data=None)',
    'rvt.blend.BlenderCombination.save_to_file': '(self, file_path)',
    'rvt.blend.BlenderCombination.to_json': '(self)',
    # rvt.blend.BlenderCombinations
    'rvt.blend.BlenderCombinations': '()',
    'rvt.blend.BlenderCombinations.add_combination': '(self, combination: rvt.blend.BlenderCombination, name=None)',
    'rvt.blend.BlenderCombinations.combination_in_combinations': '(self, input_combination: rvt.blend.BlenderCombination)',
    'rvt.blend.BlenderCombinations.combinations_names': '(self)',
    'rvt.blend.BlenderCombinations.read_from_file': '(self, file_path)',
    'rvt.blend.BlenderCombinations.remove_all_combinations': '(self)',
    'rvt.blend.BlenderCombinations.remove_combination_by_name': '(self, name)',
    'rvt.blend.BlenderCombinations.save_to_file': '(self, file_path)',
    'rvt.blend.BlenderCombinations.select_combination_by_name': '(self, name)',
    # rvt.blend.BlenderLayer
    'rvt.blend.BlenderLayer': "(vis_method=None, normalization='value', minimum=None, maximum=None, blend_mode='normal', opacity=100, colormap=None, min_colormap_cut=None, max_colormap_cut=None, image=None, image_path=None)",
    'rvt.blend.BlenderLayer.check_data': '(self)',
    # rvt.blend.TerrainSettings
    'rvt.blend.TerrainSettings': '()',
    'rvt.blend.TerrainSettings.apply_terrain': '(self, default: rvt.default.DefaultValues, combination: rvt.blend.BlenderCombination)',
    'rvt.blend.TerrainSettings.read_from_file': '(self, file_path)',
    'rvt.blend.TerrainSettings.read_from_json': '(self, json_data)',
    # rvt.blend.TerrainsSettings
    'rvt.blend.TerrainsSettings': '()',
    'rvt.blend.TerrainsSettings.read_from_file': '(self, file_path)',
    'rvt.blend.TerrainsSettings.select_terrain_settings_by_name': '(self, name)',
    # rvt.blend.color_relief_image_map
    'rvt.blend.color_relief_image_map': "(dem, resolution, default: rvt.default.DefaultValues = <rvt.default.DefaultValues object>, colormap='OrRd', min_colormap_cut=0, max_colormap_cut=1, no_data=None)",
    # rvt.blend.compare_2_combinations
    'rvt.blend.compare_2_combinations': '(combination1: rvt.blend.BlenderCombination, combination2: rvt.blend.BlenderCombination)',
    # rvt.blend.create_blender_file_example
    'rvt.blend.create_blender_file_example': '(file_path=None)',
    # rvt.blend.e3mstp
    'rvt.blend.e3mstp': '(dem, resolution, default: rvt.default.DefaultValues = <rvt.default.DefaultValues object>, no_data=None)',
    # rvt.blend.e4mstp
    'rvt.blend.e4mstp': '(dem, resolution, default: rvt.default.DefaultValues = <rvt.default.DefaultValues object>, no_data=None)',
    # rvt.blend_func.advanced_normalization
    'rvt.blend_func.advanced_normalization': '(image, minimum, maximum, normalization)',
    # rvt.blend_func.apply_opacity
    'rvt.blend_func.apply_opacity': '(active, background, opacity)',
    # rvt.blend_func.blend_images
    'rvt.blend_func.blend_images': '(blend_mode, active, background, min_c=None, max_c=None)',
    # rvt.blend_func.blend_luminosity
    'rvt.blend_func.blend_luminosity': '(active, background, min_c=None, max_c=None)',
    # rvt.blend_func.blend_multi_dim_images
    'rvt.blend_func.blend_multi_dim_images': '(blend_mode, active, background)',
    # rvt.blend_func.blend_multiply
    'rvt.blend_func.blend_multiply': '(active, background)',
    # rvt.blend_func.blend_normal
    'rvt.blend_func.blend_normal': '(active, background)',
    # rvt.blend_func.blend_overlay
    'rvt.blend_func.blend_overlay': '(active, background)',
    # rvt.blend_func.blend_screen
    'rvt.blend_func.blend_screen': '(active, background)',
    # rvt.blend_func.blend_soft_light
    'rvt.blend_func.blend_soft_light': '(active, background)',
    # rvt.blend_func.channel_max
    'rvt.blend_func.channel_max': '(r: numpy.ndarray, g: numpy.ndarray, b: numpy.ndarray)',
    # rvt.blend_func.channel_min
    'rvt.blend_func.channel_min': '(r: numpy.ndarray, g: numpy.ndarray, b: numpy.ndarray)',
    # rvt.blend_func.clip_color
    'rvt.blend_func.clip_color': '(c, min_c=None, max_c=None)',
    # rvt.blend_func.cut_off_normalize
    'rvt.blend_func.cut_off_normalize': '(image, mode, cutoff_min=None, cutoff_max=None, bool_norm=True)',
    # rvt.blend_func.equation_blend
    'rvt.blend_func.equation_blend': '(blend_mode, active, background)',
    # rvt.blend_func.gray_scale_to_color_ramp
    'rvt.blend_func.gray_scale_to_color_ramp': '(gray_scale, colormap, min_colormap_cut=None, max_colormap_cut=None, alpha=False, output_8bit=True)',
    # rvt.blend_func.lin_cutoff_calc_from_perc
    'rvt.blend_func.lin_cutoff_calc_from_perc': '(image, minimum, maximum)',
    # rvt.blend_func.lum
    'rvt.blend_func.lum': '(img)',
    # rvt.blend_func.matrix_eq_max_gt_one
    'rvt.blend_func.matrix_eq_max_gt_one': '(r: numpy.ndarray, idx_max_c_gt_one, lum_c, max_c)',
    # rvt.blend_func.matrix_eq_min_lt_zero
    'rvt.blend_func.matrix_eq_min_lt_zero': '(r: numpy.ndarray, idx_min_lt_zero, lum_c, min_c)',
    # rvt.blend_func.normalize_image
    'rvt.blend_func.normalize_image': '(visualization, image, min_norm, max_norm, normalization)',
    # rvt.blend_func.normalize_lin
    'rvt.blend_func.normalize_lin': '(image, minimum, maximum)',
    # rvt.blend_func.normalize_perc
    'rvt.blend_func.normalize_perc': '(image, minimum, maximum)',
    # rvt.blend_func.render_images
    'rvt.blend_func.render_images': '(active, background, opacity)',
    # rvt.blend_func.scale_0_to_1
    'rvt.blend_func.scale_0_to_1': '(numeric_value)',
    # rvt.blend_func.scale_strict_0_to_1
    'rvt.blend_func.scale_strict_0_to_1': '(numeric_value)',
    # rvt.blend_func.scale_within_0_and_1
    'rvt.blend_func.scale_within_0_and_1': '(numeric_value)',
    # rvt.blend_func.truncate_colormap
    'rvt.blend_func.truncate_colormap': '(cmap, minval=0.0, maxval=1.0, n=100)',
    # rvt.default.DefaultValues
    'rvt.default.DefaultValues': '()',
    'rvt.default.DefaultValues().asvf_bytscl': "('value', 0.7, 0.9)",
    'rvt.default.DefaultValues().asvf_compute': '0',
    'rvt.default.DefaultValues().asvf_dir': '315',
    'rvt.default.DefaultValues().asvf_level': '1',
    'rvt.default.DefaultValues().hs_bytscl': "('value', 0.0, 1.0)",
    'rvt.default.DefaultValues().hs_compute': '1',
    'rvt.default.DefaultValues().hs_save_8bit': '0',
    'rvt.default.DefaultValues().hs_save_float': '1',
    'rvt.default.DefaultValues().hs_shadow': '0',
    'rvt.default.DefaultValues().hs_sun_azi': '315',
    'rvt.default.DefaultValues().hs_sun_el': '35',
    'rvt.default.DefaultValues().ld_anglr_res': '15',
    'rvt.default.DefaultValues().ld_bytscl': "('value', 0.5, 1.8)",
    'rvt.default.DefaultValues().ld_compute': '0',
    'rvt.default.DefaultValues().ld_max_rad': '20',
    'rvt.default.DefaultValues().ld_min_rad': '10',
    'rvt.default.DefaultValues().ld_observer_h': '1.7',
    'rvt.default.DefaultValues().ld_rad_inc': '1',
    'rvt.default.DefaultValues().ld_save_8bit': '0',
    'rvt.default.DefaultValues().ld_save_float': '1',
    'rvt.default.DefaultValues().mhs_bytscl': "('value', 0.0, 1.0)",
    'rvt.default.DefaultValues().mhs_compute': '0',
    'rvt.default.DefaultValues().mhs_nr_dir': '16',
    'rvt.default.DefaultValues().mhs_save_8bit': '0',
    'rvt.default.DefaultValues().mhs_save_float': '1',
    'rvt.default.DefaultValues().mhs_sun_el': '35',
    'rvt.default.DefaultValues().msrm_bytscl': "('value', -2.5, 2.5)",
    'rvt.default.DefaultValues().msrm_compute': '0',
    'rvt.default.DefaultValues().msrm_feature_max': '20',
    'rvt.default.DefaultValues().msrm_feature_min': '0',
    'rvt.default.DefaultValues().msrm_save_8bit': '0',
    'rvt.default.DefaultValues().msrm_save_float': '1',
    'rvt.default.DefaultValues().msrm_scaling_factor': '2',
    'rvt.default.DefaultValues().mstp_broad_scale': '(50, 500, 50)',
    'rvt.default.DefaultValues().mstp_bytscl': "('value', 0.0, 1.0)",
    'rvt.default.DefaultValues().mstp_compute': '0',
    'rvt.default.DefaultValues().mstp_lightness': '0.9',
    'rvt.default.DefaultValues().mstp_local_scale': '(1, 5, 1)',
    'rvt.default.DefaultValues().mstp_meso_scale': '(5, 50, 5)',
    'rvt.default.DefaultValues().mstp_save_8bit': '0',
    'rvt.default.DefaultValues().mstp_save_float': '1',
    'rvt.default.DefaultValues().neg_opns_bytscl': "('value', 60.0, 95.0)",
    'rvt.default.DefaultValues().neg_opns_compute': '0',
    'rvt.default.DefaultValues().neg_opns_save_8bit': '0',
    'rvt.default.DefaultValues().neg_opns_save_float': '1',
    'rvt.default.DefaultValues().overwrite': '0',
    'rvt.default.DefaultValues().pos_opns_bytscl': "('value', 60.0, 95.0)",
    'rvt.default.DefaultValues().pos_opns_compute': '0',
    'rvt.default.DefaultValues().sim_bytscl': "('percent', 0.25, 0.0)",
    'rvt.default.DefaultValues().sim_compute': '0',
    'rvt.default.DefaultValues().sim_compute_shadow': '0',
    'rvt.default.DefaultValues().sim_nr_dir': '32',
    'rvt.default.DefaultValues().sim_save_8bit': '0',
    'rvt.default.DefaultValues().sim_save_float': '1',
    'rvt.default.DefaultValues().sim_shadow_az': '315',
    'rvt.default.DefaultValues().sim_shadow_dist': '100',
    'rvt.default.DefaultValues().sim_shadow_el': '35',
    'rvt.default.DefaultValues().sim_sky_mod': "'overcast'",
    'rvt.default.DefaultValues().slp_bytscl': "('value', 0.0, 51.0)",
    'rvt.default.DefaultValues().slp_compute': '0',
    'rvt.default.DefaultValues().slp_output_units': "'degree'",
    'rvt.default.DefaultValues().slp_save_8bit': '0',
    'rvt.default.DefaultValues().slp_save_float': '1',
    'rvt.default.DefaultValues().slrm_bytscl': "('value', -2.0, 2.0)",
    'rvt.default.DefaultValues().slrm_compute': '0',
    'rvt.default.DefaultValues().slrm_rad_cell': '20',
    'rvt.default.DefaultValues().slrm_save_8bit': '0',
    'rvt.default.DefaultValues().slrm_save_float': '1',
    'rvt.default.DefaultValues().svf_bytscl': "('value', 0.6375, 1.0)",
    'rvt.default.DefaultValues().svf_compute': '0',
    'rvt.default.DefaultValues().svf_n_dir': '16',
    'rvt.default.DefaultValues().svf_noise': '0',
    'rvt.default.DefaultValues().svf_r_max': '10',
    'rvt.default.DefaultValues().svf_save_8bit': '0',
    'rvt.default.DefaultValues().svf_save_float': '1',
    'rvt.default.DefaultValues().tile_size': '(4000, 4000)',
    'rvt.default.DefaultValues().tile_size_limit': '100000000',
    'rvt.default.DefaultValues().ve_factor': '1',
    'rvt.default.DefaultValues.calculate_visualization': '(self, visualization: rvt.default.RVTVisualization, dem: <built-in function array>, resolution_x: float, resolution_y: float, no_data: Optional[float] = None, save_float: bool = True, save_8bit: bool = False) -> Optional[Tuple[<built-in function array>, <built-in function array>]]',
    'rvt.default.DefaultValues.create_log_file': '(self, dem_path, custom_dir=None, compute_time=None)',
    'rvt.default.DefaultValues.float_to_8bit': '(self, float_arr: <built-in function array>, visualization: rvt.default.RVTVisualization, x_res: float = None, y_res: float = None, no_data: Optional[float] = None)',
    'rvt.default.DefaultValues.get_asvf_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_asvf_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_hillshade': '(self, dem_arr, resolution_x, resolution_y, no_data=None)',
    'rvt.default.DefaultValues.get_hillshade_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_hillshade_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_local_dominance': '(self, dem_arr, no_data=None)',
    'rvt.default.DefaultValues.get_local_dominance_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_local_dominance_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_msrm': '(self, dem_arr, resolution, no_data=None)',
    'rvt.default.DefaultValues.get_msrm_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_msrm_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_mstp': '(self, dem_arr, no_data=None)',
    'rvt.default.DefaultValues.get_mstp_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_mstp_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_multi_hillshade': '(self, dem_arr, resolution_x, resolution_y, no_data=None)',
    'rvt.default.DefaultValues.get_multi_hillshade_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_multi_hillshade_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_neg_opns': '(self, dem_arr, resolution, no_data=None)',
    'rvt.default.DefaultValues.get_neg_opns_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_neg_opns_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_opns_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_opns_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_shadow': '(self, dem_arr, resolution, no_data=None)',
    'rvt.default.DefaultValues.get_shadow_file_name': '(self, dem_path)',
    'rvt.default.DefaultValues.get_shadow_path': '(self, dem_path)',
    'rvt.default.DefaultValues.get_sky_illumination': '(self, dem_arr, resolution, no_data=None)',
    'rvt.default.DefaultValues.get_sky_illumination_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_sky_illumination_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_sky_view_factor': '(self, dem_arr, resolution, compute_svf=True, compute_asvf=False, compute_opns=False, no_data=None)',
    'rvt.default.DefaultValues.get_slope': '(self, dem_arr, resolution_x, resolution_y, no_data=None)',
    'rvt.default.DefaultValues.get_slope_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_slope_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_slrm': '(self, dem_arr, no_data=None)',
    'rvt.default.DefaultValues.get_slrm_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_slrm_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_svf_file_name': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_svf_path': '(self, dem_path, bit8=False)',
    'rvt.default.DefaultValues.get_visualization_file_name': '(self, rvt_visualization: rvt.default.RVTVisualization, dem_path: pathlib.Path, path_8bit: bool) -> str',
    'rvt.default.DefaultValues.get_visualization_path': '(self, rvt_visualization: rvt.default.RVTVisualization, dem_path: pathlib.Path, output_dir_path: pathlib.Path, path_8bit: bool) -> pathlib.Path',
    'rvt.default.DefaultValues.read_default_from_file': '(self, file_path)',
    'rvt.default.DefaultValues.save_default_to_file': '(self, file_path=None)',
    'rvt.default.DefaultValues.save_hillshade': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None, save_shadow=None)',
    'rvt.default.DefaultValues.save_local_dominance': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_msrm': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_mstp': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_multi_hillshade': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_neg_opns': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_sky_illumination': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_sky_view_factor': '(self, dem_path, save_svf=True, save_asvf=False, save_opns=False, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_slope': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_slrm': '(self, dem_path, custom_dir=None, save_float=None, save_8bit=None)',
    'rvt.default.DefaultValues.save_visualizations': '(self, dem_path, custom_dir=None)',
    # rvt.default.RVTVisualization
    'rvt.default.RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR': "'asvf'",
    'rvt.default.RVTVisualization.HILLSHADE': "'hs'",
    'rvt.default.RVTVisualization.LOCAL_DOMINANCE': "'ld'",
    'rvt.default.RVTVisualization.MULTI_HILLSHADE': "'mhs'",
    'rvt.default.RVTVisualization.MULTI_SCALE_RELIEF_MODEL': "'msrm'",
    'rvt.default.RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION': "'mstp'",
    'rvt.default.RVTVisualization.NEGATIVE_OPENNESS': "'neg_opns'",
    'rvt.default.RVTVisualization.POSITIVE_OPENNESS': "'pos_opns'",
    'rvt.default.RVTVisualization.SHADOW': "'shd'",
    'rvt.default.RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL': "'slrm'",
    'rvt.default.RVTVisualization.SKY_ILLUMINATION': "'sim'",
    'rvt.default.RVTVisualization.SKY_VIEW_FACTOR': "'svf'",
    'rvt.default.RVTVisualization.SLOPE': "'slp'",
    # rvt.default.get_raster_arr
    'rvt.default.get_raster_arr': '(raster_path)',
    # rvt.default.get_raster_size
    'rvt.default.get_raster_size': '(raster_path, band=1)',
    # rvt.default.save_raster
    'rvt.default.save_raster': '(src_raster_path, out_raster_path, out_raster_arr: numpy.ndarray, no_data=None, e_type=6)',
    # rvt.tile.save_rvt_visualization_tile_by_tile
    'rvt.tile.save_rvt_visualization_tile_by_tile': "(rvt_visualization: 'rvt.default.RVTVisualization', rvt_default: 'rvt.default.DefaultValues', dem_path: pathlib.Path, output_dir_path: Optional[pathlib.Path] = None, save_float: bool = True, save_8bit: bool = False) -> None",
    # rvt.tile.save_visualization_tile_by_tile
    'rvt.tile.save_visualization_tile_by_tile': '(visualization_function: Callable, function_parameters: Optional[Dict[str, Optional[Any]]], dem_path: pathlib.Path, overlap: int, tile_size_x: int, tile_size_y: int, out_raster_path: pathlib.Path, out_raster_nr_of_bands: int = 1, out_raster_e_type: int = 6, out_visualization_dict_key: Optional[str] = None) -> None',
    # rvt.vis.byte_scale
    'rvt.vis.byte_scale': '(data, c_min=None, c_max=None, high=255, low=0, no_data=None)',
    # rvt.vis.fill_where_nan
    'rvt.vis.fill_where_nan': "(dem, method='idw')",
    # rvt.vis.hillshade
    'rvt.vis.hillshade': '(dem, resolution_x, resolution_y, sun_azimuth=315, sun_elevation=35, slope=None, aspect=None, ve_factor=1, no_data=None)',
    # rvt.vis.horizon_generate_coarse_dem
    'rvt.vis.horizon_generate_coarse_dem': '(dem_fine, pyramid_scale, conv_from, conv_to, max_radius)',
    # rvt.vis.horizon_generate_pyramids
    'rvt.vis.horizon_generate_pyramids': '(dem, num_directions=4, max_fine_radius=100, max_pyramid_radius=7, pyramid_scale=3)',
    # rvt.vis.horizon_shift_vector
    'rvt.vis.horizon_shift_vector': '(num_directions=16, radius_pixels=10, min_radius=1)',
    # rvt.vis.integral_image
    'rvt.vis.integral_image': "(dem, data_type=<class 'numpy.float64'>)",
    # rvt.vis.local_dominance
    'rvt.vis.local_dominance': '(dem, min_rad=10, max_rad=20, rad_inc=1, angular_res=15, observer_height=1.7, ve_factor=1, no_data=None)',
    # rvt.vis.max_elevation_deviation
    'rvt.vis.max_elevation_deviation': '(dem, minimum_radius, maximum_radius, step)',
    # rvt.vis.mean_filter
    'rvt.vis.mean_filter': '(dem, kernel_radius)',
    # rvt.vis.msrm
    'rvt.vis.msrm': '(dem, resolution, feature_min, feature_max, scaling_factor, ve_factor=1, no_data=None)',
    # rvt.vis.mstp
    'rvt.vis.mstp': '(dem, local_scale=(1, 5, 1), meso_scale=(5, 50, 5), broad_scale=(50, 500, 50), lightness=1.2, ve_factor=1, no_data=None)',
    # rvt.vis.multi_hillshade
    'rvt.vis.multi_hillshade': '(dem, resolution_x, resolution_y, nr_directions=16, sun_elevation=35, slope=None, aspect=None, ve_factor=1, no_data=None)',
    # rvt.vis.roll_fill_nans
    'rvt.vis.roll_fill_nans': '(dem, shift, axis)',
    # rvt.vis.shadow_horizon
    'rvt.vis.shadow_horizon': '(dem, resolution, shadow_az=315, shadow_el=35, ve_factor=1, no_data=None)',
    # rvt.vis.sky_illumination
    'rvt.vis.sky_illumination': "(dem, resolution, sky_model='overcast', compute_shadow=False, shadow_horizon_only=False, max_fine_radius=100, num_directions=32, shadow_az=315, shadow_el=35, ve_factor=1, no_data=None)",
    # rvt.vis.sky_view_factor
    'rvt.vis.sky_view_factor': '(dem, resolution, compute_svf=True, compute_opns=False, compute_asvf=False, svf_n_dir=16, svf_r_max=10, svf_noise=0, asvf_dir=315, asvf_level=1, ve_factor=1, no_data=None)',
    # rvt.vis.sky_view_factor_compute
    'rvt.vis.sky_view_factor_compute': '(height_arr, radius_max=10, radius_min=1, num_directions=16, compute_svf=True, compute_opns=False, compute_asvf=False, a_main_direction=315.0, a_poly_level=4, a_min_weight=0.4)',
    # rvt.vis.slope_aspect
    'rvt.vis.slope_aspect': "(dem, resolution_x=1, resolution_y=1, output_units='radian', ve_factor=1, no_data=None)",
    # rvt.vis.slrm
    'rvt.vis.slrm': '(dem, radius_cell=20, ve_factor=1, no_data=None)',
    # rvt.vis.topographic_dev
    'rvt.vis.topographic_dev': '(dem, dem_i_nr_pixels, dem_i1, dem_i2, kernel_radius)',
}


def test_snapshot_not_empty():
    # guards against an accidentally emptied literal
    assert len(SNAPSHOT) > 200


def test_public_api_matches_snapshot():
    current = collect_api()
    removed = sorted(set(SNAPSHOT) - set(current))
    added = sorted(set(current) - set(SNAPSHOT))
    changed = sorted(k for k in set(SNAPSHOT) & set(current) if SNAPSHOT[k] != current[k])
    report = []
    report += ["removed: {}".format(k) for k in removed]
    report += ["added:   {}".format(k) for k in added]
    report += ["changed: {}\n    was {}\n    now {}".format(k, SNAPSHOT[k], current[k]) for k in changed]
    assert not report, "public API differs from the snapshot:\n" + "\n".join(report)


@pytest.mark.parametrize("mod_name", MODULES)
def test_every_module_has_entries(mod_name):
    assert any(k.startswith(mod_name + ".") for k in SNAPSHOT)
