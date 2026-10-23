"""
Relief Visualization Toolbox – Tile-by-tile processing of large rasters

Computes a visualization of a DEM that does not fit into memory: the DEM GeoTIFF is read in tiles of
``tile_size_x`` x ``tile_size_y`` px, each tile is extended by an overlap of ``overlap`` px on every side that
has a neighbour, the visualization is computed on the extended tile, the overlap is cut off again and the
remaining block is written into an output GeoTIFF at the position of the tile. With an overlap at least as
large as the neighbourhood that a pixel's value depends on, the tiled result equals the untiled one.

Public functions:

- ``save_visualization_tile_by_tile``: any function that takes ``dem=`` (for example from ``rvt.vis``), with
  its parameters as a dict and an overlap chosen by the caller;
- ``save_rvt_visualization_tile_by_tile``: one ``rvt.default.RVTVisualization`` with the parameters of an
  ``rvt.default.DefaultValues`` instance; the overlap is chosen per visualization by
  ``_get_rvt_visualization_overlap``, the float and the 8-bit raster can be written in the same pass, and file
  names follow ``DefaultValues.get_*_file_name``.

Relation to the other modules: ``rvt.vis`` holds the numerics, ``rvt.default`` holds the parameters and calls
``save_rvt_visualization_tile_by_tile`` from its ``save_*`` methods when the DEM has more pixels than
``DefaultValues.tile_size_limit``. ``rvt.blend`` does not use this module.

Tiling scheme (same code in both public functions): tiles start at multiples of the tile size from the
top-left pixel, the last tile of a row or column is smaller when the raster size is not a multiple of the
tile size, and the overlap is clipped at the raster edge (no padding is added; each ``rvt.vis`` function
treats the raster edge as it does for an untiled DEM). The output raster is created empty first and opened
in update mode once per tile.

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
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Union, Tuple
import numpy as np
from osgeo import gdal
import rvt.default


def _create_blank_raster(
        in_data_set: gdal.Dataset,
        out_raster_path: Path,
        nr_bands: int = 1,
        no_data: float = np.nan,
        e_type: int = 6,
):
    """
    Create an empty GeoTIFF with the size, projection and geotransform of ``in_data_set``.

    Parameters
    ----------
    in_data_set : osgeo.gdal.Dataset
        Open source raster (the DEM); band 1 gives the size in px.
    out_raster_path : pathlib.Path
        Path of the GeoTIFF to create; an existing file is replaced.
    nr_bands : int
        Number of bands of the new raster.
    no_data : float
        No-data value set on band 1 only (default NaN). A GeoTIFF stores one no-data value per file, so GDAL
        reports it for every band.
    e_type : int
        GDAL data type code: 6 = ``GDT_Float32``, 1 = ``GDT_Byte``.

    Notes
    -----
    Creation option ``BIGTIFF=IF_NEEDED`` only; no compression option is passed (``rvt.default.save_raster``
    passes ``COMPRESS=LZW``). The data type of the source raster is not used.
    """
    gtiff_driver = gdal.GetDriverByName("GTiff")
    band = in_data_set.GetRasterBand(1)
    x_size = band.XSize  # number of columns
    y_size = band.YSize  # number of rows
    out_ds = gtiff_driver.Create(out_raster_path.as_posix(), xsize=x_size, ysize=y_size, bands=nr_bands, eType=e_type,
                                 options=["BIGTIFF=IF_NEEDED"])
    out_ds.SetProjection(in_data_set.GetProjection())
    out_ds.SetGeoTransform(in_data_set.GetGeoTransform())
    out_ds.GetRasterBand(1).SetNoDataValue(no_data)
    out_ds.FlushCache()
    out_ds = None


def save_visualization_tile_by_tile(
        visualization_function: Callable,
        function_parameters: Optional[Dict[str, Optional[Any]]],
        dem_path: Path,
        overlap: int,
        tile_size_x: int,
        tile_size_y: int,
        out_raster_path: Path,
        out_raster_nr_of_bands: int = 1,
        out_raster_e_type: int = 6,
        out_visualization_dict_key: Optional[str] = None,
) -> None:
    """
    Compute any visualization function on a DEM GeoTIFF tile by tile and write the result to a GeoTIFF.

    Reads band 1 of ``dem_path`` in tiles of ``tile_size_x`` x ``tile_size_y`` px extended by ``overlap`` px
    towards every neighbouring tile, calls ``visualization_function(dem=tile, **function_parameters)``, cuts
    the overlap off the result and writes it at the tile position into ``out_raster_path``. Use it for DEMs that
    do not fit into memory. The tile is passed as read by GDAL (the dtype of the file); conversion to float32
    and the no-data handling are left to ``visualization_function``.

    Parameters
    ----------
    visualization_function : Callable
        Function with a keyword parameter ``dem`` (2D ``numpy.ndarray``), e.g. ``rvt.vis.slope_aspect``. It must
        return an array with the shape of its input (``(H, W)``, or ``(B, H, W)`` when
        ``out_raster_nr_of_bands`` > 1), or a dict of such arrays (see ``out_visualization_dict_key``).
    function_parameters : dict or None
        Keyword arguments for ``visualization_function`` except ``dem``. Entries ``resolution``,
        ``resolution_x``, ``resolution_y`` (m/px) and ``no_data`` whose value is None are filled from the DEM
        (x pixel size ``gt[1]``, y pixel size ``-gt[5]``, no-data of band 1). The filled values are written into
        the caller's dict (in place). None is accepted by the type hint but raises TypeError (see Raises).
    dem_path : pathlib.Path
        Path of the DEM GeoTIFF (elevation in m).
    overlap : int
        Overlap in px read from each neighbouring tile and cut off again after the computation. It must be at
        least the radius in px of the neighbourhood a pixel's value depends on, otherwise values within that
        distance of a tile seam differ from the untiled result.
    tile_size_x : int
        Tile width in px (number of columns), without overlap.
    tile_size_y : int
        Tile height in px (number of rows), without overlap.
    out_raster_path : pathlib.Path
        Path of the output GeoTIFF; an existing file is replaced. It gets the size, projection and geotransform
        of the DEM and the no-data value NaN.
    out_raster_nr_of_bands : int
        Number of bands of the result. With 1 the result is treated as ``(H, W)``, otherwise as ``(B, H, W)``.
    out_raster_e_type : int
        GDAL data type code of the output: 6 = ``GDT_Float32`` (default), 1 = ``GDT_Byte``; see
        https://gdal.org/api/raster_c_api.html#_CPPv412GDALDataType.
    out_visualization_dict_key : str or None
        None if ``visualization_function`` returns an array. If it returns a dict, the key of the array to save,
        e.g. ``"slope"`` for ``rvt.vis.slope_aspect`` (which returns ``{"slope": ..., "aspect": ...}``).

    Returns
    -------
    None
        The result is written to ``out_raster_path``.

    Raises
    ------
    TypeError
        If ``function_parameters`` is None (the membership test ``"resolution" in function_parameters`` runs
        before the None check).

    Notes
    -----
    Known issue: the checks for a missing DEM and for tile sizes below 50 px build an ``Exception`` but do not
    raise it, so they have no effect; a missing DEM fails later in GDAL. See docs/KNOWN_ISSUES.md.

    Known issue: ``function_parameters=None`` raises TypeError although the signature allows None. See
    docs/KNOWN_ISSUES.md.

    Known issue: the y pixel size is ``-gt[5]`` without ``abs`` (``rvt.default.get_raster_arr`` uses ``abs``), so
    for a raster with a positive ``gt[5]`` (south-up) the resolution passed on is negative and most
    ``rvt.vis`` functions raise. See docs/KNOWN_ISSUES.md.

    The tile is written into the output file through a new ``gdal.Open(..., GA_Update)`` per tile, and the
    file is flushed after every band, so a failed run leaves a partly written raster behind.
    """
    # Structure of this function:
    #   1. (intended) argument checks; the Exception objects are not raised (see Notes)
    #   2. open the DEM, read pixel size, no-data value and raster size
    #   3. fill None entries of function_parameters (resolution*, no_data) from the DEM, in place
    #   4. create the empty output raster
    #   5. for each tile (row-major): clip the overlap at the raster edge, read tile + overlap,
    #      compute, cut the overlap off the result, write the block at (x, y)
    if not dem_path.exists():
        Exception("rvt.tile.save_visualization_tile_by_tile: Input dem path does not exist!")
    if tile_size_x < 50 or tile_size_y < 50:
        Exception("rvt.tile.save_visualization_tile_by_tile: Tile size too small (tile_size_x, tile_size_y),"
                  " it needs to be bigger than 50 pixels!")

    dem_ds = gdal.Open(dem_path.as_posix())
    gt = dem_ds.GetGeoTransform()
    x_res = gt[1]  # x_resolution
    y_res = -gt[5]  # y_resolution
    no_data = dem_ds.GetRasterBand(1).GetNoDataValue()
    band = dem_ds.GetRasterBand(1)
    x_size = band.XSize  # number of columns
    y_size = band.YSize  # number of rows

    # set resolution and no_data function_parameters if needed (are set to None) from dem
    if "resolution" in function_parameters:
        if function_parameters["resolution"] is None:
            function_parameters["resolution"] = x_res
    if "resolution_x" in function_parameters:
        if function_parameters["resolution_x"] is None:
            function_parameters["resolution_x"] = x_res
    if "resolution_y" in function_parameters:
        if function_parameters["resolution_y"] is None:
            function_parameters["resolution_y"] = y_res
    if "no_data" in function_parameters:
        if function_parameters["no_data"] is None:
            function_parameters["no_data"] = no_data

    _create_blank_raster(in_data_set=dem_ds, out_raster_path=out_raster_path, nr_bands=out_raster_nr_of_bands,
                         e_type=out_raster_e_type)

    for y in range(0, y_size, tile_size_y):
        if y + tile_size_y < y_size:  # if rows overlap
            rows = tile_size_y
        else:
            rows = y_size - y
        for x in range(0, x_size, tile_size_x):
            if x + tile_size_x < x_size:  # if cols overlap
                cols = tile_size_x
            else:
                cols = x_size - x

            # get offset for each tile, check edges
            # (overlap only towards a neighbouring tile, clipped at the raster edge; no padding is added,
            # so pixels at the raster edge see the same neighbourhood as in an untiled run)
            left_offset = 0
            right_offset = 0
            top_offset = 0
            bottom_offset = 0
            if x != 0:  # left offset
                if x - overlap < 0:  # left overlap
                    left_offset = x
                else:
                    left_offset = overlap
            if x + cols != x_size:  # right offset
                if x + cols + overlap > x_size:  # right overlap
                    right_offset = x_size - x - cols
                else:
                    right_offset = overlap
            if y != 0:  # apply top offset
                if y - overlap < 0:  # top overlap
                    top_offset = y
                else:
                    top_offset = overlap
            if y + rows != y_size:  # bottom offset
                if y + rows + overlap > y_size:  # bottom overlap
                    bottom_offset = y_size - y - rows
                else:
                    bottom_offset = overlap

            # read tile
            x_off = x - left_offset
            y_off = y - top_offset
            cols_off = cols + left_offset + right_offset
            rows_off = rows + top_offset + bottom_offset

            tile_array = np.array(dem_ds.GetRasterBand(1).ReadAsArray(x_off, y_off, cols_off, rows_off))
            if function_parameters is not None:
                visualization_array = visualization_function(dem=tile_array, **function_parameters)
            else:
                visualization_array = visualization_function(dem=tile_array)

            if out_visualization_dict_key is not None:
                visualization_array = visualization_array[out_visualization_dict_key]

            # remove offset from visualization block
            # (four cases because a slice end of -0 would give an empty block)
            if out_raster_nr_of_bands == 1:
                if right_offset == 0 and bottom_offset == 0:
                    visualization_array = visualization_array[top_offset:, left_offset:]
                elif right_offset == 0:
                    visualization_array = visualization_array[top_offset:-bottom_offset, left_offset:]
                elif bottom_offset == 0:
                    visualization_array = visualization_array[top_offset:, left_offset:-right_offset]
                else:
                    visualization_array = visualization_array[top_offset:-bottom_offset, left_offset:-right_offset]
            else:
                if right_offset == 0 and bottom_offset == 0:
                    visualization_array = visualization_array[:, top_offset:, left_offset:]
                elif right_offset == 0:
                    visualization_array = visualization_array[:, top_offset:-bottom_offset, left_offset:]
                elif bottom_offset == 0:
                    visualization_array = visualization_array[:, top_offset:, left_offset:-right_offset]
                else:
                    visualization_array = visualization_array[:, top_offset:-bottom_offset, left_offset:-right_offset]

            # write tile
            out_ds = gdal.Open(out_raster_path.as_posix(), gdal.GA_Update)
            if out_raster_nr_of_bands == 1:  # one band
                out_ds.GetRasterBand(1).WriteArray(visualization_array, x, y)
                out_ds.FlushCache()
            else:  # multiple bands
                for i_band in range(out_raster_nr_of_bands):
                    band = i_band + 1
                    out_ds.GetRasterBand(band).WriteArray(visualization_array[i_band], x, y)
                    out_ds.FlushCache()
            out_ds = None
    dem_ds = None


def _create_rvt_visualization_blank_raster(
        rvt_visualization: "rvt.default.RVTVisualization",
        rvt_default: "rvt.default.DefaultValues",
        dem_path: Path,
        output_dir_path: Path,
        dem_ds: gdal.Dataset,
        save_float: bool,
        save_8bit: bool
) -> None:
    """
    Create the empty float and/or 8-bit GeoTIFF that the tiles of ``rvt_visualization`` are written into.

    Parameters
    ----------
    rvt_visualization : rvt.default.RVTVisualization
        Visualization; selects the file name and the number of bands.
    rvt_default : rvt.default.DefaultValues
        Parameters; give the file name (``get_visualization_path``) and, for the multi-direction hillshade, the
        number of bands (``mhs_nr_dir``).
    dem_path : pathlib.Path
        Path of the DEM GeoTIFF; its base name starts the output file names.
    output_dir_path : pathlib.Path
        Directory of the output rasters.
    dem_ds : osgeo.gdal.Dataset
        Open DEM; gives size, projection and geotransform.
    save_float : bool
        Create the float32 raster (1 band; ``mhs_nr_dir`` bands for MULTI_HILLSHADE).
    save_8bit : bool
        Create the byte raster (1 band; 3 bands for MULTI_HILLSHADE and MULTI_SCALE_TOPOGRAPHIC_POSITION).

    Notes
    -----
    Known issue: the float raster of MULTI_SCALE_TOPOGRAPHIC_POSITION gets 1 band, but ``rvt.vis.mstp`` returns
    3 bands, so writing the second band fails in ``save_rvt_visualization_tile_by_tile``. See
    docs/KNOWN_ISSUES.md.
    """
    if save_float:
        out_float_path = rvt_default.get_visualization_path(
            rvt_visualization=rvt_visualization,
            dem_path=dem_path,
            output_dir_path=output_dir_path,
            path_8bit=False
        )
        nr_bands = 1
        if rvt_visualization == rvt.default.RVTVisualization.MULTI_HILLSHADE:
            nr_bands = rvt_default.mhs_nr_dir
        _create_blank_raster(
            in_data_set=dem_ds,
            out_raster_path=out_float_path,
            nr_bands=nr_bands,
            e_type=6)
    if save_8bit:
        out_8bit_path = rvt_default.get_visualization_path(
            rvt_visualization=rvt_visualization,
            dem_path=dem_path,
            output_dir_path=output_dir_path,
            path_8bit=True
        )
        nr_bands = 1
        if rvt_visualization == rvt.default.RVTVisualization.MULTI_HILLSHADE or \
                rvt_visualization == rvt.default.RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION:
            nr_bands = 3
        _create_blank_raster(
            in_data_set=dem_ds,
            out_raster_path=out_8bit_path,
            nr_bands=nr_bands,
            e_type=1)


def _get_rvt_visualization_overlap(
        rvt_visualization: "rvt.default.RVTVisualization", rvt_default: "rvt.default.DefaultValues"
) -> int:
    """
    Return the tile overlap in px for ``rvt_visualization``.

    The overlap is the radius in px of the neighbourhood that one output pixel depends on, taken from the
    visualization's parameters in ``rvt_default``:

    ============================================  ==========================  =================================
    Visualization                                 Overlap                     Defaults
    ============================================  ==========================  =================================
    SLOPE, HILLSHADE, MULTI_HILLSHADE             1 px (3 x 3 stencil)        1 px
    SHADOW                                        1 px                        1 px (too small, see Notes)
    SIMPLE_LOCAL_RELIEF_MODEL                     ``slrm_rad_cell`` px        20 px
    SKY_VIEW_FACTOR, ANISOTROPIC_SKY_VIEW_FACTOR,
    POSITIVE_OPENNESS, NEGATIVE_OPENNESS          ``svf_r_max`` px            10 px
    SKY_ILLUMINATION                              ``sim_shadow_dist`` px      100 px
    LOCAL_DOMINANCE                               ``ld_max_rad`` px           20 px
    MULTI_SCALE_RELIEF_MODEL                      ``int(msrm_feature_max)``   20 (a value in m, see Notes)
    MULTI_SCALE_TOPOGRAPHIC_POSITION              ``mstp_broad_scale[1]`` px  500 px
    ============================================  ==========================  =================================

    For example the default SVF overlap is r = 10 px = 2.5 m at 0.25 m/px, and the default MSTP overlap is
    r = 500 px = 125 m at 0.25 m/px.

    Parameters
    ----------
    rvt_visualization : rvt.default.RVTVisualization
        Visualization to be tiled.
    rvt_default : rvt.default.DefaultValues
        Parameters that give the radius.

    Returns
    -------
    int or None
        Overlap in px; None for a value that is not a member of ``RVTVisualization``.

    Notes
    -----
    Known issue: MULTI_SCALE_RELIEF_MODEL uses ``msrm_feature_max``, which is a feature size in m, as a pixel
    count. The largest mean-filter radius of ``rvt.vis.msrm`` is ``n ** scaling_factor`` px with
    ``n = ceil(((feature_max - res) / (2 * res)) ** (1 / scaling_factor))``; for the defaults (20 m, factor 2)
    that is 25 px at 0.5 m/px (overlap 20 px too small, the tiled MSRM differs from the untiled one near tile
    seams) and 9 px at 2 m/px (overlap larger than needed). See docs/KNOWN_ISSUES.md.

    Known issue: SHADOW gets 1 px, but a shadow reaches as far as the relief casts it (tens of px for some
    metres of relief at 35 deg sun elevation), so the tiled shadow differs from the untiled one near seams.
    See docs/KNOWN_ISSUES.md.

    SKY_ILLUMINATION: ``sim_shadow_dist`` is ``max_fine_radius`` of ``rvt.vis.sky_illumination`` (horizon
    search radius on the full-resolution DEM, px). That function also searches DEM pyramids built from the whole
    tile, so the tiled result is not exactly the untiled one in general; see the notes of
    ``rvt.vis.sky_illumination`` and docs/KNOWN_ISSUES.md.
    """
    if rvt_visualization == rvt.default.RVTVisualization.SLOPE:
        return 1
    elif rvt_visualization == rvt.default.RVTVisualization.HILLSHADE:
        return 1
    elif rvt_visualization == rvt.default.RVTVisualization.SHADOW:
        return 1
    elif rvt_visualization == rvt.default.RVTVisualization.MULTI_HILLSHADE:
        return 1
    elif rvt_visualization == rvt.default.RVTVisualization.SIMPLE_LOCAL_RELIEF_MODEL:
        return int(rvt_default.slrm_rad_cell)
    elif rvt_visualization == rvt.default.RVTVisualization.SKY_VIEW_FACTOR:
        return int(rvt_default.svf_r_max)
    elif rvt_visualization == rvt.default.RVTVisualization.ANISOTROPIC_SKY_VIEW_FACTOR:
        return int(rvt_default.svf_r_max)
    elif rvt_visualization == rvt.default.RVTVisualization.POSITIVE_OPENNESS:
        return int(rvt_default.svf_r_max)
    elif rvt_visualization == rvt.default.RVTVisualization.NEGATIVE_OPENNESS:
        return int(rvt_default.svf_r_max)
    elif rvt_visualization == rvt.default.RVTVisualization.SKY_ILLUMINATION:
        return int(rvt_default.sim_shadow_dist)
    elif rvt_visualization == rvt.default.RVTVisualization.LOCAL_DOMINANCE:
        return int(rvt_default.ld_max_rad)
    elif rvt_visualization == rvt.default.RVTVisualization.MULTI_SCALE_RELIEF_MODEL:
        return int(rvt_default.msrm_feature_max)
    elif rvt_visualization == rvt.default.RVTVisualization.MULTI_SCALE_TOPOGRAPHIC_POSITION:
        return int(rvt_default.mstp_broad_scale[1])


def save_rvt_visualization_tile_by_tile(
        rvt_visualization: "rvt.default.RVTVisualization",
        rvt_default: "rvt.default.DefaultValues",
        dem_path: Path,
        output_dir_path: Optional[Path] = None,
        save_float: bool = True,
        save_8bit: bool = False
) -> None:
    """
    Compute one RVT visualization of a DEM GeoTIFF tile by tile and write its float and/or 8-bit GeoTIFF.

    Tiles of ``rvt_default.tile_size`` = (x_size, y_size) px are read with the overlap of
    ``_get_rvt_visualization_overlap`` (px), computed by ``rvt_default.calculate_visualization`` with the
    parameters held in ``rvt_default``, cut back to the tile and written at the tile position. Both outputs are
    computed from the same tile in one pass. ``rvt.default.DefaultValues.save_*`` calls this function when the
    DEM has more pixels than ``rvt_default.tile_size_limit``.

    Parameters
    ----------
    rvt_visualization : rvt.default.RVTVisualization
        Visualization to compute.
    rvt_default : rvt.default.DefaultValues
        Visualization parameters, ``ve_factor``, ``tile_size`` (x_size, y_size) px and the file-name rules.
    dem_path : pathlib.Path
        Path of the DEM GeoTIFF (band 1, elevation in m). Resolution (m/px) and no-data value are read from it.
    output_dir_path : pathlib.Path or None
        Directory of the outputs; None means the directory of ``dem_path``. File names are
        ``rvt_default.get_visualization_path(rvt_visualization, dem_path, output_dir_path, path_8bit)``.
        Existing files are replaced (``rvt_default.overwrite`` is not checked here).
    save_float : bool
        Write the float32 raster (no-data NaN).
    save_8bit : bool
        Write the byte raster (stretch of ``DefaultValues.float_to_8bit``).

    Returns
    -------
    None
        The results are written to ``output_dir_path``.

    Notes
    -----
    Known issue: the checks "no output requested", "DEM missing" and "tile size below 50 px" build an
    ``Exception`` but do not raise it; with both flags False the function reads and computes every tile and
    writes nothing. See docs/KNOWN_ISSUES.md.

    Known issue: with ``save_float`` and ``save_8bit`` both True, ``calculate_visualization`` hands the float
    array to ``float_to_8bit``, which clips it in place to the stretch range, so the float raster is clipped too
    (e.g. slope limited to 0..51 deg). See docs/KNOWN_ISSUES.md.

    Known issue: MULTI_HILLSHADE float: ``rvt.vis.multi_hillshade`` returns ``(16, H-2, W-2)`` per tile, but the
    code assumes ``(16, H, W)``; the cropped block is written at the tile origin, so pixel [i, j] holds the
    value of DEM pixel [i + 1, j + 1] (as in the untiled array) and the last 2 rows and 2 columns of every tile
    stay NaN. See docs/KNOWN_ISSUES.md.

    Known issue: MULTI_SCALE_TOPOGRAPHIC_POSITION float: the raster has 1 band for a 3-band result, writing band 2
    fails and a partly written file remains. Known issues of the overlap (SHADOW, MULTI_SCALE_RELIEF_MODEL) are
    listed in ``_get_rvt_visualization_overlap``. The y pixel size is ``-gt[5]`` without ``abs`` (see
    ``save_visualization_tile_by_tile``). See docs/KNOWN_ISSUES.md.
    """
    # Structure of this function:
    #   1. (intended) argument checks; the Exception objects are not raised (see Notes)
    #   2. tile size from rvt_default.tile_size; default output directory = DEM directory
    #   3. open the DEM, read pixel size, no-data value and raster size
    #   4. create the empty float and/or 8-bit output rasters (_create_rvt_visualization_blank_raster)
    #   5. overlap for this visualization (_get_rvt_visualization_overlap)
    #   6. for each tile (row-major): clip the overlap at the raster edge, read tile + overlap,
    #      compute float and 8-bit arrays (DefaultValues.calculate_visualization), cut the overlap off
    #      both (2D or 3D), open each output in update mode and write the block at (x, y)
    if not save_float and not save_8bit:
        Exception("rvt.tile.save_visualization_tile_by_tile: At least one of save_float or save_8bit must be true!")
    if not dem_path.exists():
        Exception("rvt.tile.save_visualization_tile_by_tile: Input dem path does not exist!")

    tile_size_x = rvt_default.tile_size[0]
    tile_size_y = rvt_default.tile_size[1]

    if tile_size_x < 50 or tile_size_y < 50:
        Exception("rvt.tile.save_visualization_tile_by_tile: Tile size too small (tile_size_x, tile_size_y),"
                  " it needs to be bigger than 50 pixels!")

    if output_dir_path is None:
        output_dir_path = dem_path.parent

    dem_ds = gdal.Open(dem_path.as_posix())
    gt = dem_ds.GetGeoTransform()
    x_res = gt[1]  # x_resolution
    y_res = -gt[5]  # y_resolution
    no_data = dem_ds.GetRasterBand(1).GetNoDataValue()
    band = dem_ds.GetRasterBand(1)
    x_size = band.XSize  # number of columns
    y_size = band.YSize  # number of rows

    _create_rvt_visualization_blank_raster(
        rvt_visualization=rvt_visualization,
        rvt_default=rvt_default,
        dem_path=dem_path,
        output_dir_path=output_dir_path,
        dem_ds=dem_ds,
        save_float=save_float,
        save_8bit=save_8bit
    )

    overlap = _get_rvt_visualization_overlap(rvt_visualization=rvt_visualization, rvt_default=rvt_default)

    for y in range(0, y_size, tile_size_y):
        if y + tile_size_y < y_size:  # if rows overlap
            rows = tile_size_y
        else:
            rows = y_size - y
        for x in range(0, x_size, tile_size_x):
            if x + tile_size_x < x_size:  # if cols overlap
                cols = tile_size_x
            else:
                cols = x_size - x

            # get offset for each tile, check edges
            # (overlap only towards a neighbouring tile, clipped at the raster edge; no padding is added,
            # so pixels at the raster edge see the same neighbourhood as in an untiled run)
            left_offset = 0
            right_offset = 0
            top_offset = 0
            bottom_offset = 0
            if x != 0:  # left offset
                if x - overlap < 0:  # left overlap
                    left_offset = x
                else:
                    left_offset = overlap
            if x + cols != x_size:  # right offset
                if x + cols + overlap > x_size:  # right overlap
                    right_offset = x_size - x - cols
                else:
                    right_offset = overlap
            if y != 0:  # apply top offset
                if y - overlap < 0:  # top overlap
                    top_offset = y
                else:
                    top_offset = overlap
            if y + rows != y_size:  # bottom offset
                if y + rows + overlap > y_size:  # bottom overlap
                    bottom_offset = y_size - y - rows
                else:
                    bottom_offset = overlap

            # read tile
            x_off = x - left_offset
            y_off = y - top_offset
            cols_off = cols + left_offset + right_offset
            rows_off = rows + top_offset + bottom_offset

            tile_array = np.array(dem_ds.GetRasterBand(1).ReadAsArray(x_off, y_off, cols_off, rows_off))

            visualization_float_arr, visualization_8bit_arr = rvt_default.calculate_visualization(
                visualization=rvt_visualization,
                dem=tile_array,
                resolution_x=x_res,
                resolution_y=y_res,
                no_data=no_data,
                save_float=save_float,
                save_8bit=save_8bit
            )

            # remove offset from visualization block
            # (four cases because a slice end of -0 would give an empty block)
            if save_float:
                if visualization_float_arr.ndim == 2:
                    if right_offset == 0 and bottom_offset == 0:
                        visualization_float_arr = visualization_float_arr[top_offset:, left_offset:]
                    elif right_offset == 0:
                        visualization_float_arr = visualization_float_arr[top_offset:-bottom_offset, left_offset:]
                    elif bottom_offset == 0:
                        visualization_float_arr = visualization_float_arr[top_offset:, left_offset:-right_offset]
                    else:
                        visualization_float_arr = visualization_float_arr[
                                                  top_offset:-bottom_offset, left_offset:-right_offset
                                                  ]
                else:
                    if right_offset == 0 and bottom_offset == 0:
                        visualization_float_arr = visualization_float_arr[:, top_offset:, left_offset:]
                    elif right_offset == 0:
                        visualization_float_arr = visualization_float_arr[:, top_offset:-bottom_offset, left_offset:]
                    elif bottom_offset == 0:
                        visualization_float_arr = visualization_float_arr[:, top_offset:, left_offset:-right_offset]
                    else:
                        visualization_float_arr = visualization_float_arr[
                                                 :, top_offset:-bottom_offset, left_offset:-right_offset
                                                 ]
            if save_8bit:
                if visualization_8bit_arr.ndim == 2:
                    if right_offset == 0 and bottom_offset == 0:
                        visualization_8bit_arr = visualization_8bit_arr[top_offset:, left_offset:]
                    elif right_offset == 0:
                        visualization_8bit_arr = visualization_8bit_arr[top_offset:-bottom_offset, left_offset:]
                    elif bottom_offset == 0:
                        visualization_8bit_arr = visualization_8bit_arr[top_offset:, left_offset:-right_offset]
                    else:
                        visualization_8bit_arr = visualization_8bit_arr[
                                                  top_offset:-bottom_offset, left_offset:-right_offset
                                                  ]
                else:
                    if right_offset == 0 and bottom_offset == 0:
                        visualization_8bit_arr = visualization_8bit_arr[:, top_offset:, left_offset:]
                    elif right_offset == 0:
                        visualization_8bit_arr = visualization_8bit_arr[:, top_offset:-bottom_offset, left_offset:]
                    elif bottom_offset == 0:
                        visualization_8bit_arr = visualization_8bit_arr[:, top_offset:, left_offset:-right_offset]
                    else:
                        visualization_8bit_arr = visualization_8bit_arr[
                                                 :, top_offset:-bottom_offset, left_offset:-right_offset
                                                 ]


            # write tile
            if save_float:
                out_visualization_float_path = rvt_default.get_visualization_path(
                    rvt_visualization=rvt_visualization,
                    dem_path=dem_path,
                    output_dir_path=output_dir_path,
                    path_8bit=False
                )
                out_ds_float = gdal.Open(out_visualization_float_path.as_posix(), gdal.GA_Update)
                if visualization_float_arr.ndim == 2:
                    out_ds_float.GetRasterBand(1).WriteArray(visualization_float_arr, x, y)
                    out_ds_float.FlushCache()
                else:
                    for i_band in range(visualization_float_arr.shape[0]):
                        band = i_band + 1
                        out_ds_float.GetRasterBand(band).WriteArray(visualization_float_arr[i_band], x, y)
                        out_ds_float.FlushCache()
                out_ds_float = None
            if save_8bit:  # multiple bands
                out_visualization_8bit_path = rvt_default.get_visualization_path(
                    rvt_visualization=rvt_visualization,
                    dem_path=dem_path,
                    output_dir_path=output_dir_path,
                    path_8bit=True
                )
                out_ds_8bit = gdal.Open(out_visualization_8bit_path.as_posix(), gdal.GA_Update)
                if visualization_8bit_arr.ndim == 2:
                    out_ds_8bit.GetRasterBand(1).WriteArray(visualization_8bit_arr, x, y)
                    out_ds_8bit.FlushCache()
                else:
                    for i_band in range(visualization_8bit_arr.shape[0]):
                        band = i_band + 1
                        out_ds_8bit.GetRasterBand(band).WriteArray(visualization_8bit_arr[i_band], x, y)
                        out_ds_8bit.FlushCache()
                out_ds_8bit = None

    dem_ds = None
