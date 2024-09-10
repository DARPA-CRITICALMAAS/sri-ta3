from typing import List, Dict, Optional
import os
from os import makedirs
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterio.fill import fillnodata
import rasterio
import subprocess
import fiona
from sri_maper.src import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from cdr_schemas.cdr_responses.prospectivity import ProspectModelMetaData

log = utils.get_pylogger(__name__)


def warp_raster(
    src_raster_path: str,
    dst_raster_path: str,
    dst_crs: str = 'ESRI:102008',
    dst_nodata: float = -999999999.0,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
    src_crs: Optional[str] = None,
):
    print(f'Warping raster: {src_raster_path}')
    cmd = ['gdalwarp', '-overwrite']
    if src_crs is not None:
        cmd += ['-s_srs', src_crs]
    cmd += ['-t_srs', str(dst_crs), '-dstnodata', str(dst_nodata),
    '-tr', str(dst_res_x), str(dst_res_y), '-r', 'bilinear', '-of', 'GTiff', src_raster_path, dst_raster_path
    ]
    subprocess.run(cmd, check=True)


def dilate_raster(
    src_raster_path: str,
    dst_raster_path: str,
    dilation_size: int = 50,
):
    print(f'Dilating raster: {src_raster_path}')
    cmd = [
    'gdal_fillnodata.py', src_raster_path, dst_raster_path, '-md', str(dilation_size), '-b', '1', '-of', 'GTiff'
    ]
    subprocess.run(cmd, check=True)


def clip_raster(
    src_raster_path: str,
    dst_raster_path: str,
    aoi_path: str,
    dst_crs: str = 'ESRI:102008',
    dst_nodata: float = -999999999.0,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
):
    print(f'Clipping raster: {src_raster_path} w/r to AOI: {aoi_path}')
    gdf = gpd.read_file(aoi_path)
    layers = fiona.listlayers(aoi_path)
    cmd = [
        'gdalwarp', '-overwrite', '-t_srs', str(dst_crs),
        '-te', str(gdf.bounds.minx[0]), str(gdf.bounds.miny[0]), str(gdf.bounds.maxx[0]), str(gdf.bounds.maxy[0]), '-te_srs', gdf.crs.to_string(),
        '-of', 'GTiff', '-tr', str(dst_res_x), str(dst_res_y), '-tap', '-cutline', aoi_path, '-cl', layers[0], '-dstnodata', str(dst_nodata),
        src_raster_path, dst_raster_path
    ]
    subprocess.run(cmd, check=True)


def remove_outliers_tukey_raster(
    src_raster_path: str,
    dst_raster_path: str,
    k: int = 1.5
):
    """
    Remove outliers from a raster image using the Tukey fences method.

    Parameters:
    - input_raster_path (str): Path to the input raster file.
    - output_raster_path (str): Path to save the output raster with outliers removed.
    - k (float): The constant to define the range for outlier detection (default is 1.5 for Tukey's rule).
    """
    # Open the input raster file
    with rasterio.open(src_raster_path) as src:
        raster_data = src.read(1)

        Q1 = np.percentile(raster_data, 25)
        Q3 = np.percentile(raster_data, 75)
        IQR = Q3 - Q1
        lower_fence = Q1 - k * IQR
        upper_fence = Q3 + k * IQR
        p5 = np.percentile(raster_data, 5)
        p95 = np.percentile(raster_data, 95)
        raster_data = np.where(raster_data < lower_fence, p5, raster_data)
        raster_data = np.where(raster_data > upper_fence, p95, raster_data)

        metadata = src.meta
        metadata.update(dtype=rasterio.float32)

    with rasterio.open(dst_raster_path, 'w', **metadata) as dst:
        dst.write(raster_data.astype(rasterio.float32), 1)

    return dst_raster_path


def scale_raster(
    src_raster_path: str,
    dst_raster_path: str,
    scaling_type: str,
):
    """
    Standard scale a raster image using scikit-learn's StandardScaler.

    Parameters:
    - src_raster_path (str): Path to the input raster file.
    - dst_raster_path (str): Path to save the output scaled raster file.
    """
    with rasterio.open(src_raster_path) as src:
        raster_data = src.read(1)

        flat_data = raster_data.flatten().reshape(-1, 1)
        if scaling_type == "standard":
            scaler = StandardScaler()
        elif scaling_type == "minmax":
            scaler = MinMaxScaler()
        else:
            Exception(f"Unknown scaling type {scaling_type}.")
        scaled_data = scaler.fit_transform(flat_data)
        scaled_raster_data = scaled_data.reshape(raster_data.shape)

        metadata = src.meta
        metadata.update(dtype=rasterio.float32)

    with rasterio.open(dst_raster_path, 'w', **metadata) as dst:
        dst.write(scaled_raster_data.astype(rasterio.float32), 1)

    return dst_raster_path


def warp_vector(
    src_vector_path: str,
    dst_vector_path: str,
    dst_crs: str = 'ESRI:102008',
):
    print(f'Warping vector: {src_vector_path}')
    cmd = [
        'ogr2ogr', '-f', 'ESRI Shapefile', '-t_srs', dst_crs, dst_vector_path, src_vector_path
    ]
    subprocess.run(cmd, check=True)


def vector_to_raster(
    src_vector_path: str,
    dst_raster_path: str,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
    dst_nodata: float = -999999999.0,
    attribute: str = None,
    burn_value: Optional[float] = 1.0,
):
    print(f'Converting vector to raster: {src_vector_path}')
    layers = fiona.listlayers(src_vector_path)
    src_layer_name = layers[0]
    cmd = ['gdal_rasterize', '-l', src_layer_name]
    if attribute is not None:
        cmd += ['-a', attribute]
    else:
        cmd += ['-burn', str(burn_value)]
    cmd += ['-tr', str(dst_res_x), str(dst_res_y),
        '-a_nodata', str(dst_nodata), '-ot', 'Float32', '-of', 'GTiff', src_vector_path, dst_raster_path]
    subprocess.run(cmd, check=True)


def proximity_raster(
    src_raster_path: str,
    dst_raster_path: str,
    src_burn_value: float = 1.0,
    dst_nodata: float = -999999999.0
):
    print(f'Calculating proximity raster: {src_raster_path}')
    cmd = [
    'gdal_proximity.py', '-srcband', '1', '-distunits', 'GEO', '-values', str(src_burn_value),
    '-nodata', str(dst_nodata), '-ot', 'Float32', '-of', 'GTiff', src_raster_path, dst_raster_path
    ]
    subprocess.run(cmd, check=True)


def fill_nodata_raster(
    src_raster_path: str,
    dst_raster_path: str,
    logic_cmd: str = "numpy.where(A>0,A,0)",
):
    print(f'Filling nodata in raster: {src_raster_path}')
    cmd = [
        'gdal_calc.py', '--overwrite', '--calc', logic_cmd, '--format', 'GTiff', '--type', 'Float32', '-A',
        src_raster_path, '--A_band', '1', '--hideNoData', '--outfile', dst_raster_path
    ]
    subprocess.run(cmd, check=True)


def preprocess_evidence_layers(
    event_obj: ProspectModelMetaData,
    layers: List[Path],
    aoi: Path,
):
    pev_lyr_paths = []
    for layer in layers:
        if layer.suffix == ".tif":
            pev_lyr_path = preprocess_raster(event_obj, layer, aoi)
        elif layer.suffix == ".zip":
            pev_lyr_path = preprocess_vector(event_obj, layer, aoi)
        pev_lyr_paths.append(pev_lyr_path)
    return pev_lyr_paths


def preprocess_raster(
    event_obj: ProspectModelMetaData,
    layer: Path,
    aoi: Path,
    dilation_px: int = 50,
):
    warped_file = layer.parent / (layer.stem +"_warped" + layer.suffix)
    imputed_file = layer.parent / (layer.stem +"_imputed" + layer.suffix)
    clipped_file = layer.parent / (layer.stem +"_clipped" + layer.suffix)
    olr_file = layer.parent / (layer.stem +"_olr" + layer.suffix)
    scaled_file = layer.parent / (layer.stem +"_processed" + layer.suffix)
    with rasterio.open(layer) as ras: 
        nodata = ras.nodata
        CRS = ras.crs if ras.crs is not None else 'EPSG:4326'
    warp_raster(
        src_raster_path = str(layer), 
        dst_raster_path = str(warped_file),
        dst_crs = event_obj.cma.crs, 
        dst_nodata = nodata,
        dst_res_x = event_obj.cma.resolution[0], 
        dst_res_y = event_obj.cma.resolution[1],
        src_crs = CRS,
    )
    dilate_raster(
        src_raster_path = warped_file, 
        dst_raster_path = imputed_file,
        dilation_size = dilation_px,
    )
    clip_raster(
        src_raster_path = imputed_file, 
        dst_raster_path = clipped_file, 
        aoi_path = str(aoi),
        dst_crs = event_obj.cma.crs, 
        dst_nodata = nodata,
        dst_res_x = event_obj.cma.resolution[0], 
        dst_res_y = event_obj.cma.resolution[1],
    )
    remove_outliers_tukey_raster(
        src_raster_path=clipped_file,
        dst_raster_path=olr_file,
    )
    scale_raster(
        src_raster_path=olr_file,
        dst_raster_path=scaled_file,
        scaling_type="standard"
    )
    return scaled_file


def find_shapefiles(directory):
    """
    Get file paths to all .shp files within a directory and its subfolders.

    Parameters:
    - directory (str): The path to the directory to search.

    Returns:
    - List of file paths to .shp files.
    """
    shapefiles = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.shp'):
                shapefiles.append(os.path.join(root, file))
    return shapefiles


def preprocess_vector(
    event_obj: ProspectModelMetaData,
    layer: Path,
    aoi: Path,
):
    shp_files = find_shapefiles(layer.parent / layer.stem)
    if len(shp_files) > 1 or len(shp_files) == 0: raise Exception(f"Cannot process vector file {layer}.")

    warped_file = layer.parent / (layer.stem +"_warped" + layer.suffix)
    imputed_file = layer.parent / (layer.stem +"_imputed" + layer.suffix)
    clipped_file = layer.parent / (layer.stem +"_clipped" + layer.suffix)
    olr_file = layer.parent / (layer.stem +"_olr" + layer.suffix)
    scaled_file = layer.parent / (layer.stem +"_processed" + layer.suffix)

    full_path = os.path.join(dirpath, filename)
    warped_dir = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_warped')
    os.makedirs(warped_dir, exist_ok=True)
    warped_file = os.path.join(warped_dir, full_path.split(dir_orig_path)[1].split('/')[1]+'_warped.shp')
    rasterized_file = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_rasterized.tif')
    proximity_file = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_proximity.tif')
    clipped_file = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_clipped.tif')

    print(full_path)
    # Reproject vector .shp file into desired CRS
    warp_vector(
        src_vector_path = full_path, dst_vector_path = warped_file, dst_crs = dst_params['crs'],
    )
    # Rasterize the reprojected .shp file
    vector_to_raster(
        src_vector_path = warped_file, dst_raster_path = rasterized_file,
        dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'],
        dst_nodata = dst_params['nodata'], attribute = None, burn_value = 1.0,
    )
    # Generate proximity raster for the rasterized ^ file
    proximity_raster(
        src_raster_path = rasterized_file, dst_raster_path = proximity_file,
        src_burn_value = 1.0, dst_nodata = dst_params['nodata']
    )
    # Clip proximity raster to aoi
    clip_raster(
        src_raster_path = proximity_file, dst_raster_path = clipped_file, aoi_path = aoi_output_path,
        dst_crs = dst_params['crs'], dst_nodata = dst_params['nodata'],
        dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'],
    )
    shutil.rmtree(warped_dir)
    os.remove(rasterized_file)
    os.remove(proximity_file)


def generate_raster_stacks(raster_stacks):
    for raster_stack in tqdm(raster_stacks):
        if not Path(raster_stack.raster_stack_path).is_file():
            generate_raster_stack(
                raster_stack.raster_stack_path,
                raster_stack.raster_files_path,
                raster_stack.dilation_size,
                raster_stack.raster_files,
            )


def generate_raster_stack(
    raster_stack_path: Path,
    raster_files_path: Path,
    dilation_size: int,
    raster_files: List[str],
):
    r"""
    Generates a multi-band GeoTiff (i.e., raster stack). Assumes each raster is already
    aligned and has imputed values.

    :param raster_stack_path: path to save the raster stack
    :type raster_stack_path: str
    :param raster_files_path: root path to library of single band rasters
    :type raster_files_path: str
    :param dilation_size: number of pixels to dilate the rasters by
    :type dilation_size: int
    :param raster_files: list of paths to single band rasters
    :type raster_files: List[str]
    ...
    :return: None
    :rtype: None
    """
    # loads the individual rasters
    rasters = load_rasters(raster_files, raster_files_path)
    rasters_data = [raster.read(1, masked=True) for raster in rasters]
    for raster_data in rasters_data: np.ma.set_fill_value(raster_data, np.nan) # changing masked fill_value from -1e9 to NaN

    # dilation of all the rasters by window_size pixels
    label_msk = np.isnan(rasters_data[-1])
    dilated_rasters_data = [fillnodata(raster_data, max_search_distance=dilation_size) for raster_data in rasters_data]
    dilated_rasters_data[-1][label_msk & ~np.isnan(dilated_rasters_data[-1])] = 0.

    # creating list of raster shapes, dilated raster masks, and overall dilated mask
    raster_shapes = [dilated_raster_data.shape for dilated_raster_data in dilated_rasters_data]

    # creates the raster dataframe
    raster_df = pd.DataFrame()
    for raster_file, dilated_raster_data in zip(raster_files, dilated_rasters_data):
        raster_df[f"{raster_file.path}"] = dilated_raster_data.flatten()
    raster_df = raster_df.replace(-1.0e+09, np.nan) # changing -1.0e+09 to NaN
    raster_df.loc[raster_df.isnull().any(axis=1), :] = np.nan # if any value in a row is NaN, set all values in that row to NaN

    # raster_df_bu = raster_df.copy()
    new_rasters_data = []
    for i, raster_file in tqdm(enumerate(raster_files)):
        raster_name = raster_file.path
        # removes outliers and normalizes
        if raster_file.outlier_removal: # and raster_file.type != "bool":
            raster_df = tukey_remove_outliers(raster_df, raster_name)
        if raster_file.normalize: # and raster_file.type != "bool":
            raster_df[f"{raster_name}"] = StandardScaler().fit_transform(raster_df[f"{raster_name}"].values.reshape(-1,1)).squeeze(-1)
        # extracts masked numpy arrays from the from dataframe
        new_raster_data = raster_df[f"{raster_name}"].values.reshape(raster_shapes[i])
        new_rasters_data.append(new_raster_data)

    # generates and saves raster stack
    raster_stack_meta = rasters[0].meta
    raster_stack_meta.update({"count": len(new_rasters_data)})
    raster_stack_meta.update({"dtype": "float32"})
    log.debug(f"Writing a raster stack with the following meta data: {raster_stack_meta}")
    makedirs(Path(raster_stack_path).parent, exist_ok=True)
    with rasterio.open(Path(raster_stack_path), "w", **raster_stack_meta) as raster_stack:
        tags = {Path(raster_file.path).stem: idx for idx, raster_file in enumerate(raster_files)}
        tags["ns"] = "evidence_layers"
        raster_stack.update_tags(**tags)
        for idx, new_raster_data in enumerate(new_rasters_data):
            raster_stack.write_band(idx+1, new_raster_data)


def load_rasters(
    raster_files: List[str],
    rasters_path: str,
):
    return [load_raster(Path(rasters_path) / Path(raster_file.path)) for raster_file in raster_files]


def load_raster(
    raster_path,
):
    raster = rasterio.open(raster_path)
    log.debug(f"-------- {raster_path} raster details --------\n")
    info = {i: dtype for i, dtype in zip(raster.indexes, raster.dtypes)}
    log.debug(f"Raster bands and dtypes:\n{info}\n\n")
    log.debug(f"Coordinate reference system:\n{raster.crs}\n\n")
    log.debug(f"Bounds:{raster.bounds},Size:{raster.shape},Resolution:{raster.res}\n\n")
    return raster
