from typing import List, Dict, Optional, Union, Literal
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from cdr_schemas.cdr_responses.prospectivity import ProspectModelMetaData
from cdr_schemas.prospectivity_input import ScalingType, TransformMethod, Impute, ImputeMethod
import yaml
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

log = utils.get_pylogger(__name__)


def format_nodata_crs(
    src_raster_path: Path,
    dst_raster_path: Path,
    default_crs: str = 'EPSG:4326',
    default_nodata: float = np.nan,
):
    """
    Load a raster, update NoData values to NaN, and save the modified raster.

    Parameters:
    - input_raster_path (str): Path to the input raster file.
    - output_raster_path (str): Path to save the output raster with NoData updated to NaN.
    """
    with rasterio.open(src_raster_path) as src:
        raster_data = src.read(1)
        nodata_value = src.nodata
        CRS = src.crs if src.crs is not None else default_crs
        # if nodata_value is not None:
        #     raster_data = np.where(raster_data == nodata_value, default_nodata, raster_data)
        # else:
        #     raise Exception(f"Raster no data value is None: {src_raster_path}")
        raster_data = np.where(raster_data == nodata_value, default_nodata, raster_data)

        metadata = src.meta
        metadata.update(dtype=rasterio.float32, nodata=default_nodata, crs=CRS)

    # Save the modified raster to the output path
    with rasterio.open(dst_raster_path, 'w', **metadata) as dst:
        dst.write(raster_data.astype(rasterio.float32), 1)


def warp_raster(
    src_raster_path: Path,
    dst_raster_path: Path,
    dst_crs: str = 'ESRI:102008',
    dst_nodata: float = np.nan,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
    resampling=rasterio.warp.Resampling.bilinear
):
    """
    Reproject a raster to a new CRS using rasterio.warp.reproject.

    Parameters:
    - src_raster_path (str): Path to the input raster file.
    - dst_raster_path (str): Path to save the reprojected raster file.
    - dst_crs (str or dict): The destination coordinate reference system.
    - dst_nodata (float or int): NoData value for the output raster.
    - dst_res_x, dst_res_y (float): Resolution of the output raster.
    - resampling (rasterio.warp.Resampling): Resampling method to use.
    """
    with rasterio.open(src_raster_path) as src:
        # Calculate transform and dimensions for output raster
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=(dst_res_x, dst_res_y) if dst_res_x and dst_res_y else None
        )

        # Update metadata for the output raster
        metadata = src.meta.copy()
        metadata.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': dst_nodata,
            'dtype': src.dtypes[0]
        })

        # Reproject and write to the output file
        with rasterio.open(dst_raster_path, 'w', **metadata) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    dst_nodata=dst_nodata
                )


def dilate_raster(
    src_raster_path: Path,
    dst_raster_path: Path,
    dilation_size: int = 100,
    smoothing_iterations: int = 0,
    label_raster: bool = False
):
    """
    Fill NoData values in a raster using rasterio's fillnodata function.

    Parameters:
    - src_raster_path (str): Path to the input raster file.
    - dst_raster_path (str): Path to save the filled raster.
    - dilation_size (int): Maximum search distance for interpolation (default is 100).
    - smoothing_iterations (int): Number of smoothing iterations (default is 0).
    - label_raster (bool): Whether or not the input raster file is a label raster.
    """
    with rasterio.open(src_raster_path) as src:
        data = src.read(1, masked=True)  # Read the first band
        if label_raster:
            label_msk = np.isnan(data)
        filled_data = rasterio.fill.fillnodata(
            data,
            max_search_distance=dilation_size,
            smoothing_iterations=smoothing_iterations
        )
        if label_raster:
            filled_data[label_msk & ~np.isnan(filled_data)] = 0.


        # Copy metadata and write the filled raster
        profile = src.profile

    with rasterio.open(dst_raster_path, 'w', **profile) as dst:
        dst.write(filled_data, 1)


def clip_raster(
    src_raster_path: Path,
    dst_raster_path: Path,
    aoi_path: Path,
):
    """
    Clip a raster to a region of interest using a shapefile.

    Parameters:
    - input_raster (str): Path to the input raster file.
    - shapefile (str): Path to the shapefile defining the region of interest.
    - output_raster (str): Path to save the clipped raster.
    """
    # Read the shapefile
    shapes = gpd.read_file(aoi_path)
    # shapes['geometry'] = shapes['geometry'].simplify(tolerance=0.1)

    # Open the raster file
    with rasterio.open(src_raster_path) as src:
        # Clip the raster with the shapes from the shapefile
        out_image, out_transform = rasterio.mask.mask(src, shapes.geometry, crop=True, all_touched=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

    # Save the clipped raster
    with rasterio.open(dst_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


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
    scaling_type: Literal["standard", "minmax", "maxabs"] = "standard",
    min_value: float = 0.0,
    max_value: float = 1.0
):
    """
    Standard/MinMax/MaxAbs scale a raster image using scikit-learn's StandardScaler.

    Parameters:
    - src_raster_path (str): Path to the input raster file.
    - dst_raster_path (str): Path to save the output scaled raster file.
    - scaling_type (str): The type of scaling to apply. Options are 'standard', 'minmax', and 'maxabs' (default is 'standard').
    - min_value (float): Minimum value for scaling, only used if scaling_type is 'minmax' (default is 0.0).
    - max_value (float): Maximum value for scaling, only used if scaling_type is 'minmax' (default is 1.0).
    """
    with rasterio.open(src_raster_path) as src:
        raster_data = src.read(1)

        flat_data = raster_data.flatten().reshape(-1, 1)
        if scaling_type == "standard":
            scaler = StandardScaler()
        elif scaling_type == "minmax":
            assert min_value < max_value, "min_value must be less than max_value."
            scaler = MinMaxScaler(
                feature_range = (min_value, max_value)
            )
        elif scaling_type == "maxabs":
            scaler = MaxAbsScaler()
        else:
            Exception(f"Unknown scaling type {scaling_type}.")
        scaled_data = scaler.fit_transform(flat_data)
        scaled_raster_data = scaled_data.reshape(raster_data.shape)

        metadata = src.meta
        metadata.update(dtype=rasterio.float32)

    with rasterio.open(dst_raster_path, 'w', **metadata) as dst:
        dst.write(scaled_raster_data.astype(rasterio.float32), 1)


def warp_vector(
    src_vector_path: Path,
    dst_vector_path: Path,
    dst_crs: str = 'ESRI:102008',
):
    """
    Reproject a vector file to a different CRS.

    Parameters:
    - input_vector (str): Path to the input vector file.
    - output_vector (str): Path to save the reprojected vector file.
    - crs (str or dict): The target CRS (e.g., 'EPSG:4326' or {'init': 'epsg:4326'}).
    """
    # Read the vector file
    gdf = gpd.read_file(src_vector_path)

    # Reproject to the target CRS
    gdf = gdf.to_crs(dst_crs)

    # Save the reprojected vector
    gdf.to_file(dst_vector_path, driver='ESRI Shapefile')


def vector_to_raster(
    src_vector_path: Path,
    dst_raster_path: Path,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
    burn_value: float = 1.0,
    fill_value: float = None,
    dst_nodata: float = np.nan,
    aoi_path: Optional[Union[Path, None]] = None
):
    """
    Rasterize a vector file to a raster with specific resolution.

    Parameters:
    - vector_path (str): Path to the input vector file.
    - output_raster (str): Path to save the output raster file.
    - x_res (float): Desired x resolution of the output raster.
    - y_res (float): Desired y resolution of the output raster.
    - burn_value (int/float): Value to burn in the raster (default is 1).
    - aoi_path (Path or None): Path to the shapefile defining the region of interest (default is None).
    """
    # Read the vector file
    gdf = gpd.read_file(src_vector_path)

    if aoi_path:
        aoi_gdf = gpd.read_file(aoi_path)

    # Get bounds and calculate transform
    minx, miny, maxx, maxy = aoi_gdf.total_bounds if aoi_path else gdf.total_bounds
    width = int((maxx - minx) / dst_res_x)
    height = int((maxy - miny) / dst_res_y)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Rasterize the geometries
    shapes = ((geom, burn_value) for geom in gdf.geometry)
    raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill_value,
    )

    # Write to output raster
    with rasterio.open(dst_raster_path, 'w', driver='GTiff', height=height, width=width, count=1,
            dtype=rasterio.float32, crs=gdf.crs, transform=transform, nodata=dst_nodata) as dst:
        dst.write(raster, 1)


def proximity_raster(
    src_raster_path: Path,
    dst_raster_path: Path,
    src_burn_value: float = 1.0,
):
    """
    Compute pixel proximity raster to values from existing raster.

    Parameters:
    - src_raster_path (str): Path to the input raster file.
    - dst_raster_path (str): Path to save the output raster file.
    - src_burn_value (float): Value from source raster to compute proximities (default is 1).
    """
    # Open the source raster
    with rasterio.open(src_raster_path) as src:
        # Read the first band
        data = src.read(1)

        # Create a mask of where the burn value exists
        burn_value_mask = data == src_burn_value

        # Calculate the proximity using the distance transform
        proximity = distance_transform_edt(~burn_value_mask, sampling=src.res)

        # Update metadata for the output raster
        dst_meta = src.meta.copy()
        dst_meta.update({
            'dtype': 'float32'
        })

        # Write the proximity raster to the destination path
        with rasterio.open(dst_raster_path, 'w', **dst_meta) as dst:
            dst.write(proximity.astype(np.float32), 1)


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


def align_rasters(
    src_raster_path,
    dst_raster_path,
    reference_raster_path,
    resampling=rasterio.warp.Resampling.bilinear
):
    """
    Aligns a target raster to a reference raster using rasterio.

    Parameters:
    - src_raster_path (str): Path to the target raster file to be aligned.
    - dst_raster_path (str): Path to save the aligned output raster file.
    - reference_raster_path (str): Path to the reference raster file.
    """
    # Open the reference raster
    with rasterio.open(reference_raster_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    # Open the target raster
    with rasterio.open(src_raster_path) as target:

        # Set up the metadata for the aligned output raster
        aligned_meta = target.meta.copy()
        aligned_meta.update({
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height
        })

        # Perform the alignment by reprojecting the target raster
        with rasterio.open(dst_raster_path, 'w', **aligned_meta) as aligned_raster:
            for i in range(1, target.count + 1):  # Loop through each band
                reproject(
                    source=rasterio.band(target, i),
                    destination=rasterio.band(aligned_raster, i),
                    src_transform=target.transform,
                    src_crs=target.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling
                )


def preprocess_evidence_layers(
    event_obj: ProspectModelMetaData,
    layers: List[Path],
    aoi: Path,
    reference_layer_path: Path,
) -> List[Path]:
    """
    Preprocessing evidence layers

    Parameters:
    event_obj (ProspectModelMetaData): The prospect model metadata object
    layers (List[Path]): List with Paths of raw/not processed layers
    aoi (Path): Path to the region of interest
    reference_layer_path (Path):

    Returns:
    List[Path]: List with Paths of fully preprocessed layers
    """
    pev_lyr_paths = []
    for idx, layer in tqdm(enumerate(layers)):
        if layer.suffix == ".tif":
            pev_lyr_path = preprocess_raster(event_obj, layer, aoi, reference_layer_path, idx)
        elif layer.suffix == ".zip":
            pev_lyr_path = preprocess_vector(event_obj, layer, aoi, reference_layer_path, idx)
        pev_lyr_paths.append(pev_lyr_path)
    return pev_lyr_paths


def transform_raster(
    src_raster_path,
    dst_raster_path,
    method: str = Literal["log","abs","sqrt"]
) -> None:
    """
    Apply a transformation function to a raster image.

    Parameters:
    - src_raster_path (str): Path to the input raster file.
    - dst_raster_path (str): Path to save the output raster file.
    - method (str): The transformation function to apply.
    """
    with rasterio.open(src_raster_path) as src:
        metadata = src.meta
        metadata.update(dtype=rasterio.float32)

        raster_data = src.read(1)
        raster_data = np.where(raster_data == metadata['nodata'], np.nan, raster_data)

        if method == "log":
            transformed_data = np.log(np.where(raster_data > 0, raster_data, np.nan))
        elif method == "abs":
            transformed_data = np.abs(raster_data)
        elif method == "sqrt":
            transformed_data = np.sqrt(np.where(raster_data >= 0, raster_data, np.nan))
        else:
            raise Exception(f"Unknown transform function {method}.")

        transformed_data = np.where(np.isnan(transformed_data), metadata['nodata'], transformed_data)

    with rasterio.open(dst_raster_path, 'w', **metadata) as dst:
        dst.write(transformed_data.astype(rasterio.float32), 1)


def preprocess_raster(
    event_obj: ProspectModelMetaData,
    layer: Path,
    aoi: Path,
    reference_layer_path: Path,
    layer_idx: int,
    imputation_size: int = 100,
    window_size: int = 5,
):
    # get UI specified preprocessing methods
    transform_methods = event_obj.evidence_layers[layer_idx].transform_methods
    transform_methods_dict = {
        'transform': None,
        'impute_method': None,
        'impute_window_size': None,
        'scaling': 'standard'
    }
    for method in transform_methods:
        if isinstance(method, TransformMethod):
            transform_methods_dict['transform'] = method.value
        elif isinstance(method, Impute):
            transform_methods_dict['impute_method'] = method.impute_method.value
            transform_methods_dict['impute_window_size'] = method.window_size
        elif isinstance(method, ScalingType):
            transform_methods_dict['scaling'] = method.value
        else:
            raise ValueError("Unknown method")

    formatted_file = layer.parent / (layer.stem + "_formatted" + layer.suffix)
    warped_file = layer.parent / (layer.stem + "_warped" + layer.suffix)
    imputed_file = layer.parent / (layer.stem + "_imputed" + layer.suffix)
    clipped_file = layer.parent / (layer.stem + "_clipped" + layer.suffix)
    aligned_file = layer.parent / (layer.stem + "_aligned" + layer.suffix)
    dilated_file = layer.parent / (layer.stem + "_dilated" + layer.suffix)
    olr_file = layer.parent / (layer.stem + "_olr" + layer.suffix)
    if transform_methods_dict['transform']:
        scaled_file = layer.parent / (layer.stem + "_scaled" + layer.suffix)
        transform_file = layer.parent / (layer.stem + "_processed" + layer.suffix)
    else:
        scaled_file = layer.parent / (layer.stem + "_processed" + layer.suffix)

    format_nodata_crs(
        src_raster_path=layer,
        dst_raster_path=formatted_file,
    )
    warp_raster(
        src_raster_path=formatted_file,
        dst_raster_path=warped_file,
        dst_crs=event_obj.cma.crs,
        dst_res_x=event_obj.cma.resolution[0],
        dst_res_y=event_obj.cma.resolution[1],
    )
    dilate_raster( # impute
        src_raster_path=warped_file,
        dst_raster_path=imputed_file,
        dilation_size=imputation_size,
    )
    clip_raster(
        src_raster_path=imputed_file,
        dst_raster_path=clipped_file,
        aoi_path = str(aoi),
    )
    align_rasters(
        src_raster_path=clipped_file,
        dst_raster_path=aligned_file,
        reference_raster_path=reference_layer_path,
    )
    dilate_raster( # dilate
        src_raster_path=aligned_file,
        dst_raster_path=dilated_file,
        dilation_size=window_size,
    )
    remove_outliers_tukey_raster(
        src_raster_path=dilated_file,
        dst_raster_path=olr_file,
    )
    scale_raster(
        src_raster_path=olr_file,
        dst_raster_path=scaled_file,
        scaling_type=transform_methods_dict['scaling']
    )
    if transform_methods_dict['transform']:
        transform_raster(
            src_raster_path=scaled_file,
            dst_raster_path=transform_file,
            method=transform_methods_dict['transform']
        )
        return transform_file
    else:
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
    reference_layer_path: Path,
    layer_idx: int,
    window_size: int = 5,
):
    # get UI specified preprocessing methods
    transform_methods = event_obj.evidence_layers[layer_idx].transform_methods
    transform_methods_dict = {
        'transform': None,
        'impute_method': None,
        'impute_window_size': None,
        'scaling': 'standard'
    }
    for method in transform_methods:
        if isinstance(method, TransformMethod):
            transform_methods_dict['transform'] = method.value
        elif isinstance(method, Impute):
            transform_methods_dict['impute_method'] = method.impute_method.value
            transform_methods_dict['impute_window_size'] = method.window_size
        elif isinstance(method, ScalingType):
            transform_methods_dict['scaling'] = method.value
        else:
            raise ValueError("Unknown method")

    # gets vector file path
    shp_file = find_shapefiles(layer.parent / layer.stem)
    if len(shp_file) > 1 or len(shp_file) == 0: raise Exception(f"Cannot process vector file {layer}.")
    shp_file = Path(shp_file[0])
    # prepares preprocessing file names
    warped_shp_file = layer.parent / layer.stem / (shp_file.stem + "_warped" + shp_file.suffix)
    rasterized_file = layer.parent / (layer.stem + "_rasterized.tif")
    proximity_file = rasterized_file.parent / (rasterized_file.stem +"_proximity" + rasterized_file.suffix)
    clipped_file = rasterized_file.parent / (rasterized_file.stem +"_clipped" + rasterized_file.suffix)
    aligned_file = rasterized_file.parent / (rasterized_file.stem +"_aligned" + rasterized_file.suffix)
    dilated_file = rasterized_file.parent / (rasterized_file.stem +"_dilated" + rasterized_file.suffix)
    olr_file = rasterized_file.parent / (rasterized_file.stem +"_olr" + rasterized_file.suffix)
    if transform_methods_dict['transform']:
        scaled_file = layer.parent / (layer.stem + "_scaled" + layer.suffix)
        transform_file = layer.parent / (layer.stem + "_processed" + layer.suffix)
    else:
        scaled_file = layer.parent / (layer.stem + "_processed" + layer.suffix)

    warp_vector(
        src_vector_path = shp_file,
        dst_vector_path = warped_shp_file,
        dst_crs = event_obj.cma.crs,
    )
    vector_to_raster(
        src_vector_path=warped_shp_file,
        dst_raster_path=rasterized_file,
        dst_res_x = event_obj.cma.resolution[0],
        dst_res_y = event_obj.cma.resolution[1],
        fill_value=np.nan
    )
    proximity_raster(
        src_raster_path=rasterized_file,
        dst_raster_path=proximity_file,
    )
    clip_raster(
        src_raster_path=proximity_file,
        dst_raster_path=clipped_file,
        aoi_path = str(aoi),
    )
    align_rasters(
        src_raster_path=clipped_file,
        dst_raster_path=aligned_file,
        reference_raster_path=reference_layer_path,
    )
    dilate_raster(
        src_raster_path=aligned_file,
        dst_raster_path=dilated_file,
        dilation_size=window_size,
    )
    remove_outliers_tukey_raster(
        src_raster_path=dilated_file,
        dst_raster_path=olr_file,
    )
    scale_raster(
        src_raster_path=olr_file,
        dst_raster_path=scaled_file,
        scaling_type=transform_methods_dict['scaling']
    )
    if transform_methods_dict['transform']:
        transform_raster(
            src_raster_path=scaled_file,
            dst_raster_path=transform_file,
            method=transform_methods_dict['transform']
        )
        return transform_file
    else:
        return scaled_file


def deposits_filtering(
    df: pd.DataFrame,
    deposit_type: str,
    confidence_threshold: float,
):
    """
    Filter df DataFrame

    Parameters:
    - df (pd.DataFrame): pandas DataFrame
    - deposit_type (str): Deposit type
    - confidence_threshold (float): Confidence threshold for deposits filtering
    """
    original_len = len(df)
    df = df[df['top1_deposit_type'].str.contains(deposit_type, case=False, na=False)]
    df = df[df['top1_deposit_classification_confidence'] >= confidence_threshold]
    df = df[df['type'].str.contains('Past Producer|Prospect|Producer|NotSpecified', na=False)]
    df = df[df['rank'].str.contains('A|B|C|U', na=False)]
    df = df.reset_index(drop=True)
    print(f'Original length: {original_len}, Filtered length: {len(df)}')
    return df


def process_label_raster(
    event_obj: ProspectModelMetaData,
    deposits_csv_path: Path,
    aoi: Path,
    reference_layer_path: Path,
    confidence_threshold: float = 0.5,
    dilation_size: int = 5,
):
    """
    Rasterize a .csv file with deposits to a raster (label raster)

    Parameters:
    - event_obj ():
    - deposits_csv_path (str): Path to the input .csv file wiht deposits.
    - aoi (str): Path to the shapefile defining the region of interest.
    - confidence_threshold (float): Confidence threshold for deposits filtering (default is 0.5).
    - dilation_size (int): Distance for interpolation (default is 5).
    """
    warped_shp_file = deposits_csv_path.parent / (deposits_csv_path.stem + '_warped.shp')
    rasterized_file = deposits_csv_path.parent / (deposits_csv_path.stem + '_rasterized.tif')
    clipped_file = deposits_csv_path.parent / (deposits_csv_path.stem + '_clipped.tif')
    aligned_file = deposits_csv_path.parent / (deposits_csv_path.stem + '_aligned.tif')
    dilated_file = deposits_csv_path.parent / (deposits_csv_path.stem + '_processed.tif')

    df = pd.read_csv(deposits_csv_path)
    deposit_type = event_obj.cma.mineral
    df = deposits_filtering(
        df,
        deposit_type,
        confidence_threshold
    )
    df.to_csv(deposits_csv_path.parent / f'{deposit_type}_filtered.csv', index=False)

    geom = gpd.GeoSeries.from_wkt(df['centroid_epsg_4326'], crs='EPSG:4326')
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    gdf = gdf.to_crs(event_obj.cma.crs)
    # clipping labels to the aoi
    aoi_gdf = gpd.read_file(aoi)
    clipped_gdf = gpd.clip(gdf, aoi_gdf, keep_geom_type=True)
    # saving labels to file
    clipped_gdf.to_file(warped_shp_file)

    vector_to_raster(
        src_vector_path=warped_shp_file,
        dst_raster_path=rasterized_file,
        dst_res_x=event_obj.cma.resolution[0],
        dst_res_y=event_obj.cma.resolution[1],
        fill_value=0.0,
        aoi_path=aoi
    )
    clip_raster(
        src_raster_path=rasterized_file,
        dst_raster_path=clipped_file,
        aoi_path=aoi
    )
    align_rasters(
        src_raster_path=clipped_file,
        dst_raster_path=aligned_file,
        reference_raster_path=reference_layer_path,
        resampling=rasterio.warp.Resampling.nearest
    )
    dilate_raster(
        src_raster_path=aligned_file, #clipped_file,
        dst_raster_path=dilated_file,
        dilation_size=dilation_size,
        label_raster=True
    )
    ### Find out the number of rasterized deposits ###
    # Load the raster data
    with rasterio.open(dilated_file) as src:
        raster_data = src.read(1)
    # Calculate the number of ones
    num_of_deposits = np.count_nonzero(raster_data == 1)

    return dilated_file, num_of_deposits


def create_raster_stack_yaml(
    event_obj: ProspectModelMetaData,
    evidence_layer_paths: List[Path],
    label_raster_path: Path,
    raster_stack_path: Path,
    data_path: Path = Path("./data"),
):
    """
    Creates .yaml file with information about rasters that go into raster stack

    Parameters:
    event_obj (ProspectModelMetaData):
    evidence_layer_paths (List[Path]): Paths to evidence layers
    label_raster_path (Path): Path to label raster
    data_path (Path, optional): Path where to output .yaml file. Defaults to Path("./data").

    Returns:
    Path: .yaml file path.
    """
    yaml_output_path = data_path / Path(event_obj.model_run_id)
    yaml_output_path.mkdir(parents=True, exist_ok=True)

    raster_files = []
    # evidence rasters
    for filename in evidence_layer_paths:
        filename = str(filename)
        if filename.endswith('.tif'):
            raster_files.append(filename)
    # label raster
    label_raster = str(label_raster_path)
    variables = {
        '_target_' : 'sri_maper.src.data.preprocessing.generate_raster_stacks',
        'raster_stacks' : [
            {
                'raster_stack_path' : str(raster_stack_path),
                'window_size' : 5,
                'evidence_layer_paths' : raster_files,
                'label_raster_path' : [label_raster],
            }
        ]
    }
    yaml_output_path = yaml_output_path / 'preprocessing.yaml'
    with open(yaml_output_path, 'w') as file:
        yaml.dump(variables, file, sort_keys=False)
    return yaml_output_path


def load_rasters(
    evidence_layer_paths: List[str],
):
    return [load_raster(evidence_layer_path) for evidence_layer_path in evidence_layer_paths]


def load_raster(
    evidence_layer_path,
):
    raster = rasterio.open(evidence_layer_path)
    log.debug(f"-------- {evidence_layer_path} raster details --------\n")
    info = {i: dtype for i, dtype in zip(raster.indexes, raster.dtypes)}
    log.debug(f"Raster bands and dtypes:\n{info}\n\n")
    log.debug(f"Coordinate reference system:\n{raster.crs}\n\n")
    log.debug(f"Bounds:{raster.bounds},Size:{raster.shape},Resolution:{raster.res}\n\n")
    return raster


def generate_raster_stacks(raster_stacks):
    pass
    # for raster_stack in tqdm(raster_stacks):
    #     if not Path(raster_stack.raster_stack_path).is_file():
    #         generate_raster_stack(
    #             raster_stack.evidence_layer_paths,
    #             raster_stack.label_raster_path,
    #             raster_stack.window_size
    #         )


def generate_raster_stack(
    evidence_layer_paths: List[Path],
    label_raster_path: Path,
    window_size: int = 5
):
    """
    Generates a multi-band GeoTiff (i.e., raster stack). Assumes each raster is already
    aligned and has imputed, outlier-removed, and scaled values.

    """
    raster_stack_path = label_raster_path.parent.parent / 'raster_stack'
    raster_stack_path.mkdir(parents=True, exist_ok=True)
    raster_stack_path = raster_stack_path / f'raster_stack_d{window_size}.tif'

    all_raster_paths = evidence_layer_paths+[label_raster_path]
    rasters = load_rasters(all_raster_paths)
    rasters_data = [raster.read(1, masked=True) for raster in rasters]

    # creating list of raster shapes, raster masks, and overall mask
    raster_shapes = [raster_data.shape for raster_data in rasters_data]

    # creates the raster dataframe
    raster_df = pd.DataFrame()
    for raster_path, raster_data in zip(all_raster_paths, rasters_data):
        raster_df[f"{raster_path.name}"] = raster_data.flatten()
    raster_df.loc[raster_df.isnull().any(axis=1), :] = np.nan # if any value in a row is NaN, set all values in that row to NaN

    new_rasters_data = []
    for i, raster_path in tqdm(enumerate(all_raster_paths)):
        # extracts masked numpy arrays from the from dataframe
        new_raster_data = raster_df[f"{raster_path.name}"].values.reshape(raster_shapes[i])
        new_rasters_data.append(new_raster_data)

    # generates and saves raster stack
    raster_stack_meta = rasters[0].meta
    raster_stack_meta.update({"count": len(new_rasters_data)})
    raster_stack_meta.update({"dtype": "float32"})
    log.debug(f"Writing a raster stack with the following meta data: {raster_stack_meta}")
    with rasterio.open(raster_stack_path, "w", **raster_stack_meta) as raster_stack:
        tags = {raster_path.stem: idx for idx, raster_path in enumerate(all_raster_paths)}
        tags["ns"] = "evidence_layers"
        raster_stack.update_tags(**tags)
        for idx, new_raster_data in enumerate(new_rasters_data):
            raster_stack.write_band(idx+1, new_raster_data)

    return raster_stack_path
