from typing import List, Dict, Optional
import os
from os import makedirs
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from rasterio.fill import fillnodata
import rasterio
from sri_maper.src import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

log = utils.get_pylogger(__name__)

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


def tukey_remove_outliers(
    df,
    col_name,
    multiplier=1.5,
    replacement_percentile=0.05
):
    # get the IQR
    Q1 = df.loc[:,col_name].quantile(0.25)
    Q3 = df.loc[:,col_name].quantile(0.75)
    IQR = Q3 - Q1
    # get the lower bound replacements and replace the values
    P05 = df.loc[:,col_name].quantile(replacement_percentile)
    mask = df.loc[:,col_name] < (Q1 - multiplier * IQR)
    df.loc[mask, col_name] = P05
    # get the upper bound replacements and replace the values
    P95 = df.loc[:,col_name].quantile(1.0-replacement_percentile)
    mask = df.loc[:,col_name] > (Q3 + multiplier * IQR)
    df.loc[mask, col_name] = P95
    return df


def normalize_df(
    df,
    col_name
):
    mean = np.nanmean(df[col_name]) #.mean()
    std = np.nanstd(df[col_name]) #.std()
    if std != 0.0:
        df[col_name] = (df[col_name]-mean) / std
    else:
        raise ValueError(f"Standard deviation of {col_name} is 0.0.")
    return df


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
