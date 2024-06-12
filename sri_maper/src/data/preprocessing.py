from typing import List, Dict
from os import makedirs
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from rasterio.fill import fillnodata
import rasterio
from sri_maper.src import utils

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
    Generates a multi-band GeoTiff (i.e. raster stack). Assumes each raster is already
    aligned and has imputed values.

    :param raster_stack_path: path to save the raster stack
    :type raster_stack_path: str
    :param raster_files_path: root path to library of single band rasters
    :type raster_files_path: str
    :param raster_files: list of paths to single band rasters
    :type raster_files: List[str]
    :param raster_files_types: list of types of single band rasters
    :type raster_files_types: List[str]
    ...
    :return: None
    :rtype: None
    """
    # loads the individual rasters
    rasters = load_rasters(raster_files, raster_files_path)
    rasters_data = [raster.read(1, masked=True) for raster in rasters]
    for raster_data in rasters_data: np.ma.set_fill_value(raster_data, np.nan)
    rasters_msk = [raster.read_masks(1) for raster in rasters]
    raster_shapes = [raster_data.shape for raster_data in rasters_data]

    # creates the raster dataframe
    raster_df = pd.DataFrame()
    for raster_file, raster_data in zip(raster_files, rasters_data):
        raster_df[f"{raster_file.path}"] = raster_data.filled().flatten()

    new_rasters_data = []
    for i, raster_file in tqdm(enumerate(raster_files)):
        raster_name = raster_file.path
        raster_type = raster_file.type
        outlier_removal = raster_file.outlier_removal
        normalize = raster_file.normalize
        # removes outliers and normalizes
        raster_df = tukey_remove_outliers(raster_df, raster_name, raster_type, outlier_removal)
        raster_df = normalize_df(raster_df, raster_name, raster_type, normalize)
        # extracts masked numpy arrays from the from dataframe
        new_raster_data = raster_df[f"{raster_name}"].values.reshape(raster_shapes[i])
        new_raster_data = np.ma.masked_array(new_raster_data, mask=~rasters_msk[i], fill_value=np.nan)
        new_rasters_data.append(new_raster_data)

    # storing pre-dilation NaNs locations
    nan_mask = np.isnan(new_rasters_data[-1])

    # dilates the rasters
    new_rasters_data = [fillnodata(new_raster_data, max_search_distance=dilation_size) for new_raster_data in new_rasters_data]

    # overwriting unwanted label dilations
    new_rasters_data[-1][nan_mask & ~np.isnan(new_rasters_data[-1])] = 0.

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
    col_type,
    remove_outliers=False,
    multiplier=1.5,
    replacement_percentile=0.05
):
    if not remove_outliers or col_type == "bool":
        return df
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
    col_name,
    col_type,
    normalize=True,
):
    if not normalize or col_type == "bool":
        return df
    std = df[col_name].std()
    if std != 0.0:
        df[col_name] = (df[col_name]-df[col_name].mean()) / std
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
