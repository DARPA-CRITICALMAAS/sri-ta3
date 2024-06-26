from typing import Union, List, Callable, Optional
from math import ceil
from glob import glob
from pathlib import Path
from functools import partial
from multiprocessing import Process, Pool, cpu_count
from tqdm import tqdm
from rasterio.windows import Window
from rasterio import open as rio_open
from rasterio import Env as rio_env
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from torch import tensor, half
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sri_maper.src import utils

log = utils.get_pylogger(__name__)

# we need some way of splitting the TiFFs into train, valid, and test splits
# that are spatially independent BUT have similar distributions

import pdb

class TiffDataset(Dataset):
    def __init__(self,
        tif_dir: Union[str, None] = None,
        tif_files: Union[List[str], None] = None,
        tif_data: Union[np.ndarray, None] = None,
        tif_tags: Union[dict, None] = None,
        tif_meta: Union[dict, None] = None,
        valid_patches: Union[np.ndarray, None] = None,
        window_size: int = 33,
        stage: Union[np.ndarray, None] = None,
        num_pca_components: int = None,
        pca_matrices: Union[np.ndarray, None] = None,
    ):
        # sets object variables
        self.window_size = window_size
        self.stage = stage
        self.num_pca_components = num_pca_components

        # loads tif files in MP compatible format
        self.tif_files, self.tif_data, self.tif_tags, self.tif_meta = self._load_tif_files(tif_dir, tif_files, tif_data, tif_tags, tif_meta)

        # loads VALID patches within all tiffs of dataset
        self.valid_patches = self._load_valid_patches(self.tif_files, window_size) if valid_patches is None else valid_patches

        # load PCA matrices if it exists, otherwise generate them
        if num_pca_components is None:
            self.pca_matrices = None
        else:
            self.pca_matrices = self._load_pca_matrices(self.tif_files, self.tif_data, self.tif_tags, num_pca_components) if pca_matrices is None else pca_matrices

    def _load_tif_files(self, tif_dir, tif_files, tif_data, tif_tags, tif_meta):
        # sets List[str] of tif files
        assert (tif_files is None and tif_dir is not None) or (tif_files is not None and tif_dir is None), "tif_dir and tif_files BOTH set."
        if tif_files is None:
            # tif files loaded with correct window size ONLY
            tif_files = glob(str(Path(tif_dir) / Path(f"*_d{self.window_size}.tif")))
            tif_data = []
            for tif_file in tif_files:
                log.info(f"Loading tif data for for {tif_file}")
                with rio_env(GDAL_CACHEMAX=0):
                    with rio_open(tif_file, driver='GTiff') as tif:
                        tif_tags = tif.tags(ns="evidence_layers")
                        tif_meta = tif.meta
                        tif_data.append(tensor(tif.read().astype("half"), dtype=half))
        return tif_files, tif_data, tif_tags, tif_meta

    def _load_valid_patches(self, tif_files, window_size):
        # loads or generates df indicating which tif patches are VALID
        ds_valid_patches = []
        for tif_idx, tif_file in  enumerate(tif_files):
            valid_patch_file = Path(tif_file).parent / Path(f"{Path(tif_file).stem}_valid_p{window_size}.npy")
            try:
                # check if valid patch dataframe already exists
                log.info(f"Loading np.ndarray enumerating valid patches for {tif_file} (~5 min)")
                valid_patches = np.load(valid_patch_file)
            except FileNotFoundError:
                # if not, generate valid patch dataframe
                log.warning(f"np.ndarray not found. Generating.")
                valid_patches = self._generate_valid_patches(tif_file, window_size)
                np.save(valid_patch_file, valid_patches)

            valid_patches = np.hstack([valid_patches, tif_idx*np.ones(shape=(valid_patches.shape[0],1))])
            ds_valid_patches.append(valid_patches)
        # returns valid patches of ALL tiffs in dataset
        return np.vstack(ds_valid_patches)

    def _load_pca_matrices(self, tif_files, tif_data, tif_tags, num_pca_components):
        pca_matrices = []
        for tif_idx, tif_file in  enumerate(tif_files):
            # loads or generates PCA matrix
            pca_matrix_file = Path(tif_file).parent / Path(Path(tif_file).name.split(".")[0]+f"_pca{num_pca_components}_matrix.npy")
            try:
                # check if PCA matrix already exists
                log.info(f"Loading PCA matrix.")
                pca_matrix = np.load(pca_matrix_file)
            except FileNotFoundError:
                # if not, generate PCA matrix
                log.warning(f"PCA matrix not found. No file: {pca_matrix_file}. Generating.")
                pca_matrix = self._generate_pca_matrix(tif_data,
                                                        tif_idx,
                                                        num_pca_components,
                                                        out_png=True,
                                                        output_path=Path(tif_file).parent / Path(Path(tif_file).name.split(".")[0]+f"_pca{num_pca_components}_matrix.png"),
                                                        raster_names=tif_tags)
                np.save(pca_matrix_file, pca_matrix)
            pca_matrices.append(pca_matrix)
        return np.array(pca_matrices)

    @staticmethod
    def _generate_valid_patches(tif_file, window_size):
        with rio_open(tif_file, "r") as tif:
            tif_height = tif.height
            tif_width = tif.width

        # extracts the pixel coords of raster
        rows, cols = np.mgrid[0:tif_height:1,0:tif_width:1].reshape((-1, (tif_width)*(tif_height)))

        # sets up multiprocessing pool
        log.warning(f"Using {cpu_count()} threads to enumerate and store valid patches to np.ndarray")
        pool = Pool(cpu_count())

        # splits data into mp.cpu_count() chunks
        chunk_size = len(rows) // cpu_count()
        chunks = [(cols[i:i+chunk_size],rows[i:i+chunk_size]) for i in range(0, len(rows), chunk_size)]

        # enumerates all valid patches with multiprocessing
        validate_patches_multi = partial(validate_patches, window_size=window_size, tif_file=tif_file)
        valid_patches = np.vstack(pool.map(validate_patches_multi, chunks))

        # closes the pool to free up resources
        pool.close()
        pool.join()

        return valid_patches

    @staticmethod
    def _generate_pca_matrix(tif_data, tif_idx, num_pca_components, out_png=False, output_path=None, raster_names=None):
        # data is a numpy array of shape (num_features, width, height)
        data = tif_data[tif_idx][:-1,:,:] # exclude the label raster data/feature
        data = data.flatten(start_dim=1).transpose(0,1).numpy()
        data = data[~np.isnan(data).any(axis=1)] # remove rows with NaNs
        pca = PCA(n_components=num_pca_components, svd_solver='full')
        _ = pca.fit_transform(data)
        transformation_matrix = pca.components_

        if out_png:
            fig, ax = plt.subplots(figsize=(transformation_matrix.shape[1]/2, transformation_matrix.shape[0]))
            im = ax.imshow(transformation_matrix, cmap='bwr', interpolation='nearest')

            # Change the axis marks
            ax.set_yticks(np.arange(0, transformation_matrix.shape[0]), np.arange(1, transformation_matrix.shape[0] + 1))
            # ax.set_xticks(np.arange(0, transformation_matrix.shape[1]), np.arange(1, transformation_matrix.shape[1] + 1))
            ax.set_xticks(np.arange(0, transformation_matrix.shape[1]), list(raster_names.keys())[1:])
            ax.xaxis.tick_top()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='left', fontsize=7)

            # insert the values into output matrix
            for i in range(transformation_matrix.shape[0]):
                for j in range(transformation_matrix.shape[1]):
                    text = ax.text(j, i, np.around(transformation_matrix[i, j], decimals=2),
                                    ha="center", va="center", color="k", fontsize=7)
            plt.colorbar(im, shrink=0.5)
            plt.savefig(output_path)
        return transformation_matrix.T

    def __len__(self):
        return self.valid_patches.shape[0]

    def __getitem__(self, idx):
        # loads the patch's location and label
        col = int(self.valid_patches[idx,0])
        row = int(self.valid_patches[idx,1])
        label = self.valid_patches[idx,2]
        source_tif = int(self.valid_patches[idx,-1])

        # loads the patch's data
        patch = self.tif_data[source_tif][:-1,row:row+self.window_size,col:col+self.window_size]

        if self.stage == "predict":
            # produce map
            lon = self.valid_patches[idx,-3]
            lat = self.valid_patches[idx,-2]
            return patch, label, lon, lat, (self.pca_matrices[source_tif] if self.pca_matrices is not None else -1)
        else:
            return patch, label, (self.pca_matrices[source_tif] if self.pca_matrices is not None else -1)

def validate_patches(chunk, window_size, tif_file):
    # creates MP friendly iterator that estimates run-time
    if Process()._identity[0] == 1:
        chunk_iter = tqdm(zip(chunk[0],chunk[1]), total=len(chunk[0]))
    else:
        chunk_iter = zip(chunk[0],chunk[1])

    # validates the patches made from pixel locations in chunks
    records = []
    with rio_open(tif_file) as f:
        for x, y in chunk_iter:
            patch = f.read(window=Window(x, y, window_size, window_size))
            if patch.shape != (f.count, window_size, window_size) or np.isnan(patch).any(): continue
            records.append([x, y, patch[-1, window_size//2, window_size//2]])
        tif_tfm = f.transform

    # creates dataframe cataloging valid patches
    records = np.asarray(records)

    # efficiently adds lat / lon to dataframe
    if records.shape[0]:
        cols = records[:,0] + 0.5 + window_size//2
        rows = records[:,1] + 0.5 + window_size//2
        pts = np.dot(np.asarray(tif_tfm.column_vectors).T, np.vstack((cols, rows, np.ones_like(rows)))).T
        records = np.hstack([records, pts])
    else:
        records = np.empty(shape=(0,5))

    return records

def spatial_cross_val_split(
    ds: Dataset,
    k: int = 5,
    test_set: int = 0,
    val_set: int = 1,
    split_col: str = "lat",
    nbins: Union[int, None] = None,
    samples_per_bin: int = 3.0
):
    log.info(f"Splitting patches with spatial cross-val (2-3 min)")
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )
    # select only the deposit/occurence/neighbor present samples
    target_df = np.unique(ds_df.loc[ds_df["label"] == True, split_col].values)
    # bin the latitudes into sizes of 1-3 samples per bin
    if nbins is None:
        nbins = ceil(len(target_df) / samples_per_bin)
    _, bins = pd.qcut(target_df, nbins, retbins=True)
    bins[0] = -float("inf")
    bins[-1] = float("inf")
    bins = pd.IntervalIndex.from_breaks(bins)
    # group the bins into k groups (folds)
    bins_df = pd.DataFrame({f"{split_col}_bin": bins})
    bins_df["group"] = np.tile(np.arange(k), (ceil(nbins / k),))[:nbins]
    # assign all data to a k+1 group using the existing bin / group assignments
    ds_df[f"{split_col}_bin"] = pd.cut(ds_df[split_col], bins)
    ds_df = pd.merge(ds_df, bins_df, on=f"{split_col}_bin")
    # split into train / test data
    test_valid_patches = ds_df[ds_df["group"] == test_set].drop(columns=[f"{split_col}_bin","group"]).reset_index(drop=True).values
    test_ds = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=test_valid_patches,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    val_valid_patches = ds_df[ds_df["group"] == val_set].drop(columns=[f"{split_col}_bin","group"]).reset_index(drop=True).values
    val_ds = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=val_valid_patches,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    ds_valid_patches = ds_df[(ds_df["group"] != test_set) & (ds_df["group"] != val_set)].drop(columns=[f"{split_col}_bin","group"]).reset_index(drop=True).values
    ds = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_valid_patches,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    return ds, val_ds, test_ds

def combine(ds1, ds2):
    ds1_df = pd.DataFrame(
        data=ds1.valid_patches,
        index=np.arange(ds1.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )

    ds2_df = pd.DataFrame(
        data=ds2.valid_patches,
        index=np.arange(ds2.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )

    ds_df = pd.concat([ds1_df, ds2_df], axis=0).reset_index(drop=True)

    ds = TiffDataset(
        tif_files=ds1.tif_files,
        tif_data=ds1.tif_data,
        tif_tags=ds1.tif_tags,
        tif_meta=ds1.tif_meta,
        window_size=ds1.window_size,
        stage=ds1.stage,
        valid_patches=ds_df.values,
        num_pca_components=ds1.num_pca_components,
        pca_matrices=ds1.pca_matrices
    )
    return ds

def random_split(
    ds: Dataset,
    train_split: float = 0.8,
    seed: int = 0,
):
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )

    ds_df_train, ds_df_temp = train_test_split(ds_df, test_size=1.0-train_split, random_state=seed)
    ds_df_valid, ds_df_test = train_test_split(ds_df_temp, test_size=0.5, random_state=seed)

    ds_train = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_train.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )

    ds_valid = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_valid.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )

    ds_test = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_test.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )

    return ds_train, ds_valid, ds_test

def random_proportionate_split(
    ds: Dataset,
    train_split: float = 0.8,
    seed: int = 0,
):
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )

    # make positive / negative datasets
    ds_df_p = ds_df[ds_df["label"] == 1]
    ds_p = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_p.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    ds_df_n = ds_df[ds_df["label"] == 0]
    ds_n = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_n.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )

    ds_p_train, ds_p_valid, ds_p_test = random_split(ds_p, train_split, seed=seed)
    ds_n_train, ds_n_valid, ds_n_test = random_split(ds_n, train_split, seed=seed)

    ds_train = combine(ds_p_train, ds_n_train)
    ds_valid = combine(ds_p_valid, ds_n_valid)
    ds_test = combine(ds_p_test, ds_n_test)

    return ds_train, ds_valid, ds_test

def select_deposits(df: pd.DataFrame, coordinates: List[List[float]]):
    # select only the deposit from pos_train_coordinates
    selected_rows = []
    for (i,j) in coordinates:
        # Select the row where 'x' and 'y' are equal to i and j
        selected_row = df[(df['x'] == i) & (df['y'] == j)]
        selected_rows.append(selected_row)
    df_selected = pd.concat(selected_rows)

    # select deposits not from pos_train_coordinates
    ds_df_p_notselected = pd.merge(df, df_selected, how='outer', indicator=True)
    ds_df_p_notselected = ds_df_p_notselected[ds_df_p_notselected['_merge'] == 'left_only']
    ds_df_p_notselected = ds_df_p_notselected.drop(columns=['_merge'])

    return df_selected, ds_df_p_notselected

def specified_split(
    ds: Dataset,
    pos_train_coordinates: List[List[float]],
    train_split: float = 0.9,
    # pos_test_coordinates: Optional[List[List[float]]] = None,
    seed: int = 0,
):
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )

    # make positive datasets
    ds_df_p = ds_df[ds_df["label"] == 1]

    assert len(pos_train_coordinates) <= len(ds_df_p), \
        "number of provided coordinates should at most be equal to the number of positive samples in the dataset."

    # seperate into selected and not selected deposits
    ds_df_p_selected, ds_df_p_notselected = select_deposits(ds_df_p, pos_train_coordinates)

    ds_df_p_train, ds_df_p_valid = train_test_split(ds_df_p_selected,
                                                        train_size=train_split,
                                                        random_state=seed)

    ds_p_train = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_p_train.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    ds_p_valid = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_p_valid.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    ds_p_test = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_p_notselected.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )

    # make negative datasets
    ds_df_n = ds_df[ds_df["label"] == 0]
    ds_df_n_train, ds_df_n_temp = train_test_split(ds_df_n,
                                                    train_size=len(ds_df_p_train)/len(ds_df_p),
                                                    random_state=seed)
    ds_df_n_valid, ds_df_n_test = train_test_split(ds_df_n_temp,
                                                    train_size=len(ds_df_p_valid)/(len(ds_df_p_valid)+len(ds_df_p_notselected)),
                                                    random_state=seed)

    ds_n_train = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_n_train.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    ds_n_valid = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_n_valid.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    ds_n_test = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_n_test.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )

    ds_train = combine(ds_p_train, ds_n_train)
    ds_valid = combine(ds_p_valid, ds_n_valid)
    ds_test = combine(ds_p_test, ds_n_test)

    return ds_train, ds_valid, ds_test

def pu_downsample(
    ds: Dataset,
    feature_extractor: Callable,
    multiplier: int = 20,
    likely_neg_range: List[float] = [0.25,0.75],
    seed: int = 0,
    log_path: str = "",
    in_pca_space: bool = False,
):
    """Function to downsample the dataset using positive-unlabeled learning.

    :params ds: The dataset to downsample.
    :params feature_extractor: The feature extractor function.
    :params multiplier: The multiplier for the number of negative samples. Defaults to `20`.
    :params likely_neg_range: The range of likely negative samples. Defaults to `[0.25,0.75]`.
    :params seed: The random seed. Defaults to `0`.
    :params log_path: The path to store the log. Defaults to `""`.
    :params in_pca_space: Whether to use PCA space for negative selection/downsampling. Defaults to `False`.

    :return: The downsampled dataset.
    """
    log.info("Using Likely Negative Downsampling!")
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )
    # extract positive features
    ds_df_p = ds_df[ds_df["label"] == 1]
    ds_df_p = ds_df_p.reset_index(drop=True)
    ds_p = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_p.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    p_feats = []
    for p_idx in range(len(ds_p)):
        p_patch, _, pca_matrix = ds_p[p_idx]
        if in_pca_space and len(pca_matrix.shape) != 1:
            p_patch = np.einsum('ijk,im->mjk', p_patch, pca_matrix)
        p_feats.append(feature_extractor(p_patch))
    # prepares unlabeled dataset
    ds_df_u = ds_df[ds_df["label"] == 0]
    ds_df_u = ds_df_u.reset_index(drop=True)
    ds_u = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df_u.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    u_feats = []
    for u_idx in range(len(ds_u)):
        u_patch, _, pca_matrix = ds_u[u_idx]
        if in_pca_space and len(pca_matrix.shape) != 1:
            u_patch = np.einsum('ijk,im->mjk', u_patch, pca_matrix)
        u_feats.append(feature_extractor(u_patch))
    # compute the distances between all positives and negatives
    np_samples = np.asarray(p_feats + u_feats)
    tree = KDTree(np_samples)
    pu_dist = np.zeros(shape=(np_samples.shape[0],))
    dists, inds = tree.query(p_feats, k=np_samples.shape[0])
    # measure the distance from each unlabaled to each positive sample
    for p_idx in range(len(p_feats)):
        pu_dist[inds[p_idx]] += dists[p_idx]
    u_dist = pu_dist[len(p_feats):]
    store_samples(ds_u, log_path, "all_unlabeled", {"name":"distances", "values":u_dist})
    # rank unlabeled by negativity likelihood, taking % most negative
    u_dist_sort_idx = np.argsort(u_dist)

    # selects the range for likely negative sampling
    likely_negatives_idx = u_dist_sort_idx[int(len(u_dist_sort_idx)*likely_neg_range[0]):int(len(u_dist_sort_idx)*likely_neg_range[1])]
    ds_df_n = ds_df_u.iloc[likely_negatives_idx]
    # randomly downsample "likely" negatives
    num_negatives = int (ds_df_p.shape[0]) * multiplier
    ds_df_n = ds_df_n.sample(n=num_negatives, replace=False, random_state=seed)
    # combine positives / negatives
    ds_df = pd.concat([ds_df_n, ds_df_p], axis=0).reset_index(drop=True)
    ds = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )
    return ds


def balance_data(
    ds: Dataset,
    multiplier: int = 20,
    downsample: bool = False,
    oversample: bool = False,
    seed: int = 0,
):
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon","lat","source"]
    )

    if downsample:
        # randomly downsample negatives
        ds_df_p = ds_df[ds_df["label"] == 1]
        ds_df_n = ds_df[ds_df["label"] == 0]
        num_negatives = int (ds_df_p.shape[0]) * multiplier
        ds_df_n = ds_df_n.sample(n=num_negatives, replace=False, random_state=seed)
        ds_df = pd.concat([ds_df_n, ds_df_p], axis=0).reset_index(drop=True)

    if oversample:
        # oversample positives
        sampler = RandomOverSampler(sampling_strategy="minority", shrinkage=None, random_state=seed)
        ds_df, y = sampler.fit_resample(ds_df.drop(columns="label"), ds_df["label"])
        ds_df.insert(2, "label", y)

    ds = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        tif_tags=ds.tif_tags,
        tif_meta=ds.tif_meta,
        window_size=ds.window_size,
        stage=ds.stage,
        valid_patches=ds_df.values,
        num_pca_components=ds.num_pca_components,
        pca_matrices=ds.pca_matrices
    )

    return ds

def filter_by_bounds(ds):
    left, top = ds.tif_meta["transform"] * (0, 0)
    right, bottom = ds.tif_meta["transform"] * (ds.tif_meta["width"], ds.tif_meta["height"])
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,3] > left]
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,4] > bottom]
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,3] < right]
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,4] < top]
    return ds

def store_samples(ds, root_path, name, optional_col=None):
    log_str = f"Spatial cross val ouput: {name} pos - {ds.valid_patches[:,2].sum()}, {name} neg - {len(ds)-ds.valid_patches[:,2].sum()}."
    log.info(log_str)
    file_path = f"{root_path}/{name}.csv"
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon","lat","source"]
    )
    if optional_col is not None:
        ds_df[optional_col["name"]] = optional_col["values"]
    ds_df.to_csv(file_path, index=False)
