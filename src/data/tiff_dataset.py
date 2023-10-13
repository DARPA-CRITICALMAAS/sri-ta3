from typing import Union, Tuple
from math import ceil
from glob import glob
from pathlib import Path
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy
import torch

import fcntl
from multiprocessing import Process, Lock

import rasterio as rio
import rasterio.windows
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision as tv

from src import utils

log = utils.get_pylogger(__name__)

# we need some way of splitting the TiFFs into train, valid, and test splits
# that are spatially independent BUT have similar distributions

import pdb

class TiffDataset(Dataset):
    def __init__(self,
        tif_dir: str = "/workspace/data/",
        patch_size: int = 33,
        patch_df: Union[pd.DataFrame, None] = None,
        transformation: Union[tv.transforms.Compose, None] = None,
    ):
        # load the raw tif files
        self.tif_dir = tif_dir

        self.tifs = {
            tif_file: rio.open(tif_file) 
            for tif_file in glob(str(Path(tif_dir) / Path("*.tif")))
        }
        # self.tifs = {}
        # for tif_file in glob(str(Path(tif_dir) / Path("*.tif"))):
        #     with open(tif_file) as src:
        #         fcntl.flock(src, fcntl.LOCK_EX)
        #         self.tifs[tif_file] = rio.open(tif_file)
        # breakpoint()
        self.patch_size = patch_size

        # load a valid patch df for each tif
        self.patch_df = self._load_valid_patch_dfs() if patch_df is None else patch_df

        self.tfm = transformation

    def _load_valid_patch_dfs(self):
        # loads or generates df indicating which tif patches are VALID
        patch_dfs = []
        for tif_file, tif in self.tifs.items():
            patch_df_file = f"{str(tif_file).split('.')[0]}_valid_p{self.patch_size}_df_debug.csv"
            try:
                # check if valid patch dataframe already exists
                log.info(f"Loading dataframe CSV enumerating valid patches (2-3 min)")
                patch_df = pd.read_csv(patch_df_file)
            except FileNotFoundError:
                # if not, generate valid patch dataframe
                patch_df = self._generate_patch_df(tif_file, tif, self.patch_size)
                patch_df["tif_file"] = tif_file
                patch_df.to_csv(patch_df_file, index=False)
            patch_dfs.append(patch_df)
        return pd.concat(patch_dfs, ignore_index=True)
    
    @staticmethod
    def _generate_patch_df(tif_file, tif, patch_size):
        # extracts the pixel coords of raster
        rows, cols = np.mgrid[0:tif.height:1,0:tif.width:1].reshape((-1, (tif.width)*(tif.height)))

        # sets up multiprocessing pool
        log.warning(f"CSV not found. Using {mp.cpu_count()} threads to enumerate and store valid patches to CSV")
        pool = mp.Pool(mp.cpu_count())

        # splits data into mp.cpu_count() chunks
        chunk_size = len(rows) // mp.cpu_count()
        chunks = [(cols[i:i+chunk_size],rows[i:i+chunk_size]) for i in range(0, len(rows), chunk_size)]

        # enumerates all valid patches with multiprocessing
        validate_patches_multi = partial(validate_patches, patch_size=patch_size, tif_file=tif_file)
        df = pd.concat(pool.map(validate_patches_multi, chunks))

        # closes the pool to free up resources
        pool.close()
        pool.join()

        return df

    def __len__(self):
        return len(self.patch_df)
    
    def __getitem__(self, idx):
        col, row, label, source_tif = self.patch_df.loc[idx, ["x", "y", "label", "tif_file"]]
        
        patch = self.tifs[source_tif].read(window=rio.windows.Window(col, row, self.patch_size, self.patch_size))[:-1,:,:]
        
        # need to add transforms where appropriate
        return self.tfm(patch), label #torch.tensor(label, dtype=torch.half)


def validate_patches(chunk, patch_size, tif_file):
    # creates MP friendly iterator that estimates run-time
    if mp.Process()._identity[0] == 1:
        chunk_iter = tqdm(zip(chunk[0],chunk[1]), total=len(chunk[0]))
    else:
        chunk_iter = zip(chunk[0],chunk[1])
    
    # validates the patches made from pixel locations in chunks
    records = []
    with rio.open(tif_file) as f:
        for x, y in chunk_iter:
            patch = f.read(window=rio.windows.Window(x, y, patch_size, patch_size))
            if patch.shape != (f.count, patch_size, patch_size) or (patch == f.nodata).any(): continue
            records.append({"x": x, "y": y, "label": patch[-1, patch_size//2, patch_size//2]})
        
        # creates dataframe cataloging valid patches
        df = pd.DataFrame.from_records(records)

        # efficiently adds lat / lon to dataframe
        if not df.empty:
            cols = df.loc[:,"x"].values + 0.5 + patch_size//2
            rows = df.loc[:,"y"].values + 0.5 + patch_size//2
            pts = np.dot(np.asarray(f.transform.column_vectors).T, np.vstack((cols, rows, np.ones_like(rows)))).T
            df["lon"] = pts[:,0]
            df["lat"] = pts[:,1]
        
    return df


def spatial_cross_val_split(
    ds: Dataset, 
    k: int = 5, 
    eval_set: int = 0, 
    split_col: str = "lat", 
    nbins: Union[int, None] = None,
    samples_per_bin: int = 3.0
):
    log.info(f"Splitting patches with spatial cross-val (2-3 min)")
    ds_df = ds.patch_df.copy()
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
    test_df = ds_df[ds_df["group"] == eval_set].drop(columns=[f"{split_col}_bin","group"]).reset_index(drop=True)
    test_ds = TiffDataset(
        tif_dir=ds.tif_dir, 
        patch_size=ds.patch_size, 
        patch_df=test_df,
        transformation=ds.tfm
    )
    ds_df = ds_df[ds_df["group"] != eval_set].drop(columns=[f"{split_col}_bin","group"]).reset_index(drop=True)
    ds = TiffDataset(
        tif_dir=ds.tif_dir, 
        patch_size=ds.patch_size, 
        patch_df=ds_df,
        transformation=ds.tfm
    )
    return ds, test_ds


if __name__ == "__main__":
    dataset = TiffDataset("/workspace/data/LAWLEY22-DATACUBE")
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1])
    tr_ds, te_ds = spatial_cross_val_split(dataset, k=6, nbins=36)
    print("Train")
    print(len(tr_ds) / len(dataset))
    print("Test")
    print(len(te_ds) /len(dataset))

