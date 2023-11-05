from typing import Union, List
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
from torch import tensor, half

from src import utils

log = utils.get_pylogger(__name__)

# we need some way of splitting the TiFFs into train, valid, and test splits
# that are spatially independent BUT have similar distributions

import pdb

class TiffDataset(Dataset):
    def __init__(self,
        tif_dir: Union[str, None] = None,
        tif_files: Union[List[str], None] = None,
        tif_data: Union[np.ndarray, None] = None,
        valid_patches: Union[np.ndarray, None] = None,
        patch_size: int = 33,
        stage: Union[np.ndarray, None] = None,
    ):
        # loads tif files in MP compatible format
        self.tif_files, self.tif_data = self._load_tif_files(tif_dir, tif_files, tif_data)

        # loads VALID patches within all tiffs of dataset
        self.valid_patches = self._load_valid_patches(self.tif_files, patch_size) if valid_patches is None else valid_patches

        # sets remaining object variables
        self.patch_size = patch_size
        self.stage = stage

    def _load_tif_files(self, tif_dir, tif_files, tif_data):
        # sets List[str] of tif files
        assert (tif_files is None and tif_dir is not None) or (tif_files is not None and tif_dir is None), "tif_dir and tif_files BOTH set."
        if tif_files is None:
            tif_files = glob(str(Path(tif_dir) / Path("*.tif")))
            tif_data = []
            for tif_file in tif_files:
                log.info(f"Loading tif data for for {tif_file}")
                with rio_env(GDAL_CACHEMAX=0):
                    with rio_open(tif_file, driver='GTiff') as tif:
                        tif_data.append(tensor(tif.read().astype("half"), dtype=half))
        return tif_files, tif_data

    def _load_valid_patches(self, tif_files, patch_size):
        # loads or generates df indicating which tif patches are VALID
        ds_valid_patches = []
        for tif_idx, tif_file in  enumerate(tif_files):
            valid_patch_file = f"{str(tif_file).split('.')[0]}_valid_p{patch_size}.npy"
            try:
                # check if valid patch dataframe already exists
                log.info(f"Loading np.ndarray enumerating valid patches for {tif_file} (~5 min)")
                valid_patches = np.load(valid_patch_file)
            except FileNotFoundError:
                # if not, generate valid patch dataframe
                log.warning(f"np.ndarray not found. Generating.")
                valid_patches = self._generate_valid_patches(tif_file, patch_size)
                np.save(valid_patch_file, valid_patches)

            valid_patches = np.hstack([valid_patches, tif_idx*np.ones(shape=(valid_patches.shape[0],1))])
            ds_valid_patches.append(valid_patches)
        # returns valid patches of ALL tiffs in dataset
        return np.vstack(ds_valid_patches)
    
    @staticmethod
    def _generate_valid_patches(tif_file, patch_size):
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
        validate_patches_multi = partial(validate_patches, patch_size=patch_size, tif_file=tif_file)
        valid_patches = np.vstack(pool.map(validate_patches_multi, chunks))

        # closes the pool to free up resources
        pool.close()
        pool.join()

        return valid_patches

    def __len__(self):
        return self.valid_patches.shape[0]
    
    def __getitem__(self, idx):
        # loads the patch's location and label
        col = int(self.valid_patches[idx,0])
        row = int(self.valid_patches[idx,1])
        label = self.valid_patches[idx,2]
        source_tif = int(self.valid_patches[idx,-1])
        
        # loads the patch's data
        patch = self.tif_data[source_tif][:-1,row:row+self.patch_size,col:col+self.patch_size]
        
        if self.stage == "predict":
            return patch, label, col, row # produce map
        else:
            return patch, label # train/val/test


def validate_patches(chunk, patch_size, tif_file):
    # creates MP friendly iterator that estimates run-time
    if Process()._identity[0] == 1:
        chunk_iter = tqdm(zip(chunk[0],chunk[1]), total=len(chunk[0]))
    else:
        chunk_iter = zip(chunk[0],chunk[1])
    
    # validates the patches made from pixel locations in chunks
    records = []
    with rio_open(tif_file) as f:
        for x, y in chunk_iter:
            patch = f.read(window=Window(x, y, patch_size, patch_size))
            if patch.shape != (f.count, patch_size, patch_size) or (patch == f.nodata).any(): continue
            records.append([x, y, patch[-1, patch_size//2, patch_size//2]])
        tif_tfm = f.transform
    
    # creates dataframe cataloging valid patches
    records = np.asarray(records)

    # efficiently adds lat / lon to dataframe
    if records.shape[0]:
        cols = records[:,0] + 0.5 + patch_size//2
        rows = records[:,1] + 0.5 + patch_size//2
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
        patch_size=ds.patch_size, 
        stage=ds.stage,
        valid_patches=test_valid_patches
    )
    val_valid_patches = ds_df[ds_df["group"] == val_set].drop(columns=[f"{split_col}_bin","group"]).reset_index(drop=True).values
    val_ds = TiffDataset(
        tif_files=ds.tif_files, 
        tif_data=ds.tif_data,
        patch_size=ds.patch_size, 
        stage=ds.stage,
        valid_patches=val_valid_patches
    )
    ds_valid_patches = ds_df[(ds_df["group"] != test_set) & (ds_df["group"] != val_set)].drop(columns=[f"{split_col}_bin","group"]).reset_index(drop=True).values
    ds = TiffDataset(
        tif_files=ds.tif_files, 
        tif_data=ds.tif_data,
        patch_size=ds.patch_size, 
        stage=ds.stage,
        valid_patches=ds_valid_patches
    )
    return ds, val_ds, test_ds


def filter_by_bounds(ds, bounds):
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,3] > bounds[0]] # left
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,4] > bounds[1]] # bottom
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,3] < bounds[2]] # right
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,4] < bounds[3]] # top
    return ds


if __name__ == "__main__":
    dataset = TiffDataset("/workspace/data/SRI-DATACUBE")
    pdb.set_trace()
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1])
    tr_ds, te_ds = spatial_cross_val_split(dataset, k=6, nbins=36)
    print("Train")
    print(len(tr_ds) / len(dataset))
    print("Test")
    print(len(te_ds) /len(dataset))

