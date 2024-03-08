from typing import Union, List, Callable
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
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

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
        valid_patches: Union[np.ndarray, None] = None,
        patch_size: int = 33,
        stage: Union[np.ndarray, None] = None,
        uscan_only: bool = False,
    ):
        self.uscan_only = uscan_only
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
            if self.uscan_only:
                log.info("Only using US and Canada datapoints.")
                tif_files = [item for item in tif_files if 'north-america' in item]
            else:
                log.info("Using all available datapoints: US, Canada, and Australia.")
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
            lon = self.valid_patches[idx,-3]
            lat = self.valid_patches[idx,-2]
            return patch, label, lon, lat # produce map
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
            if patch.shape != (f.count, patch_size, patch_size) or np.isnan(patch).any(): continue
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
        valid_patches=test_valid_patches,
        uscan_only=ds.uscan_only,
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
        valid_patches=ds_valid_patches,
        uscan_only=ds.uscan_only,
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
        patch_size=ds1.patch_size,
        stage=ds1.stage,
        valid_patches=ds_df.values,
        uscan_only=ds1.uscan_only,
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
        patch_size=ds.patch_size,
        stage=ds.stage,
        valid_patches=ds_df_train.values,
        uscan_only=ds.uscan_only,
    )

    ds_valid = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        patch_size=ds.patch_size,
        stage=ds.stage,
        valid_patches=ds_df_valid.values,
        uscan_only=ds.uscan_only,
    )

    ds_test = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        patch_size=ds.patch_size,
        stage=ds.stage,
        valid_patches=ds_df_test.values,
        uscan_only=ds.uscan_only,
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
        patch_size=ds.patch_size,
        stage=ds.stage,
        valid_patches=ds_df_p.values,
        uscan_only=ds.uscan_only,
    )
    ds_df_n = ds_df[ds_df["label"] == 0]
    ds_n = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        patch_size=ds.patch_size,
        stage=ds.stage,
        valid_patches=ds_df_n.values,
        uscan_only=ds.uscan_only,
    )

    ds_p_train, ds_p_valid, ds_p_test = random_split(ds_p, train_split, seed=seed)
    ds_n_train, ds_n_valid, ds_n_test = random_split(ds_n, train_split, seed=seed)

    ds_train = combine(ds_p_train, ds_n_train)
    ds_valid = combine(ds_p_valid, ds_n_valid)
    ds_test = combine(ds_p_test, ds_n_test)

    return ds_train, ds_valid, ds_test


def pu_downsample(
    ds: Dataset,
    feature_extractor: Callable,
    multiplier: int = 20,
    percent_likely_neg: float = 0.25,
    seed: int = 0,
):
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
        patch_size=ds.patch_size,
        stage=ds.stage,
        valid_patches=ds_df_p.values,
        uscan_only=ds.uscan_only,
    )
    p_feats = []
    for p_idx in range(len(ds_p)):
        p_patch, _ = ds_p[p_idx]
        p_feats.append(feature_extractor(p_patch))
    # prepares unlabeled dataset
    ds_df_u = ds_df[ds_df["label"] == 0]
    ds_df_u = ds_df_u.reset_index(drop=True)
    ds_u = TiffDataset(
        tif_files=ds.tif_files,
        tif_data=ds.tif_data,
        patch_size=ds.patch_size,
        stage=ds.stage,
        valid_patches=ds_df_u.values,
        uscan_only=ds.uscan_only,
    )
    u_feats = []
    for u_idx in range(len(ds_u)):
        u_patch, _ = ds_u[u_idx]
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
    # rank unlabeled by negativity likelihood, taking % most negative
    u_dist_sort_idx = np.argsort(u_dist)
    
    likely_negatives_idx = u_dist_sort_idx[-int(len(u_dist_sort_idx)*percent_likely_neg):]
    ds_df_n = ds_df_u.iloc[likely_negatives_idx]
    # randomly downsample "likely" negatives
    num_negatives = int (ds_df_p.shape[0]) * multiplier
    ds_df_n = ds_df_n.sample(n=num_negatives, replace=False, random_state=seed)
    # combine positives / negatives 
    ds_df = pd.concat([ds_df_n, ds_df_p], axis=0).reset_index(drop=True)
    ds = TiffDataset(
        tif_files=ds.tif_files, 
        tif_data=ds.tif_data,
        patch_size=ds.patch_size, 
        stage=ds.stage,
        valid_patches=ds_df.values,
        uscan_only=ds.uscan_only,
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
        columns=["x","y","label","lon", "lat","source"]
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
        patch_size=ds.patch_size, 
        stage=ds.stage,
        valid_patches=ds_df.values,
        uscan_only=ds.uscan_only,
    )

    return ds


def filter_by_bounds(ds, bounds):
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,3] > bounds[0]] # left
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,4] > bounds[1]] # bottom
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,3] < bounds[2]] # right
    ds.valid_patches = ds.valid_patches[ds.valid_patches[:,4] < bounds[3]] # top
    return ds


def store_samples(ds, root_path, name):
    log_str = f"Spatial cross val ouput: train pos - {ds.valid_patches[:,2].sum()}, train neg - {len(ds)-ds.valid_patches[:,2].sum()}."
    log.info(log_str)
    file_path = f"{root_path}/{name}.csv"
    ds_df = pd.DataFrame(
        data=ds.valid_patches,
        index=np.arange(ds.valid_patches.shape[0]),
        columns=["x","y","label","lon", "lat","source"]
    )
    ds_df.to_csv(file_path, index=False)

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

