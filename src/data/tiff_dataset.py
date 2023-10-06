from typing import Union, Tuple
from glob import glob
from pathlib import Path
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

import rasterio as rio
import rasterio.windows
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision as tv

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)

# we need some way of splitting the TiFFs into train, valid, and test splits
# that are spatially independent BUT have similar distributions

import pdb

class TiffDataset(Dataset):
    def __init__(self, 
        tif_dir: str = "/workspace/data/",
        patch_size: int = 17,
        # splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        # transformation: Union[tv.transforms.Compose, None] = None,
        # shuffle: bool = True,
        # seed: int = 3,
    ):
        # load the raw tif files
        tif_files = glob(str(Path(tif_dir) / Path("*.tif")))
        self.tifs = {tif_file: rio.open(tif_file) for tif_file in tif_files}
        self.patch_size = patch_size

        # load a valid patch df for each tif
        self.patch_df = self._load_valid_patch_dfs()

    def _load_valid_patch_dfs(self):
        # loads or generates df indicating which tif patches are VALID
        patch_dfs = []
        for tif_file, tif in self.tifs.items():
            patch_df_file = f"{str(tif_file).split('.')[0]}_valid_p{self.patch_size}_df.csv"
            try:
                # check if valid patch dataframe already exists
                patch_df = pd.read_csv(patch_df_file)
                log.info(f"Loaded dataframe enumerating valid patches")
            except FileNotFoundError:
                # if not, generate valid patch dataframe
                patch_df = self._generate_patch_df(tif_file, tif, self.patch_size)
                patch_df["tif_file"] = tif_file
                patch_df.to_csv(patch_df_file, index=False)
            patch_dfs.append(patch_df)
        return pd.concat(patch_dfs, ignore_index=True)
    
    @staticmethod
    def _generate_patch_df(tif_file, tif, patch_size):
        # extracts the lat / lons of raster
        rows, cols = np.mgrid[0:tif.height:1,0:tif.width:1].reshape((-1, (tif.width)*(tif.height)))

        # sets up multiprocessing pool
        log.info(f"Using {mp.cpu_count()} threads to enumerate and store valid patches")
        pool = mp.Pool(mp.cpu_count())

        # splits data into mp.cpu_count() chunks
        chunk_size = len(rows) // mp.cpu_count()
        chunks = [(cols[i:i+chunk_size],rows[i:i+chunk_size]) for i in range(0, len(rows), chunk_size)]

        # enumerate all valid patches with multiprocessing
        validate_patches_multi = partial(validate_patches, patch_size=patch_size, tif_file=tif_file)
        df = pd.concat(pool.map(validate_patches_multi, chunks))

        # close the pool to free up resources
        pool.close()
        pool.join()

        return df

    def __len__(self):
        return len(self.patch_df)
    
    def __getitem__(self, idx):
        col, row, label, source_tif = self.patch_df.loc[idx, ["x", "y", "label", "tif_file"]]
        patch = self.tifs[source_tif].read(window=rio.windows.Window(col, row, self.patch_size, self.patch_size))[:-1,:,:]
        # need to add transforms where appropriate
        return patch, label


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
            records.append({"x": x, "y": y, "label": patch[-1, patch.shape[1]//2, patch.shape[2]//2]})

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    dataset = TiffDataset("/workspace/data/LAWLEY22-DATACUBE")
    print(dataset[0][0].shape)
    print(dataset[0][1])
    pdb.set_trace()