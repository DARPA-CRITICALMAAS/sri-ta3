import numpy as np
import rasterio as rio

import pdb

def write_tif(results, bounds, patch_size, path):
    # normalizes pixel points
    results[:,0] = results[:,0] - results[:,0].min()
    results[:,1] = results[:,1] - results[:,1].min()
    # accounts for classifications being CENTER of patches
    results = results[results[:,0]-patch_size>=0,:]
    results = results[results[:,1]-patch_size>=0,:]
    # generates numpy data tifs
    data = results[:,2:]
    resolution = (0.01,0.01)
    height = int((bounds[3]-bounds[1]) / resolution[0])+2
    width = int((bounds[2]-bounds[0]) / resolution[1])+2
    for idx, tif_layer in enumerate(["means","stds"]):
        tif_data = -1*np.ones(shape=(height, width))
        tif_data[results[:,1].astype(int), results[:,0].astype(int)] = data[:,idx].astype(float)
        # defines the tif transform
        tif_tf = rio.transform.from_bounds(
            *bounds,
            tif_data.shape[1],
            tif_data.shape[0]
        )
        # defines the tif meta data
        tiff_meta = {
            "driver": 'GTiff',
            "height": tif_data.shape[0],
            "width": tif_data.shape[1],
            "count": 1,
            "dtype": "float64",
            "crs": rio.CRS.from_epsg(4326),
            "transform": tif_tf,
            "nodata": -1.0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
        }
        # writes the output tif
        tif_file = f"{path}/{tif_layer}_{str_bounds(bounds)}.tif"
        with rio.open(tif_file, "w", **tiff_meta) as out:
            out.write_band(1, tif_data)


def str_bounds(bounds):
    # makes linux compatible str of lon/lat bounds
    bounds_str = ""
    bounds_str += f"L{str(bounds[0]).replace('.','p')}"
    bounds_str += f"B{str(bounds[1]).replace('.','p')}"
    bounds_str += f"R{str(bounds[2]).replace('.','p')}"
    bounds_str += f"T{str(bounds[3]).replace('.','p')}"
    return bounds_str