import numpy as np
import rasterio as rio
from copy import copy
import pandas as pd


def write_tif(results, path, attributions_flag, datamodule):
    
    # defines the tif meta data
    tif_meta = copy(datamodule.data_predict.tif_meta)
    tif_meta.update({
        "count": 1,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    })

    # extracts raster pts and data
    result_pts = np.dot(
        np.asarray((~tif_meta["transform"]).column_vectors).T, 
        np.vstack((results[:,0], results[:,1], np.ones_like(results[:,1])))
    ).astype(int).T
    data = results[:,2:]
    
    output_rasters = [
        "Likelihoods", 
        "Uncertainties",
    ]

    if attributions_flag:
        attrs = [None] * len(datamodule.data_predict.tif_tags)
        for tag, idx in datamodule.data_predict.tif_tags.items(): attrs[int(idx)] = tag
        output_rasters = output_rasters + attrs[:-1] # last tag is label - doesn't exist

    tif_files = []
    for idx, tif_layer in enumerate(output_rasters):
        # forms tif ndarray
        tif_data = np.empty(shape=(tif_meta["height"], tif_meta["width"]))
        tif_data[:] = np.nan
        tif_data[result_pts[:,1], result_pts[:,0]] = data[:,idx].astype(float)
        
        # writes the output tif
        tif_file = f"{path}/{tif_layer}.tif"
        with rio.open(tif_file, "w", **tif_meta) as out:
            out.write_band(1, tif_data)
        tif_files.append(tif_file)
    return tif_files


def write_embeddings(results, path, datamodule):
    
    # defines the tif meta data
    tif_meta = copy(datamodule.data_predict.tif_meta)
    tif_meta.update({
        "count": 1,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    })

    # extracts raster pts and data
    result_pts = np.dot(
        np.asarray((~tif_meta["transform"]).column_vectors).T, 
        np.vstack((results[:,0], results[:,1], np.ones_like(results[:,1])))
    ).astype(int).T
    data = results[:,2:]

    # forms the embedding ndarray
    embeddings = np.empty(shape=(tif_meta["height"], tif_meta["width"], data.shape[-1]))
    embeddings[:] = np.nan
    embeddings[result_pts[:,1], result_pts[:,0],:] = data

    # stores the embeddings
    emb_file = f"{path}/embeddings_d{data.shape[-1]}.npy"
    np.save(emb_file, embeddings)

    return emb_file


def collect_gpu_results(predictions_part, trainer):
    # writes the prediction parts to disk
    cols = ["lon","lat","mean","std"] + [f"attr{n}" for n in range(predictions_part.shape[-1]-4)]
    res_df = pd.DataFrame(data=predictions_part, columns=cols)
    res_df.to_csv(f"gpu_{trainer.strategy.global_rank}_result.csv",index=False)
    trainer.strategy.barrier()

    # reads and combines all the prediction parts
    res_df = []
    for n in range(trainer.strategy.world_size):
        res_df.append(pd.read_csv(f"gpu_{n}_result.csv", index_col=False))
    
    return pd.concat(res_df, ignore_index=True).values