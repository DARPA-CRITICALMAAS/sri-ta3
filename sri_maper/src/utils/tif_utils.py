import numpy as np
import rasterio as rio


def write_tif(results, bounds, path, attributions_flag, dataset):
    # defines size of output rasters
    resolution = (2240.0,2240.0)
    height = int((bounds[3]-bounds[1]) / resolution[0])
    width = int((bounds[2]-bounds[0]) / resolution[1])

    # defines raster transform
    tif_tf = rio.transform.from_bounds(
        *bounds,
        width,
        height
    )

    # defines the tif meta data
    tiff_meta = {
        "driver": 'GTiff',
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        # "crs": rio.CRS.from_epsg(4326),
        "crs": rio.CRS.from_wkt('PROJCS["North_America_Albers_Equal_Area_Conic",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",40],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",20],PARAMETER["standard_parallel_2",60],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'),
        "transform": tif_tf,
        "nodata": np.nan,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    }

    # extracts raster pts and data
    result_pts = np.dot(
        np.asarray((~tif_tf).column_vectors).T, 
        np.vstack((results[:,0], results[:,1], np.ones_like(results[:,1])))
    ).astype(int).T
    data = results[:,2:]
    
    output_rasters = [
        "Likelihoods", 
        "Uncertainties",
    ]

    if attributions_flag:
        attrs = [None] * len(dataset.data_predict.tif_tags)
        for tag, idx in dataset.data_predict.tif_tags.items(): attrs[int(idx)] = tag
        output_rasters = output_rasters + attrs[:-1] # last tag is label - doesn't exist

    for idx, tif_layer in enumerate(output_rasters):
        # forms tif ndarray
        tif_data = np.empty(shape=(height, width))
        tif_data[:] = np.nan
        tif_data[result_pts[:,1], result_pts[:,0]] = data[:,idx].astype(float)
        
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


def produce_label_geotiff(datamod, bounds, path):
    # accumulates points for positives in train, test, and val
    datamod.data_train.stage = "predict"
    datamod.data_val.stage = "predict"
    datamod.data_test.stage = "predict"
    samples = []
    for i in range(len(datamod.data_train)):
        sample = datamod.data_train[i]
        if sample[1]: samples.append([0,sample[2],sample[3]])
    for i in range(len(datamod.data_val)):
        sample = datamod.data_val[i]
        if sample[1]: samples.append([1,sample[2],sample[3]])
    for i in range(len(datamod.data_test)):
        sample = datamod.data_test[i]
        if sample[1]: samples.append([2,sample[2],sample[3]])
    samples = np.asarray(samples)
    
    # defines size of output rasters
    resolution = (0.008,0.008)
    height = int((bounds[3]-bounds[1]) / resolution[0])
    width = int((bounds[2]-bounds[0]) / resolution[1])

    # defines raster transform
    tif_tf = rio.transform.from_bounds(
        *bounds,
        width,
        height
    )

    # defines the tif meta data
    tiff_meta = {
        "driver": 'GTiff',
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": rio.CRS.from_epsg(4326),
        "transform": tif_tf,
        "nodata": np.nan,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    }

    # extracts raster pts and data
    result_pts = np.dot(
        np.asarray((~tif_tf).column_vectors).T, 
        np.vstack((samples[:,1], samples[:,2], np.ones_like(samples[:,1])))
    ).astype(int).T
    data = samples[:,0]

    # forms tif ndarray
    tif_data = np.empty(shape=(height, width))
    tif_data[:] = np.nan
    tif_data[result_pts[:,1], result_pts[:,0]] = data.astype(float)
    
    # writes the output tif
    tif_file = f"{path}/split_geotiff_{str_bounds(bounds)}.tif"
    with rio.open(tif_file, "w", **tiff_meta) as out:
        out.write_band(1, tif_data)