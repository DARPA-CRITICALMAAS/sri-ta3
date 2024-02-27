import numpy as np
import rasterio as rio


def write_tif(results, bounds, path):
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

    # filters valid idx - some error with multi GPU
    # valid_idx = np.logical_and(result_pts[:,0]>=0,result_pts[:,0]<width)
    # valid_idx = np.logical_and(valid_idx,result_pts[:,1]>=0)
    # valid_idx = np.logical_and(valid_idx,result_pts[:,1]<height)
    # data = data[valid_idx,:]
    # result_pts = result_pts[valid_idx,:]
    
    output_rasters = [
        "Likelihoods", 
        "Uncertainties",
        # sources
    #     "Sedimentary_Dictionary",
    #     "Geology_BlackShale_Proximity",
    #     "Geology_Fault_Proximity",
    #     "Terrane_Proximity",
    #     "Geology_Lithology_Majority_Igneous_Extrusive",
    #     "Geology_Lithology_Majority_Igneous_Intrusive_Felsic",
    #     "Geology_Lithology_Majority_Igneous_Intrusive_Mafic",
    #     "Geology_Lithology_Majority_Metamorphic_Gneiss",
    #     "Geology_Lithology_Majority_Metamorphic_Gneiss_Paragneiss",
    #     "Geology_Lithology_Majority_Metamorphic_Schist",
    #     "Geology_Lithology_Majority_Other_Unconsolidated",
    #     "Geology_Lithology_Majority_Sedimentary_Chemical",
    #     "Geology_Lithology_Majority_Sedimentary_Siliciclastic",
    #     "Geology_Lithology_Minority_Igneous_Extrusive",
    #     "Geology_Lithology_Minority_Igneous_Intrusive_Felsic",
    #     "Geology_Lithology_Minority_Igneous_Intrusive_Mafic",
    #     "Geology_Lithology_Minority_Metamorphic_Gneiss",
    #     "Geology_Lithology_Minority_Metamorphic_Gneiss_Paragneiss",
    #     "Geology_Lithology_Minority_Metamorphic_Schist",
    #     "Geology_Lithology_Minority_Other_Unconsolidated",
    #     "Geology_Lithology_Minority_Sedimentary_Chemical",
    #     "Geology_Lithology_Minority_Sedimentary_Siliciclastic",
    #     "Geology_Period_Minimum_Majority_Cambrian",
    #     "Geology_Period_Minimum_Majority_Cretaceous",
    #     "Geology_Period_Minimum_Majority_Devonian",
    #     "Geology_Period_Minimum_Majority_Jurassic",
    #     "Geology_Period_Minimum_Majority_Mesoproterozoic",
    #     "Geology_Period_Minimum_Majority_Mississippian",
    #     "Geology_Period_Minimum_Majority_Neoarchean",
    #     "Geology_Period_Minimum_Majority_Neogene",
    #     "Geology_Period_Minimum_Majority_Neoproterozoic",
    #     "Geology_Period_Minimum_Majority_Ordovician",
    #     "Geology_Period_Minimum_Majority_Paleogene",
    #     "Geology_Period_Minimum_Majority_Paleoproterozoic",
    #     "Geology_Period_Minimum_Majority_Pennsylvanian",
    #     "Geology_Period_Minimum_Majority_Permian",
    #     "Geology_Period_Minimum_Majority_Quaternary",
    #     "Geology_Period_Minimum_Majority_Silurian",
    #     "Geology_Period_Minimum_Majority_Triassic",
    #    "Gravity_Bouguer_HGM",
    #     "Geology_Period_Maximum_Majority_Cambrian",
    #     "Geology_Period_Maximum_Majority_Cretaceous",
    #     "Geology_Period_Maximum_Majority_Devonian",
    #     "Geology_Period_Maximum_Majority_Jurassic",
    #     "Geology_Period_Maximum_Majority_Mesoproterozoic",
    #     "Geology_Period_Maximum_Majority_Mississippian",
    #     "Geology_Period_Maximum_Majority_Neoarchean",
    #     "Geology_Period_Maximum_Majority_Neogene",
    #     "Geology_Period_Maximum_Majority_Neoproterozoic",
    #     "Geology_Period_Maximum_Majority_Ordovician",
    #     "Geology_Period_Maximum_Majority_Paleogene",
    #     "Geology_Period_Maximum_Majority_Paleoproterozoic",
    #     "Geology_Period_Maximum_Majority_Pennsylvanian",
    #     "Geology_Period_Maximum_Majority_Permian",
    #     "Geology_Period_Maximum_Majority_Quaternary",
    #     "Geology_Period_Maximum_Majority_Silurian",
    #     "Geology_Period_Maximum_Majority_Triassic",
    #     "Gravity_Bouguer",
    #     "Gravity_Bouguer_UpCont30km_HGM",
    #     "Geology_PassiveMargin_Proximity",
    #     "Magnetic_HGM",
    #     "Geology_Paleolatitude_Period_Minimum",
    #     "Magnetic_LongWavelength_HGM",
    #     "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity",
    #     "Gravity_Bouguer_HGM_Worms_Proximity",
    #     "Seismic_LAB_Priestley",
    #     "Seismic_Moho",
    #     "Igneous_Dictionary",
    #     "Magnetic_LongWavelength_HGM_Worms_Proximity",
    #     "Magnetic_HGM_Worms_Proximity",
    #     "Metamorphic_Dictionary",
    #     "Gravity_GOCE_ShapeIndex",
    #     "Radiometric Potassium",
    #     "Radiometric Thorium",
    #     "Radiometric Uranium",
    #     "Isostatic Gravity HGM",
    #     "Isostatic Gravity",
    ]
    output_rasters = output_rasters + [f"attr{i}" for i in range(results[0,4:].shape[0])]

    for idx, tif_layer in enumerate(output_rasters):
        # forms tif ndarray
        tif_data = np.empty(shape=(height, width))
        tif_data[:] = np.nan
        tif_data[result_pts[:,1], result_pts[:,0]] = data[:,idx].astype(float)
        
        # writes the output tif
        tif_file = f"{path}/{tif_layer}_{str_bounds(bounds)}.tif"
        with rio.open(tif_file, "w", **tiff_meta) as out:
            out.write_band(1, tif_data)

    # outputs a feature importance map, with the most important feature per pixel
    # tif_data = np.empty(shape=(height, width))
    # tif_data[:] = np.nan
    # data_overall = np.argmax(data[:,2:].astype(float), axis=1)
    # data_overall[(data_overall>=6) & (data_overall<=14)] = 20 # Geology_Lithology_Majority
    # data_overall[(data_overall>=15) & (data_overall<=24)] = 21 # Geology_Lithology_Minority
    # data_overall[(data_overall>=25) & (data_overall<=41)] = 22 # Geology_Period_Maximum_Majority
    # data_overall[(data_overall>=43) & (data_overall<=59)] = 23 # Geology_Period_Minimum_Majority

    # tif_data[result_pts[:,1], result_pts[:,0]] = data_overall.astype(float)
    # tif_file = f"{path}/overall_{str_bounds(bounds)}.tif"
    # with rio.open(tif_file, "w", **tiff_meta) as out:
    #     out.write_band(1, tif_data)


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