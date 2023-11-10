import numpy as np
import rasterio as rio

import pdb

def write_tif(results, bounds, path):
    # defines size of output rasters
    resolution = (0.01,0.01)
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
        np.vstack((results[:,0], results[:,1], np.ones_like(results[:,1])))
    ).astype(int).T
    data = results[:,2:]
    
    output_rasters = [
        "Likelihoods", 
        "Uncertainties", 
        "Sedimentary_Dictionary",
        "Igneous_Dictionary",
        "Metamorphic_Dictionary",
        "Seismic_LAB_Priestley",
        "Seismic_Moho",
        "Gravity_GOCE_ShapeIndex",
        "Geology_Paleolatitude_Period_Minimum",
        "Terrane_Proximity",
        "Geology_PassiveMargin_Proximity",
        "Geology_BlackShale_Proximity",
        "Geology_Fault_Proximity",
        "Gravity_Bouguer",
        "Gravity_Bouguer_HGM",
        "Gravity_Bouguer_UpCont30km_HGM",
        "Gravity_Bouguer_HGM_Worms_Proximity",
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity",
        "Magnetic_HGM",
        "Magnetic_LongWavelength_HGM",
        "Magnetic_HGM_Worms_Proximity",
        "Magnetic_LongWavelength_HGM_Worms_Proximity",
        "Geology_Lithology_Majority_Igneous_Extrusive",
        "Geology_Lithology_Majority_Igneous_Intrusive_Felsic",
        "Geology_Lithology_Majority_Igneous_Intrusive_Mafic",
        "Geology_Lithology_Majority_Metamorphic_Gneiss",
        "Geology_Lithology_Majority_Metamorphic_Gneiss_Paragneiss",
        "Geology_Lithology_Majority_Metamorphic_Schist",
        "Geology_Lithology_Majority_Other_Unconsolidated",
        "Geology_Lithology_Majority_Sedimentary_Chemical",
        "Geology_Lithology_Majority_Sedimentary_Siliciclastic",
        "Geology_Lithology_Minority_Igneous_Extrusive",
        "Geology_Lithology_Minority_Igneous_Intrusive_Felsic",
        "Geology_Lithology_Minority_Igneous_Intrusive_Mafic",
        "Geology_Lithology_Minority_Metamorphic_Gneiss",
        "Geology_Lithology_Minority_Metamorphic_Gneiss_Paragneiss",
        "Geology_Lithology_Minority_Metamorphic_Schist",
        "Geology_Lithology_Minority_Other_Unconsolidated",
        "Geology_Lithology_Minority_Sedimentary_Chemical",
        "Geology_Lithology_Minority_Sedimentary_Siliciclastic",
        "Geology_Lithology_Minority_UNK",
        "Geology_Period_Maximum_Majority_Cambrian",
        "Geology_Period_Maximum_Majority_Cretaceous",
        "Geology_Period_Maximum_Majority_Devonian",
        "Geology_Period_Maximum_Majority_Jurassic",
        "Geology_Period_Maximum_Majority_Mesoproterozoic",
        "Geology_Period_Maximum_Majority_Mississippian",
        "Geology_Period_Maximum_Majority_Neoarchean",
        "Geology_Period_Maximum_Majority_Neogene",
        "Geology_Period_Maximum_Majority_Neoproterozoic",
        "Geology_Period_Maximum_Majority_Ordovician",
        "Geology_Period_Maximum_Majority_Paleogene",
        "Geology_Period_Maximum_Majority_Paleoproterozoic",
        "Geology_Period_Maximum_Majority_Pennsylvanian",
        "Geology_Period_Maximum_Majority_Permian",
        "Geology_Period_Maximum_Majority_Quaternary",
        "Geology_Period_Maximum_Majority_Silurian",
        "Geology_Period_Maximum_Majority_Triassic",
        "Geology_Period_Minimum_Majority_Cambrian",
        "Geology_Period_Minimum_Majority_Cretaceous",
        "Geology_Period_Minimum_Majority_Devonian",
        "Geology_Period_Minimum_Majority_Jurassic",
        "Geology_Period_Minimum_Majority_Mesoproterozoic",
        "Geology_Period_Minimum_Majority_Mississippian",
        "Geology_Period_Minimum_Majority_Neoarchean",
        "Geology_Period_Minimum_Majority_Neogene",
        "Geology_Period_Minimum_Majority_Neoproterozoic",
        "Geology_Period_Minimum_Majority_Ordovician",
        "Geology_Period_Minimum_Majority_Paleogene",
        "Geology_Period_Minimum_Majority_Paleoproterozoic",
        "Geology_Period_Minimum_Majority_Pennsylvanian",
        "Geology_Period_Minimum_Majority_Permian",
        "Geology_Period_Minimum_Majority_Quaternary",
        "Geology_Period_Minimum_Majority_Silurian",
        "Geology_Period_Minimum_Majority_Triassic",
    ]

    for idx, tif_layer in enumerate(output_rasters):
        # forms tif ndarray
        tif_data = np.empty(shape=(height, width))
        tif_data[:] = np.nan
        tif_data[result_pts[:,1], result_pts[:,0]] = data[:,idx].astype(float)
        print(tif_data.max())
        
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