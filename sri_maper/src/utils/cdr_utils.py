import argparse
import asyncio
import os
import shutil
from pathlib import Path
import requests
import zipfile
import geopandas as gpd
from tqdm import tqdm
import httpx
import rasterio as rio
from rasterio.mask import mask

from cdr_schemas.cdr_responses.prospectivity import ProspectModelMetaData
from cdr_schemas.prospectivity_input import (ProspectivityOutputLayer, SaveProcessedDataLayer)

from pydantic import BaseModel, Field


class CDR_Settings(BaseModel):
    system_name: str
    system_version: str
    ml_model_name: str
    ml_model_version: str
    user_api_token: str
    cdr_host: str


def get_event_payload_result(id: str, app_settings: CDR_Settings):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)
    resp = client.get(f"{app_settings.cdr_host}/v1/prospectivity/model_run?model_run_id={id}",
                      headers=headers)
    return resp.json()

def parse_event_payload_result(resp_json: dict, model_type_filter="sri_NN"):
    if resp_json.get("model_type") != model_type_filter:
        raise Exception(f"The model_type '{resp_json.get('model_type')}' is not supported.")
    
    resp_json = resp_json.get("event")

    if resp_json.get("event") != "prospectivity_model_run.process":
        raise Exception("Event is not found or is not a model run event")

    model_payload= resp_json.get("payload")
    
    prospect_model_metadata = ProspectModelMetaData(
        model_run_id = model_payload.get("model_run_id"),
        cma = model_payload.get("cma"),
        model_type = model_payload.get("model_type"),
        train_config = model_payload.get("train_config"),
        evidence_layers = model_payload.get("evidence_layers"),
    )

    return prospect_model_metadata


def download_layer(title: str, url: str, dst_dir: Path):
    local_file = f"{title}{Path(url).suffix}"
    response = requests.get(url)
    response.raise_for_status()
    dst_path = dst_dir / local_file
    with open(dst_path, 'wb') as f:
        f.write(response.content)
    return dst_path


def download_evidence_layers(
    event_obj: ProspectModelMetaData, 
    data_path: Path = Path("./data")
):
    # sets evidence layers location
    ev_lyrs_path = data_path / Path(event_obj.model_run_id) / Path("evidence_layers")
    ev_lyrs_path.mkdir(parents=True, exist_ok=True)

    # downloads evidence layers
    ev_lyrs_paths = []
    for ev_lyr in tqdm(event_obj.evidence_layers):
        ev_lyr_path = download_layer(
            title=ev_lyr.data_source.evidence_layer_raster_prefix, 
            url=ev_lyr.data_source.download_url, 
            dst_dir=ev_lyrs_path
        )
        if ev_lyr_path.suffix == '.zip':
            os.makedirs(ev_lyr_path.parent / ev_lyr_path.stem, exist_ok=True)
            with zipfile.ZipFile(ev_lyr_path, 'r') as zip_ref:
                zip_ref.extractall(ev_lyr_path.parent / ev_lyr_path.stem)
        ev_lyrs_paths.append(ev_lyr_path)
    
    return ev_lyrs_paths

def create_aoi_geopkg(
    event_obj: ProspectModelMetaData, 
    data_path: Path = Path("./data")
):
    # sets geopackage location
    geopkg_path = data_path / Path(event_obj.model_run_id) 
    geopkg_path.mkdir(parents=True, exist_ok=True)
    geopkg_path = geopkg_path / Path(f"aoi.gpkg")

    # Creating the AOI geopackage
    gdf = gpd.GeoDataFrame(
        {'id': [0]},
        crs = event_obj.cma.crs,
        geometry = [event_obj.cma.extent]
    )
    gdf.to_file(geopkg_path, driver="GPKG")

    return geopkg_path

def read_tiff(file_path):
    with rio.open(file_path) as src:
        return src.read(1), src


def clip_tiff(tiff_path, mask_tiff_path, output_path):
    # placeholder to clip tiff
    shutil.copy(mask_tiff_path, output_path)


def prepare_data_sources(payload):
    print("Downloading template cma file")
    updated_url=  payload.cma.download_url
    
    # to remove. local testing
    if "minio.cdr.geo" in payload.cma.download_url:
        updated_url= "http://0.0.0.0:9000/" + payload.cma.download_url.split(":9000/")[-1]
    
    if not os.path.exists(f"datasources/{payload.cma.download_url.split('/')[-1]}"):
        r = httpx.get(updated_url, timeout=5000)
        with open(f"datasources/{payload.cma.download_url.split('/')[-1]}", "wb") as f:
            f.write(r.content)

    print('preparing data sources')
    print("loop over evidence layers specified by the UI")
    for layer in payload.evidence_layers:
        print(f"downloading datasource layer from cdr: {layer} ")
        if not os.path.exists(f"datasources/{layer.data_source.download_url.split('/')[-1]}"):
            r = httpx.get(updated_url, timeout=5000)
            with open(f"datasources/{layer.data_source.download_url.split('/')[-1]}", "wb") as f:
                f.write(r.content)
    
    print("CMA template and layers are downloaded. Now clip them to template extent")
    for layer in payload.evidence_layers:
        clip_tiff(
            tiff_path = f"datasources/{layer.data_source.download_url.split('/')[-1]}", 
            mask_tiff_path = f"datasources/{payload.cma.download_url.split('/')[-1]}", 
            output_path = f"datasources/clipped_{layer.data_source.download_url.split('/')[-1]}"
        )

    return


def train_model(payload):
    print("Train model on new process stack ...")
    print("model is trained")  
    return


def run_model(payload):
    print('run model to generate output')
    shutil.copy(f"datasources/{payload.cma.download_url.split('/')[-1]}", "outputs/model_output_uncertainty.tif")
    shutil.copy(f"datasources/{payload.cma.download_url.split('/')[-1]}", "outputs/model_output_likelihood.tif")
    print("model runs have finished")
    return


def send_outputs(payload, app_settings):
    print("Sending Output layers to CDR...")

    #  send output layers from model run
    #  create output layer metadata
    results = ProspectivityOutputLayer(**{
        "system": app_settings.system_name,
        "system_version": app_settings.system_version,
        "model": app_settings.ml_model_name,
        "model_version": app_settings.ml_model_version,
        "model_run_id": payload.model_run_id,
        "cma_id": payload.cma.cma_id,
        "output_type": "uncertainty",
        "title":"model_ouput_uncertainty.tif"
    })
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    files_ = {"input_file": (
        "model_output_uncertainty.tif", 
        open("./outputs/model_output_uncertainty.tif", "rb"), "application/octet-stream")
        }

    resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/propectivity_output_layer",
                       data={
                        "metadata": results.model_dump_json(exclude_none=True)
                        },
                        files=files_,
                        headers=headers)
    if resp.status_code != 200 or resp.status_code != 204:
        print("An Error Occurred sending uncertainty layer")
        print(resp.text)
    else:
        print("Finished sending uncertainty!")
    
    #  additional output layer's metadata
    result_2 = ProspectivityOutputLayer(**{
        "system": app_settings.system_name,
        "system_version": app_settings.system_version,
        "model": app_settings.ml_model_name,
        "model_version": app_settings.ml_model_version,
        "model_run_id": payload.model_run_id,
        "cma_id": payload.cma.cma_id,
        "output_type": "likelihood",
        "title":"model_ouput_likelihood.tif"
    })
    files_ = {"input_file": (
        "model_output_likelihood.tif",
        open("./outputs/model_output_likelihood.tif", "rb"),
        "application/octet-stream")
        }

    resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/propectivity_output_layer",
            data={
                "metadata": result_2.model_dump_json(exclude_none=True)
                },
            files=files_,
            headers=headers)
    if resp.status_code != 200 or resp.status_code != 204:
        print("An Error Occurred sending likelihood layer")
        print(resp.text)
    else:
        
        print("Finished sending likelihood!")
    return

def send_stack(payload, app_settings):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    print("Now sending processed data layers")
    for layer in payload.evidence_layers:
        data_layer = SaveProcessedDataLayer(**{
            "system": app_settings.system_name,
            "system_version": app_settings.system_version,
            "data_source_id": layer.data_source.data_source_id,
            "model_run_id": payload.model_run_id,
            "cma_id": payload.cma.cma_id,
            "transform_methods":layer.transform_methods,
            "title":f"processed_{layer.data_source.data_source_id}"
            })
        
        files_ = {"input_file": (
                    f"{layer.data_source.download_url.split('/')[-1]}",
                    open(f"./datasources/{layer.data_source.download_url.split('/')[-1]}", "rb"),
                    "application/octet-stream"
                )}

        resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/propectivity_input_layer",
            data={
                "metadata": data_layer.model_dump_json(exclude_none=True)
                },
            files=files_,
            headers=headers
            )
        if resp.status_code != 200 or resp.status_code != 204:
            print("An Error Occurred sending input layer")
            print(resp.text)
        else:
            
            print("Finished sending input layer!")

# stub steps to mimic ta3 model code
def run_ta3_pipeline(payload, app_settings):
    prepare_data_sources(payload=payload)
    train_model(payload=payload)
    run_model(payload=payload)
    send_outputs(payload=payload, app_settings=app_settings)
    send_stack(payload=payload, app_settings=app_settings)
    print("finished!")
