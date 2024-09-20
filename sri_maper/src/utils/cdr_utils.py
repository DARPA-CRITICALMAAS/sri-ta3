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
import fiona
import glob

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
    local_port: int
    registration_id: str
    registration_secret: str
    callback_url: str


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
    # breakpoint()
    # sets geopackage location
    geopkg_path = data_path / Path(event_obj.model_run_id)
    geopkg_path.mkdir(parents=True, exist_ok=True)
    # geopkg_path = geopkg_path / Path(f"aoi.gpkg")

    # Creating the AOI geopackage
    gdf = gpd.GeoDataFrame(
        {'id': [0]},
        crs = event_obj.cma.crs,
        geometry = [event_obj.cma.extent]
    )
    try:
        gdf.to_file(geopkg_path / Path(f"aoi.gpkg"), driver="GPKG")
        return geopkg_path / Path(f"aoi.gpkg")
    except fiona.errors.TransactionError:
        gdf.to_file(geopkg_path / Path(f"aoi.shp"))
        return geopkg_path / Path(f"aoi.shp")


def download_deposits(
    event_obj: ProspectModelMetaData,
    app_settings: CDR_Settings,
    data_path: Path = Path("./data"),
    # with_location: str = None,
    # with_deposit_types_only: bool = True,
    # top_n: int = 1,
    # limit: int = -1,
):
    # breakpoint()
    # sets deposits location folder
    deposits_path = data_path / Path(event_obj.model_run_id) / Path("deposits")
    deposits_path.mkdir(parents=True, exist_ok=True)

    commodity = event_obj.cma.mineral

    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)

    link = f"{app_settings.cdr_host}/v1/minerals/dedup-site/search/{commodity}?with_location=true&with_deposit_types_only=true&top_n=1&limit=-1"

    resp = client.get(link, headers=headers)
    if resp.status_code == 200:
        # Get the filename from the 'Content-Disposition' header
        content_disposition = resp.headers.get('content-disposition')
        if content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"')
        else:
            filename = 'deposits.csv' # Use a default filename if none was provided
        deposits_path = deposits_path / filename
        # Open file in write mode
        with open(deposits_path, 'w') as f:
            # Write the response content to the file
            f.write(resp.text)
    else:
        raise Exception(f"Failed to download file: {resp.status_code}, {resp.text}")
    return deposits_path


def send_output(
    output_type: str,
    output_path: Path,
    payload,
    app_settings
):
    print(f"Sending {output_path} to CDR...")

    # checks outputs file exists
    assert output_path.is_file()
    assert "likelihood" in output_type.lower() or "uncertaint" in output_type.lower()

    #  create output layer metadata
    results = ProspectivityOutputLayer(**{
        "system": app_settings.system_name,
        "system_version": app_settings.system_version,
        "model": app_settings.ml_model_name,
        "model_version": app_settings.ml_model_version,
        "model_run_id": payload.model_run_id,
        "cma_id": payload.cma.cma_id,
        "output_type": output_type,
        "title": output_path.name
    })
    files_ = {"input_file": (
        output_path.name,
        open(output_path, "rb"), "application/octet-stream")
    }

    # prepare the CDR client
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)

    # post the request
    resp = client.post(
        url=f"{app_settings.cdr_host}/v1/prospectivity/prospectivity_output_layer",
        data={"metadata": results.model_dump_json(exclude_none=True)},
        files=files_,
        headers=headers
    )
    if resp.status_code == 200 or resp.status_code == 204:
        print(f"Finished sending {output_path.name} to CDR.")
    else:
        print(f"An Error Occurred sending {output_path} to CDR.")
        print(resp.status_code)
        print(resp.text)
        print("debug")


def send_processed_evidence_layer(
    layer_path: Path,
    layer: str,
    payload,
    app_settings
):
    print(f"Sending {layer_path.stem} to CDR...")

    # checks outputs file exists
    assert layer_path.is_file()

    #  create processed data layer metadata
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
        layer_path.name,
        open(layer_path, "rb"),
        "application/octet-stream"
    )}

    # prepare the CDR client
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)

    # post the request
    resp = client.post(
        url=f"{app_settings.cdr_host}/v1/prospectivity/prospectivity_input_layer",
        data={"metadata": data_layer.model_dump_json(exclude_none=True)},
        files=files_,
        headers=headers
    )
    if resp.status_code == 200 or resp.status_code == 204:
        print(f"Finished sending {layer_path.name} to CDR.")
    else:
        print(f"An Error Occurred sending {layer_path} to CDR.")
        print(resp.status_code)
        print(resp.text)
        print("debug")
