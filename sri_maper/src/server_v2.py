import argparse
import atexit
import hashlib
import hmac
import os

from fastapi.security import APIKeyHeader

import httpx
import ngrok

import uvicorn
import uvicorn.logging
from cdr_schemas.events import Event
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException, Request, status)
# from common import run_ta3_pipeline

from pydantic_settings import BaseSettings
from cdr_schemas.cdr_responses.prospectivity import ProspectModelMetaData

from sri_maper.src import utils
from sri_maper.src.data.preprocessing import preprocess_evidence_layers, \
                                            process_label_raster, \
                                            generate_raster_stack, \
                                            create_raster_stack_yaml

from sri_maper.src.pretrain import pretrain
from sri_maper.src.train import train
from sri_maper.src.map import build_map

from torch import set_float32_matmul_precision
set_float32_matmul_precision('medium') # reduces floating point precision for computational efficiency

from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
args = parser.parse_args()

app_settings = utils.CDR_Settings(
    system_name = os.environ["SYSTEM_NAME"],
    system_version = os.environ["SYSTEM_VERSION"],
    ml_model_name = "xcorp_prospectivity_model",
    ml_model_version = "0.0.1",
    user_api_token = os.environ["CDR_TOKEN"],
    cdr_host = os.environ["CDR_HOST"],
)

def run_ta3_pipeline(event_id):
    print("Querying CDR for event.")
    model_event_json = utils.get_event_payload_result(id=event_id, app_settings=app_settings)

    print("Parsing CDR event payload.")
    model_event_obj = utils.parse_event_payload_result(model_event_json)

    print("Generating AOI geopackage.")
    aoi_geopkg_path = utils.create_aoi_geopkg(model_event_obj)

    print("Downloading deposits.")
    deposits_path = utils.download_deposits(model_event_obj, app_settings=app_settings)
    print("Processing label raster.")
    processed_label_raster_path = process_label_raster(
        event_obj=model_event_obj,
        deposits_csv_path=deposits_path,
        aoi=aoi_geopkg_path,
    )

    print("Downloading evidence layers.")
    evidence_layer_paths = utils.download_evidence_layers(model_event_obj)
    print("Preprocessing evidence layers.")
    processed_evidence_layer_paths = preprocess_evidence_layers(
        event_obj=model_event_obj,
        layers=evidence_layer_paths,
        aoi=aoi_geopkg_path,
        reference_layer_path=processed_label_raster_path,
    )

    print("Creating a raster stack.")
    raster_stack_path = generate_raster_stack(
        evidence_layer_paths=processed_evidence_layer_paths,
        label_raster_path=processed_label_raster_path
    )

    print("Creating raster stack .yaml file.")
    raster_stack_yaml_path = create_raster_stack_yaml(
        event_obj=model_event_obj,
        evidence_layer_paths=processed_evidence_layer_paths,
        label_raster_path=processed_label_raster_path,
        raster_stack_path=raster_stack_path
    )

    print("Pretraining MAE.")
    pretrain_cfg = utils.build_hydra_config_notebook(
        overrides=[
            # f"preprocess={raster_stack_yaml_path}",
            "experiment=pretrain_template.yaml",
            # "logger=csv", # wandb logger has issues in notebooks
            "logger.wandb.name=pretrain|",
            "trainer=gpu",
            f"tags=['pretrain','mae', {str(model_event_obj.model_run_id)}, {str(model_event_obj.cma.mineral)}]",
            "task_name=pretrain-maevit",
            f"data.tif_dir={raster_stack_path.parent}",
            f"model.net.input_dim={len(processed_evidence_layer_paths)}",
            "paths.data_dir=data",
            "paths.log_dir=logs",
            "trainer.min_epochs=2",
            "trainer.max_epochs=2",
        ]
    )
    utils.print_config_tree(pretrain_cfg)
    pretrain_metrics, pretrain_objs = pretrain(pretrain_cfg)

    print("Training classifier using pretrained MAE.")
    backbone_ckpt_embeddings = pretrain_objs['trainer'].checkpoint_callback.dirpath+f"/embeddings_d{pretrain_cfg.model.net.enc_dim}.npy"
    train_cfg = utils.build_hydra_config_notebook(
        overrides=[
            "experiment=classifier_template.yaml",
            "logger.wandb.name=train|",
            "trainer=gpu",
            f"data.tif_dir={raster_stack_path.parent}",
            f"model.net.backbone_net.input_dim={len(processed_evidence_layer_paths)}",
            "paths.data_dir=data",
            "paths.log_dir=logs",
            f"model.net.backbone_ckpt_embeddings={backbone_ckpt_embeddings}",
            f"tags=['train', 'mae', {str(model_event_obj.model_run_id)}, {str(model_event_obj.cma.mineral)}]",
            "task_name=train-mae",
            "trainer.min_epochs=5",
            "trainer.max_epochs=5",
        ]
    )
    utils.print_config_tree(train_cfg)
    train_metrics, train_objs = train(train_cfg)
    train_cfg.ckpt_path = train_objs["trainer"].checkpoint_callback.best_model_path

    print("Generating maps.")
    train_cfg.data.batch_size=128
    output_map_paths, _ = build_map(train_cfg)
    output_map_paths.sort(reverse=True) # place Uncertainties.tif first
    output_map_paths = [Path(path) for path in output_map_paths]

    print("Uploading results to CDR.")
    for path in tqdm(output_map_paths):
        utils.send_output(
            output_type=path.stem,
            output_path=path,
            payload=model_event_obj,
            app_settings=app_settings
        )

    print("Uploading processed evidence layers to CDR.")
    for evidence_layer_idx, path in tqdm(enumerate(processed_evidence_layer_paths)):
        utils.send_processed_evidence_layer(
            layer_path=path,
            layer=model_event_obj.evidence_layers[evidence_layer_idx],
            payload=model_event_obj,
            app_settings=app_settings
        )

    print(f"event_id={event_id} cma is finished!")

class Settings(BaseSettings):
    # TO BE CHANGED BY TA3-4 system.
    system_name: str = os.environ["SYSTEM_NAME"]
    system_version: str = os.environ["SYSTEM_VERSION"]
    ml_model_name: str = "xcorp_prospectivity_model"
    ml_model_version: str = "0.0.1"

    # Local port to run on
    local_port: int = 9999
    # To be filled in programmatically via ngrok below.
    callback_url: str = ""
    # Secret string used for signature verification on callback.  Changed by TA3-4 system.
    registration_secret: str = "mysecret"

    # To be provided to TA3-4 system by CDR admin
    user_api_token: str = os.environ["CDR_TOKEN"]
    cdr_host: str = os.environ["CDR_HOST"]
    admin_cdr_host: str = "https://admin.cdr.land"
    # For local development
    # cdr_host: str = "http://0.0.0.0:8333"
    # admin_cdr_host: str = "http://0.0.0.0:3333"


    # To be filled in programmatically after registration process below.  Needed to remove registration.
    registration_id: str = ""
    ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
    # ngrok.set_auth_token("2mFmoyp0MKgIgpxn98X6U7EvSq5_3fbaqD6BfoAUkS4qq5Xab")

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create an instance
app_settings = Settings()



# breakpoint()
# Get ngrok to give us an endpoint
listener = ngrok.forward(app_settings.local_port, authtoken_from_env=True) # Forward the local port through ngrok and get a listener.
app_settings.callback_url = listener.url() + "/hook" # Set the callback URL to the ngrok URL plus "/hook".


def clean_up():
    # delete our registered system at CDR on program end
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'} # Define the headers for the HTTP request. The 'Authorization' header is set to 'Bearer ' followed by the user API token.
    client = httpx.Client(follow_redirects=True) # Create an HTTP client that follows redirects.
    client.delete(f"{app_settings.cdr_host}/user/me/register/{app_settings.registration_id}", headers=headers) # Send a DELETE request to the CDR host to unregister the system. The URL is constructed from the CDR host URL, the registration ID, and some static parts. The headers defined earlier are passed to the request.


# register clean_up
atexit.register(clean_up)

app = FastAPI() # creating an instance

async def event_handler(
    evt: Event
):
    try:
        match evt: # pattern matching on evt.
            case Event(event="ping"):
                print("Received PING!")
            case Event(event="prospectivity_model_run.process"):
                print("Received model run event payload!")
                print(evt.payload)
                # breakpoint()
                run_ta3_pipeline(evt.payload['model_run_id'])
                # run_ta3_pipeline(
                #     ProspectModelMetaData(
                #         model_run_id = evt.payload.get("model_run_id"),
                #         cma = evt.payload.get("cma"),
                #         model_type = evt.payload.get("model_type"),
                #         train_config = evt.payload.get("train_config"),
                #         evidence_layers = evt.payload.get("evidence_layers"),
                #         ), app_settings)
            case _:
                print("Nothing to do for event: %s", evt)

    except Exception:
        print("background processing event: %s", evt)
        raise

cdr_signiture = APIKeyHeader(name="x-cdr-signature-256")

# verify the signature of  a request
async def verify_signature(
    request: Request,
    signature_header: str = Depends(cdr_signiture)
):
    payload_body = await request.body() # retrieving the body of the request
    if not signature_header:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="x-hub-signature-256 header is missing!")
    hash_object = hmac.new(
        app_settings.registration_secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256
    ) # creating a new hmac hash object
    expected_signature = hash_object.hexdigest() # calculating the hexadecimal digest
    # Compare the expected signature with the signature in the header
    if not hmac.compare_digest(expected_signature, signature_header):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Request signatures didn't match!")
    return True


@app.post("/hook") # This decorator tells FastAPI to use this function (named hook) to handle POST requests to the "/hook" endpoint.
async def hook(
    evt: Event,
    background_tasks: BackgroundTasks, # a class provided by FastAPI that allows to add background tasks that will be run after returning the response.
    request: Request,
    verified_signature: bool = Depends(verify_signature),
):
    """Our main entry point for CDR calls"""

    background_tasks.add_task(event_handler, evt) # add a background task that will call the event_handler with evt as an argument
    return {"ok": "success"}


def run():
    """Run our web hook"""
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=app_settings.local_port,
        reload=False
    ) # start a Uvicorn server with the FastAPI application


def register_system():
    # breakpoint()
    """Register our system to the CDR using the app_settings"""
    global app_settings
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}

    registration = {
        "name": app_settings.system_name,
        "version": app_settings.system_version,
        "callback_url": app_settings.callback_url,
        "webhook_secret": app_settings.registration_secret,
        # Leave blank if callback url has no auth requirement
        "auth_header": "",
        "auth_token": "",
        # Registers for ALL events
        "events": []

    }
    # creating an httpx client
    client = httpx.Client(follow_redirects=True) # follow_redirects=True argument tells the client to automatically follow redirects

    r = client.post(f"{app_settings.cdr_host}/user/me/register",
                    json=registration, headers=headers)

    # Log our registration_id such we can delete it when we close the program.
    app_settings.registration_id = r.json()["id"]


if __name__ == "__main__":
    register_system()
    run()
