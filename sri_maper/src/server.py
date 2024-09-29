import os
import json
from pathlib import Path
from tqdm import tqdm

# CDR intergration imports
import atexit
import hashlib
import hmac
import httpx
import ngrok
import uvicorn
import uvicorn.logging

from fastapi.security import APIKeyHeader
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException, Request, status)
from cdr_schemas.events import Event

# SRI TA3 specific imports
from torch import set_float32_matmul_precision
from sri_maper.src import utils
import sri_maper.src.data.preprocessing as preprocessing
from sri_maper.src.pretrain import pretrain
from sri_maper.src.train import train
from sri_maper.src.map import build_map


def run_ta3_pipeline(
    event_id: int,
    app_settings: utils.CDR_Settings
):
    print("Querying CDR for event.")
    model_event_json = utils.get_event_payload_result(id=event_id, app_settings=app_settings)

    print("Parsing CDR event payload.")
    model_event_obj = utils.parse_event_payload_result(model_event_json)

    print("Generating AOI geopackage.")
    aoi_geopkg_path = utils.create_aoi_geopkg(model_event_obj)

    print("Downloading reference layer (aka template_raster.tif).")
    reference_layer_path = utils.download_reference_layer(model_event_obj)

    print("Downloading deposits.")
    deposits_path = utils.download_deposits(model_event_obj, app_settings=app_settings)

    print("Processing label raster.")
    processed_label_raster_path = preprocessing.process_label_raster(
        event_obj=model_event_obj,
        deposits_csv_path=deposits_path,
        aoi=aoi_geopkg_path,
        reference_layer_path=reference_layer_path
    )

    print("Downloading evidence layers.")
    evidence_layer_paths = utils.download_evidence_layers(model_event_obj)

    print("Preprocessing evidence layers.")
    processed_evidence_layer_paths = preprocessing.preprocess_evidence_layers(
        event_obj=model_event_obj,
        layers=evidence_layer_paths,
        aoi=aoi_geopkg_path,
        reference_layer_path=reference_layer_path
    )

    print("Creating a raster stack.")
    raster_stack_path = preprocessing.generate_raster_stack(
        evidence_layer_paths=processed_evidence_layer_paths,
        label_raster_path=processed_label_raster_path
    )

    print("Creating raster stack .yaml file.")
    raster_stack_yaml_path = preprocessing.create_raster_stack_yaml(
        event_obj=model_event_obj,
        evidence_layer_paths=processed_evidence_layer_paths,
        label_raster_path=processed_label_raster_path,
        raster_stack_path=raster_stack_path
    )

    print("Pretraining MAE.")
    pretrain_cfg = utils.build_hydra_config_notebook(
        overrides=[
            "experiment=pretrain_template.yaml",
            f"preprocess.raster_stacks.0.raster_stack_path={str(raster_stack_path)}",
            f"preprocess.raster_stacks.0.evidence_layer_paths={[str(layer_path) for layer_path in processed_evidence_layer_paths]}",
            f"preprocess.raster_stacks.0.label_raster_path={[str(processed_label_raster_path)]}",
            # "logger=csv", # wandb logger has issues in notebooks
            f"logger.wandb.name=pretrain|{str(model_event_obj.cma.mineral)}|{str(model_event_obj.model_run_id)}",
            f"tags=['pretrain','mae','ViT',{str(model_event_obj.model_run_id)},{str(model_event_obj.cma.mineral)}]",
            f"task_name=pretrain-{str(model_event_obj.cma.mineral)}-{str(model_event_obj.model_run_id)}",
            f"data.tif_dir={raster_stack_path.parent}",
            "data.batch_size=128",
            f"model.net.input_dim={len(processed_evidence_layer_paths)}",
            "paths.data_dir=data",
            "paths.log_dir=logs",
            "trainer=gpu",
            "trainer.min_epochs=5",
            "trainer.max_epochs=50",
        ]
    )

    utils.print_config_tree(pretrain_cfg)
    pretrain_metrics, pretrain_objs = pretrain(pretrain_cfg)
    breakpoint()
    print("Training classifier using pretrained MAE.")
    backbone_ckpt_embeddings = pretrain_objs['trainer'].checkpoint_callback.dirpath+f"/embeddings_d{pretrain_cfg.model.net.enc_dim}.npy"
    train_cfg = utils.build_hydra_config_notebook(
        overrides=[
            "experiment=classifier_template.yaml",
            f"logger.wandb.name=train|{str(model_event_obj.cma.mineral)}|{str(model_event_obj.model_run_id)}",
            "paths.data_dir=data",
            "paths.log_dir=logs",
            f"task_name=train-{str(model_event_obj.cma.mineral)}-{str(model_event_obj.model_run_id)}",
            f"tags=['train','mae','ViT','frozen',{str(model_event_obj.model_run_id)},{str(model_event_obj.cma.mineral)}]",
            # trainer args
            "trainer=gpu",
            "trainer.min_epochs=10",
            "trainer.max_epochs=100",
            # data args
            # f"data.window_size=5",
            f"data.tif_dir={raster_stack_path.parent}",
            f"data.likely_neg_range={list(model_event_obj.train_config.negative_sampling_fraction)}", #{str(model_event_obj.train_config.likely_negative_range)}",
            f"data.frac_train_split=0.8", #{model_event_obj.train_config.fraction_train_split}",
            f"data.multiplier=20", #{model_event_obj.train_config.upsample_multiplier}",
            # model args
            f"model.net.backbone_net.input_dim={len(processed_evidence_layer_paths)}",
            f"model.net.backbone_ckpt_embeddings={backbone_ckpt_embeddings}",
            f"model.net.dropout_rate=[0.0,{model_event_obj.train_config.dropout},{model_event_obj.train_config.dropout}]", #{str(model_event_obj.train_config.dropout)}",
            f"model.smoothing={model_event_obj.train_config.smoothing}",
            f"model.optimizer.lr=1e-3", #{model_event_obj.train_config.learning_rate}",
            f"model.optimizer.weight_decay=1e-2", #{model_event_obj.train_config.weight_decay}",
            # f"model.net.backbone_net.patch_size=1",
            # f"model.net.backbone_net.enc_dim=256",
            # f"model.net.backbone_net.encoder_layer=6",
            # f"model.net.backbone_net.encoder_head=8",
            # f"model.net.backbone_net.dec_dim=128",
            # f"model.net.backbone_net.decoder_layer=2",
            # f"model.net.backbone_net.decoder_head=4",
            # f"model.net.backbone_net.mask_ratio=0.0",
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


server_settings = utils.CDR_Settings(
    system_name = os.environ["SYSTEM_NAME"],
    system_version = os.environ["SYSTEM_VERSION"],
    ml_model_name = os.environ["MODEL_NAME"],
    ml_model_version = os.environ["MODEL_VERSION"],
    user_api_token = os.environ["CDR_TOKEN"],
    cdr_host = os.environ["CDR_HOST"],
    local_port = int(os.environ["NGROK_PORT"]),
    registration_id = "",
    registration_secret = os.environ["CDR_HOST"],
    callback_url = ""
)

def clean_up():
    # delete our registered system at CDR on program end
    headers = {'Authorization': f'Bearer {server_settings.user_api_token}'} # Define the headers for the HTTP request. The 'Authorization' header is set to 'Bearer ' followed by the user API token.
    client = httpx.Client(follow_redirects=True) # Create an HTTP client that follows redirects.
    client.delete(f"{server_settings.cdr_host}/user/me/register/{server_settings.registration_id}", headers=headers) # Send a DELETE request to the CDR host to unregister the system. The URL is constructed from the CDR host URL, the registration ID, and some static parts. The headers defined earlier are passed to the request.

# register clean_up
atexit.register(clean_up)


# Get ngrok to give us an endpoint
listener = ngrok.forward(server_settings.local_port, authtoken_from_env=True) # Forward the local port through ngrok and get a listener.
server_settings.callback_url = listener.url() + "/hook" # Set the callback URL to the ngrok URL plus "/hook".


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
                run_ta3_pipeline(evt.payload['model_run_id'], server_settings)
            case _:
                print("Nothing to do for event: %s", evt)

    except Exception:
        print("background processing event: %s", evt)
        raise

# verify the signature of  a request
async def verify_signature(
    request: Request,
    signature_header: str = Depends(APIKeyHeader(name="x-cdr-signature-256"))
):
    payload_body = await request.body() # retrieving the body of the request
    if not signature_header:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="x-hub-signature-256 header is missing!")
    hash_object = hmac.new(
        server_settings.registration_secret.encode("utf-8"),
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
        port=server_settings.local_port,
        reload=False
    ) # start a Uvicorn server with the FastAPI application


def register_system():
    """Register our system to the CDR using the server_settings"""
    global server_settings
    headers = {'Authorization': f'Bearer {server_settings.user_api_token}'}

    registration = {
        "name": server_settings.system_name,
        "version": server_settings.system_version,
        "callback_url": server_settings.callback_url,
        "webhook_secret": server_settings.registration_secret,
        # Leave blank if callback url has no auth requirement
        "auth_header": "",
        "auth_token": "",
        # Registers for ALL events
        "events": []

    }
    # creating an httpx client
    client = httpx.Client(follow_redirects=True) # follow_redirects=True argument tells the client to automatically follow redirects

    r = client.post(f"{server_settings.cdr_host}/user/me/register",
                    json=registration, headers=headers)

    # Log our registration_id such we can delete it when we close the program.
    server_settings.registration_id = r.json()["id"]


if __name__ == "__main__":
    set_float32_matmul_precision('medium') # reduces floating point precision for computational efficiency
    print("Registering with CDR")
    register_system()
    print("Starting TA3 server")
    run()
