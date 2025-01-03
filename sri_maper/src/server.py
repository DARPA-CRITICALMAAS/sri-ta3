import os
import glob
import json
import zipfile
import traceback
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# CDR intergration imports
import requests
import atexit
import hashlib
import hmac
import httpx
import uvicorn
import uvicorn.logging
import optuna

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

bot_token = None
auth_token = None

def get_authorized_chat_ids():
    """Fetch chat IDs of users who sent the correct token"""
    url = f'https://api.telegram.org/bot{bot_token}/getUpdates'
    response = requests.get(url)
    if response.status_code == 200:
        updates = response.json()['result']
        # Extract chat IDs only from messages containing the correct token
        chat_ids = list(set(str(update['message']['chat']['id'])
                        for update in updates
                        if 'message' in update
                        and 'text' in update['message']
                        and update['message']['text'] == auth_token))
        return chat_ids if chat_ids else ['5186897455']  # Fallback to original chat ID
    return ['5186897455']

def send_telegram_message(message_body):
    chat_ids = get_authorized_chat_ids()
    for chat_id in chat_ids:
        url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
        payload = {
            'chat_id': chat_id,
            'text': message_body
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Failed to send message to chat ID {chat_id}: {response.text}")

def run_ta3_pipeline(
    event_id: int,
    app_settings: utils.CDR_Settings
):
    try:
        print("Querying CDR for event.")
        model_event_json = utils.get_event_payload_result(id=event_id, app_settings=app_settings)
        if bot_token: send_telegram_message(f"[1/16]: New CMA: {model_event_json['event']['payload']['cma']['description']} ({event_id}) 👀")
        if bot_token: send_telegram_message(f"[2/16]: Queried CDR for event ✅")

        print("Parsing CDR event payload.")
        model_event_obj = utils.parse_event_payload_result(model_event_json)
        if bot_token: send_telegram_message(f"[3/16]: Parsed CDR event payload ✅")

        print("Generating AOI geopackage.")
        aoi_geopkg_path = utils.create_aoi_geopkg(model_event_obj)
        if bot_token: send_telegram_message(f"[4/16]: Generated AOI geopackage ✅")

        print("Downloading preprocessed evidence and label rasters.")
        processed_evidence_layer_paths, processed_label_raster_path, number_of_deposits, num_of_pixels = utils.download_preprocessed_layers(model_event_obj)
        print(f'This CMA has {number_of_deposits} deposits.')
        if bot_token: send_telegram_message(f"[5/16]: Downloaded {len(processed_evidence_layer_paths)} preprocessed evidence and 1 label rasters | # of depos.={number_of_deposits} | # of pixels={num_of_pixels} ✅")

        print("Creating a raster stack.")
        raster_stack_path = preprocessing.generate_raster_stack(
            evidence_layer_paths=processed_evidence_layer_paths,
            label_raster_path=processed_label_raster_path
        )
        if bot_token: send_telegram_message(f"[6/16]: Created a raster stack ✅")

        print("Creating raster stack .yaml file.")
        raster_stack_yaml_path = preprocessing.create_raster_stack_yaml(
            event_obj=model_event_obj,
            evidence_layer_paths=processed_evidence_layer_paths,
            label_raster_path=processed_label_raster_path,
            raster_stack_path=raster_stack_path
        )
        if bot_token: send_telegram_message(f"[7/16]: Created raster stack .yaml file ✅")

        print("Pretraining MAE.")
        pretrain_cfg = utils.build_hydra_config_notebook(
            overrides=[
                "experiment=pretrain_template.yaml",
                f"preprocess.raster_stacks.0.raster_stack_path={str(raster_stack_path)}",
                f"preprocess.raster_stacks.0.evidence_layer_paths={[str(layer_path) for layer_path in processed_evidence_layer_paths]}",
                f"preprocess.raster_stacks.0.label_raster_path={[str(processed_label_raster_path)]}",
                "logger=csv", # wandb logger has issues in notebooks
                f"logger.wandb.name=pretrain|{str(model_event_obj.cma.mineral)}|{str(model_event_obj.model_run_id)}",
                f"tags=['pretrain','mae','ViT',{str(model_event_obj.model_run_id)},{str(model_event_obj.cma.mineral)}]",
                f"task_name=pretrain-{str(model_event_obj.cma.mineral)}-{str(model_event_obj.model_run_id)}",
                f"data.tif_dir={raster_stack_path.parent}",
                f"data.batch_size={64 if num_of_pixels < 50000 else 128 if num_of_pixels < 150000 else 256 if num_of_pixels < 500000 else 512 if num_of_pixels < 1000000 else 1024}",
                f"model.net.input_dim={len(processed_evidence_layer_paths)}",
                "paths.data_dir=data",
                "paths.log_dir=logs",
                "trainer=gpu",
                "trainer.min_epochs=1",
                f"trainer.max_epochs={40 if num_of_pixels < 50000 else 30 if num_of_pixels < 150000 else 20 if num_of_pixels < 300000 else 10}",
            ]
        )
        utils.print_config_tree(pretrain_cfg)
        pretrain_metrics, pretrain_objs = pretrain(pretrain_cfg)
        if bot_token: send_telegram_message(f"[8/16]: Finished pretraining MAE ✅")

        print("Preparing classifier overrides")
        backbone_ckpt_embeddings =  glob.glob(os.path.join(pretrain_objs['trainer'].checkpoint_callback.dirpath, '*.npy'))[0]
        backbone_ckpt = glob.glob(os.path.join(pretrain_objs['trainer'].checkpoint_callback.dirpath, '*psnr*.ckpt'))[0]

        fixed_overrides = [
            "experiment=classifier_template.yaml",
            f"preprocess.raster_stacks.0.raster_stack_path={str(raster_stack_path)}",
            f"preprocess.raster_stacks.0.evidence_layer_paths={[str(layer_path) for layer_path in processed_evidence_layer_paths]}",
            f"preprocess.raster_stacks.0.label_raster_path={[str(processed_label_raster_path)]}",
            "logger=csv",
            f"logger.wandb.name=train|{str(model_event_obj.cma.mineral)}|{str(model_event_obj.model_run_id)}",
            "paths.data_dir=data",
            "paths.log_dir=logs",
            f"task_name=train-{str(model_event_obj.cma.mineral)}-{str(model_event_obj.model_run_id)}",
            f"tags=['train','mae','ViT','frozen',{str(model_event_obj.model_run_id)},{str(model_event_obj.cma.mineral)}]",
            # trainer args
            "trainer=gpu",
            "trainer.min_epochs=25",
            "trainer.max_epochs=50",
            # data args
            f"data.tif_dir={raster_stack_path.parent}",
            f"data.batch_size={16 if number_of_deposits < 25 else 32}", # if number_of_deposits < 50 else 128 if number_of_deposits > 100 else 64}",
            # model args
            f"model.net.backbone_net.input_dim={len(processed_evidence_layer_paths)}",
            f"model.net.backbone_ckpt_embeddings={backbone_ckpt_embeddings}",
        ]

        exposed_params_dict = model_event_obj.train_config.__dict__
        exposed_overrides = []
        optuna_params_dict = {}
        for key, value in exposed_params_dict.items():
            if key == "fraction_train_split":
                if value:
                    exposed_overrides.append(f"data.frac_train_split={model_event_obj.train_config.fraction_train_split}")
                else:
                    optuna_params_dict[key] = lambda x: f"data.frac_train_split={x}"
            elif key == "upsample_multiplier":
                if value:
                    exposed_overrides.append(f"data.multiplier={model_event_obj.train_config.upsample_multiplier}")
                else:
                    optuna_params_dict[key] = lambda x: f"data.multiplier={x}"
            elif key == "learning_rate":
                if value:
                    exposed_overrides.append(f"model.optimizer.lr={model_event_obj.train_config.learning_rate}")
                else:
                    optuna_params_dict[key] = lambda x: f"model.optimizer.lr={x}"
            elif key == "weight_decay":
                if value:
                    exposed_overrides.append(f"model.optimizer.weight_decay={model_event_obj.train_config.weight_decay}")
                else:
                    optuna_params_dict[key] = lambda x: f"model.optimizer.weight_decay={x}"
            elif key == "smoothing":
                if value:
                    exposed_overrides.append(f"model.smoothing={model_event_obj.train_config.smoothing}")
                else:
                    optuna_params_dict[key] = lambda x: f"model.smoothing={x}"
            elif key == "likely_negative_range":
                if value:
                    exposed_overrides.append(f"data.likely_neg_range={list(model_event_obj.train_config.likely_negative_range)}")
                else:
                    optuna_params_dict[key] = lambda x,y: f"data.likely_neg_range={[x,y]}"
            elif key == "dropout_tuple":
                if value:
                    exposed_overrides.append(f"model.net.dropout_rate={list(model_event_obj.train_config.dropout_tuple)}")
                else:
                    optuna_params_dict[key] = lambda x,y,z: f"model.net.dropout_rate={[x,y,z]}"
            else:
                raise ValueError(f"Unexpected key: {key}")

        if bot_token: send_telegram_message(f"[9/16]: Prepared {len(exposed_overrides)} GUI provided overrides ✅")

        # add exposed (user provided) train configs (no optuna yet)
        fixed_overrides += exposed_overrides

        if len(optuna_params_dict) > 0:
            print(f"Running hyperparameter search for params: {list(optuna_params_dict.keys())}")
            optuna_overrides, optuna_trial = utils.run_optuna_study(fixed_overrides,
                                                                    optuna_params_dict,
                                                                    num_deposits=number_of_deposits,
                                                                    n_trials=min(30,10*int(len(optuna_params_dict))))

            fixed_overrides += optuna_overrides
            if bot_token: send_telegram_message(f"[9.1/16]: Prepared {len(optuna_params_dict)} OPTUNA overrides ✅")

        print("Training classifier using pretrained MAE.")
        fixed_overrides.remove(f"model.net.backbone_ckpt_embeddings={backbone_ckpt_embeddings}")
        fixed_overrides.append(f"model.net.backbone_ckpt={backbone_ckpt}")
        fixed_overrides.append("enable_attributions=True")

        train_cfg = utils.build_hydra_config_notebook(overrides=fixed_overrides)
        utils.print_config_tree(train_cfg)
        train_metrics, train_objs = train(train_cfg)
        train_cfg.ckpt_path = train_objs["trainer"].checkpoint_callback.best_model_path
        if bot_token: send_telegram_message(f"[10/16]: Finished training classifier ✅")

        print("Generating maps.")
        train_cfg.data.batch_size=128
        output_map_paths, _ = build_map(train_cfg)
        lklhoods_n_uncerts_paths = [output_map_paths.pop(1), output_map_paths.pop(0)] # place Uncertainties.tif first
        feat_attr_paths = [Path(path) for path in output_map_paths]
        lklhoods_n_uncerts_paths = [Path(path) for path in lklhoods_n_uncerts_paths]
        if bot_token: send_telegram_message(f"[11/16]: Generated {len(output_map_paths)} FA and {len(lklhoods_n_uncerts_paths)} LHD/UNCT maps ✅")

        print("Uploading Likelihoods and Uncertainties to CDR.")
        for path in tqdm(lklhoods_n_uncerts_paths):
            utils.send_output(
                output_type=path.stem,
                output_path=path,
                payload=model_event_obj,
                app_settings=app_settings
            )
        if bot_token: send_telegram_message(f"[12/16]: Uploaded LHD/UNCT to CDR ✅")

        print("Uploading feature attributes to CDR.")
        for path in tqdm(feat_attr_paths):
            utils.send_output(
                output_type=path.stem,
                output_path=path,
                payload=model_event_obj,
                app_settings=app_settings
            )
        if bot_token: send_telegram_message(f"[13/16]: Uploaded FA to CDR ✅")

        print("Packing and uploading .csv files into .zip.")
        base_path = lklhoods_n_uncerts_paths[0].parent
        zip_split_path = base_path / Path('splits.zip')
        files_splits = [base_path / Path('train.csv'), base_path / Path('valid.csv'), base_path / Path('test.csv')]
        utils.create_zip_file(zip_split_path, files_splits)
        utils.send_output(
            output_type=zip_split_path.stem,
            output_path=zip_split_path,
            payload=model_event_obj,
            app_settings=app_settings
        )
        if bot_token: send_telegram_message(f"[14/16]: Uploaded split_files.zip to CDR ✅")

        print("Packing and uploading metric .json file into .zip.")
        zip_metric_path = base_path / Path('metrics.zip')
        output_metrics_file = base_path / Path('metrics.json')
        utils.reorganize_metrics_file(train_metrics, output_metrics_file)
        utils.create_zip_file(zip_metric_path, [output_metrics_file])
        utils.send_output(
            output_type=zip_metric_path.stem,
            output_path=zip_metric_path,
            payload=model_event_obj,
            app_settings=app_settings
        )
        if bot_token: send_telegram_message(f"[15/16]: Uploaded metric_files.zip to CDR ✅")

        if len(optuna_params_dict) > 0:
            print("Packing and uploading optuna .json file into .zip.")
            zip_optuna_path = base_path / Path('optuna_search_values.zip')
            optuna_file = base_path / Path('optuna_search_values.json')
            with open(optuna_file, 'w') as file:
                json.dump(optuna_overrides, file, indent=4)
            files_metrics = [optuna_file]
            utils.create_zip_file(zip_optuna_path, files_metrics)
            utils.send_output(
                output_type=zip_optuna_path.stem,
                output_path=zip_optuna_path,
                payload=model_event_obj,
                app_settings=app_settings
            )
            if bot_token: send_telegram_message(f"[15.1/16]: Uploaded optuna_search_values.zip to CDR ✅")

        print(f"event_id={event_id} cma is finished!")
        if bot_token: send_telegram_message(f"[16/16]: {model_event_json['event']['payload']['cma']['description']} ({event_id}) CMA is finished 🎉")
        if bot_token: send_telegram_message(f"")

    except Exception as e:
        print(f"Houston we have a problem! {e}")
        if bot_token: send_telegram_message(f"Houston we have a problem! {e}")

        error_log_path = Path("./data") / Path(event_id) / Path('error_log.txt')
        with open(error_log_path, 'w') as f:
            traceback.print_exc(file=f)
        zip_error_path = error_log_path.parent / Path('error_logs.zip')
        files_errors = [error_log_path]
        utils.create_zip_file(zip_error_path, files_errors)
        utils.send_output(
            output_type=zip_error_path.stem,
            output_path=zip_error_path,
            payload=model_event_obj,
            app_settings=app_settings
        )



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
    callback_url = os.environ["CALLBACK_URL"],
)

def clean_up():
    # delete our registered system at CDR on program end
    headers = {'Authorization': f'Bearer {server_settings.user_api_token}'} # Define the headers for the HTTP request. The 'Authorization' header is set to 'Bearer ' followed by the user API token.
    client = httpx.Client(follow_redirects=True) # Create an HTTP client that follows redirects.
    client.delete(f"{server_settings.cdr_host}/user/me/register/{server_settings.registration_id}", headers=headers) # Send a DELETE request to the CDR host to unregister the system. The URL is constructed from the CDR host URL, the registration ID, and some static parts. The headers defined earlier are passed to the request.

# register clean_up
atexit.register(clean_up)

if not server_settings.callback_url:
    import ngrok
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
                if evt.payload['model_type'] == 'sri_NN':
                    print("Received model run event payload with model_type 'sri_NN'!")
                    print(evt.payload)
                    run_ta3_pipeline(evt.payload['model_run_id'], server_settings)
                else:
                    print(f"Received model run event payload with {evt.payload['model_type']} model_type!")
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
