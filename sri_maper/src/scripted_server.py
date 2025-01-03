import os
import argparse
import requests
import torch
import glob
import json
import zipfile
import traceback
from pathlib import Path

# SRI TA3 specific imports
from torch import set_float32_matmul_precision
from sri_maper.src import utils
# from sri_maper.src.server import run_ta3_pipeline
from tqdm import tqdm

from cdr_schemas.events import Event

from sri_maper.src import utils
import sri_maper.src.data.preprocessing as preprocessing
from sri_maper.src.pretrain import pretrain
from sri_maper.src.train import train
from sri_maper.src.map import build_map

log = utils.get_pylogger(__name__)

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


def run_ta3_pipeline(
    event_id: int,
    app_settings: utils.CDR_Settings
):
    try:
        print("Querying CDR for event.")
        model_event_json = utils.get_event_payload_result(id=event_id, app_settings=app_settings)

        print("Parsing CDR event payload.")
        model_event_obj = utils.parse_event_payload_result(model_event_json)

        print("Generating AOI geopackage.")
        aoi_geopkg_path = utils.create_aoi_geopkg(model_event_obj)

        print("Downloading preprocessed evidence and label rasters.")
        processed_evidence_layer_paths, processed_label_raster_path, number_of_deposits, num_of_pixels = utils.download_preprocessed_layers(model_event_obj)
        print(f'This CMA has {number_of_deposits} deposits.')

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
                f"trainer.max_epochs={40 if num_of_pixels < 50000 else 30 if num_of_pixels < 150000 else 20 if num_of_pixels < 300000 else 10}"
            ]
        )
        utils.print_config_tree(pretrain_cfg)
        pretrain_metrics, pretrain_objs = pretrain(pretrain_cfg)

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
                if value is not None:
                    exposed_overrides.append(f"data.frac_train_split={model_event_obj.train_config.fraction_train_split}")
                else:
                    optuna_params_dict[key] = lambda x: f"data.frac_train_split={x}"
            elif key == "upsample_multiplier":
                if value is not None:
                    exposed_overrides.append(f"data.multiplier={model_event_obj.train_config.upsample_multiplier}")
                else:
                    optuna_params_dict[key] = lambda x: f"data.multiplier={x}"
            elif key == "learning_rate":
                if value is not None:
                    exposed_overrides.append(f"model.optimizer.lr={model_event_obj.train_config.learning_rate}")
                else:
                    optuna_params_dict[key] = lambda x: f"model.optimizer.lr={x}"
            elif key == "weight_decay":
                if value is not None:
                    exposed_overrides.append(f"model.optimizer.weight_decay={model_event_obj.train_config.weight_decay}")
                else:
                    optuna_params_dict[key] = lambda x: f"model.optimizer.weight_decay={x}"
            elif key == "smoothing":
                if value is not None:
                    exposed_overrides.append(f"model.smoothing={model_event_obj.train_config.smoothing}")
                else:
                    optuna_params_dict[key] = lambda x: f"model.smoothing={x}"
            elif key == "likely_negative_range":
                if value is not None:
                    exposed_overrides.append(f"data.likely_neg_range={list(model_event_obj.train_config.likely_negative_range)}")
                else:
                    optuna_params_dict[key] = lambda x,y: f"data.likely_neg_range={[x,y]}"
            elif key == "dropout_tuple":
                if value is not None:
                    exposed_overrides.append(f"model.net.dropout_rate={list(model_event_obj.train_config.dropout_tuple)}")
                else:
                    optuna_params_dict[key] = lambda x,y,z: f"model.net.dropout_rate={[x,y,z]}"
            else:
                raise ValueError(f"Unexpected key: {key}")


        # add exposed (user provided) train configs (no optuna yet)
        fixed_overrides += exposed_overrides

        if len(optuna_params_dict) > 0:
            print(f"Running hyperparameter search for params: {list(optuna_params_dict.keys())}")
            optuna_overrides, optuna_trial = utils.run_optuna_study(fixed_overrides,
                                                                    optuna_params_dict,
                                                                    num_deposits=number_of_deposits,
                                                                    n_trials=min(25,10*int(len(optuna_params_dict))))

            fixed_overrides += optuna_overrides

        print("Training classifier using pretrained MAE.")
        fixed_overrides.remove(f"model.net.backbone_ckpt_embeddings={backbone_ckpt_embeddings}")
        fixed_overrides.append(f"model.net.backbone_ckpt={backbone_ckpt}")
        fixed_overrides.append("enable_attributions=True")

        train_cfg = utils.build_hydra_config_notebook(overrides=fixed_overrides)
        utils.print_config_tree(train_cfg)
        train_metrics, train_objs = train(train_cfg)
        train_cfg.ckpt_path = train_objs["trainer"].checkpoint_callback.best_model_path

        print("Generating maps.")
        train_cfg.data.batch_size=128
        output_map_paths, _ = build_map(train_cfg)
        lklhoods_n_uncerts_paths = [output_map_paths.pop(1), output_map_paths.pop(0)] # place Uncertainties.tif first
        feat_attr_paths = [Path(path) for path in output_map_paths]
        lklhoods_n_uncerts_paths = [Path(path) for path in lklhoods_n_uncerts_paths]

        print("Uploading Likelihoods and Uncertainties to CDR.")
        for path in tqdm(lklhoods_n_uncerts_paths):
            utils.send_output(
                output_type=path.stem,
                output_path=path,
                payload=model_event_obj,
                app_settings=app_settings
            )

        print("Uploading feature attributes to CDR.")
        for path in tqdm(feat_attr_paths):
            utils.send_output(
                output_type=path.stem,
                output_path=path,
                payload=model_event_obj,
                app_settings=app_settings
            )

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

        print(f"event_id={event_id} cma is finished!")

    except Exception as e:
        print(f"Houston we have a problem! {e}")

        error_log_path = Path("./data") / Path(event_id) / Path('error_log.txt')
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    set_float32_matmul_precision('medium') # reduces floating point precision for computational efficiency
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id", type=str, help="Event ID from CDR")
    args = parser.parse_args()

    # event id is created from a new model run event. Should be provided by ta4
    if args.event_id:
        event_id = args.event_id
        run_ta3_pipeline(event_id=event_id, app_settings=server_settings)
