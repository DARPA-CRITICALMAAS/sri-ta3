import argparse
import os
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

log = utils.get_pylogger(__name__)

app_settings = utils.CDR_Settings(
    system_name = os.environ["SYSTEM_NAME"],
    system_version = os.environ["SYSTEM_VERSION"],
    ml_model_name = "xcorp_prospectivity_model",
    ml_model_version = "0.0.1",
    user_api_token = os.environ["CDR_TOKEN"],
    cdr_host = os.environ["CDR_HOST"],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id")
    args = parser.parse_args()

    # event id is created from a new model run event. Should be provided by ta4
    if args.event_id:
        event_id = args.event_id

        print("Querying CDR for event.")
        model_event_json = utils.get_event_payload_result(id=event_id, app_settings=app_settings)

        print("Parsing CDR event payload.")
        model_event_obj = utils.parse_event_payload_result(model_event_json)

        print("Generating AOI geopackage.")
        aoi_geopkg_path = utils.create_aoi_geopkg(model_event_obj)
        # aoi_geopkg_path = Path("data/49d7a30705fc478d9fbcea371da4627a/aoi.gpkg")

        print("Downloading deposits.")
        deposits_path = utils.download_deposits(model_event_obj, app_settings=app_settings)
        # deposits_path = Path("data/49d7a30705fc478d9fbcea371da4627a/deposits/Tungsten.csv")

        print("Processing label raster.")
        processed_label_raster_path = process_label_raster(
            event_obj=model_event_obj,
            deposits_csv_path=deposits_path,
            aoi=aoi_geopkg_path,
        )
        # processed_label_raster_path = Path("data/49d7a30705fc478d9fbcea371da4627a/deposits/Tungsten_processed.tif")

        print("Downloading evidence layers.")
        evidence_layer_paths = utils.download_evidence_layers(model_event_obj)
        print("Preprocessing evidence layers.")
        processed_evidence_layer_paths = preprocess_evidence_layers(
            event_obj=model_event_obj,
            layers=evidence_layer_paths,
            aoi=aoi_geopkg_path,
            reference_layer_path=processed_label_raster_path,
        )
        # processed_evidence_layer_paths = [
        #     Path("data/49d7a30705fc478d9fbcea371da4627a/evidence_layers/Geophysics_Gravity_Isostatic_processed.tif"),
        #     Path("data/49d7a30705fc478d9fbcea371da4627a/evidence_layers/Geophysics_Mag_RTP_processed.tif"),
        #     Path("data/49d7a30705fc478d9fbcea371da4627a/evidence_layers/Geophysics_Gravity_Bouguer_Up30km_HGM_processed.tif"),
        #     Path("data/49d7a30705fc478d9fbcea371da4627a/evidence_layers/Geophysics_Gravity_Bouguer_HGM_Worms_rasterized_processed.tif"),
        #     Path("data/49d7a30705fc478d9fbcea371da4627a/evidence_layers/Geophysics_MT2023_30km_processed.tif"),
        # ]

        print("Creating a raster stack.")
        raster_stack_path = generate_raster_stack(
            evidence_layer_paths=processed_evidence_layer_paths,
            label_raster_path=processed_label_raster_path
        )
        # raster_stack_path = Path("data/49d7a30705fc478d9fbcea371da4627a/raster_stack/raster_stack_d5.tif")

        print("Creating raster stack .yaml file.")
        raster_stack_yaml_path = create_raster_stack_yaml(
            event_obj=model_event_obj,
            evidence_layer_paths=processed_evidence_layer_paths,
            label_raster_path=processed_label_raster_path,
            raster_stack_path=raster_stack_path
        )
        # raster_stack_yaml_path = Path("data/49d7a30705fc478d9fbcea371da4627a/preprocessing.yaml")
        breakpoint()
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
                "trainer.min_epochs=5",
                "trainer.max_epochs=10",
            ]
        )
        utils.print_config_tree(pretrain_cfg)
        pretrain_metrics, pretrain_objs = pretrain(pretrain_cfg)
        # pretrain_objs = Path("logs/pretrain-maevit/runs/2024-09-16_04-07-10/checkpoints/psnr_41.716.ckpt")
        # pretrain_objs = Path("logs/pretrain-maevit/runs/2024-09-16_04-07-10/checkpoints/embeddings_d256.npy")

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
                "trainer.min_epochs=30",
                "trainer.max_epochs=100",
            ]
        )
        utils.print_config_tree(train_cfg)
        train_metrics, train_objs = train(train_cfg)
        train_cfg.ckpt_path = train_objs["trainer"].checkpoint_callback.best_model_path

        print("Generating maps.")
        train_cfg.data.batch_size=128
        output_map_paths, _ = build_map(train_cfg)
        # output_map_paths = ['logs/train-mae/runs/2024-09-16_14-34-05/Uncertainties.tif', 'logs/train-mae/runs/2024-09-16_14-34-05/Likelihoods.tif']
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

        # process label raster -> need to query CDR, filter CSV using gdf, then rasterize
        # after have all evidence and label layers, create raster stack
        # then create the preprocess YAML
        # then run pretrain
        # then run train
        # then generate map
        # then upload map results to CDR
        # then upload processed evidence layers to CDR (not labels and not feature attributions!)

        # import pdb
        # pdb.set_trace()

        # utils.run_ta3_pipeline(
        #     ProspectModelMetaData(
        #         model_run_id = model_payload.get("model_run_id"),
        #         cma = model_payload.get("cma"),
        #         model_type = model_payload.get("model_type"),
        #         train_config = model_payload.get("train_config"),
        #         evidence_layers = model_payload.get("evidence_layers"),
        #         ),
        #         app_settings=app_settings)

