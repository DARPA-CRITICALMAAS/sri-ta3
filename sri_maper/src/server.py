import argparse
import os
from sri_maper.src import utils
from sri_maper.src.data.preprocessing import preprocess_evidence_layers

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
        
        model_event_json = utils.get_event_payload_result(id=event_id, app_settings=app_settings)

        model_event_obj = utils.parse_event_payload_result(model_event_json)

        evidence_layer_paths = utils.download_evidence_layers(model_event_obj)

        aoi_geopkg_path = utils.create_aoi_geopkg(model_event_obj)

        processed_evidence_layer_paths = preprocess_evidence_layers(
            event_obj=model_event_obj,
            layers=evidence_layer_paths,
            aoi=aoi_geopkg_path,
        )

        import pdb
        pdb.set_trace()

        # set the destination parameters: crs, resolution, nodata
        dst_params = {}
        dst_params['aoi_path'] = aoi_geopkg_path
        dst_params['crs'] = data["cma"]["crs"]
        dst_params['res_x'] = data["cma"]["resolution"][0]
        dst_params['res_y'] = data["cma"]["resolution"][1]
        dst_params['nodata'] = -999999999.0
        dst_params['description'] = data["cma"]["description"].lower().replace(" ", "_")

        pev_lyrs_path = data_path / Path(event_obj.model_run_id) / Path("processed_evidence_layers")
        pev_lyrs_path.mkdir(parents=True, exist_ok=True)

        # utils.run_ta3_pipeline(
        #     ProspectModelMetaData(
        #         model_run_id = model_payload.get("model_run_id"),
        #         cma = model_payload.get("cma"),
        #         model_type = model_payload.get("model_type"),
        #         train_config = model_payload.get("train_config"),
        #         evidence_layers = model_payload.get("evidence_layers"),
        #         ), 
        #         app_settings=app_settings)
        
