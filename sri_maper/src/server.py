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
        
        print("Querying CDR for event.")
        model_event_json = utils.get_event_payload_result(id=event_id, app_settings=app_settings)

        print("Parsing CDR event payload.")
        model_event_obj = utils.parse_event_payload_result(model_event_json)

        print("Downloading evidence layers.")
        evidence_layer_paths = utils.download_evidence_layers(model_event_obj)

        print("Generating AOI geopackage.")
        aoi_geopkg_path = utils.create_aoi_geopkg(model_event_obj)

        print("Preprocessing evidence layers.")
        processed_evidence_layer_paths = preprocess_evidence_layers(
            event_obj=model_event_obj,
            layers=evidence_layer_paths,
            aoi=aoi_geopkg_path,
        )

        # process label raster -> need to query CDR, filter CSV using gdf, then rasterize
        # after have all evidence and label layers, create raster stack
        # then create the preprocess YAML
        # then run pretrain
        # then run train
        # then generate map
        # then upload map results to CDR
        # then upload processed evidence layers to CDR (not labels!)

        import pdb
        pdb.set_trace()

        # utils.run_ta3_pipeline(
        #     ProspectModelMetaData(
        #         model_run_id = model_payload.get("model_run_id"),
        #         cma = model_payload.get("cma"),
        #         model_type = model_payload.get("model_type"),
        #         train_config = model_payload.get("train_config"),
        #         evidence_layers = model_payload.get("evidence_layers"),
        #         ), 
        #         app_settings=app_settings)
        
