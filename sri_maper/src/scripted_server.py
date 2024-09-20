import os
import argparse

# SRI TA3 specific imports
from torch import set_float32_matmul_precision
from sri_maper.src import utils
from sri_maper.src.server import run_ta3_pipeline

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

if __name__ == "__main__":
    set_float32_matmul_precision('medium') # reduces floating point precision for computational efficiency
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id")
    args = parser.parse_args()

    # event id is created from a new model run event. Should be provided by ta4
    if args.event_id:
        event_id = args.event_id

        run_ta3_pipeline(event_id=event_id, app_settings=server_settings)
