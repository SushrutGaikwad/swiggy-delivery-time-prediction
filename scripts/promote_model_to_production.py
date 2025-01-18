import json
import mlflow
import dagshub

from pathlib import Path
from mlflow import MlflowClient


# DagsHub and MLFlow configuration
dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/SushrutGaikwad/swiggy-delivery-time-prediction.mlflow"
)


def load_run_info(run_info_file_path: Path):
    with open(run_info_file_path) as file:
        run_info = json.load(file)
    return run_info


# Model name and model stage
model_name = load_run_info(Path("run_info.json"))["model_name"]
stage = "Staging"

# Get the latest model version from the 'Staging' stage
client = MlflowClient()
latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
latest_model_version_staging = latest_versions[0].version

# Promote the model
promotion_stage = "Production"
client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True,
)
