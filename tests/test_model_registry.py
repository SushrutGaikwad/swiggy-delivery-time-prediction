import json
import pytest
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


# Set model name
model_name = load_run_info(Path("run_info.json"))["model_name"]


@pytest.mark.parametrize(argnames="model_name, stage", argvalues=[(model_name, "Staging")])
def test_load_model_from_registry(model_name, stage):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
    if latest_versions:
        latest_version = latest_versions[0].version
    else:
        latest_version = None

    assert latest_version is not None, f"No model in the {stage} stage."

    # model uri
    model_uri = f"models:/{model_name}/{stage}"

    # load the latest model from model registry
    model = mlflow.sklearn.load_model(model_uri)

    assert model is not None, "Failed to load model from the model registry."
    print(f"The '{model_name}' model, version {latest_version}, was loaded successfully.")
