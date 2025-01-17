import json
import pytest
import joblib
import mlflow
import dagshub

import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


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


def load_preprocessor(preprocessor_file_path: Path):
    preprocessor = joblib.load(preprocessor_file_path)
    return preprocessor


# Model name and stage
model_name = load_run_info(Path("run_info.json"))["model_name"]
stage = "Staging"

# Load the latest model from MLFlow model registry
model_uri = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_uri)

# Root path
root_path = Path(__file__).parent.parent

# Load the preprocessor
preprocessor_file_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_preprocessor(preprocessor_file_path=preprocessor_file_path)

# Build model pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# Test data path
test_data_file_path = root_path / "data" / "interim" / "test.csv"


@pytest.mark.parametrize(
    argnames="model_pipeline, test_data_file_path, error_threshold",
    argvalues=[(model_pipeline, test_data_file_path, 5)],
)
def test_model_performance(model_pipeline, test_data_file_path, error_threshold):
    # Loading the test data
    df = pd.read_csv(test_data_file_path)

    # Dropping missing values
    df.dropna(inplace=True)

    # X-y split
    X = df.drop(columns=["time_taken"])
    y = df["time_taken"]

    # Predictions
    y_pred = model_pipeline.predict(X)

    # Calculate MAE
    mae = mean_absolute_error(y_true=y, y_pred=y_pred)

    # Check for performance
    assert (
        mae <= error_threshold
    ), f"The model MAE is {mae} mins., which does not pass the performance error threshold of {error_threshold} mins."

    print(f"MAE of the model '{model_name}' is {mae} mins.")
    print(f"The model '{model_name}' has passed the performance test.")
