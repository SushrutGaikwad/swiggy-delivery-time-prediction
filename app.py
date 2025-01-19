import json
import mlflow
import joblib
import uvicorn
import dagshub
import logging

import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from typing import Dict, Any
from pydantic import BaseModel
from mlflow import MlflowClient
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from swiggy_delivery_time_prediction.data.data_cleaning import DataCleaner

set_config(transform_output="pandas")


# Create logger
logger = logging.getLogger("fastapi_app")
logger.setLevel(logging.INFO)

# Console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)

# Create a formatter
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add formatter to handler
handler.setFormatter(formatter)


# FastAPI and MLFlow configuration
dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/SushrutGaikwad/swiggy-delivery-time-prediction.mlflow"
)


# Pydantic model for input data
class Data(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


# Service
class ModelService:
    def __init__(self, run_info_file_path: Path, stage: str = "Production"):
        self.logger = logging.getLogger("fastapi_app")
        self.logger.info("Instantiating a `ModelService` object...")

        self.run_info_file_path = run_info_file_path
        self.stage = stage
        self.client = MlflowClient()
        self._model_pipeline = None
        self._preprocessor = None

        self.data_cleaner = DataCleaner(modify_target=False)

        self.logger.info("`ModelService` object instantiated.")

    def load_run_info(self) -> Dict[str, Any]:
        """Loads the model run information from the 'run_info.json' file.

        Returns:
            Dict[str, Any]: The 'run_info.json' file content.
        """
        try:
            self.logger.info("Executing `load_run_info`...")

            run_info_file_path = self.run_info_file_path
            with open(run_info_file_path, "r") as file:
                model_run_info = json.load(file)

            self.logger.info("Execution of `load_run_info` complete.")
            return model_run_info
        except FileNotFoundError:
            self.logger.error(f"[load_run_info] The file {run_info_file_path} does not exist.")
            raise
        except Exception as e:
            self.logger.error(f"[load_run_info] Error loading JSON: {e}")
            raise

    def load_preprocessor(self, preprocessor_file_path: Path) -> ColumnTransformer:
        """Loads a preprocessor from a joblib file.

        Args:
            preprocessor_file_path (Path): Preprocessor file path.

        Returns:
            ColumnTransformer: Preprocessor.
        """
        try:
            self.logger.info("Executing `load_preprocessor`...")

            preprocessor = joblib.load(preprocessor_file_path)

            self.logger.info("Execution of `load_preprocessor` complete.")
            return preprocessor
        except FileNotFoundError:
            self.logger.error(f"[load_preprocessor] File not found at: {preprocessor_file_path}")
            raise
        except Exception as e:
            self.logger.error(f"[load_preprocessor] Error loading preprocessor: {e}")
            raise

    def build_pipeline(self) -> Pipeline:
        """Builds a pipeline containing the preprocessor and the model.

        Returns:
            Pipeline: The pipeline.
        """
        try:
            self.logger.info("Executing `build_pipeline`...")

            # Load run info
            run_info = self.load_run_info()
            model_name = run_info["model_name"]

            # Load model from the MLFlow registry
            model_uri = f"models:/{model_name}/{self.stage}"
            mlflow_model = mlflow.sklearn.load_model(model_uri)

            # Load preprocessor
            preprocessor_file_path = Path("models/preprocessor.joblib")
            preprocessor = self.load_preprocessor(preprocessor_file_path=preprocessor_file_path)

            # Create pipeline
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", mlflow_model),
                ]
            )

            self.logger.info("Execution of `build_pipeline` complete.")
            return pipeline
        except Exception as e:
            self.logger.error(f"[build_pipeline] Error: {e}")
            raise

    @property
    def model_pipeline(self) -> Pipeline:
        """Provides a cached instance of the pipeline (load once; reuse across requests).

        Returns:
            Pipeline: The pipeline.
        """
        try:
            self.logger.info("Executing `model_pipeline`...")

            if self._model_pipeline is None:
                self.logger.info("`model_pipeline` is None, building the pipeline...")
                self._model_pipeline = self.build_pipeline()
                self.logger.info("Pipeline is built.")

            self.logger.info("Execution of `model_pipeline` complete.")
            return self._model_pipeline
        except Exception as e:
            self.logger.error(f"[model_pipe property] Error: {e}")
            raise

    def predict(self, raw_df: pd.DataFrame) -> float:
        try:
            self.logger.info("Executing `predict`...")

            cleaned_df = self.data_cleaner.clean_data(raw_df=raw_df)
            self.logger.info("Data cleaned successfully.")

            # Prediction
            self.logger.info("Making prediction...")
            preds = self.model_pipeline.predict(cleaned_df)
            self.logger.info("Prediction complete.")

            self.logger.info("Execution of `predict` complete.")
            return preds[0]
        except Exception as e:
            self.logger.error(f"[predict] Error during prediction: {e}")
            raise


# Initialize the service
model_service = ModelService(
    run_info_file_path=Path("run_info.json"),
    # stage="Staging",
)

# FastAPI application
app = FastAPI()


@app.get("/")
def home() -> dict:
    """A simple welcome page.

    Returns:
        dict: Welcome message.
    """
    return {"message": "Welcome to the Swiggy Delivery Time Prediction App!"}


@app.post("/predict")
def do_predictions(data: Data) -> dict:
    """Does prediction on a single data row.

    Args:
        data (Data): Data.

    Returns:
        dict: Prediction.
    """
    try:
        logger.info("Executing `do_predictions`...")

        logger.info("Received predict request ...")

        # Convert input to pandas DataFrame (single row)
        pred_df = pd.DataFrame([data.dict()])

        # Model inference
        prediction = model_service.predict(pred_df)
        logger.info("Prediction response ready.")

        logger.info("Execution of `do_predictions` complete.")
        return {"time_taken_prediction": prediction}
    except Exception as e:
        logger.error(f"[do_predictions] Error: {e}")
        raise


# Run the server (local dev)
if __name__ == "__main__":
    uvicorn.run(
        app="app:app",
        host="0.0.0.0",
        port=8000,
        # reload=True,
    )
