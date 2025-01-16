import json
import mlflow
import dagshub
import logging

from pathlib import Path
from typing import Dict, Any
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion


# Create logger
logger = logging.getLogger("model_registration")
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


# Initialize dagshub for MLFlow
dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True,
)

# Set the MLFlow tracking server and experiment name
mlflow.set_tracking_uri(
    "https://dagshub.com/SushrutGaikwad/swiggy-delivery-time-prediction.mlflow"
)


class ModelRegistrar:
    def __init__(self) -> None:
        self.logger = logging.getLogger("model_registration")
        self.logger.info("Instantiating a `ModelRegistrar` object...")

        self.client = MlflowClient()

        self.logger.info("`ModelRegistrar` object instantiated.")

    def load_run_info(self, file_path: Path) -> Dict[str, Any]:
        """Loads the model run information from a JSON file.

        Args:
            file_path (Path): Path of the 'run_info.json' file.

        Returns:
            Dict[str, Any]: The 'run_info.json' file content.
        """
        try:
            self.logger.info("Executing `load_run_info`...")

            with open(file_path, "r") as file:
                model_run_info = json.load(file)

            self.logger.info("Execution of `load_run_info` complete.")
            return model_run_info
        except FileNotFoundError:
            self.logger.error(f"[load_run_info] The file {file_path} does not exist.")
            raise
        except Exception as e:
            self.logger.error(f"[load_run_info] Error loading JSON: {e}")
            raise

    def register_model(self, run_id: str, model_name: str) -> ModelVersion:
        """Registers the model in MLFlow model registry.

        Args:
            run_id (str): MLFlow run ID.
            model_name (str): Name of the model in MLFlow to register.

        Returns:
            ModelVersion: Model version object from MLFlow after registration.
        """
        try:
            self.logger.info("Executing `register_model`...")

            model_registry_path = f"runs:/{run_id}/{model_name}"
            model_version = mlflow.register_model(
                model_uri=model_registry_path,
                name=model_name,
            )
            self.logger.info(f"Registered model version: {model_version.version}")

            self.logger.info("Execution of `register_model` complete.")
            return model_version
        except Exception as e:
            self.logger.error(f"[register_model] Error: {e}")
            raise

    def transition_model_stage(
        self, model_name: str, model_version: str, stage: str = "Staging"
    ) -> None:
        """Transitions the model version to a specified stage.

        Args:
            model_name (str): Registered model name.
            model_version (str): Model version to transition.
            stage (str, optional): The stage to transition to. Defaults to "Staging".
        """
        try:
            self.logger.info("Executing `transition_model_stage`...")

            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage,
            )
            self.logger.info(f"Model version {model_version} transitioned to stage '{stage}'.")

            self.logger.info("Execution of `transition_model_stage` complete.")
        except Exception as e:
            self.logger.error(f"[transition_model_stage] Error: {e}")
            raise

    def register_and_stage_model(self, run_info_file_path: Path, stage: str = "Staging") -> None:
        """Registers a model and transitions it to a specified stage.

        Args:
            run_info_file_path (Path): Path of the 'run_info.json' file.
            stage (str, optional): The stage to transition to. Defaults to "Staging".
        """
        self.logger.info("Executing `register_and_stage_model`...")

        # Load the model run info
        model_run_info = self.load_run_info(file_path=run_info_file_path)
        run_id = model_run_info["run_id"]
        model_name = model_run_info["model_name"]

        # Register model
        model_version = self.register_model(run_id=run_id, model_name=model_name)
        registered_version = model_version.version
        registered_name = model_version.name

        # Transition the model stage
        self.transition_model_stage(
            model_name=registered_name, model_version=registered_version, stage=stage
        )

        self.logger.info("Execution of `register_and_stage_model` complete.")


if __name__ == "__main__":

    # Root path
    root_path = Path(__file__).parent.parent.parent
    run_info_file_path = root_path / "run_info.json"

    # Initiate model registration
    registrar = ModelRegistrar()
    registrar.register_and_stage_model(run_info_file_path=run_info_file_path, stage="Staging")
