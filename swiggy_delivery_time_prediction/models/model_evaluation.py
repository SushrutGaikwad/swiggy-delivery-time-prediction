import json
import joblib
import dagshub
import mlflow
import logging

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Tuple
from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score


# Create logger
logger = logging.getLogger("model_evaluation")
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
mlflow.set_experiment("DVC Pipeline")


class ModelEvaluator:
    def __init__(self, target_col: str = "time_taken"):
        self.logger = logging.getLogger("model_evaluation")
        self.logger.info("Instantiating a `ModelEvaluator` object...")

        self.target_col = target_col

        self.logger.info("`ModelEvaluator` object instantiated.")

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Loads the data from a given path.

        Args:
            path (Path): Path of the data.

        Returns:
            pd.DataFrame: Data.
        """
        try:
            self.logger.info("Executing `load_data`...")

            df = pd.read_csv(data_path)

            self.logger.info("Execution of `load_data` complete.")
            return df
        except FileNotFoundError:
            self.logger.error(f"[load_data] The file {data_path} does not exist.")
            raise
        except Exception as e:
            self.logger.error(f"[load_data] Error reading data at {data_path}: {e}")
            raise

    def split_X_and_y(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Splits the data into features (X) and target (y).

        Args:
            df (pd.DataFrame): Data.
            target_col (str): Target name.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        try:
            self.logger.info("Executing `split_X_and_y`...")

            X = df.drop(columns=[target_col])
            y = df[target_col]

            self.logger.info("Execution of `split_X_and_y` complete.")
            return X, y
        except Exception as e:
            self.logger.error(f"[split_X_and_y] Error: {e}")
            raise

    def load_trained_model(self, trained_model_file_path: Path) -> RegressorMixin:
        """Loads a trained model.

        Args:
            trained_model_file_path (Path): Path of the trained model.

        Returns:
            RegressorMixin: Trained model.
        """
        try:
            self.logger.info("Executing `load_trained_model`...")

            trained_model = joblib.load(trained_model_file_path)

            self.logger.info("Execution of `load_trained_model` complete.")
            return trained_model
        except FileNotFoundError:
            self.logger.error(
                f"[load_trained_model] The file {trained_model_file_path} does not exist."
            )
            raise
        except Exception as e:
            self.logger.error(f"[load_trained_model] Error loading model: {e}")
            raise

    def compute_predictions(self, trained_model: RegressorMixin, X: pd.DataFrame):
        """Uses the trained model and compute predictions on features (X).

        Args:
            trained_model (RegressorMixin): Trained model.
            X (pd.DataFrame): Features.

        Returns:
            np.ndarray or pd.Series: Predictions.
        """
        try:
            self.logger.info("Executing `compute_predictions`...")

            predictions = trained_model.predict(X)

            self.logger.info("Execution of `compute_predictions` complete.")
            return predictions
        except Exception as e:
            self.logger.error(f"[compute_predictions] Error during prediction: {e}")
            raise

    def calculate_metrics(self, y_true_train, y_pred_train, y_true_test, y_pred_test) -> dict:
        """Calculates MAE and R2 for the training and test predictions.

        Args:
            y_true_train (array-like): Training ground truth (correct) target values.
            y_pred_train (array-like): Estimated training target values.
            y_true_test (array-like): Test ground truth (correct) target values.
            y_pred_test (array-like): Estimated test target values.

        Returns:
            dict: MAE and R2 for training and test predictions.
        """
        try:
            self.logger.info("Executing `calculate_metrics`...")

            train_mae = mean_absolute_error(y_true=y_true_train, y_pred=y_pred_train)
            test_mae = mean_absolute_error(y_true=y_true_test, y_pred=y_pred_test)
            train_r2 = r2_score(y_true=y_true_train, y_pred=y_pred_train)
            test_r2 = r2_score(y_true=y_true_test, y_pred=y_pred_test)

            metrics = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2,
            }

            self.logger.info("Execution of `calculate_metrics` complete.")
            return metrics
        except Exception as e:
            self.logger.error(f"[calculate_metrics] Error calculating metrics: {e}")
            raise

    def cross_validate_trained_model(
        self,
        trained_model: RegressorMixin,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = "neg_mean_absolute_error",
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Performs cross-validation on the training data using a trained model.

        Args:
            trained_model (RegressorMixin): Trained model.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            cv (int, optional): Determines the cross-validation splitting strategy. Defaults to 5.
            scoring (str, optional): Scoring parameter. Defaults to "neg_mean_absolute_error".
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.

        Returns:
            np.ndarray: Cross-validation scores.
        """
        try:
            self.logger.info("Executing `cross_validate_trained_model`...")

            cv_scores = cross_val_score(
                estimator=trained_model,
                X=X_train,
                y=y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
            )

            self.logger.info("Execution of `cross_validate_trained_model` complete.")
            return cv_scores
        except Exception as e:
            self.logger.error(f"[cross_validate_trained_model] Error during cross-validation: {e}")
            raise

    def log_trained_model_info(
        self, json_file_path: Path, run_id: str, artifact_uri: str, trained_model_name: str
    ) -> None:
        try:
            self.logger.info("Executing `log_trained_model_info`...")

            info_dict = {
                "run_id": run_id,
                "artifact_path": artifact_uri,
                "model_name": trained_model_name,
            }

            with open(json_file_path, "w") as file:
                json.dump(info_dict, file, indent=4)

            self.logger.info("Execution of `log_trained_model_info` complete.")
        except Exception as e:
            self.logger.error(f"[log_trained_model_info] Error saving model info: {e}")
            raise

    def evaluate_trained_model(
        self,
        transformed_train_file_path: Path,
        transformed_test_file_path: Path,
        trained_model_file_path: Path,
        json_run_info_file_path: Path,
    ) -> None:
        self.logger.info("Executing `evaluate_trained_model`...")

        # Loading the transformed training and transformed test data
        train_df = self.load_data(data_path=transformed_train_file_path)
        test_df = self.load_data(data_path=transformed_test_file_path)

        # X-y split
        X_train, y_train = self.split_X_and_y(df=train_df, target_col=self.target_col)
        X_test, y_test = self.split_X_and_y(df=test_df, target_col=self.target_col)

        # Load the trained model
        trained_model = self.load_trained_model(trained_model_file_path=trained_model_file_path)

        # Make predictions
        y_pred_train = self.compute_predictions(trained_model=trained_model, X=X_train)
        y_pred_test = self.compute_predictions(trained_model=trained_model, X=X_test)

        # Calculate metrics
        metrics = self.calculate_metrics(
            y_true_train=y_train,
            y_pred_train=y_pred_train,
            y_true_test=y_test,
            y_pred_test=y_pred_test,
        )
        train_mae = metrics["train_mae"]
        test_mae = metrics["test_mae"]
        train_r2 = metrics["train_r2"]
        test_r2 = metrics["test_r2"]

        # Cross-validation
        cv_scores = self.cross_validate_trained_model(
            trained_model=trained_model,
            X_train=X_train,
            y_train=y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        # Log with MLFlow
        with mlflow.start_run() as run:
            # Basic tagging
            mlflow.set_tag("model", "Food Delivery Time Regressor")

            # Log parameters (model params, if accessible via `get_params()`)
            if hasattr(trained_model, "get_params"):
                mlflow.log_params(trained_model.get_params())

            # Log main metrics
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("mean_cv_scores", -(cv_scores.mean()))

            # Log individual cv scores
            mlflow.log_metrics(
                {f"CV {num}": -score for num, score in enumerate(cv_scores, start=1)}
            )

            # Log the training and test data as MLFlow inputs
            train_df_input = mlflow.data.from_pandas(train_df, targets=self.target_col)
            test_df_input = mlflow.data.from_pandas(test_df, targets=self.target_col)
            mlflow.log_input(dataset=train_df_input, context="training")
            mlflow.log_input(dataset=test_df_input, context="validation")

            # Infer a model signature for logging
            sample_X = X_train.sample(min(20, len(X_train)), random_state=42)
            sample_y = trained_model.predict(sample_X)
            model_signature = mlflow.models.infer_signature(sample_X, sample_y)

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=trained_model,
                artifact_path="swiggy_delivery_time_prediction_model",
                signature=model_signature,
            )

            artifact_uri = mlflow.get_artifact_uri()

        run_id = run.info.run_id
        model_name = "swiggy_delivery_time_prediction_model"

        self.log_trained_model_info(
            json_file_path=json_run_info_file_path,
            run_id=run_id,
            artifact_uri=artifact_uri,
            trained_model_name=model_name,
        )


if __name__ == "__main__":

    # Root path
    root_path = Path(__file__).parent.parent.parent

    # Transformed training and transformed test data paths
    transformed_train_file_path = root_path / "data" / "processed" / "transformed_train.csv"
    transformed_test_file_path = root_path / "data" / "processed" / "transformed_test.csv"

    # Model path
    trained_model_file_path = root_path / "models" / "model.joblib"

    # JSON file path
    json_run_info_file_path = root_path / "run_info.json"

    # Initiate model evaluation
    evaluator = ModelEvaluator(target_col="time_taken")
    evaluator.evaluate_trained_model(
        transformed_train_file_path=transformed_train_file_path,
        transformed_test_file_path=transformed_test_file_path,
        trained_model_file_path=trained_model_file_path,
        json_run_info_file_path=json_run_info_file_path,
    )
