import yaml
import joblib
import logging

import pandas as pd

from pathlib import Path
from typing import Dict, Any, Tuple
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor


# Create logger
logger = logging.getLogger("model_training")
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


class ModelTrainer:
    def __init__(self, target_col: str = "time_taken") -> None:
        """Initializes a `ModelTrainer` object.

        Args:
            target_col (str, optional): Name of the target column. Defaults to "time_taken".
        """
        self.logger = logging.getLogger("model_training")
        self.logger.info("Instantiating a `ModelTrainer` object...")

        self.target_col = target_col

        self.logger.info("`ModelTrainer` object instantiated.")

    def read_params(self, file_path: Path) -> Dict[str, Any]:
        """Reads parameters from a YAML file.

        Args:
            file_path (Path): YAML file path.

        Returns:
            Dict[str, Any]: Content of the YAML file.
        """
        try:
            self.logger.info("Executing `read_params`...")

            with open(file_path, "r") as file:
                params = yaml.safe_load(file)

            self.logger.info("Execution of `read_params` complete.")
            return params
        except Exception as e:
            self.logger.error(f"[read_params] Error: {e}")
            raise

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

    def train_model(self, model: object, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Fits the given model on the given training data.

        Args:
            model (object): An sklearn-compatible regressor or pipeline.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            object: Fitted model.
        """
        try:
            self.logger.info("Executing `train_model`...")

            model.fit(X_train, y_train)

            self.logger.info("Execution of `train_model` complete.")
            return model
        except Exception as e:
            self.logger.error(f"[train_model] Error: {e}")
            raise

    def save_model(self, model: object, models_dir_path: Path, model_file_name: str) -> None:
        """Saves a trained model.

        Args:
            model (object): Trained model.
            models_dir_path (Path): Path of the directory where the model should be saved.
            model_file_name (str): Model file name.
        """
        try:
            self.logger.info("Executing `save_model`...")

            models_dir_path.mkdir(exist_ok=True, parents=True)
            model_file_path = models_dir_path / model_file_name
            joblib.dump(model, model_file_path)
            self.logger.info(f"Model saved to {model_file_path}")

            self.logger.info("Execution of `save_model` complete.")
        except Exception as e:
            self.logger.error(f"[save_model] Error: {e}")
            raise

    def save_transformer(
        self, transformer: object, transformer_dir_path: Path, transformer_file_name: str
    ) -> None:
        """Saves a transformer.

        Args:
            transformer (object): Transformer object to save.
            transformer_dir_path (Path): Path of the directory where the transformer object should be saved.
            transformer_file_name (str): Transformer file name.
        """
        try:
            self.logger.info("Executing `save_transformer`...")

            transformer_dir_path.mkdir(exist_ok=True, parents=True)
            transformer_file_path = transformer_dir_path / transformer_file_name
            joblib.dump(transformer, transformer_file_path)
            self.logger.info(f"Transformer saved to {transformer_file_path}")

            self.logger.info("Execution of `save_transformer` complete.")
        except Exception as e:
            self.logger.error(f"[save_transformer] Error: {e}")
            raise

    def train_and_save_model(
        self,
        transformed_train_file_path: Path,
        params_file_path: Path,
        models_dir_path: Path,
        model_file_name: str = "model.joblib",
        stacking_regressor_file_name: str = "stacking_regressor.joblib",
        transformer_file_name: str = "power_transformer.joblib",
    ) -> None:
        # Loading the training data
        train_df = self.load_data(data_path=transformed_train_file_path)

        # X-y split
        X_train, y_train = self.split_X_and_y(df=train_df, target_col=self.target_col)

        # Read parameters from 'params.yaml'
        model_params = self.read_params(file_path=params_file_path)["Model_Training"]

        # Extract random forest parameters to build random forest regressor
        rf_params = model_params["Random_Forest"]
        rf = RandomForestRegressor(**rf_params)

        # Extract LightGBM parameters to build LightGBM regressor
        lgbm_params = model_params["LightGBM"]
        lgbm = LGBMRegressor(**lgbm_params)

        # Create the meta-model
        lr = LinearRegression()

        # Create the stacking regressor
        sr = StackingRegressor(
            estimators=[
                ("rf_model", rf),
                ("lgbm_model", lgbm),
            ],
            final_estimator=lr,
            cv=5,
            n_jobs=-1,
        )

        # Create the power transformer for the target
        pt = PowerTransformer(method="yeo-johnson")

        # Wrapping the stacking regressor in a TransformedTargetRegressor to
        # build the final model
        model = TransformedTargetRegressor(regressor=sr, transformer=pt)

        # Training the model
        model = self.train_model(model=model, X_train=X_train, y_train=y_train)

        # Saving the model
        self.save_model(
            model=model, models_dir_path=models_dir_path, model_file_name=model_file_name
        )

        # Extracting and saving the stacking regressor and transformer separately
        stacking_model = model.regressor_
        self.save_model(
            model=stacking_model,
            models_dir_path=models_dir_path,
            model_file_name=stacking_regressor_file_name,
        )

        transformer = model.transformer_
        self.save_model(
            model=transformer,
            models_dir_path=models_dir_path,
            model_file_name=transformer_file_name,
        )


if __name__ == "__main__":

    # Root path
    root_path = Path(__file__).parent.parent.parent

    # Paths
    transformed_train_file_path = root_path / "data" / "processed" / "transformed_train.csv"
    params_file_path = root_path / "params.yaml"
    models_dir_path = root_path / "models"

    # Initiate model training
    trainer = ModelTrainer(target_col="time_taken")
    trainer.train_and_save_model(
        transformed_train_file_path=transformed_train_file_path,
        params_file_path=params_file_path,
        models_dir_path=models_dir_path,
        model_file_name="model.joblib",
        stacking_regressor_file_name="stacking_regressor.joblib",
        transformer_file_name="power_transformer.joblib",
    )
