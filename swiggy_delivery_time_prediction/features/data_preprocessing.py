import yaml
import joblib
import logging

import pandas as pd

from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder

set_config(transform_output="pandas")


# Create logger
logger = logging.getLogger("data_preprocessing")
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


class DataPreprocessor:
    def __init__(self) -> None:
        """Initializes a `DataPreprocessor` object."""
        self.logger = logging.getLogger("data_preprocessing")
        self.logger.info("Instantiating a `DataPreprocessor` object...")

        self.num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]
        self.nominal_cat_cols = [
            "weather",
            "type_of_order",
            "type_of_vehicle",
            "festival",
            "city_type",
            "order_day_is_weekend",
            "order_time_of_day",
        ]
        self.ordinal_cat_cols = ["traffic", "distance_type"]
        self.target_col = "time_taken"

        self.traffic_order = ["low", "medium", "high", "jam"]
        self.distance_type_order = ["short", "medium", "long", "very_long"]

        self.preprocessor = None

        self.logger.info("`DataPreprocessor` object instantiated.")

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
        except Exception as e:
            self.logger.error(f"[load_data] Error reading data at {data_path}: {e}")
            raise

    def drop_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Executing `drop_missing_values`...")

            self.logger.info(f"Shape of the data BEFORE dropping missing values: {df.shape}.")

            df_dropped = df.dropna()
            self.logger.info(
                f"Shape of the data AFTER dropping missing values: {df_dropped.shape}."
            )
            if df_dropped.isna().sum().sum() > 0:
                raise ValueError("The data has missing values even after dropping them.")

            self.logger.info("Execution of `drop_missing_values` complete.")
            return df_dropped
        except Exception as e:
            self.logger.error(f"[drop_missing_values] Error: {e}")
            raise

    def split_X_and_y(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into features (X) and target (y).

        Args:
            df (pd.DataFrame): Data.
            target_col (str): Target name.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (X, y)
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

    def join_X_and_y(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Joins features (X) and target (y) along the same index to produce a single data frame.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.

        Returns:
            pd.DataFrame: Joined data frame.
        """
        try:
            self.logger.info("Executing `join_X_and_y`...")

            joined_df = X.join(y, how="inner")

            self.logger.info("Execution of `join_X_and_y` complete.")
            return joined_df
        except Exception as e:
            self.logger.error(f"[join_X_and_y] Error: {e}")
            raise

    def create_preprocessor(self) -> ColumnTransformer:
        """Creates and returns a preprocessor that performs preprocessing on the numerical and categorical columns.

        Returns:
            ColumnTransformer: Preprocessor.
        """
        try:
            self.logger.info("Executing `create_preprocessor`...")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("scaler", MinMaxScaler(), self.num_cols),
                    (
                        "nominal_encoder",
                        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                        self.nominal_cat_cols,
                    ),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            categories=[self.traffic_order, self.distance_type_order],
                            encoded_missing_value=-999,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                        self.ordinal_cat_cols,
                    ),
                ],
                remainder="passthrough",
                n_jobs=-1,
                verbose_feature_names_out=False,
            )

            self.logger.info("Execution of `create_preprocessor` complete.")
            return preprocessor
        except Exception as e:
            self.logger.error(f"[create_preprocessor] Error: {e}")
            raise

    def train_preprocessor(
        self, preprocessor: ColumnTransformer, X: pd.DataFrame
    ) -> ColumnTransformer:
        """Fits the preprocessor on the features.

        Args:
            preprocessor (ColumnTransformer): Untrained preprocessor.
            X (pd.DataFrame): Features.

        Returns:
            ColumnTransformer: Fitted preprocessor.
        """
        try:
            self.logger.info("Executing `train_preprocessor`...")

            preprocessor.fit(X)

            self.logger.info("Execution of `train_preprocessor` complete.")
            return preprocessor
        except Exception as e:
            self.logger.error(f"[train_preprocessor] Error: {e}")
            raise

    def transform_data(self, preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the features (X) using the fitted preprocessor.

        Args:
            preprocessor (ColumnTransformer): Fitted preprocessor.
            X (pd.DataFrame): Features.

        Returns:
            pd.DataFrame: Transformed features.
        """
        try:
            self.logger.info("Executing `transform_data`...")

            transformed_df = preprocessor.transform(X)

            self.logger.info("Execution of `transform_data` complete.")
            return transformed_df
        except Exception as e:
            self.logger.error(f"[transform_data] Error: {e}")
            raise

    def save_preprocessor(
        self, preprocessor: ColumnTransformer, dir_path: Path, preprocessor_file_name: str
    ) -> None:
        """Saves the preprocessor.

        Args:
            preprocessor (ColumnTransformer): Fitted preprocessor.
            dir_path (Path): Path of the directory where the preprocessor should be saved.
            preprocessor_file_name (str): Name of the preprocessor file.
        """
        try:
            self.logger.info("Executing `save_preprocessor`...")

            dir_path.mkdir(exist_ok=True, parents=True)
            preprocessor_path = dir_path / preprocessor_file_name
            joblib.dump(preprocessor, preprocessor_path)

            self.logger.info("Execution of `save_preprocessor` complete.")
        except Exception as e:
            self.logger.error(f"[save_transformer] Error: {e}")
            raise

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """Saves data to a CSV file.

        Args:
            df (pd.DataFrame): Data.
            file_path (Path): Path of the file to be saved.
        """
        try:
            self.logger.info("Executing `save_data`...")

            df.to_csv(file_path, index=False)

            self.logger.info("Execution of `save_data` complete.")
        except Exception as e:
            self.logger.error(f"[save_data] Error: {e}")
            raise

    def preprocess_data(
        self,
        train_file_path: Path,
        test_file_path: Path,
        processed_dir_path: Path,
        preprocessor_dir_path: Path,
        preprocessor_file_name: str,
    ) -> None:
        self.logger.info("Executing `preprocess_data`...")

        # Loading the training and test data
        train_df = self.load_data(path=train_file_path)
        test_df = self.load_data(path=test_file_path)

        # Dropping the missing values
        train_df = self.drop_missing_values(df=train_df)
        test_df = self.drop_missing_values(df=test_df)

        # X-y split
        X_train, y_train = self.split_X_and_y(df=train_df, target_col=self.target_col)
        X_test, y_test = self.split_X_and_y(df=test_df, target_col=self.target_col)

        # Creating and training preprocessor
        self.preprocessor = self.create_preprocessor()
        self.preprocessor = self.train_preprocessor(preprocessor=self.preprocessor, X=X_train)

        # Transforming the data using the preprocessor
        X_train_transformed = self.transform_data(preprocessor=self.preprocessor, X=X_train)
        X_test_transformed = self.transform_data(preprocessor=self.preprocessor, X=X_test)

        # Joining X-y to form the transformed training and transformed test sets
        transformed_train_df = self.join_X_and_y(X=X_train_transformed, y=y_train)
        transformed_test_df = self.join_X_and_y(X=X_test_transformed, y=y_test)

        # Saving the transformed sets
        processed_dir_path.mkdir(exist_ok=True, parents=True)
        transformed_train_file_path = processed_dir_path / "transformed_train.csv"
        transformed_test_file_path = processed_dir_path / "transformed_test.csv"
        self.save_data(df=transformed_train_df, file_path=transformed_train_file_path)
        self.save_data(df=transformed_test_df, file_path=transformed_test_file_path)

        # Saving the preprocessor
        self.save_preprocessor(
            preprocessor=self.preprocessor,
            dir_path=preprocessor_dir_path,
            preprocessor_file_name=preprocessor_file_name,
        )

        self.logger.info("Execution of `preprocess_data` complete.")


if __name__ == "__main__":

    # Root path
    root_path = Path(__file__).parent.parent.parent

    # Paths of train.csv and test.csv
    train_file_path = root_path / "data" / "interim" / "train.csv"
    test_file_path = root_path / "data" / "interim" / "test.csv"

    # Path of 'processed' directory
    processed_dir_path = root_path / "data" / "processed"

    # Create the 'processed' directory if it does not exist
    processed_dir_path.mkdir(exist_ok=True, parents=True)

    # Directory path and name of the preprocessor
    preprocessor_dir_path = root_path / "models"
    preprocessor_file_name = "preprocessor.joblib"

    # Create the 'models' directory if it does not exist
    preprocessor_dir_path.mkdir(exist_ok=True, parents=True)

    # Initiating data preprocessing
    data_preprocessor = DataPreprocessor()
    data_preprocessor.preprocess_data(
        train_file_path=train_file_path,
        test_file_path=test_file_path,
        processed_dir_path=processed_dir_path,
        preprocessor_dir_path=preprocessor_dir_path,
        preprocessor_file_name=preprocessor_file_name,
    )
