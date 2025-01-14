import yaml
import logging

import pandas as pd

from typing import Tuple, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split


# Create logger
logger = logging.getLogger("training_test_split")
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


class TrainingTestSplitter:
    def __init__(self) -> None:
        """Initializes a `TrainingTestSplitter` object."""
        self.logger = logging.getLogger("training_test_split")
        self.logger.info("Instantiating a `TrainingTestSplitter` object...")
        self.logger.info("`TrainingTestSplitter` object instantiated.")

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

    def load_data(self, path: Path) -> pd.DataFrame:
        """Loads the data from a given path.

        Args:
            path (Path): Path of the data.

        Returns:
            pd.DataFrame: Data.
        """
        try:
            self.logger.info("Executing `load_data`...")

            df = pd.read_csv(path)

            self.logger.info("Execution of `load_data` complete.")
            return df
        except Exception as e:
            self.logger.error(f"[load_data] Error: {e}")
            raise

    def training_test_split(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and test sets.

        Args:
            df (pd.DataFrame): Data.
            test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training set and test set.
        """
        try:
            self.logger.info("Executing `training_test_split`...")

            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )

            self.logger.info("Execution of `training_test_split` complete.")
            return train_df, test_df
        except Exception as e:
            self.logger.error(f"[training_test_split] Error: {e}")
            raise

    def save_df(self, df: pd.DataFrame, path: Path) -> None:
        """Saves the data to a CSV file.

        Args:
            df (pd.DataFrame): Data.
            path (Path): Path to save.
        """
        try:
            self.logger.info("Executing `save_df`...")

            df.to_csv(path, index=False)

            self.logger.info("Execution of `save_df` complete.")
        except Exception as e:
            self.logger.error(f"[save_df] Error: {e}")
            raise

    def split_and_save(
        self,
        cleaned_data_path: Path,
        params_file_path: Path,
        train_file_path: Path,
        test_file_path: Path,
        params_section: str = "Train_Test_Split",
    ) -> None:
        """Loads the data, performs the training and test split, and saves the training and test sets.

        Args:
            cleaned_data_path (Path): Path of cleaned data.
            params_file_path (Path): Path of 'params.yaml' file.
            train_file_path (Path): Path of training data.
            test_file_path (Path): Path of test data.
            params_section (str, optional): The key of the YAML file that holds relevant parameters. Defaults to "Train_Test_Split".
        """
        self.logger.info("Executing `split_and_save`...")

        df = self.load_data(path=cleaned_data_path)

        params = self.read_params(file_path=params_file_path)
        test_size = params[params_section]["test_size"]
        random_state = params[params_section]["random_state"]

        train_df, test_df = self.training_test_split(
            df=df, test_size=test_size, random_state=random_state
        )

        self.save_df(df=train_df, path=train_file_path)
        self.save_df(df=test_df, path=test_file_path)

        self.logger.info("Execution of `split_and_save` complete.")


if __name__ == "__main__":

    # Root path
    root_path = Path(__file__).parent.parent.parent

    # Cleaned data path
    cleaned_data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"

    # Training and test data directory path
    train_test_dir = root_path / "data" / "interim"

    # Create the 'interim' directory if it does not exist
    train_test_dir.mkdir(exist_ok=True, parents=True)

    # Training and test data file names
    train_file_name = "train.csv"
    test_file_name = "test.csv"

    # Paths of training and test data files
    train_file_path = train_test_dir / train_file_name
    test_file_path = train_test_dir / test_file_name

    # 'params.yaml' file path
    params_file_path = root_path / "params.yaml"

    train_test_splitter = TrainingTestSplitter()
    train_test_splitter.split_and_save(
        cleaned_data_path=cleaned_data_path,
        params_file_path=params_file_path,
        train_file_path=train_file_path,
        test_file_path=test_file_path,
        params_section="Train_Test_Split",
    )
