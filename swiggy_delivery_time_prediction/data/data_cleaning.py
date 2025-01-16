import logging

import numpy as np
import pandas as pd

from pathlib import Path


# Create logger
logger = logging.getLogger("data_cleaning")
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


class DataCleaner:
    def __init__(self) -> None:
        """Initializes a `DataCleaner` object."""
        self.logger = logging.getLogger("data_cleaning")
        self.logger.info("Instantiating a `DataCleaner` object...")
        self.rename_mapping = {
            "delivery_person_id": "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken",
        }
        self.cols_to_drop = [
            "rider_id",
            "restaurant_latitude",
            "restaurant_longitude",
            "delivery_latitude",
            "delivery_longitude",
            "order_date",
            "order_time_hour",
            "order_day",
            "city_name",
            "order_day_of_week",
            "order_month",
        ]
        self.logger.info("`DataCleaner` object instantiated.")

    def remove_trailing_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes trailing spaces from all the entries of the data in the object-type columns.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after the removal.
        """
        try:
            self.logger.info("Executing `remove_trailing_spaces`...")

            object_type_cols = df.select_dtypes(include=["object"]).columns
            for col in object_type_cols:
                df[col] = df[col].str.strip()

            self.logger.info("Execution of `remove_trailing_spaces` complete.")
            return df
        except Exception as e:
            self.logger.error(f"[remove_trailing_spaces] Error: {e}")
            raise

    def lowercase_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts all the entries of the data in the object-type columns to lowercase.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after the conversion.
        """
        try:
            self.logger.info("Executing `lowercase_strings`...")

            object_type_cols = df.select_dtypes(include=["object"]).columns
            for col in object_type_cols:
                if col != "Delivery_person_ID":
                    df[col] = df[col].str.lower()

            self.logger.info("Execution of `lowercase_strings` complete.")
            return df
        except Exception as e:
            self.logger.error(f"[lowercase_strings] Error: {e}")
            raise

    def replace_string_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces all the entries of the data in the object-type columns containing the string "nan" to `np.nan`.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after replacement.
        """
        try:
            self.logger.info("Executing `replace_string_nans`...")

            object_type_cols = df.select_dtypes(include=["object"]).columns
            df[object_type_cols] = df[object_type_cols].map(
                lambda x: np.nan if isinstance(x, str) and "nan" in x else x
            )

            self.logger.info("Execution of `replace_string_nans` complete.")
            return df
        except Exception as e:
            self.logger.error(f"[replace_string_nans] Error: {e}")
            raise

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames all the columns of the data.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after renaming.
        """
        try:
            self.logger.info("Executing `rename_columns`...")

            df = df.rename(str.lower, axis=1).rename(self.rename_mapping, axis=1)

            self.logger.info("Execution of `rename_columns` complete.")
            return df
        except Exception as e:
            self.logger.error(f"[rename_columns] Error: {e}")
            raise

    def drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops the 'id' column from the data.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after dropping.
        """
        try:
            self.logger.info("Executing `drop_id_column`...")

            df.drop(columns=["id"], inplace=True)

            self.logger.info("Execution of `drop_id_column` complete.")
            return df
        except KeyError:
            self.logger.error("[drop_id_column] Column 'id' not found.")
            raise
        except Exception as e:
            self.logger.error(f"[drop_id_column] Error dropping 'id': {e}")
            raise

    def drop_minors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops data of all the minors (i.e., people with age < 18).

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after dropping.
        """
        try:
            self.logger.info("Executing `drop_minors`...")

            df["age"] = df["age"].astype(float)
            minors_idxs = df.loc[df["age"] < 18].index
            df.drop(minors_idxs, inplace=True)

            self.logger.info("Execution of `drop_minors` complete.")
            return df
        except KeyError:
            self.logger.error("[drop_minors] Column 'age' not found.")
            raise
        except ValueError as ve:
            self.logger.error(f"[drop_minors] Could not convert 'age' to float: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"[drop_minors] Error dropping minors: {e}")
            raise

    def drop_six_star_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops the data with 6-star ratings.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after dropping.
        """
        try:
            self.logger.info("Executing `drop_six_star_ratings`...")

            df["ratings"] = df["ratings"].astype(float)
            six_star_ratings_idxs = df.loc[df["ratings"] == 6].index
            df.drop(six_star_ratings_idxs, inplace=True)

            self.logger.info("Execution of `drop_six_star_ratings` complete.")
            return df
        except KeyError:
            self.logger.error("[drop_six_star_ratings] Column 'ratings' not found.")
            raise
        except ValueError as ve:
            self.logger.error(
                f"[drop_six_star_ratings] Could not convert 'ratings' to float: {ve}"
            )
            raise
        except Exception as e:
            self.logger.error(f"[drop_six_star_ratings] Error dropping 6-star ratings: {e}")
            raise

    def create_city_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates the column 'city_name' using the column 'rider_id'.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after creation.
        """
        try:
            self.logger.info("Executing `create_city_name`...")

            df["city_name"] = df["rider_id"].str.split("RES").str.get(0)

            self.logger.info("Execution of `create_city_name` complete.")
            return df
        except KeyError:
            self.logger.error(
                "[create_city_name] Column 'rider_id' not found; cannot create 'city_name'."
            )
            raise
        except Exception as e:
            self.logger.error(f"[create_city_name] Error creating 'city_name': {e}")
            raise

    def clean_location_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the location columns.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after cleaning.
        """
        try:
            self.logger.info("Executing `clean_location_columns`...")

            location_columns = df.columns[3:7]
            df[location_columns] = df[location_columns].abs()
            df[location_columns] = df[location_columns].map(lambda x: x if x >= 1 else np.nan)

            self.logger.info("Execution of `clean_location_columns` complete.")
            return df
        except IndexError as ie:
            self.logger.error(f"[clean_location_columns] location_columns indexing error: {ie}")
            raise
        except Exception as e:
            self.logger.error(f"[clean_location_columns] Error cleaning location columns: {e}")
            raise

    def compute_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes the haversine distance using the location columns and saves it in a new column called 'distance'.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after creation.
        """
        try:
            self.logger.info("Executing `compute_distance`...")

            location_columns = df.columns[3:7]
            lat1 = df[location_columns[0]]
            lon1 = df[location_columns[1]]
            lat2 = df[location_columns[2]]
            lon2 = df[location_columns[3]]

            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = (np.sin(dlat / 2.0) ** 2) + (np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371 * c

            df["distance"] = distance

            self.logger.info("Execution of `compute_distance` complete.")
            return df
        except KeyError:
            self.logger.error(
                "[compute_distance] Some location columns not found; cannot compute distance."
            )
            raise
        except Exception as e:
            self.logger.error(f"[compute_distance] Error computing distance: {e}")
            raise

    def bin_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a new column 'distance_type', which is a binned version of the column 'distance'.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after binning.
        """
        try:
            self.logger.info("Executing `bin_distance`...")

            bins = [0, 5, 10, 15, np.inf]
            labels = ["short", "medium", "long", "very_long"]
            df["distance_type"] = pd.cut(df["distance"], bins=bins, right=False, labels=labels)

            self.logger.info("Execution of `bin_distance` complete.")
            return df
        except KeyError:
            self.logger.error("[bin_distance] Column 'distance' not found.")
            raise
        except Exception as e:
            self.logger.error(f"[bin_distance] Error binning distance: {e}")
            raise

    def clean_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and creates new date time columns.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after cleaning and creation.
        """
        try:
            self.logger.info("Executing `clean_datetime_columns`...")

            df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
            df["order_day"] = df["order_date"].dt.day
            df["order_month"] = df["order_date"].dt.month
            df["order_day_of_week"] = df["order_date"].dt.day_name().str.lower()
            df["order_day_is_weekend"] = (
                df["order_day_of_week"].isin(["saturday", "sunday"]).astype(int)
            )

            df["order_time"] = pd.to_datetime(df["order_time"], format="%H:%M:%S", errors="coerce")
            df["order_picked_time"] = pd.to_datetime(df["order_picked_time"], format="%H:%M:%S")

            df["pickup_time_minutes"] = (
                df["order_picked_time"] - df["order_time"]
            ).dt.seconds / 60

            df["order_time_hour"] = df["order_time"].dt.hour

            condlist = [
                (df["order_time_hour"].between(6, 12, inclusive="left")),
                (df["order_time_hour"].between(12, 17, inclusive="left")),
                (df["order_time_hour"].between(17, 20, inclusive="left")),
                (df["order_time_hour"].between(20, 24, inclusive="left")),
            ]
            choicelist = ["morning", "afternoon", "evening", "night"]
            default = "after_midnight"
            time_of_day_info = pd.Series(
                np.select(condlist=condlist, choicelist=choicelist, default=default),
                index=df.index,
            )
            df["order_time_of_day"] = time_of_day_info.where(df["order_time"].notna(), np.nan)

            df.drop(columns=["order_time", "order_picked_time"], inplace=True)

            self.logger.info("Execution of `clean_datetime_columns` complete.")
            return df
        except KeyError as ke:
            self.logger.error(f"[clean_datetime_columns] Missing datetime columns: {ke}")
            raise
        except Exception as e:
            self.logger.error(f"[clean_datetime_columns] Error cleaning datetime columns: {e}")
            raise

    def clean_weather_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the column 'weather'.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after cleaning.
        """
        try:
            self.logger.info("Executing `clean_weather_column`...")

            df["weather"] = df["weather"].str.replace("conditions ", "")

            self.logger.info("Execution of `clean_weather_column` complete.")
            return df
        except KeyError:
            self.logger.error("[clean_weather_column] Column 'weather' not found.")
            raise
        except Exception as e:
            self.logger.error(f"[clean_weather_column] Error cleaning 'weather': {e}")
            raise

    def convert_multiple_deliveries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts the column 'multiple_deliveries' to `float`.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after conversion.
        """
        try:
            self.logger.info("Executing `convert_multiple_deliveries`...")

            df["multiple_deliveries"] = df["multiple_deliveries"].astype(float)

            self.logger.info("Execution of `convert_multiple_deliveries` complete.")
            return df
        except KeyError:
            self.logger.error("[convert_multiple_deliveries] 'multiple_deliveries' not found.")
            raise
        except ValueError as ve:
            self.logger.error(f"[convert_multiple_deliveries] Could not convert to float: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"[convert_multiple_deliveries] Error: {e}")
            raise

    def clean_time_taken_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the column 'time_taken'.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after cleaning.
        """
        try:
            self.logger.info("Executing `clean_time_taken_column`...")

            df["time_taken"] = df["time_taken"].str.replace("(min) ", "").astype(int)

            self.logger.info("Execution of `clean_time_taken_column` complete.")
            return df
        except KeyError:
            self.logger.error("[clean_time_taken_column] Column 'time_taken' not found.")
            raise
        except ValueError as ve:
            self.logger.error(f"[clean_time_taken_column] Could not convert to int: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"[clean_time_taken_column] Error: {e}")
            raise

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unnecessary columns from the data.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.DataFrame: Data after dropping.
        """
        try:
            self.logger.info("Executing `drop_unnecessary_columns`...")

            df.drop(columns=self.cols_to_drop, inplace=True, errors="ignore")

            self.logger.info("Execution of `drop_unnecessary_columns` complete.")
            return df
        except Exception as e:
            self.logger.error(f"[drop_final_columns] Error dropping columns: {e}")
            raise

    def clean_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the raw data.

        Args:
            raw_df (pd.DataFrame): Raw data.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        self.logger.info("Executing `clean_data`...")

        df = raw_df.copy()

        df = self.remove_trailing_spaces(df)
        df = self.lowercase_strings(df)
        df = self.replace_string_nans(df)
        df = self.rename_columns(df)
        df = self.drop_id_column(df)
        df = self.drop_minors(df)
        df = self.drop_six_star_ratings(df)
        df = self.create_city_name(df)
        df = self.clean_location_columns(df)
        df = self.compute_distance(df)
        df = self.bin_distance(df)
        df = self.clean_datetime_columns(df)
        df = self.clean_weather_column(df)
        df = self.convert_multiple_deliveries(df)
        df = self.clean_time_taken_column(df)
        df = self.drop_unnecessary_columns(df)

        self.logger.info("Execution of `clean_data` complete.")
        return df


if __name__ == "__main__":

    # Root path
    root_path = Path(__file__).parent.parent.parent

    # 'cleaned' directory path
    cleaned_dir_path = root_path / "data" / "cleaned"

    # Create the 'cleaned' directory if it does not exist
    cleaned_dir_path.mkdir(exist_ok=True, parents=True)

    # Cleaned data file name
    cleaned_data_file_name = "swiggy_cleaned.csv"

    # Path of the cleaned data file
    cleaned_data_file_path = cleaned_dir_path / cleaned_data_file_name

    # Raw data file name
    raw_data_file_name = "swiggy.csv"

    # Raw data path
    raw_data_file_path = root_path / "data" / "raw" / raw_data_file_name

    # Reading the raw data
    raw_df = pd.read_csv(raw_data_file_path)

    # Cleaning the raw data and saving the cleaned data
    data_cleaner = DataCleaner()
    clean_df = data_cleaner.clean_data(raw_df=raw_df)
    clean_df.to_csv(cleaned_data_file_path, index=False)
