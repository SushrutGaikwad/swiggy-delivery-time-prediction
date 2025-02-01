import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame):
    """Cleans the given data.

    Args:
        df (pd.DataFrame): Data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    df = df.copy()

    # Removing trailing spaces from all the object-type columns
    object_type_cols = df.select_dtypes(include=["object"]).columns
    for col in object_type_cols:
        df[col] = df[col].str.strip()

    # Changing all the values to lower case
    for col in object_type_cols:
        if col not in ["Delivery_person_ID"]:
            df[col] = df[col].str.lower()

    # Changing the string `"nan"` and `"conditions nan"` to `np.nan`
    df[object_type_cols] = df[object_type_cols].map(
        lambda x: np.nan if isinstance(x, str) and "nan" in x else x
    )

    # Shortening the column names and changing them to lower case
    rename_mapping = {
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
    df = df.rename(str.lower, axis=1).rename(rename_mapping, axis=1)

    # Dropping the column `id`
    df.drop(columns=["id"], inplace=True)

    # Dropping minors

    ## First converting the `age` column to numeric
    df["age"] = df["age"].astype("float")
    minors = df.loc[df["age"] < 18]
    minors_idxs = minors.index
    df.drop(minors_idxs, inplace=True)

    # Dropping 6-star ratings

    ## First converting the `ratings` column to numeric
    df["ratings"] = df["ratings"].astype(float)
    six_star_ratings = df.loc[df["ratings"] == 6]
    six_star_ratings_idxs = six_star_ratings.index
    df.drop(six_star_ratings_idxs, inplace=True)

    # Creating a new column `city_name` using the column `rider_id`
    df["city_name"] = df["rider_id"].str.split("RES").str.get(0)

    # Cleaning location columns
    location_columns = df.columns[3:7]
    df[location_columns] = df[location_columns].abs()
    df[location_columns] = df[location_columns].map(lambda x: x if x >= 1 else np.nan)

    # Creating a new column called `distance` (haversine distance)
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    distance = 6371 * c

    df["distance"] = distance

    bins = [0, 5, 10, 15, np.inf]
    labels = ["short", "medium", "long", "very_long"]
    df["distance_type"] = pd.cut(df["distance"], bins=bins, right=False, labels=labels)

    # Cleaning the datetime related columns
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
    df["order_day"] = df["order_date"].dt.day
    df["order_month"] = df["order_date"].dt.month
    df["order_day_of_week"] = df["order_date"].dt.day_name().str.lower()
    df["order_day_is_weekend"] = df["order_day_of_week"].isin(["saturday", "sunday"]).astype(int)

    df["order_time"] = pd.to_datetime(df["order_time"], format="%H:%M:%S", errors="coerce")
    df["order_picked_time"] = pd.to_datetime(df["order_picked_time"], format="%H:%M:%S")
    df["pickup_time_minutes"] = ((df["order_picked_time"] - df["order_time"]).dt.seconds) / 60

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
        np.select(condlist, choicelist, default=default), index=df["order_time"].index
    )
    # order_time_of_day = np.select(condlist=condlist, choicelist=choicelist, default=default)
    df["order_time_of_day"] = time_of_day_info.where(df["order_time"].notna(), np.nan)
    df.drop(columns=["order_time", "order_picked_time"], inplace=True)

    # Cleaning the `weather` column
    df["weather"] = df["weather"].str.replace("conditions ", "")

    # Changing dtype of `multiple_deliveries`
    df["multiple_deliveries"] = df["multiple_deliveries"].astype(float)

    # Cleaning the target column `time_taken`
    df["time_taken"] = df["time_taken"].str.replace("(min) ", "").astype(int)
    return df
