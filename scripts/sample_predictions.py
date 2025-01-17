import requests

import pandas as pd

from pathlib import Path

# Paths for the raw data

root_path = Path(__file__).parent.parent
raw_data_file_path = root_path / "data" / "raw" / "swiggy.csv"

# Prediction endpoint
predict_url = "http://127.0.0.1:8000/predict"

# Sample row for testing
sample_row = pd.read_csv(raw_data_file_path).dropna().sample(1)
print(f"The target value is {sample_row.iloc[:, -1].values.item().replace("(min) ", "")}.")

# Removing the target
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).dropna().squeeze().to_dict()

# Response from the API
response = requests.post(url=predict_url, json=data)

print(f"The status code for response is {response.status_code}.")

if response.status_code == 200:
    print(f"The prediction value by the API is {response.text} mins.")
else:
    print("Error:", response.status_code, f"{response.text}.")
