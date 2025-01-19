# Set the base image
FROM python:3.12-slim

# Install LightGBM dependency
RUN apt-get update && apt-get install -y libgomp1

# Set up the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements-dockers.txt ./

# Install the packages
RUN pip install -r requirements-dockers.txt

# Copy the app content
COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./swiggy_delivery_time_prediction/data/data_cleaning.py ./swiggy_delivery_time_prediction/data/data_cleaning.py
COPY ./run_info.json ./

# Expose the port
EXPOSE 8000

# Run the file using the command
CMD [ "python", "./app.py" ]