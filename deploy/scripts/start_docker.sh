#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 872515288060.dkr.ecr.us-east-2.amazonaws.com

echo "Pulling Docker image..."
docker pull 872515288060.dkr.ecr.us-east-2.amazonaws.com/ml/swiggy-delivery-time-prediction:latest

echo "Checking if container already exists..."
if [ "$(docker ps -q -f name=swiggy-delivery-time-prediction)" ]; then
    echo "Stopping existing container..."
    docker stop swiggy-delivery-time-prediction
fi

if [ "$(docker ps -aq -f name=swiggy-delivery-time-prediction)" ]; then
    echo "Removing existing container..."
    docker rm swiggy-delivery-time-prediction
fi

echo "Starting a new container..."
docker run -d -p 80:8000 --name swiggy-delivery-time-prediction -e DAGSHUB_USER_TOKEN=3fc5e02e498cb98801a03673de990da9dbb2f96d 872515288060.dkr.ecr.us-east-2.amazonaws.com/ml/swiggy-delivery-time-prediction:latest

echo "Container started successfully."