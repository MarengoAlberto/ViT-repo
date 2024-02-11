#!/bin/bash

# Load ENV variables
set -a
source .env

# Check if the service name is set
if [ -z "${MODEL_NAME}" ]; then
    echo "Error: Environment variable MODEL_NAME is not set."
    exit 1
fi
# Get the current git commit SHA
COMMIT_SHA=$(git rev-parse --short HEAD)

# Check if git rev-parse was successful
if [ $? -ne 0 ]; then
    echo "Error: Unable to get the current commit SHA."
    exit 1
fi

# Construct the full image name for GCP Artifact Registry
FULL_IMAGE_NAME="${CUSTOM_PREDICTOR_IMAGE_URI}"

# Tag the image with the full name
docker tag "${MODEL_NAME}:${COMMIT_SHA}" "$FULL_IMAGE_NAME"

# Check if docker tag was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker tagging failed."
    exit 1
fi
# Push the image to the GCP Artifact Registry
docker push "$FULL_IMAGE_NAME"

# Check if docker push was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker push failed."
    exit 1
fi

echo "Docker image pushed successfully to GCP Artifact Registry: $FULL_IMAGE_NAME"