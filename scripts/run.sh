#!/bin/bash

# Load ENV variables
set -a
source environments/.env

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

# Combine the service name with the commit SHA for the image tag
IMAGE_TAG="${MODEL_NAME}:${COMMIT_SHA}"

# Run the Docker container
docker run -p 7080:7080 -ti -v ~/.config:/home/model-server/.config "$IMAGE_TAG"

echo "Docker container running successfully: $IMAGE_TAG"
