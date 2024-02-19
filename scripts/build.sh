#!/bin/bash

# Prep mar file
set -a
source environments/.env
rm -rf serve
python deployment/create_mar_locally.py
gsutil cp serve/model-store/ViT-model.mar gs://$BUCKET_NAME/$MAR_MODEL_OUT_PATH/$MODEL_DISPLAY_NAME.mar

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

# Build the Docker image
docker build -f predictor/Dockerfile -t "${IMAGE_TAG}" \
                                      --build-arg MODEL_NAME=${MODEL_NAME} \
                                      --build-arg PROJECT_ID=${PROJECT_ID} \
                                      --build-arg REGION=${REGION} \
                                      --build-arg BUCKET_NAME=${BUCKET_NAME} \
                                      --build-arg EMBED_DIM=${EMBED_DIM} \
                                      --build-arg HIDDEN_DIM=${HIDDEN_DIM} \
                                      --build-arg NUM_HEADS=${NUM_HEADS} \
                                      --build-arg NUM_LAYERS=${NUM_LAYERS} \
                                      --build-arg PATCH_SIZE=${PATCH_SIZE} \
                                      --build-arg NUM_CHANNELS=${NUM_CHANNELS} \
                                      --build-arg NUM_PATCHES=${NUM_PATCHES} \
                                      --build-arg NUM_CLASSES=${NUM_CLASSES} \
                                      --build-arg DROPOUT=${DROPOUT} \
                                      --build-arg MAP_CLASSES_PATH=${MAP_CLASSES_PATH} \
                                      .
# Check if docker build was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed."
    exit 1
fi

echo "Docker image built successfully: $IMAGE_TAG"