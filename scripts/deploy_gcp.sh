#!/bin/bash

set -a
source .env
rm -rf serve
python deployment/create_mar_locally.py
gsutil cp serve/model-store/ViT-model.mar gs://$BUCKET_NAME/$MAR_MODEL_OUT_PATH/$MODEL_DISPLAY_NAME.mar
docker build -f predictor/Dockerfile -t $CUSTOM_PREDICTOR_IMAGE_URI \
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
docker push $CUSTOM_PREDICTOR_IMAGE_URI
python deployment/deploy_gcp.py