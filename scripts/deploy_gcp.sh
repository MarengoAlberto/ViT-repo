#!/bin/bash

source .env
rm -rf serve
python deployment/create_mar_locally.py
gsutil cp serve/model-store/ViT-model.mar gs://$BUCKET_NAME/$MAR_MODEL_OUT_PATH/$MODEL_DISPLAY_NAME.mar
docker build -f predictor/Dockerfile -t $CUSTOM_PREDICTOR_IMAGE_URI ./
docker push $CUSTOM_PREDICTOR_IMAGE_URI
python deployment/deploy_gcp.py