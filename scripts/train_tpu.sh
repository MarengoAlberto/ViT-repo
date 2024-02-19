#!/bin/bash

set -a
source environments/.env
docker build -f tpu/Dockerfile -t $CUSTOM_TPU_TRAINING_IMAGE_URI ./
docker push $CUSTOM_TPU_TRAINING_IMAGE_URI
gcloud beta ai-platform jobs submit training vit_tpu_training — config=tpu/tpu_training.yaml — region us-central1
