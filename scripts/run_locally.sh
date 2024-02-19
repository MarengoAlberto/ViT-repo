#!/bin/bash

set -a
source environments/.env
rm -rf serve
python deployment/create_mar_locally.py
torchserve --start --model-store serve/model-store --models $MODEL_DISPLAY_NAME.mar