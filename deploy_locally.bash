rm -rf serve
python deployment/create_mar_locally.py
torchserve --start --model-store serve/model-store --models ViT-model.mar