rm -rf serve
python create_mar_locally.py
torchserve --start --model-store serve/model-store --models ViT-model.mar