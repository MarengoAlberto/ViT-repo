import os
import logging
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import base64
from io import BytesIO
from PIL import Image
import json

def keystoint(x):
    return {int(k): v for k, v in x.items()}


TEMP_FILE_NAME = '/tmp/image.png'

transforms = T.Resize(size=(32, 32))
softmax = torch.nn.Softmax(dim=-1)

logger = logging.getLogger('__main__')

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME')
model_kwargs = {
                "embed_dim": int(os.getenv('EMBED_DIM')),
                "hidden_dim": int(os.getenv('HIDDEN_DIM')),
                "num_heads": int(os.getenv('NUM_HEADS')),
                "num_layers": int(os.getenv('NUM_LAYERS')),
                "patch_size": int(os.getenv('PATCH_SIZE')),
                "num_channels": int(os.getenv('NUM_CHANNELS')),
                "num_patches": int(os.getenv('NUM_PATCHES')),
                "num_classes": int(os.getenv('NUM_CLASSES')),
                "dropout": float(os.getenv('DROPOUT')),
            }

map_class_path_ext = os.getenv('MAP_CLASSES_PATH')
map_class_path = map_class_path_ext.split('/')[1]
with open(map_class_path, 'r') as fp:
    CLASS_MAPPING = json.load(fp)

# Makes sure keys are integer
CLASS_MAPPING = keystoint(CLASS_MAPPING)


def get_input(file):
    if isinstance(file, dict):
        logger.info(file.keys())
        im = Image.open(BytesIO(base64.b64decode(file['content'])))
        im.save(TEMP_FILE_NAME)
    else:
        image = Image.open(BytesIO(file))
        image.save(TEMP_FILE_NAME)
    input_img = read_image(TEMP_FILE_NAME)
    input_img = transforms(input_img)
    input_img= torch.unsqueeze(input_img, 0)
    logger.info(f"Tensor shape: {input_img.shape}")
    os.remove(TEMP_FILE_NAME)
    return input_img
