import os
import logging
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image

TEMP_FILE_NAME = '/tmp/image.png'

transforms = T.Resize(size=(32, 32))
softmax = torch.nn.Softmax(dim=-1)

load_dotenv()
logger = logging.getLogger('__main__')


PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME')
model_kwargs = {
                "embed_dim": os.getenv('EMBED_DIM'),
                "hidden_dim": os.getenv('HIDDEN_DIM'),
                "num_heads": os.getenv('NUM_HEADS'),
                "num_layers": os.getenv('NUM_LAYERS'),
                "patch_size": os.getenv('PATCH_SIZE'),
                "num_channels": os.getenv('NUM_CHANNELS'),
                "num_patches": os.getenv('NUM_PATCHES'),
                "num_classes": os.getenv('NUM_CLASSES'),
                "dropout": os.getenv('DROPOUT'),
            }


CLASS_MAPPING = {
                0: 'airplane',
                1: 'automobile',
                2: 'bird',
                3: 'cat',
                4: 'deer',
                5: 'dog',
                6: 'frog',
                7: 'horse',
                8: 'ship',
                9: 'truck'
}


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
