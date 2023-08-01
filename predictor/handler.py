import os
import logging
import base64
from io import BytesIO
from PIL import Image
import json
import torch
from torchvision.io import read_image
import torchvision.transforms as T
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger('__main__')

try:
    from google.cloud import storage
except:
    logger.warning('No GCS loaded, only local prediction available')

from model import VisionTransformer


transforms = T.Resize(size=(32, 32))
softmax = torch.nn.Softmax(dim=-1)

PROJECT_ID = 'alberto-playground'
BUCKET_NAME = 'alberto-vit-playground'
TEMP_FILE_NAME = '/tmp/image.png'
model_kwargs={
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 3,
            "num_patches": 64,
            "num_classes": 10,
            "dropout": 0.2,
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

try:
    client = storage.Client(project=PROJECT_ID)
    bucket = storage.Client().bucket(BUCKET_NAME)
except:
    logger.warning('No GCS loaded, only local prediction available')

def get_input(file):
    if isinstance(file, dict):
        logger.info(file.keys())
        im = Image.open(BytesIO(base64.b64decode(file['content'])))
        im.save(TEMP_FILE_NAME)
    else:
        image = Image.open(BytesIO(file))
        image.save(TEMP_FILE_NAME)
    input = read_image(TEMP_FILE_NAME)
    input = transforms(input)
    input= torch.unsqueeze(input, 0)
    logger.info(f"Tensor shape: {input.shape}")
    os.remove(TEMP_FILE_NAME)
    return input


class TransformersClassifierHandler(BaseHandler):
    """
    The handler takes an input string and returns the class based on Vit model.
    """

    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest

        properties = ctx.system_properties

        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")

        # Load model
        self.model = VisionTransformer(**model_kwargs)
        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.to(self.device)
        self.model.eval()
        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """
        if isinstance(data, list):
            path = data[0].get("data")
            if path is None:
                path = data[0].get("body")
            if path is None:
                root = data[0]
            else:
                root = path
            file = root.get("file")
            if file is None:
                link = root.get("link")
                if isinstance(link, str) and link.startswith('gs://'):
                    sub_folder = os.path.relpath(link.replace('gs://', ''), BUCKET_NAME)
                    blob = bucket.get_blob(sub_folder)    
                    file = blob.download_as_bytes()
                else:
                    raise ValueError()
        else:
            file = data
        inputs = get_input(file)
        return inputs.type(torch.float32)

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        outputs = self.model(inputs.to(self.device))
        return outputs

    def postprocess(self, inference_output):
        proba = softmax(inference_output)
        prediction = proba.argmax(dim=-1).item()
        proba_ret = {v: proba[:, int(k)].item() for k, v in CLASS_MAPPING.items()}
        return [{'response': CLASS_MAPPING[prediction],
                 'probabilities': proba_ret}]
