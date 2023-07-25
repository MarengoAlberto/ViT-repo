import os
import logging
import torch
from torchvision.io import read_image
import torchvision.transforms as T
from ts.torch_handler.base_handler import BaseHandler

from model import VisionTransformer

logger = logging.getLogger('__main__')
transforms = T.Resize(size=(32, 32))
softmax = torch.nn.Softmax(dim=-1)

VALID_IMAGE_FORMATS = [".jpg", ".gif", ".png", ".tga", ".jpeg"]

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


def get_batch(path):
    imgs = []
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in VALID_IMAGE_FORMATS:
            continue
        imgs.append(os.path.join(path, filename))
    batch = []
    logger.info("Processing images: '%s'", imgs)
    for image in imgs:
        X = read_image(image)
        X = transforms(X)
        batch.append(X)
    batch = torch.stack(batch, 0)
    logger.info(f"Tensor shape: {batch.shape}")
    return batch


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
        path = data[0].get("data")
        if path is None:
            path = data[0].get("body")
        logger.info("Received path: '%s'", path)
        inputs = get_batch(path['input'])
        return inputs.type(torch.float32)

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        outputs = self.model(inputs.to(self.device))
        predictions = softmax(outputs).argmax(dim=-1).tolist()
        logger.info("Model predicted: '%s'", predictions)
        return predictions

    def postprocess(self, inference_output):
        return [{'response': [CLASS_MAPPING[prediction] for prediction in inference_output]}]
