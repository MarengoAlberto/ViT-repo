import os
import sys
import json
import logging

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

sys.path.append(".")

from src.model.model import VisionTransformer


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
        self.model = VisionTransformer.load_state_dict(torch.load(model_pt_path))
        self.model.to(self.device)
        self.model.eval()
        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """
        pass

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        prediction = self.model(inputs.to(self.device)).argmax().item()

        logger.info("Model predicted: '%s'", prediction)
        return [prediction]

    def postprocess(self, inference_output):
        return inference_output
