import os
import logging
import torch
from ts.torch_handler.base_handler import BaseHandler
from model import VisionTransformer
from utils import softmax, get_input
from utils import PROJECT_ID, BUCKET_NAME, CLASS_MAPPING, model_kwargs

logger = logging.getLogger('__main__')


try:
    from google.cloud import storage
    client = storage.Client(project=PROJECT_ID)
    bucket = storage.Client().bucket(BUCKET_NAME)
except:
    logger.warning('No GCS loaded, only local prediction available')


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
        logger.info(data)
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
