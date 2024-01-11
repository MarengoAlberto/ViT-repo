import os
import google.cloud.aiplatform as aiplatform
from pytorch_lightning.loggers import TensorBoardLogger

PROJECT_ID = 'alberto-playground-395414'
REGION = 'us-east1'
BUCKET_URI = 'vit-bucket'
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/VisionTransformers/")
STORAGE_BUCKET = 'gs://vit-bucket/outputs'
TENSORBOARD_NAME = 'tbViT'
EXPERIMENT_NAME = 'vitmodel'


def create_tensorboard(is_local):
    if is_local:
        return "tb_logs"
    tensorboard_log_dir = f'gs://{BUCKET_URI}/{EXPERIMENT_NAME}'
    tensorboard = aiplatform.Tensorboard.create(display_name=TENSORBOARD_NAME,
                                                project=PROJECT_ID, location=REGION)
    aiplatform.init(location=REGION, project=PROJECT_ID, experiment_tensorboard=tensorboard)
    aiplatform.start_upload_tb_log(
        tensorboard_id=tensorboard.gca_resource.name.split('/')[-1],
        tensorboard_experiment_name=EXPERIMENT_NAME,
        logdir=tensorboard_log_dir
    )
    return tensorboard_log_dir
