import os
import logging
from dotenv import load_dotenv
from multiprocessing import cpu_count
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from .src.lightning_model import ViT
from .src import utils, project_setup

load_dotenv()

is_local = True if "LOCAL_ENV" in os.environ else False
if not is_local:
    from google.cloud import storage
    import google.cloud.aiplatform as aiplatform

logger = logging.getLogger(__name__)

PROJECT_ID = 'alberto-playground-395414'
REGION = 'us-east1'
BUCKET_URI = 'vit-bucket'
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/VisionTransformers/")
STORAGE_BUCKET = 'gs://vit-bucket/outputs'
TENSORBOARD_NAME = 'tbViT'
EXPERIMENT_NAME = 'vitmodel'

for k, v in os.environ.items():
    print(f'{k}={v}')


def train_model(**kwargs):
    rank = int(os.environ.get("GLOBAL_RANK", 0))
    AIP_TENSORBOARD_LOG_DIR = project_setup.create_tensorboard(is_local)
    logger = TensorBoardLogger(AIP_TENSORBOARD_LOG_DIR, name=EXPERIMENT_NAME)
    model_kwargs = kwargs.get('model_kwargs')
    loader_kwargs = kwargs.get('loader_kwargs')
    trainer_kwargs = kwargs.get('trainer_kwargs')
    trainer_kwargs['logger'] = logger
    lr = kwargs.get('lr')
    train_loader, val_loader, test_loader = utils.get_loaders(**loader_kwargs)
    trainer = utils.get_trainer(**trainer_kwargs)
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        if rank == 0:
            # Automatically loads the model with the saved hyperparameters
            model = ViT.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = ViT(model_kwargs, lr)
        trainer.fit(model, train_loader, val_loader)
        # model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    print(result)
    
    if rank == 0:
        # Save the trained model locally
        # trainer.save_checkpoint(pretrained_filename)
        model_filepath = os.path.join(CHECKPOINT_PATH, 'ViT.pt')
        torch.save(model.model.state_dict(), model_filepath)

    if not is_local:
        if rank == 0:
            # Upload the trained model to Cloud storage
            storage_path = os.path.join(STORAGE_BUCKET, 'ViT-model.pt')
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_URI)
            blob = bucket.blob('model/ViT-model.pt')
            # blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
            blob.upload_from_filename(model_filepath)
            print(f"Saved model files in {storage_path}")

        aiplatform.end_upload_tb_log()


if __name__=="__main__":
    isExist = os.path.exists(CHECKPOINT_PATH)
    if not isExist:
        os.makedirs(CHECKPOINT_PATH)
    # world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    world_size = 2
    tpu = True if "XRT_TPU_CONFIG" in os.environ else False
    if tpu:
        accelerator = "tpu"
        strategy = "auto"
        num_workers = world_size * utils.get_n_tpus()
    else:
        accelerator = "auto"
        num_cpus = cpu_count()
        num_gpus = torch.cuda.device_count()
        if torch.cuda.is_available():
            accelerator = "gpu"
            num_workers = num_gpus
            num_dataloader_workers = world_size * num_gpus
            num_nodes = world_size
            strategy = "ddp"
        else:
            num_workers = num_cpus
            strategy = "auto"
    logger.info('__________')
    logger.info(world_size)
    logger.info(accelerator)
    logger.info(num_workers)
    logger.info(strategy)
    logger.info('__________')
    train_model(
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
        },
        trainer_kwargs={
            "default_root_dir": os.path.join(CHECKPOINT_PATH, "ViT"),
            "accelerator": accelerator,
            "strategy": strategy,
            "devices": num_workers,
            "num_nodes": num_nodes,
            "max_epochs": 180,
            "callbacks": [
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc",
                                dirpath=os.path.join(CHECKPOINT_PATH, "ViT")),
                LearningRateMonitor("epoch"),
            ],
        },
        loader_kwargs={
            "dataset_path": DATASET_PATH,
            "batch_size": 128,
            "num_workers": num_dataloader_workers
        },
        lr=3e-4,
    )
