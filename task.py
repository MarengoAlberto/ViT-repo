import os
from multiprocessing import cpu_count
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from google.cloud import storage

from src.lightning_model import ViT
from src import utils

# TODO: Fill these out
DATASET_PATH = ''
CHECKPOINT_PATH = ''
STORAGE_BUCKET = ''


def train_model(**kwargs):
    model_kwargs = kwargs.get('model_kwargs')
    loader_kwargs = kwargs.get('loader_kwargs')
    trainer_kwargs = kwargs.get('trainer_kwargs')
    lr = kwargs.get('lr')
    train_loader, val_loader, test_loader = utils.get_loaders(**loader_kwargs)
    trainer = utils.get_trainer(**trainer_kwargs)
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = ViT.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = ViT(model_kwargs, lr)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    print(result)

    # Save the trained model locally
    trainer.save_checkpoint(pretrained_filename)

    if os.path.isfile(pretrained_filename):
        # Upload the trained model to Cloud storage
        storage_path = os.path.join(STORAGE_BUCKET, 'ViT-model')
        blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
        blob.upload_from_filename(pretrained_filename)
        print(f"Saved model files in {storage_path}")


if __name__=="__main__":
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_cpus = cpu_count()
    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        num_workers = world_size * num_gpus
    else:
        num_workers = world_size * num_cpus

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
            "accelerator": "auto",
            "devices": num_workers,
            "max_epochs": 180,
            "callbacks": [
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
            ],
        },
        loader_kwargs={
            "dataset_path": DATASET_PATH,
            "batch_size": 128,
            "num_workers": num_workers
        },
        lr=3e-4,
    )
