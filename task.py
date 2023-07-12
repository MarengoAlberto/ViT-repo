import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.lightning_model import ViT
from src import utils

DATASET_PATH = ''
CHECKPOINT_PATH = ''


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

    return model, result


if __name__=="__main__":
    model, results = train_model(
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
            "devices": 1,
            "max_epochs": 180,
            "callbacks": [
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
            ],
        },
        loader_kwargs={
            "dataset_path": DATASET_PATH,
            "batch_size": 128,
            "num_workers": 4,
            "is_parallel": False
        },
        lr=3e-4,
    )
    print("ViT results", results)
