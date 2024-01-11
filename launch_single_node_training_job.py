import os
from multiprocessing import cpu_count
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from trainer.task import train_model

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/VisionTransformers/")

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run():
    num_cpus = cpu_count()
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda') if num_gpus else 'cpu'
    print(f'Device: {device}')
    print(f'CPUs: {num_cpus}')
    print(f'GPUs: {num_gpus}')
    if torch.cuda.is_available():
        trainer_num_workers = num_gpus
        dataloader_num_workers = num_gpus
    else:
        trainer_num_workers = "auto"
        dataloader_num_workers = num_cpus
    # Check whether the specified path exists or not
    isExist = os.path.exists(CHECKPOINT_PATH)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(CHECKPOINT_PATH)
       print("The new directory is created!")
    args = {'model_kwargs': {
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
    'trainer_kwargs':{
        "default_root_dir": os.path.join(CHECKPOINT_PATH, "ViT"),
        "accelerator": "auto",
        "strategy": "auto",
        "devices": trainer_num_workers,
        "max_epochs": 25,
        "callbacks": [
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc",
                            dirpath=os.path.join(CHECKPOINT_PATH, "ViT")),
            LearningRateMonitor("epoch"),
        ],
    },
    'loader_kwargs':{
        "dataset_path": DATASET_PATH,
        "batch_size": 128,
        "num_workers": dataloader_num_workers
    }}
    train_model(**args, lr=3e-4,)


if __name__ == "__main__":
    run()
