import os
from dotenv import load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

load_dotenv()

checkpoint_path = os.environ.get("CHECKPOINT_PATH", "saved_models/VisionTransformers/")
isExist = os.path.exists(checkpoint_path)
if not isExist:
    os.makedirs(checkpoint_path)

model_kwargs = {
            "embed_dim": int(os.environ.get("EMBED_DIM", 256)),
            "hidden_dim": int(os.environ.get("HIDDEN_DIM", 512)),
            "num_heads": int(os.environ.get("NUM_HEADS", 8)),
            "num_layers": int(os.environ.get("NUM_LAYERS", 6)),
            "patch_size": int(os.environ.get("PATCH_SIZE", 4)),
            "num_channels": int(os.environ.get("NUM_CHANNELS", 3)),
            "num_patches": int(os.environ.get("NUM_PATCHES", 64)),
            "num_classes": int(os.environ.get("NUM_CLASSES", 10)),
            "dropout": int(os.environ.get("DROPOUT", 0.2))
        }

loader_kwargs = {
    "dataset_path": os.environ.get("DATASET_PATH", "data/"),
    "batch_size": int(os.environ.get("BATCH_SIZE", 128))
}

trainer_kwargs = {
            "default_root_dir": os.path.join(checkpoint_path, "ViT"),
            "num_nodes": int(os.environ.get("NUM_NODES", 1)),
            "max_epochs": int(os.environ.get("MAX_EPOCHS", 180)),
            "callbacks": [
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc",
                                dirpath=os.path.join(checkpoint_path, "ViT")),
                LearningRateMonitor("epoch"),
            ]
        }

runner_kwargs = {
        "rank": int(os.environ.get("GLOBAL_RANK", 0)),
        "is_local": True if "GOOGLE_VM_CONFIG_LOCK_FILE" not in os.environ else False,
        "lr": float(os.environ.get("LR", 3e-4)),
        "model_name": os.environ.get("MODEL_NAME", 'ViT-model'),
        "experiment_name": os.environ.get("EXPERIMENT_NAME", 'vitmodel'),
        "checkpoint_path": checkpoint_path,
        "project_id": os.environ.get("PROJECT_ID", 'alberto-playground-395414'),
        "bucket_name": os.environ.get("BUCKET_NAME", 'vit-bucket'),
        "region": os.environ.get("REGION", 'us-east1'),
        "tensorboard_name": os.environ.get("TENSORBOARD_NAME", 'tbViT')
        }
