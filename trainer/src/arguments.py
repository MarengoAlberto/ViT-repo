from dataclasses import dataclass
from typing import List
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


@dataclass
class ModelArguments:
    embed_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    patch_size: int
    num_channels: int
    num_patches: int
    num_classes: int
    dropout: float


@dataclass
class LoaderArguments:
    dataset_path: str
    batch_size: int
    num_workers: int = None


@dataclass
class TrainerArguments:
    default_root_dir: str
    num_nodes: int
    max_epochs: int
    callbacks: List[ModelCheckpoint]
    accelerator: str = None
    strategy: str = None
    devices: int = None
    logger: TensorBoardLogger = None


@dataclass
class RunnerArguments:
    rank: int
    is_local: bool
    lr: float
    model_name: str
    experiment_name:str
    checkpoint_path: str
    bucket_name: str
    project_id: str
    region: str
    tensorboard_name: str
    model_kwargs: ModelArguments = None
    loader_kwargs: LoaderArguments = None
    trainer_kwargs: TrainerArguments = None
