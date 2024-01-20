from dataclasses import dataclass


@dataclass
class RunnerArguments:
    n_clusters: int
    rank: int
    is_local: bool
    model_kwargs: dict
    loader_kwargs: dict
    trainer_kwargs: dict
    lr: float
    model_name: str
    experiment_name:str
    checkpoint_path: str
    bucket_name: str
    project_id: str
    region: str
    tensorboard_name: str


@dataclass
class ModelArguments:
    pass


@dataclass
class LoaderArguments:
    pass


@dataclass
class TrainerArguments:
    pass
