import os
import logging
import torch
from multiprocessing import cpu_count
import google.cloud.aiplatform as aiplatform
from dotenv import load_dotenv

from .arguments import RunnerArguments, TrainerArguments, LoaderArguments, ModelArguments
from .variables import model_kwargs, loader_kwargs, trainer_kwargs, runner_kwargs
from . import utils

load_dotenv('../.env')
logger = logging.getLogger(__name__)


def create_tensorboard(args: RunnerArguments) -> str:
    if args.is_local:
        return "tb_logs"
    tensorboard_log_dir = f'gs://{args.experiment_name}/{args.experiment_name}'
    tensorboard = aiplatform.Tensorboard.create(display_name=args.tensorboard_name,
                                                project=args.project_id, location=args.region)
    aiplatform.init(location=args.region, project=args.project_id, experiment_tensorboard=tensorboard)
    aiplatform.start_upload_tb_log(
        tensorboard_id=tensorboard.gca_resource.name.split('/')[-1],
        tensorboard_experiment_name=args.experiment_name,
        logdir=tensorboard_log_dir
    )
    return tensorboard_log_dir


def get_arguments() -> RunnerArguments:
    model_args = ModelArguments(**model_kwargs)
    loader_args = LoaderArguments(**loader_kwargs)
    trainer_args = TrainerArguments(**trainer_kwargs)
    runner_args = RunnerArguments(**runner_kwargs)
    world_size = trainer_args.num_nodes
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
            strategy = "ddp"
        else:
            num_dataloader_workers = num_cpus
            num_workers = num_cpus
            strategy = "auto"
    loader_args.num_workers = num_dataloader_workers
    trainer_args.accelerator = accelerator
    trainer_args.strategy = strategy
    trainer_args.devices = num_workers
    runner_args.model_kwargs = model_args
    runner_args.loader_kwargs = loader_args
    runner_args.trainer_kwargs = trainer_args
    return runner_args
