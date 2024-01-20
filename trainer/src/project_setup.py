import os
import logging
import google.cloud.aiplatform as aiplatform
from dotenv import load_dotenv

from .arguments import RunnerArguments, TrainerArguments, LoaderArguments, ModelArguments

load_dotenv()
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


# TODO: Implement get arguments function
def get_arguments() -> RunnerArguments:
    self.rank = kwargs.get('rank')
    self.is_local = kwargs.get('is_local')
    self.model_kwargs = kwargs.get('model_kwargs')
    self.loader_kwargs = kwargs.get('loader_kwargs')
    self.trainer_kwargs = kwargs.get('trainer_kwargs')
    self.lr = kwargs.get('lr')
    self.model_name = kwargs.get('model_name', 'ViT-model')
    self.experiment_name = kwargs.get('experiment_name', 'vitmodel')
    self.checkpoint_path = kwargs.get('checkpoint_path', 'saved_models/VisionTransformers/')
    self.bucket_name = kwargs.get('bucket_name')
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
