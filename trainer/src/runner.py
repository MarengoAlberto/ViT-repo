import os
from google.cloud import storage
import google.cloud.aiplatform as aiplatform
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from .lightning_model import ViT
from . import utils, project_setup, arguments


class Runner:

    def __init__(self, runner_arguments: arguments.RunnerArguments) -> None:
        self.args = runner_arguments

    def train_model(self):
        print(self.args)
        aip_tensorboard_log_dir = project_setup.create_tensorboard(self.args)
        logger = TensorBoardLogger(aip_tensorboard_log_dir, name=self.args.experiment_name)
        self.args.trainer_kwargs.logger = logger
        train_loader, val_loader, test_loader = utils.get_loaders(self.args.loader_kwargs)
        trainer = utils.get_trainer(self.args.trainer_kwargs)
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(self.args.checkpoint_path, f'{self.args.model_name}.ckpt')
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model at %s, loading..." % pretrained_filename)
            if self.args.rank == 0:
                # Automatically loads the model with the saved hyperparameters
                model = ViT.load_from_checkpoint(pretrained_filename)
        else:
            L.seed_everything(42)
            model = ViT(self.args.model_kwargs, self.args.lr)
            trainer.fit(model, train_loader, val_loader)
        # Test best model on validation and test set
        val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
        test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
        result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
        print(self.args)
        if self.args.rank == 0:
            model_filepath = os.path.join(self.args.checkpoint_path, f'{self.args.model_name}.pt')
            torch.save(model.model.state_dict(), model_filepath)
        if not self.args.is_local:
            if self.args.rank == 0:
                # Upload the trained model to Cloud storage
                storage_path = os.path.join(f'gs://{self.args.bucket_name}/outputs', f'{self.args.model_name}.pt')
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.args.bucket_name)
                blob = bucket.blob(f'model/{self.args.model_name}.pt')
                blob.upload_from_filename(model_filepath)
                print(f"Saved model files in {storage_path}")
            aiplatform.end_upload_tb_log()
