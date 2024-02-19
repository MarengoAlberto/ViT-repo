import torch
import os
import json
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import lightning as L

from .dist_evironment import KubeflowEnvironment


def get_datasets(dataset_path):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
        ]
    )
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
        ]
    )
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=dataset_path, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=dataset_path, train=True, transform=test_transform, download=True)
    L.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    L.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    # Loading the test set
    test_set = CIFAR10(root=dataset_path, train=False, transform=test_transform, download=True)
    return train_set, val_set, test_set


def get_loaders(args):
    train_set, val_set, test_set = get_datasets(args.dataset_path)
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=True,
                                   pin_memory=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader


def get_trainer(args):
    if args.num_nodes > 1:
        return L.Trainer(
            default_root_dir=args.default_root_dir,
            logger=args.logger,
            accelerator=args.accelerator,
            strategy=args.strategy,
            devices=args.devices,
            num_nodes=args.num_nodes,
            max_epochs=args.max_epochs,
            callbacks=args.callbacks,
            enable_checkpointing=True,
            plugins=KubeflowEnvironment(),
        )
    else:
        return L.Trainer(
            default_root_dir=args.default_root_dir,
            logger=args.logger,
            accelerator=args.accelerator,
            strategy=args.strategy,
            devices=args.devices,
            num_nodes=args.num_nodes,
            max_epochs=args.max_epochs,
            callbacks=args.callbacks,
            enable_checkpointing=True,
        )


def get_n_tpus():
    tf_config_str = os.environ.get('TF_CONFIG')
    tf_config_dict = json.loads(tf_config_str)
    return int(tf_config_dict['job']['worker_config']['accelerator_config']['count'])
