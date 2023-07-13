import os
from setuptools import find_packages
from setuptools import setup
import setuptools
from distutils.command.build import build as _build
import subprocess

BUCKET_URI = ""
APP_NAME = "ViT-model"
PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = ("us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest")
PYTHON_PACKAGE_APPLICATION_DIR = "."
source_package_file_name = "/dist/trainer-0.1.tar.gz"
python_package_gcs_uri = (f"{BUCKET_URI}/pytorch-on-gcp/{APP_NAME}/train/python_package/trainer-0.1.tar.gz")
python_module_name = "task"
REQUIRED_PACKAGES = [
    'torch>=1.8.1, <1.14.0',
    'lightning>=2.0.0rc0',
    'setuptools==67.4.0',
    'torchmetrics>=0.7, <0.12',
    'torchvision',
    'pytorch-lightning>=1.4, <2.0.0',
    'google-cloud-storage'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Vertex AI | Training | PyTorch | Image Classification | Python Package | Vis Transformer'
)
