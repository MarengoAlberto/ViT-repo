from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'torch>=1.8.1, <1.14.0',
    'lightning>=2.0.0rc0',
    'setuptools==67.4.0',
    'torchmetrics>=0.7, <0.12',
    'torchvision',
    'pytorch-lightning>=1.4, <2.0.0',
    'google-cloud-storage',
    'torch-xla',
    'python-dotenv'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Vertex AI | Training | PyTorch | Image Classification | Python Package | Vis Transformer'
)
