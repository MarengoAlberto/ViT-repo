from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==2.13.0',
    'tensorboard',
    'torch>=1.8.1, <=1.13.0',
    'lightning>=2.0.0rc0',
    'setuptools==67.4.0',
    'torchmetrics>=0.7, <0.12',
    'torchvision',
    'pytorch-lightning>=2.1.0.rc0',
    'google-cloud-aiplatform==1.29.0',
    'google-cloud-aiplatform[tensorboard]',
    'protobuf==3.20.*',
    'google-cloud-storage',
    'torch-xla',
    'python-dotenv',
    'websocket-client',
    'python-json-logger'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Vertex AI | Training | PyTorch | Image Classification | Python Package | Vis Transformer'
)
