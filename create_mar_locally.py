import logging
import os
import json
import subprocess
import time
from collections import namedtuple
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)

MODEL_PT_FILEPATH = 'saved_models/VisionTransformers'
MAR_MODEL_OUT_PATH = 'serve'
MODEL_FILE_PATH = 'predictor'
handler = 'predictor/handler.py'
MODEL_DISPLAY_NAME = 'ViT-model'
model_version = 1
PROJECT_ID = 'alberto-playground'
BUCKET_NAME = 'alberto-vit-playground'
CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{PROJECT_ID}/pytorch_predict_{APP_NAME}"

# create directory to save model archive file
model_output_root = MODEL_PT_FILEPATH
mar_output_root = MAR_MODEL_OUT_PATH
additiona_files_base_dir = 'src/model'
export_path = f"{mar_output_root}/model-store"
try:
    Path(export_path).mkdir(parents=True, exist_ok=True)
except Exception as e:
    logging.warning(e)
    # retry after pause
    time.sleep(2)
    Path(export_path).mkdir(parents=True, exist_ok=True)

# parse and configure paths for model archive config
handler_path = (
    handler.replace("gs://", "/gcs/") + "predictor/handler.py"
    if handler.startswith("gs://")
    else handler
)
model_artifacts_dir = model_output_root
extra_files = [f'{MODEL_FILE_PATH}/modules.py']

# define model archive config
mar_config = {
    "MODEL_NAME": MODEL_DISPLAY_NAME,
    "HANDLER": handler_path,
    "SERIALIZED_FILE": f'{model_artifacts_dir}/ViT.pt',
    "MODEL_FILE": f'{MODEL_FILE_PATH}/model.py',
    "VERSION": model_version,
    "EXTRA_FILES": ",".join(extra_files),
    "EXPORT_PATH": export_path,
}

# generate model archive command
archiver_cmd = (
    "torch-model-archiver --force "
    f"--model-name {mar_config['MODEL_NAME']} "
    f"--serialized-file {mar_config['SERIALIZED_FILE']} "
    f"--model-file {mar_config['MODEL_FILE']} "
    f"--handler {mar_config['HANDLER']} "
    f"--version {mar_config['VERSION']}"
)
if "EXPORT_PATH" in mar_config:
    archiver_cmd += f" --export-path {mar_config['EXPORT_PATH']}"
if "EXTRA_FILES" in mar_config:
    archiver_cmd += f" --extra-files {mar_config['EXTRA_FILES']}"
if "REQUIREMENTS_FILE" in mar_config:
    archiver_cmd += f" --requirements-file {mar_config['REQUIREMENTS_FILE']}"

# run archiver command
logging.warning("Running archiver command: %s", archiver_cmd)
with subprocess.Popen(
        archiver_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
) as p:
    _, err = p.communicate()
    if err:
        raise ValueError(err)