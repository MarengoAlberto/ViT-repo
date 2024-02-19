#!/bin/bash

set -a
source environments/.env
python setup.py sdist --formats=gztar
gsutil cp $SOURCE_PACKAGE_FILE_NAME $PYTHON_PACKAGE_GCS_URI
gsutil ls -l $PYTHON_PACKAGE_GCS_URI
python distributed_utils/launch_distributed_training_job.py