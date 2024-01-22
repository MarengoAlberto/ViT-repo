import logging
import os
from dotenv import load_dotenv

from google.cloud import aiplatform

logging.getLogger().setLevel(logging.INFO)
load_dotenv()

PROJECT_ID = os.environ['PROJECT_ID']
REGION = os.environ['REGION']
BUCKET_URI = os.environ['BUCKET_NAME']
APP_NAME = os.environ['APP_NAME']
JOB_NAME = os.environ['JOB_NAME']
PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = os.environ['PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI']
PYTHON_PACKAGE_GCS_URI = os.environ['PYTHON_PACKAGE_GCS_URI']
python_module_name = "trainer.task"


print(f"APP_NAME={APP_NAME}")
print(f"PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI={PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI}")
print(f"python_package_gcs_uri={PYTHON_PACKAGE_GCS_URI}")
print(f"python_module_name={python_module_name}")

print(f"JOB_NAME={JOB_NAME}")

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=f"{JOB_NAME}",
    python_package_gcs_uri=PYTHON_PACKAGE_GCS_URI,
    python_module_name=python_module_name,
    container_uri=PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI,
)

# Training cluster worker pool configuration
REPLICA_COUNT = os.environ['NUM_NODES']
MACHINE_TYPE = os.environ.get('MACHINE_TYPE', "n1-standard-32")
ACCELERATOR_TYPE = os.environ.get('ACCELERATOR_TYPE', "NVIDIA_TESLA_P100")
ACCELERATOR_COUNT = os.environ.get('ACCELERATOR_COUNT', 4)

SERVICE_ACCOUNT = os.environ['SERVICE_ACCOUNT']

# Reduction Server configuration
REDUCTION_SERVER_COUNT = os.environ.get("REDUCTION_SERVER_COUNT", 4)
REDUCTION_SERVER_MACHINE_TYPE = os.environ.get("REDUCTION_SERVER_MACHINE_TYPE", "n1-highcpu-16")
REDUCTION_SERVER_IMAGE_URI = os.environ.get("REDUCTION_SERVER_IMAGE_URI",
                                            "us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest")
ENVIRONMENT_VARIABLES = {"NCCL_DEBUG": "INFO"}

model = job.run(
    replica_count=REPLICA_COUNT,
    machine_type=MACHINE_TYPE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
    # reduction_server_replica_count=REDUCTION_SERVER_COUNT,
    # reduction_server_machine_type=REDUCTION_SERVER_MACHINE_TYPE,
    # reduction_server_container_uri=REDUCTION_SERVER_IMAGE_URI,
    environment_variables=ENVIRONMENT_VARIABLES,
    service_account=SERVICE_ACCOUNT,
    sync=True,
)
