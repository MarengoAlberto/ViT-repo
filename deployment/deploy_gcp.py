import logging
import os

from dotenv import load_dotenv

import google.cloud.aiplatform as aiplatform

logging.getLogger().setLevel(logging.INFO)

load_dotenv()

APP_NAME = os.environ['APP_NAME']
MAR_MODEL_OUT_PATH = os.environ['MAR_MODEL_OUT_PATH']
MODEL_VERSION = os.environ['MODEL_VERSION']
PROJECT_ID = os.environ['PROJECT_ID']
BUCKET_NAME = os.environ['BUCKET_NAME']
CUSTOM_PREDICTOR_IMAGE_URI = os.environ['CUSTOM_PREDICTOR_IMAGE_URI']

aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_NAME)

model_display_name = f"{APP_NAME}-v{MODEL_VERSION}"
model_description = "PyTorch Image classifier with custom container"

MODEL_NAME = APP_NAME
health_route = "/ping"
predict_route = f"/predictions/{MODEL_NAME}"
serving_container_ports = [7080]

model = aiplatform.Model.upload(
    display_name=model_display_name,
    description=model_description,
    serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
    serving_container_predict_route=predict_route,
    serving_container_health_route=health_route,
    serving_container_ports=serving_container_ports,
    artifact_uri=f'gs://{BUCKET_NAME}/{MAR_MODEL_OUT_PATH}',
)

model.wait()

print(model.display_name)
print(model.resource_name)

endpoint_display_name = f"{APP_NAME}-endpoint"
endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

traffic_percentage = 100
machine_type = "n1-standard-4"
deployed_model_display_name = model_display_name
min_replica_count = 1
max_replica_count = 3
sync = True

model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=deployed_model_display_name,
    machine_type=machine_type,
    traffic_percentage=traffic_percentage,
    sync=sync,
)

endpoint_display_name = f"{APP_NAME}-endpoint"
filter = f'display_name="{endpoint_display_name}"'

for endpoint_info in aiplatform.Endpoint.list(filter=filter):
    print(f"Endpoint display name = {endpoint_info.display_name} resource id ={endpoint_info.resource_name} ")

endpoint = aiplatform.Endpoint(endpoint_info.resource_name)

endpoint.list_models()

