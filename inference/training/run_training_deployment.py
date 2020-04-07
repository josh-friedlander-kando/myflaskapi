import os
import time

from dotenv import load_dotenv
from gradient import sdk_client

load_dotenv()
client = sdk_client.SdkClient(os.getenv('APIKEY'))
deploy_param = {
    "name": "xgboost_model_training_deployment",
    "deployment_type": "Custom",
    "image_url": "kandoenv/training:latest",
    "ports": 3000,
    "machine_type": "c5.xlarge",
    "instance_count": 1,
    "cluster_id": "cljdd692n",
    "model_id": "mozz3f5i9x1k5a",
    "container_model_path": "/models",
    "method": "/",
    "auth_username": os.getenv('AUTH_USERNAME'),
    "auth_password": os.getenv('AUTH_PASSWORD'),
    "image_server": "index.docker.io",
    "image_username": os.getenv('IMAGE_USERNAME'),
    "image_password": os.getenv('IMAGE_PASSWORD'),
}
deployment_id = client.deployments.create(**deploy_param)
time.sleep(5)
client.deployments.start(deployment_id)
