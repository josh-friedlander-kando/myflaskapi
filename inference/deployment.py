import os
import time

from dotenv import load_dotenv
from gradient import sdk_client

load_dotenv()
client = sdk_client.SdkClient(os.getenv('APIKEY'))
deploy_param = {
    "name": "deploy_25_3",
    "model_id": "mosjclv7k2se3pz",
    "deployment_type": "Custom",
    "image_url": "kandoenv/inference:latest",
    "ports": 3000,
    "machine_type": "c5.xlarge",
    "instance_count": 1,
    "cluster_id": "cljdd692n",
    "container_model_path": "/models",
    "method": "/",
    "auth_username": os.getenv('AUTH_USERNAME'),
    "auth_password": os.getenv('AUTH_PASSWORD'),
    # "docker_args": ["python", "/code/app_xgboost.py"]
}
deployment_id = client.deployments.create(**deploy_param)
time.sleep(5)
client.deployments.start(deployment_id)
