import os
import time
import argparse

from dotenv import load_dotenv
from gradient import sdk_client

parser = argparse.ArgumentParser()
parser.add_argument('model', help='Gradient ID for model to be deployed')
args = parser.parse_args()

load_dotenv()
client = sdk_client.SdkClient(os.getenv('APIKEY'))
deploy_param = {
    "name": "xgboost_model_deployment",
    "model_id": args.model,
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
}
deployment_id = client.deployments.create(**deploy_param)
time.sleep(5)
client.deployments.start(deployment_id)
