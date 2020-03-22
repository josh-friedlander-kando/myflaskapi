import os
import time

from dotenv import load_dotenv
from gradient import sdk_client

load_dotenv()
client = sdk_client.SdkClient(os.getenv('APIKEY'))
deploy_param = {
    "deployment_type": "Custom",
    "image_url": "kandoenv/inference:latest",
    "name": "deploy_22_3",
    "ports": 3000,
    "machine_type": "c5.xlarge",
    "instance_count": 1,
    "cluster_id": "cljdd692n",
    "container_model_path": "/models",
    "method": "/",
    "auth_username": "kando", # make env variable
    "auth_password": "kando",
    "model_id": "mop9t799dilr29"
}
deployment_id = client.deployments.create(**deploy_param)
time.sleep(5)
client.deployments.start(deployment_id)
