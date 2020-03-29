import os
from dotenv import load_dotenv
from gradient import sdk_client

load_dotenv()
client = sdk_client.SdkClient(os.getenv('APIKEY'))
experiment_param = {
    "name": "xgboost_training",
    "project_id": "prfuvnyks",
    "container": "kandoenv/training:latest",
    "machine_type": "c5.xlarge",
    "cluster_id": "cljdd692n",
    "command": "python /code/trainer.py xgboost cloud --point_id 1012 --start 1554182371 --end 1582008447 --prediction_param EC"
}
handle = client.experiments.run_single_node(**experiment_param)
print(handle)
