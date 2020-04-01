import os
import argparse
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
    # "workspace_url": "https://github.com/kando-env/flask_api/",
    # "workspace_ref": "master",
    # "workspace_username": os.getenv('GIT_USERNAME'),
    # "workspace_password": os.getenv('GIT_PASSWORD'),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="which command to use")
    args = parser.parse_args()
    experiment_param['command'] = args.command
    print(client.experiments.run_single_node(**experiment_param))
