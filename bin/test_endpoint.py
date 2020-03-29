import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

# env = 'local'
env = 'gradient'
model = 'xgboost'
# model = 'prophet'

base_url = 'https://gradient-trial-2.paperspace.com/model-serving/desf7ero8k1uk3y'
if env == 'local':
    base_url = 'http://127.0.0.1:3000/'

load_dotenv()
model_address = "models/"
prediction_param = 'EC'
auth = (os.getenv('AUTH_USERNAME'), os.getenv('AUTH_PASSWORD'))
s = (datetime.now() - timedelta(weeks=16)).replace(hour=0, minute=0, second=0, microsecond=0)
start = int(s.timestamp())
end = int((s + timedelta(weeks=16)).timestamp())

request_header = {'pred_hours': 12,
                  'pred_only': True,
                  'model_address': model_address,
                  'model_name': 'model_predict_COD'}
if model == 'prophet':
    request_header['model_name'] = 'model_1012_EC'

start_time = time.time()
assert requests.get(base_url, auth=auth).ok  # check GET returns 200
assert requests.post(base_url + '/predict', auth=auth, json=request_header).ok
print(f'Finished in {time.time() - start_time} seconds')
