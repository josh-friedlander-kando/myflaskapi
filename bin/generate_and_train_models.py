import time
import requests
from datetime import datetime, timedelta

all_points = [925, 888, 896, 906, 911, 920, 926, 927, 889, 891]

base_url = 'http://127.0.0.1:5000/'
start = (datetime.now() - timedelta(weeks=16)).replace(hour=0, minute=0, second=0, microsecond=0)
end = int((start + timedelta(weeks=16)).timestamp())
start = int(start.timestamp())

prediction_param = 'EC'

train_params = {'unit_id': '', 'start': start, 'end': end, 'prediction_param': prediction_param}
pred_params = {'pred_hours': 12, 'pred_only': True}

for p in all_points:
    train_params['point_id'] = p
    train_params['model_id'] = '_'.join([str(x) for x in [p, start, end]]) + '_' + prediction_param
    start_time = time.time()
    r = requests.post(base_url + 'create', json=train_params)
    assert r.ok
    pred_params['point_id'] = p
    pred_params['model_id'] = '_'.join([str(x) for x in [p, start, end]]) + '_' + prediction_param
    r = requests.post(base_url + 'predict', json=pred_params)
    print(f'Finished point {p}, took {time.time() - start_time} seconds')