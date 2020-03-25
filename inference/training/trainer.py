import os
import pickle

from dotenv import load_dotenv
from gradient import sdk_client
from kando import kando_client
from ml_models.xgboost_model import XGBoostTemplate
from ml_models.prophet_model import ProphetTemplate


def save_and_upload_model(model, name, gradi_client=None):
    print('saving model')
    try:
        model.model.stan_backend.logger = None  # https://github.com/facebook/prophet/issues/1361 (!!)
    except AttributeError:
        pass
    export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '/../../models'))
    path = export_dir + '/' + name + '.pkl'
    with open(path, 'wb+') as f:
        pickle.dump(model, f)
    if gradi_client is not None:
        _ = gradi_client.models.upload(path, name, 'Custom')
        print(f'Successfully uploaded model ID {_}')


if __name__ == '__main__':
    load_dotenv()
    base_url = "https://kando.herokuapp.com"
    client_ = kando_client.client(base_url, os.getenv('KEY'), os.getenv('SECRET'))
    gradient_client = sdk_client.SdkClient(os.getenv('APIKEY'))

    model = XGBoostTemplate
    # model = ProphetTemplate
    m = model()
    point_ids = os.getenv("POINT_IDS", [1012])
    for point in point_ids:
        m.do_train(client_, {
            "point_id": point,
            "start": os.getenv('START', 1554182371),
            "end": os.getenv('END', 1582008447),
            "prediction_param": os.getenv('PREDICTION_PARAM', 'EC')
        })
        m.save_metadata()
        name = f'model_{point}_{os.getenv("PREDICTION_PARAM", "no_param")}'
        name = f'model_predict_COD'
        save_and_upload_model(m, name, gradient_client)
