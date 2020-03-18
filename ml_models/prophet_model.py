import os
import pickle
from abc import ABC
from dotenv import load_dotenv
from kando import kando_client

import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics, cross_validation

from model_template import ModelTemplate

export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd()))


def fetch_and_process_data(client, context):
    prediction_param = context['prediction_param']
    client_context = {k: v for k, v in context.items() if k in ['point_id', 'start', 'end']}
    data = client.get_all(**client_context)
    if len(data['samplings']) == 0:
        print(f'No data found at point {context["point_id"]}')
        return None
    # TODO if model not fit bc of missing data, pass this on to predict method
    df = pd.DataFrame(data['samplings']).T[[prediction_param]]
    df.index = pd.to_datetime(df.index, unit='s')
    df = df.sort_index().astype(float)
    print('finished processing data successfully')
    return df.reset_index().rename(columns={'index': 'ds', prediction_param: 'y'}).copy()


def save_model(model):
    print('saving model')
    model.model.stan_backend.logger = None  # https://github.com/facebook/prophet/issues/1361 (!!)
    with open(export_dir + '/model.pkl', 'wb+') as f:
        pickle.dump(model, f)


class ProphetTemplate(ModelTemplate, ABC):
    def __init__(self):
        super().__init__()
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)

    def do_train(self, client, context):
        data = fetch_and_process_data(client, context)
        if data is not None:
            self.model.fit(data)
            print('finished fitting model')
        df_cv = cross_validation(self.model, period='180 days', initial='112 days', horizon='12H')
        metadata = performance_metrics(df_cv)
        metadata.to_json(export_dir + '/gradient-model-metadata.json')

    def do_predict(self, context):
        future = self.model.make_future_dataframe(periods=12 * context['pred_hours'], freq='5min')
        forecast = self.model.predict(future)
        print('finished prediction')
        if context['pred_only']:
            return forecast[['yhat']].to_dict()

        const_2, const_3 = 1.5, 2
        forecast = forecast.rename(columns={'yhat_upper': 'H1', 'yhat_lower': 'L1'}).copy()
        forecast['H2'], forecast['H3'] = const_2 * forecast['H1'], const_3 * forecast['H1']
        forecast['L2'], forecast['L3'] = const_2 * forecast['L1'], const_3 * forecast['L1']
        return forecast[['yhat', 'H1', 'H2', 'H3', 'L1', 'L2', 'L3']].to_dict()


if __name__ == '__main__':
    load_dotenv()
    base_url = "https://kando.herokuapp.com"
    p = ProphetTemplate()
    p.do_train(kando_client.client(base_url, os.getenv('KEY'), os.getenv('SECRET')), {
        "point_id": 1012,
        "unit_id": "",
        "start": 1554182371,
        "end": 1582008447,
        "prediction_param": "EC",
        "model_id": "ABC"
    })
    save_model(p)
