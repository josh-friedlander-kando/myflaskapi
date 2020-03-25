import os
from abc import ABC
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics, cross_validation

from model_template import ModelTemplate


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


class ProphetTemplate(ModelTemplate, ABC):
    def __init__(self):
        super().__init__()
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
        self.df_cv = None

    def do_train(self, client, context):
        data = fetch_and_process_data(client, context)
        if data is not None:
            self.model.fit(data)
            print('finished fitting model')
        self.df_cv = cross_validation(self.model, period='180 days', initial='112 days', horizon='12H')

    def save_metadata(self):
        metadata = performance_metrics(self.df_cv)
        export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '/../../models'))
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
