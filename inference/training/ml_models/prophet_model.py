from abc import ABC
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics, cross_validation

from model_template import ModelTemplate, fetch_data


def process_data(**kwargs):
    prediction_param = kwargs['prediction_param']
    data = fetch_data(kwargs["point_id"], kwargs["start"], kwargs["end"])
    df = pd.DataFrame(data['samplings']).T[[prediction_param]]
    df.index = pd.to_datetime(df.index, unit='s')
    df = df.sort_index().astype(float)
    return df.reset_index().rename(columns={'index': 'ds', prediction_param: 'y'}).copy()


class ProphetTemplate(ModelTemplate, ABC):
    def __init__(self):
        super().__init__()
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
        self.model.stan_backend.logger = None  # https://github.com/facebook/prophet/issues/1361 (!!)

    def do_train(self, **kwargs):
        data = process_data(**kwargs)
        if data is not None:
            self.model.fit(data)
            print('finished fitting model')

    def get_metadata(self):
        df_cv = cross_validation(self.model, period='180 days', initial='112 days', horizon='12H')
        metadata = performance_metrics(df_cv)
        return metadata.to_json()

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
