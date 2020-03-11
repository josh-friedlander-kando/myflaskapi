from abc import ABC

import pandas as pd
from fbprophet import Prophet

from .model_template import ModelTemplate


def fetch_and_process_data(client, context):
    prediction_param = context['prediction_param']
    client_context = {k: v for k, v in context.items() if k in ['point_id', 'start', 'end']}
    data = client.get_all(**client_context)
    df = pd.DataFrame(data['samplings']).T[[prediction_param]]
    df.index = pd.to_datetime(df.index, unit='s')
    df = df.sort_index().astype(float)
    print('finished processing data successfully')
    return df.reset_index().rename(columns={'index': 'ds', prediction_param: 'y'}).copy()


class ProphetTemplate(ModelTemplate, ABC):
    def __init__(self):
        super().__init__()
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)

    def do_train(self, client, context):
        self.model.fit(fetch_and_process_data(client, context))
        print('finished fitting model')

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
    p = ProphetTemplate()
    print('hi')
