import os
import pickle
from os.path import dirname

import pandas as pd
from dotenv import load_dotenv
from fbprophet import Prophet
from flask import Flask, request
from kando import kando_client

app = Flask(__name__)


@app.route("/train", methods=['POST'])
def train():
    """
    accepts POST arguments of point_id, start, end, prediction_param, and optionally unit_id
    note that the pop() method removes the prediction_param from the dict, so we can pass it to client.get_all()
    """
    assert all(x in request.json for x in ['point_id', 'start', 'end', 'prediction_param'])  # unit_id is optional
    prediction_param = request.json.pop('prediction_param')
    load_dotenv(dirname(__file__), '.env')
    base_url = "https://kando.herokuapp.com"
    client = kando_client.client(base_url, os.getenv('KEY'), os.getenv('SECRET'))
    data = client.get_all(**request.json)
    print('finished pulling data successfully')

    # prep data
    df = pd.DataFrame(data['samplings']).T[[prediction_param]]
    df.index = pd.to_datetime(df.index, unit='s')
    df = df.sort_index().astype(float)
    df_prophet = df.reset_index().rename(columns={'index': 'ds', prediction_param: 'y'}).copy()
    print('finished processing data successfully')

    # instantiate prophet and and train model
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    m.fit(df_prophet)
    # TODO we want to have multiple models, how do we ID and load the correct model?
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{request.json["point_id"]}.pkl', 'wb+') as f:
        pickle.dump(m, f)
    return 'finished fitting model'


@app.route("/predict", methods=['POST'])
def predict():
    # generates forecast df and returns either pred_only, or else range of thresholds derived from yhat_upper/lower
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{request.json.get("point_id")}.pkl', 'rb') as f:
        m = pickle.load(f)

    future = m.make_future_dataframe(periods=12 * request.json.get('pred_hours'), freq='5min')
    forecast = m.predict(future)
    print('finished prediction')
    if request.json['pred_only']:
        return forecast[['yhat']].to_dict()

    const_2, const_3 = 1.5, 2
    forecast = forecast.rename(columns={'yhat_upper': 'H1', 'yhat_lower': 'L1'}).copy()
    forecast['H2'], forecast['H3'] = const_2 * forecast['H1'], const_3 * forecast['H1']
    forecast['L2'], forecast['L3'] = const_2 * forecast['L1'], const_3 * forecast['L1']
    return forecast[['yhat', 'H1', 'H2', 'H3', 'L1', 'L2', 'L3']].to_dict()
