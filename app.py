import os
import pickle
from os.path import dirname

import pandas as pd
from dotenv import load_dotenv
from fbprophet import Prophet
from flask import Flask, request
from kando import kando_client

app = Flask(__name__)


@app.route("/")
def hello_world():
    return 'hello world'


@app.route("/train", methods=['POST'])
def train():
    point_id, unit_id, start, end, prediction_param = [request.json.get(x) for x in ['point_id', 'unit_id', 'start',
                                                                                     'end', 'prediction_param']]
    load_dotenv(dirname(__file__), '.env')
    base_url = "https://kando.herokuapp.com"
    client = kando_client.client(base_url, os.getenv('KEY'), os.getenv('SECRET'))
    data = client.get_all(point_id, unit_id, start, end)
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
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{point_id}.pkl', 'wb+') as f:
        pickle.dump(m, f)
    return 'finished fitting model'


@app.route("/predict", methods=['POST'])
def predict():
    point_id, pred_hours = request.json.get('point_id'), request.json.get('pred_hours')
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{point_id}.pkl', 'rb') as f:
        m = pickle.load(f)

    future = m.make_future_dataframe(periods=12 * pred_hours, freq='5min')
    forecast = m.predict(future)
    print('finished prediction')
    return forecast.to_dict()


