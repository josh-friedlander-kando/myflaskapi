import pickle

import pandas as pd
from fbprophet import Prophet
from flask import Flask, request
from kando import kando_client

import os
from os.path import join, dirname
from dotenv import load_dotenv

app = Flask(__name__)


@app.route("/")
def hello_world():
    return 'hello world'


@app.route("/predict", methods=['POST'])
def predict():
    point_id, periods = request.json.get('point_id'), request.json.get('periods')
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{point_id}.pkl', 'rb') as f:
        m = pickle.load(f)

    future = m.make_future_dataframe(periods=periods, freq='H')
    forecast = m.predict(future)
    print('finished prediction')
    return forecast.to_dict()


@app.route("/train", methods=['POST'])
def train():
    point_id, unit_id, start, end, prediction_param = request.json.get('point_id'), request.json.get('unit_id'), \
                                    request.json.get('start'), request.json.get('end'), request.json.get('prediction_param')
    load_dotenv(dirname(__file__), '.env')
    client = kando_client.client("https://kando.herokuapp.com", os.getenv('KEY'), os.getenv('SECRET'))
    data = client.get_all(point_id, unit_id, start, end)
    print('finished pulling data successfully')

    # prep data
    df = pd.DataFrame(data['samplings']).T[[prediction_param]]
    df.index = pd.to_datetime(df.index, unit='s')
    df = df.sort_index().astype(float)
    df.loc[df.COD == 10, 'COD'] = 2000
    df_prophet = df.reset_index().rename(columns={'index': 'ds', prediction_param: 'y'}).copy()
    print('finished processing data successfully')

    # instantiate prophet and and return prediction
    m = Prophet(changepoint_prior_scale=0.05, seasonality_mode='additive', yearly_seasonality=False,
                weekly_seasonality=True, daily_seasonality=True, seasonality_prior_scale=10)
    m.fit(df_prophet)
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{point_id}.pkl', 'wb+') as f:
        pickle.dump(m, f)
    return 'finished fitting model'


# 1012, '', 1554182371, 1582008447, 'COD'
