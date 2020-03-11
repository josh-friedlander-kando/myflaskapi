import pickle

from flask import Flask, request

from ml_models.prophet_model import ProphetTemplate

app = Flask(__name__)


@app.route("/train", methods=['POST'])
def train():
    model = ProphetTemplate()
    context = request.get_json()
    model.train(context)
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{request.json["point_id"]}.pkl', 'wb+') as f:
        pickle.dump(model, f)
    return 'finished fitting model'


@app.route("/predict", methods=['POST'])
def predict():
    context = request.get_json()
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{request.json["point_id"]}.pkl', 'rb') as f:
        model = pickle.load(f)
    pred = model.predict(context)
    return pred
