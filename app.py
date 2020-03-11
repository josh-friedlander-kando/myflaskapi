import pickle

from flask import Flask, request, jsonify

from ml_models.prophet_model import ProphetTemplate
from ml_models.lin_reg import LinRegTemplate


app = Flask(__name__)
flask_model = ProphetTemplate  # LinRegTemplate


@app.route("/create", methods=['POST'])
def create():
    model = flask_model()
    context = request.get_json()
    model.train(context)
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{context["model_id"]}.pkl', 'wb+') as f:
        pickle.dump(model, f)
    return 'created and trained model'


@app.route("/train", methods=['POST'])
def train():
    context = request.get_json()
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{context["model_id"]}.pkl', 'rb') as f:
        model = pickle.load(f)
    model.train(context)
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{context["model_id"]}.pkl', 'wb+') as f:
        pickle.dump(model, f)
    return 'finished fitting model'


@app.route("/predict", methods=['POST'])
def predict():
    context = request.get_json()
    with open(f'/Users/joshfriedlander/Documents/research/models/model_{context["model_id"]}.pkl', 'rb') as f:
        model = pickle.load(f)
    pred = model.predict(context)
    return jsonify(pred)
