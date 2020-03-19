import os
import pickle
from flask import Flask, request
from prophet_model import ProphetTemplate

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    context = request.get_json()
    with open(f'/models/model_{context["POINT_ID"]}_{context["PREDICTION_PARAM"]}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model.do_predict(context)


@app.route("/", methods=['GET'])
def health():
    return 'healthy', 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
