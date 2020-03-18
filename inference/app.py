import pickle
from flask import Flask, request

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    context = request.get_json()
    with open('/models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model.do_predict(context).to_json()


@app.route("/", methods=['GET'])
def health():
    return 'healthy', 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
