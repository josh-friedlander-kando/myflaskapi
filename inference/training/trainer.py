import os
import pickle
from importlib import import_module

from dotenv import load_dotenv
from flask import Flask, request
from gradient import sdk_client

app = Flask(__name__)


def save_model(model, path):
    with open(path, 'wb+') as f:
        pickle.dump(model, f)
    print(f'Model saved locally at {path}')


def upload_model(path, name):
    load_dotenv()
    gradient_client = sdk_client.SdkClient(os.getenv('APIKEY'))
    _ = gradient_client.models.upload(path, name, 'Custom')
    print(f'Successfully uploaded model ID {_}')
    return _


def train(context):
    environment, model = context.pop("environment"), context.pop("model")
    assert environment in ["cloud", "local"], "environment must be either 'cloud' or 'local'"

    # import model module and instantiate it
    model_module = "ml_models." + model + '_model'
    model_class = model.title() + 'Template'
    model_module = import_module(model_module)
    my_model = getattr(model_module, model_class)
    m = my_model()
    print(context)
    m.train(**context)

    # save model and potentially upload model + metadata
    model_name = model#+ '_' + '_'.join([str(x) for x in context.values()])  # TODO pick a better naming convention!
    export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '../../../models'))
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)
    model_path = export_dir + '/' + model_name + '.pkl'
    save_model(m, model_path)
    if environment == "cloud":
        m.save_metadata()
        return upload_model(model_path, model_name)
    return model_path


@app.route("/train", methods=['POST'])
def train_model():
    context = request.get_json()
    return train(context)


@app.route("/", methods=['GET'])
def health():
    return 'healthy', 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
