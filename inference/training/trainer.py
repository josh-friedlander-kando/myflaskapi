import argparse
import os
import pickle
from importlib import import_module

from dotenv import load_dotenv
from gradient import sdk_client


def save_model(model, path):
    with open(path, 'wb+') as f:
        pickle.dump(model, f)
    print(f'Model saved locally at {path}')


def upload_model(path, name):
    load_dotenv()
    gradient_client = sdk_client.SdkClient(os.getenv('APIKEY'))
    _ = gradient_client.models.upload(path, name, 'Custom')
    print(f'Successfully uploaded model ID {_}')


def main(args):
    # separate model args into key-value pairs
    model_args_orig = dict(zip(args.model_args[::2], args.model_args[1::2]))
    model_args = {}
    for k, v in model_args_orig.items():
        try:
            model_args[k.lstrip('-')] = int(v)
        except ValueError:
            model_args[k.lstrip('-')] = v

    # import model module and instantiate it
    model_module = "ml_models." + args.model + '_model'
    model_class = args.model.title() + 'Template'
    model_module = import_module(model_module)
    my_model = getattr(model_module, model_class)
    m = my_model()
    m.train(**model_args)

    # save model and potentially upload model + metadata
    model_name = args.model + '_' + '_'.join([str(x) for x in model_args.values()])
    export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '../../../models'))
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)
    model_path = export_dir + '/' + model_name + '.pkl'
    save_model(m, model_path)
    assert args.environment in ["cloud", "local"], "environment must be either 'cloud' or 'local'"
    if args.environment == "cloud":
        m.save_metadata()
        upload_model(model_path, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", nargs="?", default="local",
                        help="if working locally or in gradient. if gradient, upload model + metadata")
    parser.add_argument("model", help="which model to instantiate")
    parser.add_argument('model_args', nargs=argparse.REMAINDER, help="all other key-value args for the specific model")
    args_ = parser.parse_args()
    main(args_)
