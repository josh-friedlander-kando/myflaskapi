# Machine Learning with Gradient

Firstly: [this](https://docs.paperspace.com/) is the correct link for the documentation. Other stuff is out of date!

### Communicating with Gradient

- UI. OK for basic tasks but can't handle most custom stuff
- Can use the CLI. If so convenient to define all parameters in a YAML file
- Python SDK. Define all variables in the file

### Setting up a model

Once you train a model, you want to save it in a location where you can find it so that you can access it for retraining or deployment. The model path is usually `/artifacts` in Gradient's server; the best way to save it is 

`export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd()))`

which will get you either the environment variable of `PS_MODEL_PATH` or your local directory. You can define the path in the `config.yaml` file.

Once the model is uploaded to Gradient, you can **deploy** it. The best way to do this is with a custom deployment - we use a Flask container. After creating it, you should make a Dockerfile, run Docker build (potentially Docker run locally first to test it works), and Docker push it.

Then you can deploy based on oyur current code or a Git workspace (if you don't give a url for the workspace, your current code is zipped and uploaded as you deploy), so you can make changes and iterate very quickly. 

