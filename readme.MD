# Flask app for deploying machine learning models.

You can add new models in `inference/training/ml_models` that expose a `do_train` and `do_predict` method.

The app accepts a `context` with all relevant parameters. Models are uploaded to Gradient automatically upon training, and `inference/deployment.py` deploys them.

For more see `bin/Machine Learning with Gradient`.
