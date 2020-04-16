# Flask servers for training/deploying ML models.

### Add, train and deploy new models, using REST APIs.

There are two servers - training and inference. To deploy on Gradient (using the Docker image `kandoenv/x`), use `deploy_x_server.py`. To run on localhost for testing, simply call the appropriate Python script, `x_server.py` 

You can add new models in `api/training/ml_models`. They must expose a `do_train` and `do_predict` method. `do_predict` must return a JSON-like dict.

### Deployment
Calling `api/training/deploy_training_server.py` spins up a generic model training deployment on Gradient. It takes (via HTTP JSON POST request) a model name and training parameters.

Calling `api/deploy_inference_server.py` with a model ID deploys that model on Gradient, allowing inference via HTTP requests with JSON parameters.
 
In either case the JSON parameters can include unnecessary parameters (which are just discarded).

More details at [Machine Learning with Gradient](bin/Machine Learning with Gradient.md).

### Sample code:
1. `python api/training/deploy_training_server.py`
2. Wait a few minutes and confirm api is healthy. First, we'll get the address:
run `gradient deployments list` and find ID of "generic_model_training_deployment" - in this example `deqes4z6pg480g`
3. Convenience: `export URL=https://gradient-trial-2.paperspace.com/model-serving/`
4. Health check with username:password `curl "${URL}deqes4z6pg480g" --user kando:kando`
5. Confirm that it returns 200. (If it takes longer than 3-4 minutes job has probably failed.)
6. Train model!
`curl "${URL}deqes4z6pg480g/train" -X POST -d '{"model":"xgboost_cod", "point_id": 1012, "start": 1554182371, "end": 1582008447}' -H 'Content-Type: application/json' --user kando:kando`
7. Returned model ID will be printed, in this case `mo5vdb75h9hu0s`
8. Let's deploy it `python api/deploy_inference_server.py mo5vdb75h9hu0s`
9. Get deployment id (in this case `deqes4z6pg480g`) and do health check: `curl "${URL}desamwjv3vg0gbb" --user kando:kand` 
10. Confirm that it returns 200.
11. Inference! `curl "${URL}deqes4z6pg480g/predict" -X POST -d '{"model":"xgboost_cod", "point_id": 1012, "start": 1554182371, "end": 1582008447}' -H 'Content-Type: application/json' --user kando:kando > res.json`
12. In this example, the return JSON is large so it is piped to `res.json`
