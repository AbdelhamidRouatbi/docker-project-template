"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
#imports
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
#import ift6758
import wandb
import json
import io

# notes
#to run (when in ./serving): gunicorn --bind 0.0.0.0:8000 app:app


#global variables
MODEL_NAMES=["lr-angle", "lr-distance", "lr-both"]

LOG_FILE = "./flask.log"
FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_WORKSPACE="IFT6758_team4"
DEFAULT_PROJECT="milestone_2"
DEFAULT_MODEL= MODEL_NAMES[0]
CURRENT_MODEL_STRING = MODEL_NAMES[0]
DEFAULT_VERSION="v1"
ARTIFACT_DIR = f"../artifacts"
MODEL=None
os.makedirs(ARTIFACT_DIR, exist_ok=True)


app = Flask(__name__)

def download_model(workspace, project, model_name, version):
    model_path = f'{ARTIFACT_DIR}/{model_name}:{version}/{model_name}.joblib'
    if os.path.exists(model_path):

        log_message=f'Model {model_name} already downloaded.'
        app.logger.info(log_message) # debug, info, warning, error, critical
        log_message=f'Model {model_name} is loaded'
        app.logger.info(log_message) 
        print(log_message)
        CURRENT_MODEL_STRING=model_name
        return joblib.load(model_path), log_message
    
    else:
        try:
            run = wandb.init(project=project) 

            wb_path = f"{workspace}/{project}/{model_name}:{version}"
            artifact = run.use_artifact(wb_path, type='model')
            artifact.download(root=f'{ARTIFACT_DIR}/{model_name}:{version}')

            log_message=f'Model {model_name} was successfully downloaded. '
            app.logger.info(log_message) 
            try:
                MODEL = joblib.load(model_path)
                log_message=f'Model {model_name} is loaded. '
                app.logger.info(log_message)
                CURRENT_MODEL_STRING=model_name
            except Exception as e:
                log_message=f'Model {model_name} failed loaded.'
                app.logger.error(log_message)
                return MODEL, log_message
            return MODEL, log_message
        
        except Exception as e:

            log_message=f'Model {model_name} failed to download. Exception: {e}'
            app.logger.error(log_message)
            print(log_message)

            return MODEL, log_message

        finally:
            wandb.finish()

def before_first_request():

    """
    Load default model, 
    setup logging handler)
    """
    logging.basicConfig(filename=LOG_FILE, 
                        filemode="w",
                        level=logging.INFO, format=FORMAT)
    
    return download_model(DEFAULT_WORKSPACE, DEFAULT_PROJECT, DEFAULT_MODEL, DEFAULT_VERSION)

def is_missing(value):
    return value is None or (isinstance(value, str) and value.strip() == "")

MODEL, mes = before_first_request()


@app.route("/")
def hello():
    log_message="Hi, there. This is team 4"
    app.logger.info(log_message)
    return "Hi, there. This is team 4. Pending..."


@app.route("/logs", methods=["GET"])
def logs():

    """Reads data from the log file 
    and returns them as the response"""
    N = 100
    logs=[]
    # read the log file specified and return the data
    with open("flask.log", "r") as f:
        lines = f.readlines()[-N:]
        for line in lines:
            logs.append(line.strip())
    response=logs

    return jsonify(response)  


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    Json schema expected:
        {
            workspace: (required),
            project: (required),
            model: (required),
            version: (required),
        }
    COMET API KEY instructions bellow were ignored
    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    #comet_api_key = os.getenv("COMET_API_KEY")
    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    """

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    required = ["workspace", "project", "model", "version"]
    missing = [key for key in required if is_missing(json.get(key))]

    if missing:
        log_message=f"Missing required fields: {', '.join(missing)}"
        app.logger.error(log_message)
        return jsonify(log_message)

    workspace=json.get("workspace")
    project=json.get("project")
    model_name = json.get("model")
    version=json.get("version")

    # Load the model
    MODEL, response = download_model(workspace,project,model_name,version) 
    #print(MODEL.feature_names_in_)
    return jsonify(response)



@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json_data = request.get_json()
    #app.logger.info(json)

    if json_data is None:
        log_message="No data received"
        app.logger.error(log_message)
        return jsonify(log_message)
    
    app.logger.info("json received")
    app.logger.info("/predicting...")
    app.logger.info(f"/predict: received keys: {list(json_data.keys())}")

    try:
        X = pd.read_json(io.StringIO(json.dumps(json_data)))

    except Exception as e:
        log_message=f"Could not parse json data: {e}"
        app.logger.error(log_message)
        return jsonify(log_message)
    
    log_message="Parsing json data successful"
    app.logger.info(log_message)

    if CURRENT_MODEL_STRING == "lr-distance":
        required = ["distance_from_net"]
    elif CURRENT_MODEL_STRING == "lr-angle":
        required = ["shot_angle"]
    elif CURRENT_MODEL_STRING == "lr-both":
        required = ["distance_from_net", "shot_angle"]

    missing = []
    for i in required:
        if i not in X.columns:
            missing.append(i)

    if missing:
        log_message = f"Missing required features: {missing}"
        app.logger.error(log_message)
        return jsonify(log_message)

    app.logger.info(f"/predict: DataFrame shape: {X.shape}")

    X_model = X[required]
    X_arr = X_model.values

    try:
        y_proba = MODEL.predict_proba(X_arr).tolist()   
        y_pred = MODEL.predict(X_arr).tolist()        
    except Exception as e:
        log_message = f"Prediction failed: {e}"
        app.logger.error(log_message)
        return jsonify({"error": log_message})

    app.logger.info(f"Predicted {len(y_pred)} shots.")

    # Return response

    response = {
        "predictions": y_pred,
        "probabilities": y_proba,
        "n_samples": len(y_pred),
        "used_features": required,
        "model_name": CURRENT_MODEL_STRING,
    }

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

if __name__ == "__main__":
    app.run()
