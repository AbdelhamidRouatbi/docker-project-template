import json
import requests
import pandas as pd
import logging

input_feature_1="distance_from_net"
input_feature_2="shot_angle"

DEFAULT_PORT=8000
DEFAULT_WORKSPACE="IFT6758_team4"
DEFAULT_PROJECT="milestone_2"
DEFAULT_MODEL="lr-angle"
DEFAULT_VERSION="v1"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 8000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = input_feature_1
        self.features = features

        # any other potential initialization
           


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """ 
        df_empty = pd.DataFrame()       

        X_model = X[self.features].copy()
        logger.info(f"Senting Data with {self.features} to app")

        try:
            r = requests.post(f"{self.base_url}/predict", json=json.loads(X_model.to_json()) )
        except Exception as e:
            logger.error(f"Failed to connect to app. {e}")
            return df_empty 
        
        try:
            response = r.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return df_empty
        
        logger.info("Predictions received")
        try:
            y_pred = response["predictions"]
            y_proba = response["probabilities"] 
        except Exception as e:
            logger.error(f"incorect data received: {e}, {response}")
            return df_empty

        df_app = pd.DataFrame({
            "prediction": y_pred,
            "proba_non_goal": [p[0] for p in y_proba],
            "proba_goal": [p[1] for p in y_proba]
        }, index=X_model.index)

        return X.join(df_app)


    def logs(self) -> dict:
        """Get server logs"""
        try:
            res = requests.get(f"{self.base_url}/logs")
        except Exception as e:
            log_message = f"Failed: {e}"
            logger.error(log_message)
            return {"error": log_message}
        
        return res.json()



    def download_registry_model(self, workspace: str = "IFT6758_team4",model: str = "lr-distance",version: str = "v1",project: str = "milestone_2",) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
    


        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        payload = {
            "workspace": workspace,
            "project": project,
            "model": model,
            "version": version
            }
        try:
            res = requests.post(f"{self.base_url}/download_registry_model", json=payload)
        except Exception as e:
            log_message=f'Request to choose model {model}, version {version} failed'
            logger.error(log_message)
            return {"error" : log_message}
        
        log_message=f'Request to swap model {model}, version {version} sent'
        logger.info(log_message)
        return (res.json())

#PATH="./ift6758/data/nhl/csv/processed/test_sample.csv"
#df=pd.read_csv(PATH)
#df_out = client.predict(df)
#print(df_out)

#mes =client.download_registry_model(model="lr-both")
#print(mes)

#logs_out =client.logs()
#print(logs_out)


