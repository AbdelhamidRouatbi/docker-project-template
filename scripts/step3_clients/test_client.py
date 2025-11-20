import pandas as pd
import sys, os

project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ift6758.ift6758.client.serving_client import ServingClient

client = ServingClient(ip="0.0.0.0", port=8000, features=["distance_from_net"])


PATH="./ift6758/data/nhl/csv/processed/test_sample.csv"
df=pd.read_csv(PATH)
df_out = client.predict(df)
print(df_out)

mes =client.download_registry_model(model="lr-both")
print(mes)

logs_out =client.logs()
print(logs_out)