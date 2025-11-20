import requests
import json
from utils.input_function import get_input_features_df

input_feature_1=["distance_from_net"]
input_feature_2=["shot_angle"]
input_features=input_feature_1+input_feature_2
print(input_features)
DEFAULT_PORT=8000

X = get_input_features_df(input_features)

r = requests.post(f"http://0.0.0.0:{DEFAULT_PORT}/predict", json=json.loads(X.to_json()) )
print(r.json())