import requests
import json
from input_example import get_input_features_df


DEFAULT_PORT=8000
PATH="..."






X = get_input_features_df()

r = requests.post( "http://0.0.0.0:{DEFAULT_PORT}/predict", json=json.loads(X.to_json()) )
print(r.json())