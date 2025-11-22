import requests
import pandas as pd

import pandas as pd
import sys, os

project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ift6758.ift6758.client.serving_client import ServingClient
from scripts.step1_data.nhl_pbp.transform import _iter_rows_from_game_json, EVENT_COLUMNS
from scripts.step1_data.feature_engineering_milestone_3 import FeatureEngineering


last_seen = {}
client = ServingClient(ip="0.0.0.0", port=8000, features=["distance_from_net"])


def get_all_game_events(game_id):

    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

    data = requests.get(url).json()
    return data 


def build_dataframe_for_predict(events):
    "To complete based on the exact requirements in step 4 and 5. "
    "The output can be any df, since the filtering for the features (distance, angle) takes place when you built the client service below"
    
    rows = list(_iter_rows_from_game_json(events))

    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    fe = FeatureEngineering()

    df = fe.calculate_distance_from_net(df)
    df = fe.calculate_empty_net(df)
    
    return df


def poll_and_predict(game_id: int):
    global last_seen

    all_game_data = get_all_game_events(game_id)  # Full game dict
    if not all_game_data:
        return None

    all_events = all_game_data.get("plays", [])  # Extract plays list
    if not all_events:
        return None

    last_index = last_seen.get(game_id, -1)
    new_events = all_events[last_index + 1 :]

    if not new_events:
        return None

    game_data_subset = all_game_data.copy()
    game_data_subset["plays"] = new_events

    df_input = build_dataframe_for_predict(game_data_subset)
    num_predicted = len(new_events)

    df_output = client.predict(df_input)

    last_seen[game_id] = len(all_events) - 1

    return df_output, num_predicted




#Example
if __name__ == "__main__":
    game_id = 2023020345

    # Everytime we call it, it sends a prediction for new events. As requested, feature engineering comes later.
    df_predictions, num =poll_and_predict(game_id)

    if num == 0:
        print("No new events predicted.")
    else:
        print(f"{num} new events predicted.")
        print(df_predictions.head())


