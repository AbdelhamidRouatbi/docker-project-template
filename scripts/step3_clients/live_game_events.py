import requests
import pandas as pd

import pandas as pd
import sys, os

project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ift6758.ift6758.client.serving_client import ServingClient


last_seen = {}
client = ServingClient(ip="0.0.0.0", port=8000, features=["distance_from_net"])


def get_all_game_events(game_id):

    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

    data = requests.get(url).json()

    events = data.get("plays", [])
    return events


def build_dataframe_for_predict(events):
    "To complete based on the exact requirements in step 4 and 5. "
    "The output can be any df, since the filtering for the features (distance, angle) takes place when you built the client service below"
    df_out = pd.DataFrame(events)
    return df_out


def poll_and_predict(game_id: int):
    global last_seen

    all_events = get_all_game_events(game_id)
    if not all_events:
        return None

    last_index = last_seen.get(game_id, -1)
    new_events = all_events[last_index + 1 :]

    if not new_events:
        return None


    # build features and sent to predict (this will be built later in step4 and 5)
    df_input = build_dataframe_for_predict(new_events)
    num_predicted = len(new_events)

    df_output = client.predict(df_input)

    # update tracker
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


