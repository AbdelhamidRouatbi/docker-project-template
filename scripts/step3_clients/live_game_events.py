import requests
import pandas as pd
import sys, os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ift6758.ift6758.client.serving_client import ServingClient
from scripts.step1_data.feature_engineering_milestone_3 import FeatureEngineering

SERVING_HOST = os.getenv("SERVING_HOST", "127.0.0.1")
SERVING_PORT = int(os.getenv("SERVING_PORT", "5000"))


# Track last seen index per game
last_seen = {}

# Local ServingClient
client = ServingClient(
    ip=SERVING_HOST,
    port=SERVING_PORT,
    features=["distance_from_net", "shot_angle", "empty_net"]
)

# -------------------------------------------------------------
# Fetch game JSON
# -------------------------------------------------------------
def get_game_json(game_id):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    return requests.get(url).json()


# -------------------------------------------------------------
# Extract plays (supports new NHL API)
# -------------------------------------------------------------
def extract_all_plays(game_json):

    # NEW NHL API FORMAT 
    if "plays" in game_json and isinstance(game_json["plays"], list):
        return game_json["plays"]

    # OLD FORMAT 
    try:
        return game_json["liveData"]["plays"]["allPlays"]
    except:
        return []


# -------------------------------------------------------------
# Build dataframe 
# -------------------------------------------------------------
def build_dataframe_for_predict(new_events, game_json):

    rows = []

    home_id = game_json["homeTeam"]["id"]
    away_id = game_json["awayTeam"]["id"]

    for ev in new_events:

        type_key = ev.get("typeDescKey", "").lower()

        # we only model shot-like events
        if type_key not in ("shot-on-goal", "missed-shot", "blocked-shot", "goal"):
            continue
        d = ev.get("details", {})
        x = d.get("xCoord")
        y = d.get("yCoord")
        shooter_team = d.get("eventOwnerTeamId")

        if x is None or y is None or shooter_team is None:
            continue

        # home / away flags
        is_home = shooter_team == home_id
        is_away = shooter_team == away_id

        situation_code = ev.get("situationCode", "")

        rows.append({
            "event_type": type_key.upper(),
            "x_coord": x,
            "y_coord": y,
            "event_team": "home" if is_home else "away",

            "home": is_home,
            "away": is_away,

            "period": ev["periodDescriptor"]["number"],
            "time_remaining": ev["timeRemaining"],
            "goalie_name": None,
            "situation_code": situation_code,
            "is_goal": 1 if type_key == "goal" else 0
        })

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ---- APPLY FEATURE ENGINEERING ----
    fe = FeatureEngineering()
    df = fe.calculate_distance_from_net(df)
    df = fe.calculate_empty_net(df)

    print(f"DF BUILT FOR MODEL: {df.shape[0]} rows")
    return df


# -------------------------------------------------------------
# Poll + Predict
# -------------------------------------------------------------
def poll_and_predict(game_id: int):
    global last_seen

    game_json = get_game_json(game_id)
    all_plays = extract_all_plays(game_json)

    if not all_plays:
        return None, 0

    last_idx = last_seen.get(game_id, -1)
    new_events = all_plays[last_idx + 1:]

    print("NEW EVENT TYPES:", [ev.get("typeDescKey") for ev in new_events])

    if not new_events:
        return None, 0

    df_input = build_dataframe_for_predict(new_events, game_json)

    if df_input.empty:
        last_seen[game_id] = len(all_plays) - 1
        return None, 0

    df_output = client.predict(df_input)

    last_seen[game_id] = len(all_plays) - 1

    return df_output, len(new_events)
