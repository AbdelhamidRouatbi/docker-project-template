import streamlit as st
import pandas as pd
import requests

from ift6758.ift6758.client.serving_client import ServingClient
from scripts.step3_clients.live_game_events import poll_and_predict


# ================================================
# SESSION STATE
# ================================================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "teams" not in st.session_state:
    st.session_state.teams = {"home": "", "away": ""}

if "logos" not in st.session_state:
    st.session_state.logos = {"home": "", "away": ""}

if "score" not in st.session_state:
    st.session_state.score = {"home": 0, "away": 0}

if "meta" not in st.session_state:
    st.session_state.meta = {"period": None, "time_left": None}


# ================================================
# CLIENT
# ================================================
serving = ServingClient(ip="127.0.0.1", port=5000)

st.title("Hockey Expected Goals Dashboard")


# ================================================
# MODEL SELECTION
# ================================================
with st.sidebar:
    st.header("Model Selection")
    workspace = st.text_input("Workspace")
    model = st.text_input("Model")
    version = st.text_input("Version")

    if st.button("Download Model"):
        serving.download_registry_model(workspace, model, version)
        st.success("Model downloaded!")


# ================================================
# GAME SELECTION
# ================================================
st.subheader("Game Selection")
game_id = st.text_input("Game ID", placeholder="e.g. 2021020329")
ping = st.button("Ping Game")


# ================================================
# GAME PING LOGIC
# ================================================
st.subheader("Game info / xG Summary")

if ping and game_id:

    df_output, num = poll_and_predict(int(game_id))
    if df_output is None or num == 0:
        st.info("No new events.")
    else:
        st.session_state.df = pd.concat([st.session_state.df, df_output], ignore_index=True)

    # Metadata 
    meta_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    meta = requests.get(meta_url).json()

    # Team names
    home = meta["homeTeam"]["commonName"]["default"]
    away = meta["awayTeam"]["commonName"]["default"]
    st.session_state.teams = {"home": home, "away": away}

    # Logos
    st.session_state.logos = {
        "home": meta["homeTeam"]["logo"],
        "away": meta["awayTeam"]["logo"]
    }

    # Score
    st.session_state.score = {
        "home": meta["homeTeam"]["score"],
        "away": meta["awayTeam"]["score"],
    }

    # Time / Period
    st.session_state.meta = {
        "period": meta["periodDescriptor"]["number"],
        "time_left": meta["clock"]["timeRemaining"],
    }

    df = st.session_state.df

    # Compute xG
    if not df.empty:
        home_xg = df[df["event_team"] == "home"]["proba_goal"].sum()
        away_xg = df[df["event_team"] == "away"]["proba_goal"].sum()
    else:
        home_xg = away_xg = 0.0

    home_score = st.session_state.score["home"]
    away_score = st.session_state.score["away"]

    # Differences
    home_diff = home_score - home_xg
    away_diff = away_score - away_xg

    # ============================================
    # 3-COLUMN LAYOUT WITH CENTERED ELEMENTS
    # ============================================
    col1, col2, col3 = st.columns([2, 2, 2])

    # LEFT COLUMN (HOME)
    with col1:
        st.image(st.session_state.logos["home"], width=90)
        st.markdown(f"<h2 style='text-align:center'>{home}</h2>", unsafe_allow_html=True)
        st.metric("xG", f"{home_xg:.2f}", f"{home_diff:.2f}")

    # CENTER COLUMN (SCORE + PERIOD)
    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:15px;">
                <div style="font-size:18px; color:gray;">
                    Period {st.session_state.meta['period']} — {st.session_state.meta['time_left']}
                </div>
                <div style="font-size:55px; font-weight:700; margin:10px 0;">
                    {home_score}  -  {away_score}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # RIGHT COLUMN (AWAY)
    with col3:
        st.image(st.session_state.logos["away"], width=90)
        st.markdown(f"<h2 style='text-align:center'>{away}</h2>", unsafe_allow_html=True)
        st.metric("xG", f"{away_xg:.2f}", f"{away_diff:.2f}")


# ===================================================
# TABLE
# ===================================================
st.subheader("Shot Events Used For Predictions")

if len(st.session_state.df) > 0:

    SHOW_COLS = [
        "event_team", "period", "time_remaining",
        "distance_from_net", "shot_angle", "empty_net",
        "prediction", "proba_goal",
    ]

    existing_cols = [c for c in SHOW_COLS if c in st.session_state.df.columns]

    df_display = (
        st.session_state.df[existing_cols]
        .reset_index(drop=True)
        .sort_index(ascending=False)
    )

    st.dataframe(df_display)

else:
    st.info("No data yet — ping a game.")

