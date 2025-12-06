import os
import streamlit as st
import pandas as pd
import requests

from ift6758.ift6758.client.serving_client import ServingClient
from scripts.step3_clients.live_game_events import poll_and_predict
import bonus as bonus


SERVING_HOST = os.getenv("SERVING_HOST", "127.0.0.1")
SERVING_PORT = int(os.getenv("SERVING_PORT", "5000"))

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
# serving = ServingClient(ip="127.0.0.1", port=5000) # Local flask app
serving = ServingClient(ip=SERVING_HOST, port=SERVING_PORT)

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
        "proba_goal", "prediction"
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

# ===================================================
# CUMULATIVE XG TIMELINE
# ===================================================
st.subheader("Cumulative xG Timeline")

bonus.plot_cumulative_xg(st.session_state.df)

# ===================================================
# HEATMAPS
# ===================================================
st.subheader("Shot Heatmaps (xG Density)")

df = st.session_state.df

if len(df) == 0:
    st.info("No events yet. Ping a game first.")
else:
    home_img, away_img = bonus.compute_heatmaps(df)

    col_h, col_a = st.columns(2)

    with col_h:
        st.markdown("### Home Heatmap")
        fig_home = bonus.overlay_rink_on_heatmap(home_img, alpha_heatmap=0.90)
        st.pyplot(fig_home)

    with col_a:
        st.markdown("### Away Heatmap")
        fig_away = bonus.overlay_rink_on_heatmap(away_img, alpha_heatmap=0.90)
        st.pyplot(fig_away)

# ===================================================
# FOOTER
# ===================================================
st.text(
    "Bonus Features:\n"
    "We implemented bonus features that go beyond the basic requirements to make our app better. "
    "We retrieve the team logos from the NHL API and display them in the main layout. "
    "We compute the cumulative xG timeline by converting game time and applying a cumulative sum of predicted goal probabilities. "
    "We generate xG heatmaps by building xG-weighted histograms, smoothing them with a Gaussian method and interpolation, and overlay the result on a rink image. This is similar to what we did in milestone 1 for the advanced visualizations. This can help teams identify which areas are generating more threat on their defense, and which areas in the opponent's defense are considered weak."
)
