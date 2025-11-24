import plotly.graph_objects as go
import streamlit as st
import numpy as np

def plot_cumulative_xg(df):
    if df.empty:
        st.info("No xG events yet.")
        return

    # Sort by game flow
    def to_seconds(t):
        if isinstance(t, str) and ":" in t:
            m, s = t.split(":")
            return int(m) * 60 + int(s)
        return np.nan

    df = df.copy()
    df["time_sec"] = df["time_remaining"].apply(to_seconds)

    # Convert countdown time → elapsed time
    df["game_clock"] = (df["period"] - 1) * 20 * 60 + (20*60 - df["time_sec"])
    df["game_clock_min"] = df["game_clock"] / 60.0  # <-- MINUTES

    df = df.sort_values("game_clock")

    # cumulative xG
    df["cum_home"] = (df["event_team"] == "home").astype(int) * df["proba_goal"]
    df["cum_away"] = (df["event_team"] == "away").astype(int) * df["proba_goal"]

    df["cum_home"] = df["cum_home"].cumsum()
    df["cum_away"] = df["cum_away"].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["game_clock_min"],
        y=df["cum_home"],
        mode="lines+markers",
        name="Home xG",
        line=dict(color="blue", width=3),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=df["game_clock_min"],
        y=df["cum_away"],
        mode="lines+markers",
        name="Away xG",
        line=dict(color="red", width=3),
        marker=dict(size=6)
    ))

    # Period break vertical lines IN MINUTES
    period_breaks_sec = [20*60, 40*60, 60*60]
    period_breaks_min = [p/60 for p in period_breaks_sec]

    for p in period_breaks_min:
        fig.add_vline(
            x=p,
            line=dict(color="gray", width=1.5, dash="dash"),
            opacity=0.5
        )

    fig.update_layout(
        title="Cumulative xG Over Time",
        xaxis_title="Game Time (minutes)",
        yaxis_title="Cumulative xG",
        height=450,
        margin=dict(l=30, r=30, t=50, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)

