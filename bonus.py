import plotly.graph_objects as go
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom


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

    df["game_clock"] = (df["period"] - 1) * 20 * 60 + (20*60 - df["time_sec"])
    df["game_clock_min"] = df["game_clock"] / 60.0 

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


def gaussian_smooth(hist, smoothing_factor=4, sigma=1.0):
    """
    Manual Gaussian smoothing on a finer grid.
    hist: (n_bins_x, n_bins_y)
    """
    n_bins_x, n_bins_y = hist.shape

    # --- 1. Fine grid dimensions ---
    fine_x = n_bins_x * smoothing_factor
    fine_y = n_bins_y * smoothing_factor

    # --- 2. Bin centers (normalized 0..1 space) ---
    bin_x_centers = np.linspace(0, 1, n_bins_x, endpoint=False) + 0.5/n_bins_x
    bin_y_centers = np.linspace(0, 1, n_bins_y, endpoint=False) + 0.5/n_bins_y
    bin_X, bin_Y = np.meshgrid(bin_x_centers, bin_y_centers, indexing="ij")
    bin_centers = np.column_stack((bin_X.ravel(), bin_Y.ravel()))
    bin_values = hist.ravel()

    # --- 3. Fine grid cell centers ---
    cell_x_centers = np.linspace(0, 1, fine_x, endpoint=False) + 0.5/fine_x
    cell_y_centers = np.linspace(0, 1, fine_y, endpoint=False) + 0.5/fine_y
    cell_X, cell_Y = np.meshgrid(cell_x_centers, cell_y_centers, indexing="ij")
    cell_centers = np.column_stack((cell_X.ravel(), cell_Y.ravel()))

    # --- 4. Gaussian kernel weights ---
    diff = cell_centers[:, None, :] - bin_centers[None, :, :]
    squared_distances = np.sum(diff**2, axis=2)
    weights = np.exp(-squared_distances / (2 * sigma * sigma))

    # weighted sum from all bins
    fine_values = weights @ bin_values
    fine_grid = fine_values.reshape(fine_x, fine_y)
    return fine_grid
    
def compute_heatmaps(df, 
                     n_bins=20, 
                     smoothing_factor=10, 
                     variance=0.04,
                     out_width=400, 
                     out_height=170):

    def make_team_map(df_team, flip_mask):
        # Flip coordinates based on mask
        df_team = df_team.copy()
        df_team.loc[flip_mask, "x_coord"] *= -1
        df_team.loc[flip_mask, "y_coord"] *= -1

        # Bin ranges
        x_min, x_max = 0, 100
        y_min, y_max = -42.5, 42.5
        x_bins = np.linspace(x_min, x_max, n_bins + 1)
        y_bins = np.linspace(y_min, y_max, n_bins + 1)

        # xG-weighted histogram
        hist, _, _ = np.histogram2d(
            df_team["x_coord"],
            df_team["y_coord"],
            bins=[x_bins, y_bins],
            weights=df_team["proba_goal"]
        )

        # Smoothing
        smooth_hist = gaussian_smooth(hist, smoothing_factor, sigma=variance)

        # Interpolate to final size
        current_h, current_w = smooth_hist.shape
        zoom_x = out_width  / current_w
        zoom_y = out_height / current_h
        final_img = zoom(smooth_hist, (zoom_y, zoom_x))

        return final_img

    # Filter valid shot events
    df = df[df["event_type"].isin(["SHOT-ON-GOAL", "MISSED-SHOT", "BLOCKED-SHOT"])].copy()

    # HOME TEAM
    df_home = df[df["home"] == True]
    home_flip_mask = (df_home["period"] % 2 == 1)   # flip on odd periods
    home_img = make_team_map(df_home, home_flip_mask)

    # AWAY TEAM
    df_away = df[df["away"] == True]
    away_flip_mask = (df_away["period"] % 2 == 0)   # flip on even periods
    away_img = make_team_map(df_away, away_flip_mask)

    return home_img, away_img

def overlay_rink_on_heatmap(heatmap,
                            path="figures/nhl_rink-no_background.png",
                            x_min=0, x_max=100,
                            y_min=-42.5, y_max=42.5,
                            alpha_heatmap=1):

    img = mpimg.imread(path)
    h, w, c = img.shape

    # right half
    img = img[:, w//2:, :]        

    # rotate heatmap + rink CCW 90
    heatmap = np.rot90(heatmap.T, k=-1)
    img = np.rot90(img, k=-1)

    # white → red
    white_to_red = LinearSegmentedColormap.from_list(
        "white_to_red", [(1,1,1), (1,0,0)]
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # heatmap
    ax.imshow(
        heatmap,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=white_to_red,
        alpha=alpha_heatmap,
        aspect="auto"
    )

    # rink overlay
    ax.imshow(
        img,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto"
    )

    ax.set_xlabel("Y (ft)")
    ax.set_ylabel("Y (ft)")

    xticks = [0, 25, 50, 75, 100]
    xtick_labels = np.interp(
        xticks,
        [0, 100],
        [42.5, -42.5]
    )
    xtick_labels = [round(v, 2) for v in xtick_labels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    y_display = [0, 20, 40, 60, 80, 100]
    yticks = np.interp(
        y_display,
        [0, 100],         
        [-42.5, 42.5]    
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in y_display])

    fig.tight_layout()
    return fig
