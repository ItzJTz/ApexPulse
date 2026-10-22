import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_math_channels(csv_path="data/clean_data.csv"):
    df = pd.read_csv(csv_path)

    # 1. Auto-detect columns
    speed_col = next(c for c in df.columns if 'speed' in c.lower())
    lap_col = next(
        (c for c in df.columns if 'lap' in c.lower() and 'num' in c.lower()),
        None,
    )
    lap_dist_col = next(
        (c for c in df.columns if 'lap' in c.lower() and 'dist' in c.lower()),
        None,
    )
    if lap_col is None or lap_dist_col is None:
        raise ValueError("Telemetry must include lap number and lap distance columns.")

    df[lap_dist_col] = pd.to_numeric(df[lap_dist_col], errors="coerce")
    df = df.dropna(subset=[lap_col, lap_dist_col]).copy()
    
    # Normalize distance per lap
    df["Normalized_Dist"] = df.groupby(lap_col)[lap_dist_col].transform(
        lambda distance: distance - distance.min()
    )
    
    # Extract valid complete laps
    lap_lengths = df.groupby(lap_col)["Normalized_Dist"].max()
    max_lap_length = lap_lengths.max()
    complete_laps = lap_lengths[lap_lengths >= 0.90 * max_lap_length].index
    df = df[df[lap_col].isin(complete_laps)].copy()
    
    # Extract single fastest lap
    speed_ms = np.maximum(pd.to_numeric(df[speed_col], errors="coerce") / 3.6, 1.0)
    distance_delta = (
        df.groupby(lap_col)["Normalized_Dist"].diff().clip(lower=0).fillna(0)
    )
    lap_times = (distance_delta / speed_ms).groupby(df[lap_col]).sum()
    fastest_lap = lap_times.idxmin()
    
    df = df[df[lap_col] == fastest_lap].copy()
    df = df.sort_values(by="Normalized_Dist").reset_index(drop=True)

    # --- FILTER SPIKES / SENSOR DROPOUTS ---
    # Apply a 3-point rolling median to eliminate single-frame dropouts
    df[speed_col] = df[speed_col].rolling(window=3, center=True, min_periods=1).median()

    # 2. Dynamic Brake Bias (%)
    brake_col = next((c for c in df.columns if 'brake' in c.lower()), None)
    if brake_col:
        df["Math_BrakeBias"] = np.where(df[brake_col] > 0.05, 58.5, 0.0)
    else:
        df["Math_BrakeBias"] = 0.0

    # 3. Aero Downforce Estimation (N)
    v_ms = df[speed_col] / 3.6
    c_aero = 0.85  # Aerodynamic constant (0.5 * rho * A * Cl)
    df["Math_AeroDownforce_N"] = c_aero * (v_ms ** 2)

    # Save cleaned CSV with math channels
    df.to_csv("data/data_with_math_channels.csv", index=False)

    # 4. Plot Dark-Theme Dashboard
    DARK_BG, PANEL_BG, GRID_COLOR = "#0e1117", "#161b22", "#21262d"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.tick_params(colors="#c9d1d9", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # Speed vs Downforce
    ax1.plot(df["Normalized_Dist"], df[speed_col], color="#00ff66", label="Filtered Ground Speed (km/h)", linewidth=1.5)
    ax1.set_ylabel("Speed (km/h)", color="#c9d1d9", fontweight="bold")
    ax1.set_title("ApexPulse | Vehicle Dynamics Math Channels (Derived Aero Downforce)", color="#ffffff", fontweight="bold", pad=15)
    ax1.legend(loc="upper right", facecolor=DARK_BG, edgecolor="#30363d", labelcolor="#ffffff")

    ax2.plot(df["Normalized_Dist"], df["Math_AeroDownforce_N"], color="#ff0055", label="Calculated Aero Downforce (N)", linewidth=1.8)
    ax2.set_xlabel("Track Distance (m)", color="#c9d1d9", fontweight="bold")
    ax2.set_ylabel("Downforce (N)", color="#c9d1d9", fontweight="bold")
    ax2.legend(loc="upper right", facecolor=DARK_BG, edgecolor="#30363d", labelcolor="#ffffff")

    plt.tight_layout()
    plt.savefig("assets/math_channels.png", dpi=300, facecolor=DARK_BG)
    print("Exported smooth, noise-filtered dashboard to assets/math_channels.png")
    plt.show()

if __name__ == "__main__":
    generate_math_channels()