import numpy as np
import matplotlib.pyplot as plt
if __package__:
    from .telemetry_io import fastest_lap_numbers, load_telemetry
else:
    from telemetry_io import fastest_lap_numbers, load_telemetry

def compare_laps(csv_path="data/raw_telemetrySilverstone.csv", grid_step=1.0):
    if grid_step <= 0:
        raise ValueError("grid_step must be greater than zero.")
    df = load_telemetry(csv_path)
    selected_laps = fastest_lap_numbers(df)
    if len(selected_laps) < 2:
        raise ValueError(f"Need at least two complete laps to compare; found {len(selected_laps)}.")

    best_lap_num, second_best_lap_num = selected_laps
    lap_a = df[df["Lap Number"] == best_lap_num].copy()
    lap_b = df[df["Lap Number"] == second_best_lap_num].copy()

    # 6. Spatial Grid Interpolation
    dist_a = lap_a["Normalized_Dist"].to_numpy()
    speed_a_raw = lap_a["Ground Speed"].to_numpy()
    dist_b = lap_b["Normalized_Dist"].to_numpy()
    speed_b_raw = lap_b["Ground Speed"].to_numpy()

    max_track_length = min(dist_a.max(), dist_b.max())
    spatial_grid = np.arange(0, max_track_length, grid_step)
    if len(spatial_grid) and spatial_grid[-1] < max_track_length:
        spatial_grid = np.append(spatial_grid, max_track_length)
    if len(spatial_grid) < 2:
        raise ValueError("The selected laps do not contain enough distance for comparison.")

    speed_a = np.interp(spatial_grid, dist_a, speed_a_raw)
    speed_b = np.interp(spatial_grid, dist_b, speed_b_raw)

    v_a_ms = np.maximum(speed_a / 3.6, 1.0)
    v_b_ms = np.maximum(speed_b / 3.6, 1.0)

    interval_distances = np.diff(spatial_grid)
    intervals_a = interval_distances * (0.5 / v_a_ms[:-1] + 0.5 / v_a_ms[1:])
    intervals_b = interval_distances * (0.5 / v_b_ms[:-1] + 0.5 / v_b_ms[1:])
    time_delta = np.concatenate(([0.0], np.cumsum(intervals_a - intervals_b)))

    # 7. Match G-G Friction Circle Dark Theme Colors
    DARK_BG = "#0e1117"
    PANEL_BG = "#161b22"
    GRID_COLOR = "#21262d"
    TEXT_COLOR = "#c9d1d9"
    ACCENT_GREEN = "#00ff66"
    ACCENT_RED = "#ff0055"
    ACCENT_CYAN = "#00f2fe"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # Panel 1: Speed Overlay
    ax1.plot(spatial_grid, speed_a, label=f"Fastest Lap ({best_lap_num:g})", color=ACCENT_GREEN, linewidth=1.8)
    ax1.plot(spatial_grid, speed_b, label=f"Second Fastest ({second_best_lap_num:g})", color=ACCENT_RED, linewidth=1.8, linestyle="--")
    ax1.set_ylabel("Speed (km/h)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax1.set_title(f"ApexPulse | Lap {best_lap_num:g} vs Lap {second_best_lap_num:g} Spatial Overlay", color="#ffffff", fontsize=14, fontweight="bold", pad=15)

    legend = ax1.legend(loc="upper right", facecolor=DARK_BG, edgecolor="#30363d")
    plt.setp(legend.get_texts(), color="#ffffff", fontweight="bold")

    # Panel 2: Continuous Time Delta Line
    ax2.plot(spatial_grid, time_delta, color=ACCENT_CYAN, linewidth=2.0)
    ax2.axhline(0, color="#8b949e", linestyle=":", alpha=0.8)
    ax2.set_xlabel("Track Distance (m)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Time Delta (s)", color=TEXT_COLOR, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("assets/lap_comparison.png", dpi=300, facecolor=DARK_BG)
    print(f"Comparison complete! Saved matching dark-theme export to assets/lap_comparison.png")

    plt.show()

if __name__ == "__main__":
    compare_laps()