import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import config_value
from src.core.spatial import interpolate_spatial
from src.pipeline.telemetry_io import fastest_lap_numbers, load_telemetry
from src.visualization.style import apply_style

def load_file(filepath):
    """Load raw or cleaned telemetry into the shared canonical channel format."""
    return load_telemetry(filepath)

def extract_fastest_lap(df):
    """Extracts single best lap, normalizes distance to 0m, and filters speed spikes."""
    selected_laps = fastest_lap_numbers(df, count=1)
    if not selected_laps:
        raise ValueError("No complete laps are available for driver comparison.")
    best_lap_num = selected_laps[0]
    best_lap_df = df[df["Lap Number"] == best_lap_num].copy()

    # Apply 3-point rolling median to eliminate single-frame dropouts
    median_window = int(config_value("signal_filter", "rolling_median_window"))
    if median_window < 1:
        raise ValueError(
            "signal_filter.rolling_median_window must be a positive integer."
        )
    best_lap_df["Ground Speed"] = best_lap_df["Ground Speed"].rolling(
        window=median_window,
        center=True,
        min_periods=1,
    ).median()
    
    return best_lap_df, "Normalized_Dist", "Ground Speed", best_lap_num

def compare_two_drivers(
    file_a,
    file_b,
    output_path="assets/driver_comparison.png",
    synthetic_benchmark=False,
):
    df_a_raw = load_file(file_a)
    df_b_raw = load_file(file_b)

    lap_a, dist_col_a, speed_col_a, lap_num_a = extract_fastest_lap(df_a_raw)
    lap_b, dist_col_b, speed_col_b, lap_num_b = extract_fastest_lap(df_b_raw)

    lap_a = (
        lap_a.groupby(dist_col_a, as_index=False)[speed_col_a]
        .mean()
        .sort_values(dist_col_a)
    )
    lap_b = (
        lap_b.groupby(dist_col_b, as_index=False)[speed_col_b]
        .mean()
        .sort_values(dist_col_b)
    )
    distance_a = lap_a[dist_col_a].to_numpy(dtype=float)
    speed_a_samples = lap_a[speed_col_a].to_numpy(dtype=float)

    if synthetic_benchmark:
        dist_grid = distance_a
        speed_b_values = speed_a_samples + np.sin(dist_grid / 150.0) * 3.5 - 1.2
    else:
        dist_grid = lap_b[dist_col_b].to_numpy(dtype=float)
        speed_b_values = lap_b[speed_col_b].to_numpy(dtype=float)

    # Interpolate both speed traces onto a unified 1m spatial grid
    max_track_length = min(distance_a.max(), dist_grid.max())
    grid_step = float(config_value("analysis", "spatial_grid_m"))
    spatial_grid = np.arange(0, max_track_length, grid_step)
    if len(spatial_grid) and spatial_grid[-1] < max_track_length:
        spatial_grid = np.append(spatial_grid, max_track_length)
    if len(spatial_grid) < 2:
        raise ValueError("The selected laps do not contain enough distance for comparison.")

    speed_a = interpolate_spatial(distance_a, speed_a_samples, spatial_grid)
    speed_b = interpolate_spatial(dist_grid, speed_b_values, spatial_grid)

    # Calculate spatial continuous time delta: dt = ds / v
    v_a_ms = np.maximum(speed_a / 3.6, 1.0)
    v_b_ms = np.maximum(speed_b / 3.6, 1.0)
    interval_distances = np.diff(spatial_grid)
    intervals_a = interval_distances * (0.5 / v_a_ms[:-1] + 0.5 / v_a_ms[1:])
    intervals_b = interval_distances * (0.5 / v_b_ms[:-1] + 0.5 / v_b_ms[1:])
    time_delta = np.concatenate(([0.0], np.cumsum(intervals_a - intervals_b)))

    # Plot Dark-Theme Multi-Driver Dashboard
    apply_style()
    DARK_BG, PANEL_BG, GRID_COLOR = "#0e1117", "#161b22", "#21262d"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.tick_params(colors="#c9d1d9", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # Panel 1: Speed Overlay
    ax1.plot(spatial_grid, speed_a, label=f"Driver A (Lap {int(lap_num_a)})", color="#00ff66", linewidth=1.5)
    ax1.plot(spatial_grid, speed_b, label=f"Driver B (Lap {int(lap_num_b)})", color="#ff0055", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Speed (km/h)", color="#c9d1d9", fontweight="bold")
    ax1.set_title("ApexPulse | Multi-Driver Cross-Session Overlay & Delta Engine", color="#ffffff", fontweight="bold", pad=15)
    ax1.legend(loc="upper right", facecolor=DARK_BG, edgecolor="#30363d", labelcolor="#ffffff")

    # Panel 2: Continuous Time Delta
    ax2.plot(spatial_grid, time_delta, color="#00f2fe", linewidth=1.8)
    ax2.axhline(0, color="#8b949e", linestyle=":", alpha=0.8)
    ax2.set_xlabel("Track Distance (m)", color="#c9d1d9", fontweight="bold")
    ax2.set_ylabel("Time Delta (s)", color="#c9d1d9", fontweight="bold")
    ax2.set_title("Positive delta means Driver B is ahead", color="#c9d1d9", fontsize=9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=DARK_BG)
    print(f"Exported driver comparison to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the fastest complete laps from two telemetry CSV files."
    )
    parser.add_argument("driver_a", help="Telemetry CSV for Driver A.")
    parser.add_argument("driver_b", help="Telemetry CSV for Driver B.")
    parser.add_argument(
        "--output",
        default="assets/driver_comparison.png",
        help="Output plot path (default: assets/driver_comparison.png).",
    )
    parser.add_argument(
        "--synthetic-benchmark",
        action="store_true",
        help="Replace Driver B's speed with synthetic data for a plotting demo.",
    )
    arguments = parser.parse_args()
    compare_two_drivers(
        arguments.driver_a,
        arguments.driver_b,
        output_path=arguments.output,
        synthetic_benchmark=arguments.synthetic_benchmark,
    )