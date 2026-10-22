import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import config_value
from src.core.spatial import interpolate_spatial
from src.visualization.style import apply_style
# =========================================================
# CONFIGURATION & OVERRIDES
# Set to specific lap numbers (e.g., LAP_A = 8, LAP_B = 9) or None for auto-select
LAP_A = None
LAP_B = None
# =========================================================

# 1. Locate clean telemetry file
clean_file_path = (
    os.path.join('data', 'clean_data.csv')
    if os.path.exists(os.path.join('data', 'clean_data.csv'))
    else 'clean_data.csv'
)

if not os.path.exists(clean_file_path):
    print("Error: 'clean_data.csv' not found. Run python -m src.pipeline.clean_telemetry first.")
    exit()

df = pd.read_csv(clean_file_path)

# 2. Filter complete laps
lap_max_distances = df.groupby('Lap Number')['Lap Distance'].max()
track_max_dist = lap_max_distances.max()
valid_laps = lap_max_distances[
    lap_max_distances
    >= (track_max_dist * float(config_value("data", "minimum_lap_fraction")))
].index.tolist()

if len(valid_laps) < 2:
    print("Error: Less than two complete flying laps found in dataset.")
    exit()

lap_A = LAP_A if LAP_A is not None else int(valid_laps[-2])
lap_B = LAP_B if LAP_B is not None else int(valid_laps[-1])

df_lapA = df[df['Lap Number'] == lap_A].sort_values(by='Lap Distance')
df_lapB = df[df['Lap Number'] == lap_B].sort_values(by='Lap Distance')

# 3. 1-Meter Spatial Interpolation
max_dist = min(df_lapA['Lap Distance'].max(), df_lapB['Lap Distance'].max())
grid_step = float(config_value("analysis", "spatial_grid_m"))
min_dist = max(
    df_lapA['Lap Distance'].min(),
    df_lapB['Lap Distance'].min(),
)
if max_dist <= min_dist:
    raise ValueError("The selected laps do not have overlapping distance coverage.")
common_dist = np.arange(min_dist, max_dist, grid_step)
if len(common_dist) == 0 or common_dist[-1] < max_dist:
    common_dist = np.append(common_dist, max_dist)

speed_A = (
    interpolate_spatial(
        df_lapA['Lap Distance'], df_lapA['Ground Speed'], common_dist
    )
    / 3.6
)
speed_B = (
    interpolate_spatial(
        df_lapB['Lap Distance'], df_lapB['Ground Speed'], common_dist
    )
    / 3.6
)

speed_A = np.maximum(speed_A, 1.0)
speed_B = np.maximum(speed_B, 1.0)

interval_distances = np.diff(common_dist)
interval_time_a = interval_distances * (
    0.5 / speed_A[:-1] + 0.5 / speed_A[1:]
)
interval_time_b = interval_distances * (
    0.5 / speed_B[:-1] + 0.5 / speed_B[1:]
)
time_A = np.concatenate(([0.0], np.cumsum(interval_time_a)))
time_B = np.concatenate(([0.0], np.cumsum(interval_time_b)))
delta = time_A - time_B  # Positive = Lap B faster

# 4. Pro Telemetry Dark Theme Plot
apply_style()
fig, ax = plt.subplots(figsize=(13, 6), facecolor='#0f111a')
ax.set_facecolor('#161925')

ax.plot(
    common_dist,
    delta,
    color='#00ffcc',
    linewidth=1.8,
    label=f'Lap {lap_B} Delta vs Reference Lap {lap_A}',
)
ax.axhline(0, color='#ffffff', linestyle='--', linewidth=1, alpha=0.6)

# Highlight gain/loss zones
ax.fill_between(
    common_dist,
    delta,
    0,
    where=(delta >= 0),
    color='#00ffcc',
    alpha=0.25,
    label=f'Lap {lap_B} Ahead',
)
ax.fill_between(
    common_dist,
    delta,
    0,
    where=(delta < 0),
    color='#ff0055',
    alpha=0.25,
    label=f'Lap {lap_A} Ahead',
)

# Stat callout summary
final_delta = delta[-1]
status_str = (
    f'Lap {lap_B} Faster by {final_delta:.3f}s'
    if final_delta > 0
    else f'Lap {lap_A} Faster by {abs(final_delta):.3f}s'
)
ax.text(
    0.02,
    0.92,
    f'NET DELTA: {status_str}',
    transform=ax.transAxes,
    fontsize=11,
    fontweight='bold',
    color='#ffffff',
    bbox=dict(
        boxstyle='round,pad=0.5', facecolor='#212638', edgecolor='#00ffcc'
    ),
)

ax.set_title(
    f'ApexPulse Spatial Time-Delta Engine: Lap {lap_A} vs Lap {lap_B}',
    fontsize=14,
    fontweight='bold',
    pad=15,
    color='#ffffff',
)
ax.set_xlabel('Lap Distance (meters)', fontsize=11, color='#a0a5b5')
ax.set_ylabel('Time Gap (seconds)', fontsize=11, color='#a0a5b5')
ax.legend(loc='lower right', frameon=True, facecolor='#212638', edgecolor='none')
ax.grid(True, linestyle=':', alpha=0.3, color='#444c63')

plt.tight_layout()

os.makedirs('assets', exist_ok=True)
output_img = os.path.join('assets', 'time_delta.png')
plt.savefig(
    output_img, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor()
)
print(f"[SUCCESS] High-res time delta plot saved to '{output_img}'.")

plt.show()