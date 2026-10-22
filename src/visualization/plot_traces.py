import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import config_value
from src.core.spatial import interpolate_spatial
from src.visualization.style import apply_style
# =========================================================
# CONFIGURATION & OVERRIDES
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
    print("Error: 'clean_data.csv' not found.")
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

df_lapA = df[df['Lap Number'] == lap_A].sort_values(by='Lap Distance').copy()
df_lapB = df[df['Lap Number'] == lap_B].sort_values(by='Lap Distance').copy()

# 3. Clean Data: Strict monotonicity and rolling median
df_lapA = df_lapA[df_lapA['Lap Distance'].diff().fillna(1) > 0]
df_lapB = df_lapB[df_lapB['Lap Distance'].diff().fillna(1) > 0]

channels_to_smooth = ['Ground Speed', 'Throttle Pos', 'Brake Pos', 'Steering', 'G Force Lat']
for col in channels_to_smooth:
    if col in df_lapA.columns:
        df_lapA[col] = df_lapA[col].rolling(window=5, min_periods=1, center=True).median()
        df_lapB[col] = df_lapB[col].rolling(window=5, min_periods=1, center=True).median()

# 4. Interpolate onto 1-meter spatial grid
max_common_dist = min(df_lapA['Lap Distance'].max(), df_lapB['Lap Distance'].max())
min_common_dist = max(
    df_lapA['Lap Distance'].min(),
    df_lapB['Lap Distance'].min(),
)
if max_common_dist <= min_common_dist:
    raise ValueError("The selected laps do not have overlapping distance coverage.")
grid_step = float(config_value("analysis", "spatial_grid_m"))
common_dist = np.arange(min_common_dist, max_common_dist, grid_step)
if len(common_dist) == 0 or common_dist[-1] < max_common_dist:
    common_dist = np.append(common_dist, max_common_dist)

speed_A = interpolate_spatial(df_lapA['Lap Distance'], df_lapA['Ground Speed'], common_dist)
speed_B = interpolate_spatial(df_lapB['Lap Distance'], df_lapB['Ground Speed'], common_dist)
throttle_A = interpolate_spatial(df_lapA['Lap Distance'], df_lapA['Throttle Pos'], common_dist)
throttle_B = interpolate_spatial(df_lapB['Lap Distance'], df_lapB['Throttle Pos'], common_dist)
brake_A = interpolate_spatial(df_lapA['Lap Distance'], df_lapA['Brake Pos'], common_dist)
brake_B = interpolate_spatial(df_lapB['Lap Distance'], df_lapB['Brake Pos'], common_dist)

has_steering = 'Steering' in df.columns
if has_steering:
    steer_A = interpolate_spatial(df_lapA['Lap Distance'], df_lapA['Steering'], common_dist)
    steer_B = interpolate_spatial(df_lapB['Lap Distance'], df_lapB['Steering'], common_dist)
else:
    lat_g_A = interpolate_spatial(df_lapA['Lap Distance'], df_lapA['G Force Lat'], common_dist)
    lat_g_B = interpolate_spatial(df_lapB['Lap Distance'], df_lapB['G Force Lat'], common_dist)

# 5. Render Multi-Channel Dark Theme Plot
apply_style()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10), sharex=True, facecolor='#0f111a')

color_A = '#00d2ff'
color_B = '#ff007f'

for ax in (ax1, ax2, ax3, ax4):
    ax.set_facecolor('#161925')
    ax.grid(True, linestyle=':', alpha=0.3, color='#444c63')

# Panel 1: Speed with Shading
ax1.plot(common_dist, speed_A, color=color_A, label=f'Lap {lap_A}', linewidth=1.5)
ax1.plot(common_dist, speed_B, color=color_B, label=f'Lap {lap_B}', linewidth=1.5)
ax1.fill_between(common_dist, speed_A, speed_B, where=(speed_B > speed_A), color=color_B, alpha=0.3)
ax1.fill_between(common_dist, speed_A, speed_B, where=(speed_A > speed_B), color=color_A, alpha=0.3)
ax1.set_ylabel('Speed\n(km/h)', fontsize=10, color='#a0a5b5')
ax1.legend(loc='upper right', frameon=True, facecolor='#212638')
ax1.set_title(
    f'ApexPulse Full Telemetry Dashboard: Lap {lap_A} vs Lap {lap_B}',
    fontsize=14, fontweight='bold', pad=12,
)

# Panel 2: Throttle
ax2.plot(common_dist, throttle_A, color=color_A, label=f'Lap {lap_A}', linewidth=1.2, alpha=0.9)
ax2.plot(common_dist, throttle_B, color=color_B, label=f'Lap {lap_B}', linewidth=1.2, alpha=0.9)
ax2.set_ylabel('Throttle\n(%)', fontsize=10, color='#a0a5b5')
ax2.legend(loc='upper right', frameon=True, facecolor='#212638')

# Panel 3: Brake
ax3.plot(common_dist, brake_A, color=color_A, label=f'Lap {lap_A}', linewidth=1.2, alpha=0.9)
ax3.plot(common_dist, brake_B, color=color_B, label=f'Lap {lap_B}', linewidth=1.2, alpha=0.9)
ax3.set_ylabel('Brake\n(%)', fontsize=10, color='#a0a5b5')
ax3.legend(loc='upper right', frameon=True, facecolor='#212638')

# Panel 4: Steering or Lateral G
if has_steering:
    ax4.plot(common_dist, steer_A, color=color_A, label=f'Lap {lap_A}', linewidth=1.2, alpha=0.9)
    ax4.plot(common_dist, steer_B, color=color_B, label=f'Lap {lap_B}', linewidth=1.2, alpha=0.9)
    ax4.set_ylabel('Steering\n(Deg)', fontsize=10, color='#a0a5b5')
else:
    ax4.plot(common_dist, lat_g_A, color=color_A, label=f'Lap {lap_A}', linewidth=1.2, alpha=0.9)
    ax4.plot(common_dist, lat_g_B, color=color_B, label=f'Lap {lap_B}', linewidth=1.2, alpha=0.9)
    ax4.set_ylabel('Lat Accel\n(G)', fontsize=10, color='#a0a5b5')

ax4.legend(loc='upper right', frameon=True, facecolor='#212638')
ax4.set_xlabel('Lap Distance (meters)', fontsize=11, color='#a0a5b5')

plt.tight_layout()

os.makedirs('assets', exist_ok=True)
output_img = os.path.join('assets', 'telemetry_traces.png')
plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"[SUCCESS] Multi-channel traces saved to '{output_img}'.")

plt.show()