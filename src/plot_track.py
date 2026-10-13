import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION & OVERRIDES
# Set to a specific lap number (e.g., TARGET_LAP = 8) or None for latest complete lap
TARGET_LAP = None
# =========================================================

# 1. Locate clean telemetry file
clean_file_path = (
    os.path.join('data', 'clean_data.csv')
    if os.path.exists(os.path.join('data', 'clean_data.csv'))
    else 'clean_data.csv'
)

if not os.path.exists(clean_file_path):
    print("Error: 'clean_data.csv' not found. Run src/clean_telemetry.py first.")
    exit()

df = pd.read_csv(clean_file_path)

# 2. Filter complete laps
lap_max_distances = df.groupby('Lap Number')['Lap Distance'].max()
track_max_dist = lap_max_distances.max()
valid_laps = lap_max_distances[
    lap_max_distances >= (track_max_dist * 0.90)
].index.tolist()

if not valid_laps:
    print("Error: No complete flying laps found in dataset.")
    exit()

selected_lap = TARGET_LAP if TARGET_LAP is not None else int(valid_laps[-1])
df_lap = (
    df[df['Lap Number'] == selected_lap]
    .sort_values(by='Lap Distance')
    .copy()
)

# Remove duplicate distance rows for clean integration
df_lap = df_lap.drop_duplicates(subset=['Lap Distance']).reset_index(drop=True)

# 3. Reconstruct True 2D Trajectory via Kinematic Integration
if 'Pos X' in df_lap.columns and 'Pos Y' in df_lap.columns:
    x = df_lap['Pos X'].values
    y = df_lap['Pos Y'].values
else:
    # Kinematic integration from telemetry
    ds = np.diff(df_lap['Lap Distance'].values, prepend=0)
    speed_ms = np.maximum(df_lap['Ground Speed'].values / 3.6, 2.0)
    lat_g = df_lap['G Force Lat'].values * 9.81

    # Corner curvature: d_theta = (a_lat / v^2) * ds
    d_theta = (lat_g / (speed_ms**2)) * ds
    heading = np.cumsum(d_theta)

    # Spatial steps
    dx = np.cos(heading) * ds
    dy = np.sin(heading) * ds

    x_raw = np.cumsum(dx)
    y_raw = np.cumsum(dy)

    # Loop Closure Correction
    progress = df_lap['Lap Distance'].values / df_lap['Lap Distance'].max()
    x_drift = x_raw[-1] - x_raw[0]
    y_drift = y_raw[-1] - y_raw[0]

    x = x_raw - (progress * x_drift)
    y = y_raw - (progress * y_drift)

speed = df_lap['Ground Speed'].values

# 4. Pro Telemetry Render
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 8), facecolor='#0f111a')
ax.set_facecolor('#161925')

sc = ax.scatter(x, y, c=speed, cmap='magma', s=18, alpha=0.9, edgecolors='none')

cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Ground Speed (km/h)', color='#ffffff', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='#ffffff')
plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#ffffff')

# Annotate Start/Finish & V-Max
max_sp_idx = np.argmax(speed)
ax.scatter(
    x[max_sp_idx], y[max_sp_idx], color='#00ffcc', s=110, zorder=5,
    label=f'V-Max ({speed[max_sp_idx]:.1f} km/h)'
)
ax.scatter(x[0], y[0], color='#ff0055', s=110, zorder=5, label='Start / Finish')

track_name = df_lap['Track'].iloc[0] if 'Track' in df_lap.columns else 'Circuit'

ax.set_title(
    f'ApexPulse Reconstructed 2D Track Map: {track_name} (Lap {selected_lap})',
    fontsize=14, fontweight='bold', pad=15,
)

# Move legend to the upper left to avoid blocking the track map
ax.legend(loc='upper left', frameon=True, facecolor='#212638', edgecolor='none', bbox_to_anchor=(0.02, 0.98))
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

os.makedirs('assets', exist_ok=True)
output_img = os.path.join('assets', 'track_map.png')
plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"[SUCCESS] Reconstructed track map saved to '{output_img}'.")

plt.show()