import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION & OVERRIDES
# Set to a specific lap number (e.g., TARGET_LAP = 8) or None for latest complete lap
TARGET_LAP = None
# =========================================================

# 1. Locate clean telemetry file in data/ or root directory
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
df_lap = df[df['Lap Number'] == selected_lap].dropna(
    subset=['G Force Lat', 'G Force Long']
)

# 3. Pro Telemetry Dark Theme Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9, 9), facecolor='#0f111a')
ax.set_facecolor('#161925')

sc = ax.scatter(
    df_lap['G Force Lat'],
    df_lap['G Force Long'],
    c=df_lap['Ground Speed'],
    cmap='plasma',
    s=16,
    alpha=0.75,
    edgecolors='none',
)

cbar = plt.colorbar(sc, ax=ax, pad=0.03)
cbar.set_label('Ground Speed (km/h)', color='#ffffff', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='#ffffff')
plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#ffffff')

# Reference G boundaries
circle_10g = plt.Circle(
    (0, 0), 1.0, color='#ffffff', fill=False, linestyle='--', alpha=0.3
)
circle_15g = plt.Circle(
    (0, 0),
    1.5,
    color='#00ffcc',
    fill=False,
    linestyle='--',
    alpha=0.6,
    label='1.5G Envelope',
)
circle_20g = plt.Circle(
    (0, 0),
    2.0,
    color='#ff0055',
    fill=False,
    linestyle=':',
    alpha=0.5,
    label='2.0G Limit',
)
ax.add_patch(circle_10g)
ax.add_patch(circle_15g)
ax.add_patch(circle_20g)

# Calculate Peak G Metrics
max_lat_g = df_lap['G Force Lat'].abs().max()
max_long_accel = df_lap['G Force Long'].max()
max_long_decel = df_lap['G Force Long'].min()

stat_text = (
    f'PEAK LATERAL: {max_lat_g:.2f} G\n'
    f'MAX ACCEL: +{max_long_accel:.2f} G\n'
    f'MAX BRAKING: {max_long_decel:.2f} G'
)

ax.text(
    0.03,
    0.93,
    stat_text,
    transform=ax.transAxes,
    fontsize=10,
    fontweight='bold',
    color='#ffffff',
    bbox=dict(
        boxstyle='round,pad=0.5', facecolor='#212638', edgecolor='#00ffcc'
    ),
)

ax.set_title(
    f'ApexPulse G-G Friction Circle & Limit Utilization (Lap {selected_lap})',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
ax.set_xlabel('Lateral Acceleration (G)', fontsize=11, color='#a0a5b5')
ax.set_ylabel('Longitudinal Acceleration (G)', fontsize=11, color='#a0a5b5')
ax.axhline(0, color='#ffffff', linestyle=':', alpha=0.3)
ax.axvline(0, color='#ffffff', linestyle=':', alpha=0.3)
ax.grid(True, linestyle=':', alpha=0.3, color='#444c63')
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='lower right', frameon=True, facecolor='#212638', edgecolor='none')

plt.tight_layout()

os.makedirs('assets', exist_ok=True)
output_img = os.path.join('assets', 'friction_circle.png')
plt.savefig(
    output_img, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor()
)
print(f"[SUCCESS] Friction circle plot saved to '{output_img}'.")

plt.show()