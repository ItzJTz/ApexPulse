import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION & OVERRIDES
# Set to None to automatically detect and plot the FASTEST lap of the session.
# Or set to a specific integer (e.g., TARGET_LAP = 9) to force that specific lap.
TARGET_LAP = None
# =========================================================

def generate_friction_circle(file_path, target_lap=None, output_dir='assets'):
    """
    Generates a professional 2D Density Heatmap Friction Circle using a 
    Superellipse model. Automatically detects the fastest valid lap in the dataset.
    """
    if not os.path.exists(file_path):
        print(f"Error: Telemetry file '{file_path}' not found.")
        return

    df = pd.read_csv(file_path)

    # 1. Filter out incomplete laps (in-laps / out-laps) using track distance
    lap_max_distances = df.groupby('Lap Number')['Lap Distance'].max()
    track_max_dist = lap_max_distances.max()
    valid_laps = lap_max_distances[lap_max_distances >= (track_max_dist * 0.90)].index.tolist()

    if not valid_laps:
        print("Error: No complete flying laps found in dataset.")
        return

    # 2. Automatically calculate lap times to find the absolute BEST flying lap
    if 'Time' in df.columns:
        lap_times = df.groupby('Lap Number')['Time'].max() - df.groupby('Lap Number')['Time'].min()
    elif 'Lap Time' in df.columns:
        lap_times = df.groupby('Lap Number')['Lap Time'].max()
    else:
        # Fallback if time channel is missing: use spatial distance sampling density as a proxy
        lap_times = df.groupby('Lap Number')['Lap Distance'].count()

    # Isolate lap time data to only consider valid, complete laps
    valid_lap_times = lap_times.loc[valid_laps]
    
    # Auto-detect fastest lap if no override is provided
    if target_lap is None:
        selected_lap = int(valid_lap_times.idxmin())
        print(f"[AUTO-DETECT] Fastest valid lap found: Lap {selected_lap}")
    else:
        selected_lap = target_lap

    df_lap = df[df['Lap Number'] == selected_lap].dropna(subset=['G Force Lat', 'G Force Long']).copy()

    # 3. Correct channel inversion (Braking = negative/down, Accel = positive/up)
    df_lap['G Force Long Corr'] = df_lap['G Force Long'] * -1

    # 4. Dynamic boundaries (Safeguarded against acceleration overfitting)
    max_lat_g = df_lap['G Force Lat'].abs().max() 
    max_long_decel = df_lap['G Force Long Corr'].min() 
    max_long_accel = df_lap['G Force Long Corr'].quantile(0.99)

    # Raw absolute limits for statistics display block
    true_peak_lat = df_lap['G Force Lat'].abs().max()
    true_peak_accel = df_lap['G Force Long Corr'].max()
    true_peak_decel = df_lap['G Force Long Corr'].min()

    # 5. Calculate Superellipse Limit & Utilization Scores
    lat_data = df_lap['G Force Lat'].values
    long_data = df_lap['G Force Long Corr'].values
    angles = np.arctan2(long_data, lat_data)

    n_exponent = 1.5 
    max_long_allowed = np.where(long_data >= 0, max_long_accel, np.abs(max_long_decel))

    # Superellipse radius math
    theoretical_limit_radius = 1.0 / (
        (np.abs(np.cos(angles)) / max_lat_g) ** n_exponent + 
        (np.abs(np.sin(angles)) / max_long_allowed) ** n_exponent
    ) ** (1.0 / n_exponent)

    actual_g_magnitude = np.sqrt(lat_data**2 + long_data**2)
    df_lap['Utilization %'] = np.clip((actual_g_magnitude / theoretical_limit_radius) * 100, 0, 105)

    # Isolate performance scores to dynamic cornering/braking zones
    cornering_zones = (df_lap['G Force Lat'].abs() > 0.4) | (df_lap['G Force Long Corr'] < -0.4)
    lap_utilization_score = df_lap.loc[cornering_zones, 'Utilization %'].mean()

    blended_inputs = (df_lap['G Force Lat'].abs() > 0.6) & (df_lap['G Force Long Corr'] < -0.4)
    trail_brake_score = df_lap.loc[blended_inputs, 'Utilization %'].mean() if blended_inputs.any() else 0.0

    # 6. Plotting Architecture
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 9), facecolor='#0f111a')
    ax.set_facecolor('#161925')

    # 2D Density Heatmap
    hb = ax.hexbin(
        df_lap['G Force Lat'], df_lap['G Force Long Corr'],
        gridsize=45, cmap='inferno', mincnt=1, edgecolors='none', alpha=0.85, zorder=2
    )

    cbar = plt.colorbar(hb, ax=ax, pad=0.03)
    cbar.set_label('Data Point Density (Time Spent)', color='#ffffff', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='#ffffff')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#ffffff')

    # Draw Superellipse Envelope Line
    env_angles = np.linspace(-np.pi, np.pi, 300)
    env_long_allowed = np.where(env_angles >= 0, max_long_accel, np.abs(max_long_decel))
    env_radius = 1.0 / (
        (np.abs(np.cos(env_angles)) / max_lat_g) ** n_exponent + 
        (np.abs(np.sin(env_angles)) / env_long_allowed) ** n_exponent
    ) ** (1.0 / n_exponent)

    ax.plot(env_radius * np.cos(env_angles), env_radius * np.sin(env_angles), 
            color='#00ffcc', linestyle='-', linewidth=2, alpha=0.9, 
            label=f'Superellipse Grip Limit (n={n_exponent})', zorder=3)

    # Reference background rings
    max_g_rounded = np.ceil(true_peak_lat)
    for g_limit in np.arange(1.0, max_g_rounded, 1.0):
        circle = plt.Circle((0, 0), g_limit, color='#444c63', fill=False, linestyle='--', alpha=0.15, zorder=1)
        ax.add_patch(circle)

    # Text Display Box
    stat_text = (
        f'CORNERING UTILIZATION: {lap_utilization_score:.1f}%\n'
        f'TRAIL BRAKE EFFICIENCY: {trail_brake_score:.1f}%\n\n'
        f'PEAK LATERAL: {true_peak_lat:.2f} G\n'
        f'MAX ACCEL: +{true_peak_accel:.2f} G\n'
        f'MAX BRAKING: {true_peak_decel:.2f} G'
    )
    ax.text(0.03, 0.85, stat_text, transform=ax.transAxes, fontsize=10, fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#212638', edgecolor='#00ffcc'))

    ax.set_title(f'ApexPulse Friction Circle & Density Heatmap (Lap {selected_lap})', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Lateral Acceleration (G)', fontsize=11, color='#a0a5b5')
    ax.set_ylabel('Longitudinal Acceleration (G)', fontsize=11, color='#a0a5b5')

    axis_limit = max_g_rounded + 0.5
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.axhline(0, color='#ffffff', linestyle=':', alpha=0.2)
    ax.axvline(0, color='#ffffff', linestyle=':', alpha=0.2)
    ax.grid(True, linestyle=':', alpha=0.15, color='#444c63')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', frameon=True, facecolor='#212638', edgecolor='none')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_img = os.path.join(output_dir, f'friction_circle_lap_{selected_lap}.png')
    plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SUCCESS] Plot saved to '{output_img}'.")
    plt.show()

# Execution entry point
if __name__ == '__main__':
    # Locate clean telemetry file in data/ or root directory
    data_path = (
        os.path.join('data', 'clean_data.csv')
        if os.path.exists(os.path.join('data', 'clean_data.csv'))
        else 'clean_data.csv'
    )
    
    generate_friction_circle(data_path, target_lap=TARGET_LAP)