import json
import os
import pandas as pd

# 1. Locate clean telemetry file in data/ or root directory
clean_file_path = os.path.join('data', 'clean_data.csv') if os.path.exists(os.path.join('data', 'clean_data.csv')) else 'clean_data.csv'

if not os.path.exists(clean_file_path):
    print("Error: 'clean_data.csv' not found. Run src/clean_telemetry.py first.")
    exit()

df = pd.read_csv(clean_file_path)

# 2. Auto-detect track name embedded by clean_telemetry.py
detected_track = df['Track'].iloc[0] if 'Track' in df.columns else "Silverstone"
config_path = os.path.join("tracks", f"{detected_track}.json")

if not os.path.exists(config_path):
    print(f"Error: Configuration file '{config_path}' not found in /tracks folder.")
    exit()

with open(config_path, "r") as f:
    track_config = json.load(f)

# 3. Filter out partial/incomplete laps automatically
track_length = track_config.get("total_length_m", 5000)
lap_max_distances = df.groupby('Lap Number')['Lap Distance'].max()
valid_laps = lap_max_distances[lap_max_distances >= (track_length * 0.90)].index.tolist()

if len(valid_laps) < 2:
    print(f"Error: Less than two complete flying laps found in dataset for {track_config['circuit_name']}.")
    exit()

lap_A, lap_B = int(valid_laps[-2]), int(valid_laps[-1])

df_lapA = df[df['Lap Number'] == lap_A].sort_values(by='Lap Distance')
df_lapB = df[df['Lap Number'] == lap_B].sort_values(by='Lap Distance')

print(f"\n=========================================================================================")
print(f"       APEXPULSE SECTOR & LAP TIME DELTA ANALYSIS: {track_config['circuit_name']}")
print(f"       Comparing Lap {lap_A} vs Lap {lap_B} (Auto-Filtered Full Flying Laps)")
print(f"=========================================================================================")
print(f"{'Corner Sector':<30} | {f'Lap {lap_A} V-Min':<11} | {f'Lap {lap_B} V-Min':<11} | {'V-Min Delta':<11} | {'Est. Sector Delta':<16}")
print("-" * 101)

# 4. Calculate metrics per sector dynamically from JSON config
for sector_name, (start_dist, end_dist) in track_config["sectors"].items():
    zone_A = df_lapA[(df_lapA['Lap Distance'] >= start_dist) & (df_lapA['Lap Distance'] <= end_dist)]
    zone_B = df_lapB[(df_lapB['Lap Distance'] >= start_dist) & (df_lapB['Lap Distance'] <= end_dist)]
    
    if not zone_A.empty and not zone_B.empty:
        vmin_A = zone_A['Ground Speed'].min()
        vmin_B = zone_B['Ground Speed'].min()
        vmin_delta = vmin_B - vmin_A
        vmin_str = f"+{vmin_delta:.1f}" if vmin_delta > 0 else f"{vmin_delta:.1f}"
        
        dist_span = end_dist - start_dist
        avg_speed_A = zone_A['Ground Speed'].mean() / 3.6
        avg_speed_B = zone_B['Ground Speed'].mean() / 3.6
        
        time_A = dist_span / avg_speed_A if avg_speed_A > 0 else 0
        time_B = dist_span / avg_speed_B if avg_speed_B > 0 else 0
        time_delta = time_A - time_B
        
        time_str = f"+{time_delta:.3f}s (B faster)" if time_delta > 0 else f"-{abs(time_delta):.3f}s (A faster)"

        print(f"{sector_name:<30} | {vmin_A:5.1f} km/h  | {vmin_B:5.1f} km/h  | {vmin_str:>5} km/h  | {time_str}")

print("=========================================================================================\n")