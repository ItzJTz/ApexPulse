import pandas as pd
import numpy as np
import glob
import os

# 1. Search for raw telemetry CSV files in both data/ folder and root directory
search_patterns = ['data/raw_*.csv', 'data/*.csv', 'raw_*.csv', '*.csv']
raw_files = []

for pattern in search_patterns:
    found = glob.glob(pattern)
    # Ignore any clean_data.csv files from search
    found = [f for f in found if not f.endswith('clean_data.csv')]
    if found:
        raw_files.extend(found)

# Remove duplicates while preserving order
raw_files = list(dict.fromkeys(raw_files))

if not raw_files:
    print("Error: No raw CSV telemetry file found in project or data/ folder.")
    exit()

# Pick the most recently modified file
input_file = max(raw_files, key=os.path.getmtime)
print(f"[AUTO-DETECT] Selected input file: '{input_file}'")

# 2. Detect Track Name from filename
filename_lower = os.path.basename(input_file).lower()
known_tracks = ['silverstone', 'spa', 'monza', 'suzuka', 'imola', 'nurburgring', 'bathurst', 'barcelona']
detected_track = "Silverstone"  # Default fallback

for track in known_tracks:
    if track in filename_lower:
        detected_track = track.capitalize()
        break

print(f"[AUTO-DETECT] Identified Track: '{detected_track}'")

# 3. Load raw telemetry data (skipping MoTeC header lines)
try:
    df = pd.read_csv(input_file, skiprows=14, on_bad_lines='skip')
    print(f"Successfully loaded {input_file} ({len(df)} rows).")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Drop units row if present
df = df.iloc[1:].reset_index(drop=True)

# Coerce numerical columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Filter and clean telemetry data
target_columns = [
    'Distance', 'Lap Distance', 'Lap Number', 'Ground Speed', 
    'Throttle Pos', 'Brake Pos', 'Steering', 'G Force Lat', 'G Force Long'
]

available_cols = [c for c in target_columns if c in df.columns]
df = df[available_cols].dropna(subset=['Lap Distance', 'Ground Speed'])

# Smooth out sensor noise using a rolling median filter
if 'Ground Speed' in df.columns:
    df['Ground Speed'] = df['Ground Speed'].rolling(window=5, min_periods=1).median()

# Embed the detected track name into the dataset metadata
df['Track'] = detected_track

# 5. Save cleaned pipeline output into data/ folder if it exists
output_dir = 'data' if os.path.exists('data') else '.'
output_file = os.path.join(output_dir, 'clean_data.csv')
df.to_csv(output_file, index=False)
print(f"[SUCCESS] Cleaned data saved to '{output_file}' with track tag: '{detected_track}'.\n")