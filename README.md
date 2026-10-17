# ApexPulse | Track-Agnostic Telemetry & Dynamics Suite

**ApexPulse** is an automated telemetry engineering pipeline and vehicle dynamics analysis toolkit built for sim racing (Assetto Corsa Competizione) and real-world logging systems (e.g., MoTeC). Developed in Python using Pandas, NumPy, Matplotlib, Plotly, and Streamlit, it converts raw time-series telemetry into normalized spatial datasets, $G\text{-}G$ plots, continuous spatial time-delta curves, and corner-by-corner performance breakdowns.

By decoupling analytical logic from circuit geometries through dynamic JSON configurations (`/tracks`), ApexPulse automatically detects track identity from dataset metadata and adapts to new circuits without any code modification.

---

## Key Technical Features

* **Zero-Touch Automated Data Pipeline:** Parses raw multi-row header metadata, filters sensor noise via rolling medians, auto-detects track identity from filenames, and exports clean normalized datasets.
* **Shared Telemetry Ingestion:** Canonicalizes lap, distance, and speed channels, sorts samples, reports invalid and repeated-distance rows, and uses a single lap-distance definition across ingestion and comparison. Valid samples at repeated distances are retained for time-based analysis.
* **Interactive Session Dashboard:** Presents lap speed/delta, driver inputs, cross-session driver comparison, G-G sample density, a track view, sector estimates, theoretical optimal-lap analysis, and data-quality details in one browser UI.
* **Reproducible Session Reports:** Writes a cleaned CSV, a data-quality JSON report, and a session report with selected-lap and sector comparisons.
* **1-Meter Spatial Grid Alignment:** Interpolates asynchronous time-series telemetry onto a uniform spatial grid ($1\text{m}$ resolution) to allow direct distance-aligned lap comparisons without time-base distortion.
* **Sample-Preserving Spatial Alignment:** Retains valid time samples even when distance is repeated at the logger's resolution, averaging by distance only for spatial interpolation.
* **Kinematic 2D Track Map Reconstruction:** Provides track visualization from logged GPS coordinates or an explicitly approximate path reconstructed from lateral acceleration when GPS is unavailable.
* **Vehicle Dynamics ($G\text{-}G$ Plots):** Visualizes lateral vs. longitudinal acceleration as sample density in the dashboard. The standalone friction-circle script also draws a superellipse envelope based on observed lap data; this is not a validated tire-grip limit.
* **Config-Driven Corner Sector Analysis:** Reads JSON distance gates to evaluate corner minimum speed ($\text{V-Min}$) and sector time deltas.
* **Automated Incomplete Lap Filtering:** Excludes laps below a configurable minimum distance-coverage threshold (90% by default) before analysis.

---

## Repository Architecture

    ApexPulse/
    ├── assets/                  # Example dark-theme analysis plots (.png)
    │   ├── driver_comparison.png
    │   ├── friction_circle_density.png
    │   ├── lap_comparison.png
    │   ├── math_channels.png
    │   ├── telemetry_traces.png
    │   ├── time_delta.png
    │   └── track_map.png
    ├── data/                    # Raw telemetry inputs and cleaned datasets (.csv)
    │   ├── clean_data.csv
    │   ├── data_with_math_channels.csv
    │   └── raw_telemetrySilverstone.csv
    ├── src/                     # Core analysis, plotting, CLI, and dashboard modules
    │   ├── __init__.py          # Python package marker
    │   ├── analysis.py          # Shared lap alignment, sector metrics & session report
    │   ├── cli.py               # Configurable clean and dashboard commands
    │   ├── dashboard.py         # Interactive browser-based session dashboard
    │   ├── telemetry_io.py      # Shared CSV ingestion, channel normalization & lap timing
    │   ├── clean_telemetry.py   # Raw data ingestion, noise filtering, auto-tagging
    │   ├── compare_laps.py      # Spatial grid alignment & differential speed shading
    │   ├── driver_comparison.py # Fastest-lap driver comparison
    │   ├── math_channels.py     # Derived telemetry channels and plots
    │   ├── optimal_lap.py       # Theoretical optimal-lap analysis
    │   ├── plot_circle.py       # Standalone G-G friction-circle plot
    │   ├── plot_delta.py        # Cumulative spatial time-delta engine
    │   ├── plot_traces.py       # Multi-panel driver input dashboard
    │   ├── plot_track.py        # Track visualization from telemetry
    │   └── sector_analysis.py   # Config-driven corner V-Min & sector breakdown
    ├── tests/                   # Telemetry, analysis, CLI, and dashboard tests
    ├── tracks/                  # Circuit sector configuration files (.json)
    │   ├── Barcelona.json
    │   ├── Bathurst.json
    │   ├── Imola.json
    │   ├── Monza.json
    │   ├── Nurburgring.json
    │   ├── Silverstone.json
    │   ├── Spa.json
    │   └── Suzuka.json
    ├── .github/workflows/       # Automated tests and compilation on pushes/PRs
    ├── .streamlit/              # Dashboard theme configuration
    ├── .vscode/settings.json    # Default project interpreter and Explorer exclusions
    ├── .gitignore               # System cache & build exclusion filters
    ├── README.md                # Project documentation
    └── requirements.txt         # Runtime dependencies

---

The top-level folders separate application code, tests, telemetry data, track configurations, and image assets. Generated reports go in the ignored `reports/` folder; the local `.venv/` environment is also ignored and should not be committed. Source modules stay under `src/` so the documented `python -m src...` commands continue to work.

## File Naming Convention

To enable zero-touch track detection, ensure your input CSV file includes the circuit name in its filename (e.g., `raw_telemetrySilverstone.csv`, `spa_session.csv`, or `monza_hotlap.csv`). 

The pipeline parses the filename, tags the data, and loads the matching geometry configuration from `/tracks` automatically.

---

## Quickstart

### 1. Installation
Clone the repository and install required dependencies:

    git clone [https://github.com/ItzJTz/ApexPulse.git](https://github.com/ItzJTz/ApexPulse.git)
    cd ApexPulse
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter** if it is not already selected. The workspace settings provide this as the default and hide `.venv` from the Explorer; `.gitignore` prevents it from being committed.


### 2. Clean a session and create reports

Run from the repository root. The input may be a raw MoTeC-style CSV or a previously cleaned CSV:

    .\.venv\Scripts\python.exe -m src.cli clean data\raw_telemetrySilverstone.csv --output-dir reports

The command writes three artifacts into `reports/`: a normalized, complete-lap CSV; a data-quality JSON file with row exclusions, channel mappings, missing optional-channel rates, and lap coverage; and a session JSON report with lap, sector, and theoretical optimal-lap analysis when enough valid laps are available. Use `--minimum-lap-fraction` to adjust the complete-lap threshold, or `--report-json path.json` to choose the session-report location. `python src/clean_telemetry.py [input.csv] --output path.csv` remains available for the simple one-file cleaning workflow.

Track identity is resolved from the input filename or MoTeC `Venue` metadata against JSON files in `tracks/`. If the track is unknown, generic lap analysis still works, but track-specific sectors are not guessed.

### 3. Explore the session in one dashboard

Start the dashboard from the repository root:

    .\.venv\Scripts\python.exe -m src.cli dashboard

Then follow these steps:

1. In the left sidebar, either upload a telemetry CSV or choose a CSV already in `data/`. If a file is uploaded, it takes precedence over the local-file selector.
2. Set **Minimum lap coverage**. The default is 90%; increase it to exclude more partial laps or lower it only when appropriate for your session.
3. Select a **Reference lap** and a **Compared lap**. The dashboard initially selects the two fastest valid laps; you can change either selection.
4. Open the tabs to review the session:

   * **Overview:** lap speed traces and cumulative time delta. Positive delta means the compared lap is ahead.
   * **Driver inputs:** available throttle, brake, steering, and G channels aligned by distance.
   * **G-G:** measured lateral/longitudinal acceleration sample density. Brighter cells mean more samples in that acceleration range; empty ranges are blank. Lateral G is signed left/right. Longitudinal G is sign-inverted, so positive means acceleration and negative means braking. It does not claim to estimate a validated tire-grip envelope.
   * **Track map:** logged GPS positions when usable; otherwise an explicitly labeled approximate path from lateral acceleration.
   * **Sectors:** minimum speeds and distance-integrated sector times when the track is identified and a matching configuration is available.
   * **Data quality:** included/excluded laps, dropped/duplicate rows, optional-channel completeness, and downloads for the cleaned CSV and JSON report.
   * **Driver comparison:** upload a second session to compare its fastest complete lap with the fastest complete lap from the main session. Both files should be from the same circuit and use compatible distance units. If track metadata is missing, confirm this manually in the tab.
   * **Optimal lap:** combines each distance micro-sector's fastest observed time across complete laps. This is a theoretical benchmark assembled from different laps, not a lap achieved in one run.

5. Use the data-quality tab to check which laps and samples were retained before interpreting comparisons or downloading the cleaned data and report.

The dashboard reports sector times as estimates. A reconstructed path is not surveyed circuit geometry, and GPS coordinate values are shown as supplied by the data source. Track-specific sectors require the filename or source metadata to identify a track with a matching JSON configuration in `tracks/`.

To run the separate command-line sector report, provide a raw or cleaned CSV whose filename or Venue metadata identifies its track:

    python -m src.sector_analysis data/raw_telemetrySilverstone.csv

For the supplied Silverstone sample, this compares the last two complete laps. A generic filename such as `clean_data.csv` without Venue metadata cannot identify the circuit; use a track-named file or the original telemetry file instead.

The dashboard and command-line tools share lap alignment and telemetry processing. Synthetic-data unit tests cover fastest-lap selection, spatial alignment, repeated-distance handling, and optimal-lap calculations; real cross-driver sessions are still needed to validate results against actual driver data.

To compare two real sessions, provide one CSV for each driver. The script selects the fastest complete lap in each file, aligns the speed traces by distance, and saves the plot to `assets/driver_comparison.png`:

    .\.venv\Scripts\python.exe -m src.driver_comparison data\driverA_Silverstone.csv data\driverB_Silverstone.csv

Both files should be from the same circuit and use compatible distance units. Use `--output path\to\comparison.png` to choose a different image path. For a plot-only demo without a second driver's data, `--synthetic-benchmark` generates synthetic speed for Driver B; do not treat that result as a real driver comparison.

`python -m src.math_channels` is an optional standalone derived-channel plot. Its downforce estimate uses a fixed aerodynamic coefficient and its brake-bias output is a fixed assumption, so treat those channels as illustrative until calibrated for the vehicle.

### 4. Run automated checks

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
    .\.venv\Scripts\python.exe -m compileall -q src tests

GitHub Actions runs the same tests and Python compilation check for pushes and pull requests.
---

## Visual Analytics Showcase

### 1. Reconstructed 2D Track Map & Speed Heatmap
![Track Map](assets/track_map.png)

### 2. Multi-Channel Telemetry Dashboard
![Telemetry Traces](assets/telemetry_traces.png)

### 3. G-G Sample-Density Plot
![G-G Sample Density](assets/friction_circle_density.png)

### 4. Continuous Spatial Time-Delta Engine
![Time Delta](assets/time_delta.png)

### 5. Spatial Lap Comparison with Differential Shading
![Lap Comparison](assets/lap_comparison.png)

Creator: JT
Focus: Data Engineering | Vehicle Dynamics | GT Motorsport
