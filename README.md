# ApexPulse | Track-Agnostic Telemetry & Dynamics Suite

**ApexPulse** is an automated telemetry engineering pipeline and vehicle dynamics analysis toolkit built for sim racing (Assetto Corsa Competizione) and real-world logging systems (e.g., MoTeC). Developed in Python using Pandas, NumPy, and Matplotlib, it converts raw time-series telemetry into normalized spatial datasets, $G\text{-}G$ friction circles, continuous spatial time-delta curves, and corner-by-corner performance breakdowns.

By decoupling analytical logic from circuit geometries through dynamic JSON configurations (`/tracks`), ApexPulse automatically detects track identity from dataset metadata and adapts to new circuits without any code modification.

---

## Key Technical Features

* **Zero-Touch Automated Data Pipeline:** Parses raw multi-row header metadata, filters sensor noise via rolling medians, auto-detects track identity from filenames, and exports clean normalized datasets.
* **1-Meter Spatial Grid Alignment:** Interpolates asynchronous time-series telemetry onto a uniform spatial grid ($1\text{m}$ resolution) to allow direct distance-aligned lap comparisons without time-base distortion.
* **Sensor Spike Suppression & Monotonic Filtering:** Strips single-frame telemetry dropouts and enforces strict spatial monotonicity to prevent mathematical artifacts during interpolation.
* **Kinematic 2D Track Map Reconstruction:** Reconstructs true circuit geometry and corner curvature via dead-reckoning physics integration ($d\theta = \frac{a_{\text{lat}}}{v^2} ds$) with loop-closure drift correction.
* **Vehicle Dynamics & Limit Utilization ($G\text{-}G$ Circles):** Renders lateral vs. longitudinal acceleration friction circles color-coded by ground speed to evaluate tire load management and driver performance at the grip limit.
* **Config-Driven Corner Sector Analysis:** Reads JSON distance gates to evaluate corner minimum speed ($\text{V-Min}$) and sector time deltas.
* **Automated Incomplete Lap Filtering:** Automatically ignores in-laps, out-laps, and crash data by enforcing a $90\%$ minimum lap distance threshold prior to evaluation.

---

## Repository Architecture

```text
ApexPulse/
├── assets/                  # High-resolution dark-theme dashboard exports (.png)
│   ├── friction_circle.png
│   ├── lap_comparison.png
│   ├── telemetry_traces.png
│   ├── time_delta.png
│   └── track_map.png
├── data/                    # Raw telemetry inputs and cleaned datasets (.csv)
│   ├── clean_data.csv
│   └── raw_telemetrySilverstone.csv
├── src/                     # Core analytical and plotting modules
│   ├── clean_telemetry.py   # Raw data ingestion, noise filtering, auto-tagging
│   ├── compare_laps.py      # Spatial grid alignment & differential speed shading
│   ├── plot_circle.py       # G-G Friction Circle and peak force analysis
│   ├── plot_delta.py        # Cumulative spatial time-delta engine
│   ├── plot_traces.py       # Multi-panel driver input dashboard
│   ├── plot_track.py        # Kinematic 2D track layout reconstruction & heatmap
│   └── sector_analysis.py   # Config-driven corner V-Min & sector breakdown
├── tracks/                  # Circuit sector configuration files (.json)
│   ├── Barcelona.json
│   ├── Bathurst.json
│   ├── Imola.json
│   ├── Monza.json
│   ├── Nurburgring.json
│   ├── Silverstone.json
│   ├── Spa.json
│   └── Suzuka.json
├── .gitignore               # System cache & build exclusion filters
├── README.md                # Project documentation
└── requirements.txt         # Environment dependencies
