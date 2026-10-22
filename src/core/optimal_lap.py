import pandas as pd
import numpy as np

if __package__:
    from .telemetry_io import load_telemetry
else:
    from telemetry_io import load_telemetry


def calculate_optimal_lap(csv_path="data/raw_telemetrySilverstone.csv", micro_sector_meters=50):
    if micro_sector_meters <= 0:
        raise ValueError("micro_sector_meters must be greater than zero.")
    df = load_telemetry(csv_path)
    dist_col = "Lap Distance"
    speed_col = "Ground Speed"
    lap_col = "Lap Number"

    for col in (dist_col, speed_col, lap_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[dist_col, speed_col, lap_col]).copy()

    df["Normalized_Dist"] = df.groupby(lap_col)[dist_col].transform(lambda x: x - x.min())
    max_dist = df["Normalized_Dist"].max()
    valid_laps_df = df.groupby(lap_col).filter(lambda x: x["Normalized_Dist"].max() >= 0.90 * max_dist).copy()

    # Time calculation: dt = ds / v
    v_ms = np.maximum(valid_laps_df[speed_col] / 3.6, 1.0)
    ds = valid_laps_df.groupby(lap_col)["Normalized_Dist"].diff().fillna(0)
    ds = np.where(ds < 0, 0, ds)
    valid_laps_df["dt"] = ds / v_ms

    bins = np.arange(0, max_dist + micro_sector_meters, micro_sector_meters)
    valid_laps_df["MicroSector"] = pd.cut(valid_laps_df["Normalized_Dist"], bins=bins, labels=False)

    sector_matrix = valid_laps_df.groupby(["MicroSector", lap_col])["dt"].sum().unstack()
    optimal_sector_times = sector_matrix.min(axis=1)
    theoretical_best = optimal_sector_times.sum()

    lap_totals = valid_laps_df.groupby(lap_col)["dt"].sum()
    actual_best_lap = lap_totals.idxmin()
    actual_best_sec = lap_totals.min()

    def fmt(sec):
        return f"{int(sec // 60):02d}:{sec % 60:06.3f}"

    print("=" * 55)
    print("      APEXPULSE THEORETICAL OPTIMAL LAP ENGINE      ")
    print("=" * 55)
    print(f" Analyzed Laps        : {len(lap_totals)} Valid Laps")
    print(f" Micro-Sector Resolution: {micro_sector_meters} Meters")
    print("-" * 55)
    print(f" Actual Best Lap (Lap {int(actual_best_lap)}): {fmt(actual_best_sec)}")
    print(f" Theoretical Best Lap : {fmt(theoretical_best)}")
    print(f" Potential Time Gain  : -{(actual_best_sec - theoretical_best):.3f} s")
    print("=" * 55)

if __name__ == "__main__":
    calculate_optimal_lap()