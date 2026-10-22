from src.config import config_value
from src.core.analysis import optimal_lap_analysis
from src.pipeline.telemetry_io import load_telemetry, process_session


def calculate_optimal_lap(
    csv_path="data/raw_telemetrySilverstone.csv",
    micro_sector_meters=None,
):
    micro_sector_meters = (
        config_value("analysis", "micro_sector_m")
        if micro_sector_meters is None
        else micro_sector_meters
    )
    if micro_sector_meters <= 0:
        raise ValueError("micro_sector_meters must be greater than zero.")
    frame, _ = process_session(load_telemetry(csv_path))
    result = optimal_lap_analysis(frame, micro_sector_meters)

    def fmt(sec):
        return f"{int(sec // 60):02d}:{sec % 60:06.3f}"

    print("=" * 55)
    print("      APEXPULSE THEORETICAL OPTIMAL LAP ENGINE      ")
    print("=" * 55)
    print(f" Analyzed Laps        : {len(result['valid_laps'])} Valid Laps")
    print(f" Micro-Sector Resolution: {micro_sector_meters} Meters")
    print("-" * 55)
    print(
        f" Actual Best Lap (Lap {int(result['actual_best_lap'])}): "
        f"{fmt(result['actual_best_seconds'])}"
    )
    print(f" Realizable Theoretical Best: {fmt(result['theoretical_best_seconds'])}")
    print(f" Potential Time Gain  : -{result['potential_gain_seconds']:.3f} s")
    print("=" * 55)

if __name__ == "__main__":
    calculate_optimal_lap()