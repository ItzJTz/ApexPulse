import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.spatial import interpolate_spatial
from src.pipeline.telemetry_io import load_telemetry, valid_lap_numbers


def _sector_time(lap, start_distance, end_distance):
    spatial_time = None
    if "Time" in lap.columns:
        timed_samples = (
            lap.groupby("Normalized_Dist", as_index=False)["Time"]
            .mean()
            .sort_values("Normalized_Dist")
        )
        distance = timed_samples["Normalized_Dist"].to_numpy(dtype=float)
        times = pd.to_numeric(
            timed_samples["Time"], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(distance) & np.isfinite(times)
        distance = distance[finite]
        times = times[finite]
        if (
            len(distance) >= 2
            and distance[0] <= start_distance
            and distance[-1] >= end_distance
            and np.all(np.diff(times) >= 0)
            and times[-1] > times[0]
        ):
            spatial_time = float(
                np.interp(end_distance, distance, times)
                - np.interp(start_distance, distance, times)
            )
    if spatial_time is not None:
        return spatial_time

    spatial_samples = (
        lap.groupby("Normalized_Dist", as_index=False)["Ground Speed"]
        .mean()
        .sort_values("Normalized_Dist")
    )
    distance = spatial_samples["Normalized_Dist"].to_numpy(dtype=float)
    speed_ms = np.maximum(spatial_samples["Ground Speed"].to_numpy(dtype=float) / 3.6, 1.0)
    if distance[0] > start_distance or distance[-1] < end_distance:
        return None

    interior = (distance > start_distance) & (distance < end_distance)
    sector_distances = np.concatenate(
        (
            [start_distance],
            distance[interior],
            [end_distance],
        )
    )
    sector_speed = interpolate_spatial(distance, speed_ms, sector_distances)
    intervals = np.diff(sector_distances)
    return float(np.sum(intervals * (0.5 / sector_speed[:-1] + 0.5 / sector_speed[1:])))


def scale_track_config_to_distance(track_config, distance_m):
    """Scale configured spatial gates to the measured lap distance."""
    if not track_config:
        return track_config
    try:
        distance = float(distance_m)
        nominal_length = float(
            track_config.get(
                "_nominal_total_length_m",
                track_config.get("total_length_m"),
            )
        )
    except (TypeError, ValueError):
        return dict(track_config)
    if (
        not math.isfinite(distance)
        or distance <= 0
        or not math.isfinite(nominal_length)
        or nominal_length <= 0
    ):
        return dict(track_config)

    scale = distance / nominal_length
    scaled = dict(track_config)
    scaled["_nominal_total_length_m"] = nominal_length
    scaled["distance_scale_factor"] = scale
    scaled["total_length_m"] = distance
    for key in ("sectors", "official_sectors"):
        if key in track_config:
            scaled[key] = {
                name: [float(bound) * scale for bound in bounds]
                for name, bounds in track_config[key].items()
            }
    if "corner_metadata" in track_config:
        scaled["corner_metadata"] = []
        for corner in track_config["corner_metadata"]:
            scaled_corner = dict(corner)
            for key in ("start_m", "apex_m", "end_m"):
                if isinstance(corner.get(key), (int, float)):
                    scaled_corner[key] = float(corner[key]) * scale
            scaled["corner_metadata"].append(scaled_corner)
    return scaled


def analyze_sectors(clean_file_path="data/2_24_638_PB.csv", tracks_dir=None):
    df = load_telemetry(clean_file_path, tracks_dir)
    if "Track" not in df.columns or df["Track"].nunique() != 1:
        raise ValueError(
            f"Could not identify exactly one configured track from '{clean_file_path}'. "
            "Use a filename containing a known track name or check the Venue metadata. "
            "A generic filename such as 'clean_data.csv' needs to be renamed with the "
            "track name or replaced with the original telemetry file."
        )

    track = df["Track"].iloc[0]
    tracks_path = (
        Path(tracks_dir)
        if tracks_dir
        else Path(__file__).resolve().parents[2] / "tracks"
    )
    config_path = tracks_path / f"{track}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Track configuration not found: '{config_path}'.")
    with config_path.open("r", encoding="utf-8") as stream:
        track_config = json.load(stream)

    track_length = track_config.get("total_length_m")
    if not isinstance(track_length, (int, float)) or track_length <= 0:
        raise ValueError(f"Track configuration '{config_path}' has an invalid total_length_m.")
    valid_laps = valid_lap_numbers(df, track_length=track_length)
    if len(valid_laps) < 2:
        raise ValueError(
            f"Less than two complete flying laps found for {track_config.get('circuit_name', track)}."
        )

    lap_a, lap_b = valid_laps[-2:]
    shared_distance = min(
        float(df.loc[df["Lap Number"] == lap, "Normalized_Dist"].max())
        for lap in (lap_a, lap_b)
    )
    track_config = scale_track_config_to_distance(track_config, shared_distance)
    laps = {
        lap_number: df[df["Lap Number"] == lap_number].sort_values("Normalized_Dist")
        for lap_number in (lap_a, lap_b)
    }

    print("=" * 89)
    print(f" APEXPULSE SECTOR ANALYSIS: {track_config.get('circuit_name', track)}")
    print(f" Comparing Lap {lap_a:g} vs Lap {lap_b:g} (last two complete laps)")
    print("=" * 89)
    print(
        f"{'Corner Sector':<30} | {f'Lap {lap_a:g} V-Min':<14} | "
        f"{f'Lap {lap_b:g} V-Min':<14} | {'V-Min Delta':<13} | {'Est. Sector Delta':<20}"
    )
    print("-" * 109)

    for sector_name, bounds in track_config.get("sectors", {}).items():
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or bounds[0] < 0
            or bounds[1] <= bounds[0]
        ):
            raise ValueError(f"Invalid distance bounds for sector '{sector_name}' in '{config_path}'.")
        start_distance, end_distance = bounds
        sector_data = {}
        for lap_number, lap in laps.items():
            covered = lap[
                lap["Normalized_Dist"].between(start_distance, end_distance, inclusive="both")
            ]
            sector_time = _sector_time(lap, start_distance, end_distance)
            if covered.empty or sector_time is None:
                sector_data = {}
                break
            sector_data[lap_number] = (float(covered["Ground Speed"].min()), sector_time)

        if not sector_data:
            print(f"{sector_name:<30} | Not enough telemetry to cover {start_distance:g}-{end_distance:g} m")
            continue

        vmin_a, time_a = sector_data[lap_a]
        vmin_b, time_b = sector_data[lap_b]
        vmin_delta = vmin_b - vmin_a
        time_delta = time_a - time_b
        displayed_delta = round(time_delta, 3)
        if displayed_delta > 0:
            faster = "B faster"
        elif displayed_delta < 0:
            faster = "A faster"
        else:
            faster = "tie"
        print(
            f"{sector_name:<30} | {vmin_a:5.1f} km/h       | {vmin_b:5.1f} km/h       | "
            f"{vmin_delta:+6.1f} km/h   | {displayed_delta:+.3f}s ({faster})"
        )
    print("=" * 89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the sector performance of the last two complete laps."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/2_24_638_PB.csv",
        help=(
            "MoTeC LD or telemetry CSV. Its filename or Venue metadata must identify "
            "a configured track (default: data/2_24_638_PB.csv)."
        ),
    )
    parser.add_argument(
        "--tracks-dir",
        help="Optional directory containing track configuration JSON files.",
    )
    arguments = parser.parse_args()
    analyze_sectors(arguments.input, arguments.tracks_dir)
