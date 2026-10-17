import json
from pathlib import Path

import numpy as np

if __package__:
    from .sector_analysis import _sector_time
    from .telemetry_io import fastest_lap_numbers, lap_time_seconds, valid_lap_numbers
else:
    from sector_analysis import _sector_time
    from telemetry_io import fastest_lap_numbers, lap_time_seconds, valid_lap_numbers


def align_laps(frame, lap_a, lap_b, grid_step=1.0):
    if grid_step <= 0:
        raise ValueError("grid_step must be greater than zero.")
    if lap_a == lap_b:
        raise ValueError("Choose two different laps to compare.")

    first = (
        frame[frame["Lap Number"] == lap_a]
        .groupby("Normalized_Dist", as_index=False)["Ground Speed"]
        .mean()
        .sort_values("Normalized_Dist")
    )
    second = (
        frame[frame["Lap Number"] == lap_b]
        .groupby("Normalized_Dist", as_index=False)["Ground Speed"]
        .mean()
        .sort_values("Normalized_Dist")
    )
    if first.empty or second.empty:
        raise ValueError("Both selected laps must exist in the telemetry.")
    max_distance = min(first["Normalized_Dist"].max(), second["Normalized_Dist"].max())
    distance = np.arange(0, max_distance, grid_step)
    if len(distance) == 0 or distance[-1] < max_distance:
        distance = np.append(distance, max_distance)
    if len(distance) < 2:
        raise ValueError("The selected laps do not contain enough distance for comparison.")

    speed_a = np.interp(
        distance,
        first["Normalized_Dist"],
        first["Ground Speed"],
    )
    speed_b = np.interp(
        distance,
        second["Normalized_Dist"],
        second["Ground Speed"],
    )
    speed_a_ms = np.maximum(speed_a / 3.6, 1.0)
    speed_b_ms = np.maximum(speed_b / 3.6, 1.0)
    intervals = np.diff(distance)
    time_a = intervals * (0.5 / speed_a_ms[:-1] + 0.5 / speed_a_ms[1:])
    time_b = intervals * (0.5 / speed_b_ms[:-1] + 0.5 / speed_b_ms[1:])
    delta_b_ahead = np.concatenate(([0.0], np.cumsum(time_a - time_b)))

    return {
        "distance_m": distance,
        "speed_a_kmh": speed_a,
        "speed_b_kmh": speed_b,
        "delta_b_ahead_s": delta_b_ahead,
        "lap_a_seconds": lap_time_seconds(
            frame[frame["Lap Number"] == lap_a]
        ),
        "lap_b_seconds": lap_time_seconds(
            frame[frame["Lap Number"] == lap_b]
        ),
    }


def align_driver_sessions(frame_a, frame_b, grid_step=1.0):
    if grid_step <= 0:
        raise ValueError("grid_step must be greater than zero.")

    track_a = (
        frame_a["Track"].dropna().astype(str).unique()
        if "Track" in frame_a.columns
        else []
    )
    track_b = (
        frame_b["Track"].dropna().astype(str).unique()
        if "Track" in frame_b.columns
        else []
    )
    if len(track_a) > 1 or len(track_b) > 1:
        raise ValueError("Each telemetry file must identify at most one track.")
    if len(track_a) and len(track_b) and track_a[0] != track_b[0]:
        raise ValueError(
            f"Cannot compare different tracks ({track_a[0]} and {track_b[0]})."
        )

    laps_a = fastest_lap_numbers(frame_a, count=1)
    laps_b = fastest_lap_numbers(frame_b, count=1)
    if not laps_a or not laps_b:
        raise ValueError("Both sessions must contain at least one complete lap.")
    lap_a = (
        frame_a[frame_a["Lap Number"] == laps_a[0]]
        .groupby("Normalized_Dist", as_index=False)["Ground Speed"]
        .mean()
        .sort_values("Normalized_Dist")
    )
    lap_b = (
        frame_b[frame_b["Lap Number"] == laps_b[0]]
        .groupby("Normalized_Dist", as_index=False)["Ground Speed"]
        .mean()
        .sort_values("Normalized_Dist")
    )
    max_distance = min(
        float(lap_a["Normalized_Dist"].max()),
        float(lap_b["Normalized_Dist"].max()),
    )
    distance = np.arange(0, max_distance, grid_step)
    if len(distance) == 0 or distance[-1] < max_distance:
        distance = np.append(distance, max_distance)
    if len(distance) < 2:
        raise ValueError("The selected sessions do not contain enough overlapping distance.")

    speed_a = np.interp(distance, lap_a["Normalized_Dist"], lap_a["Ground Speed"])
    speed_b = np.interp(distance, lap_b["Normalized_Dist"], lap_b["Ground Speed"])
    speed_a_ms = np.maximum(speed_a / 3.6, 1.0)
    speed_b_ms = np.maximum(speed_b / 3.6, 1.0)
    intervals = np.diff(distance)
    time_a = intervals * (0.5 / speed_a_ms[:-1] + 0.5 / speed_a_ms[1:])
    time_b = intervals * (0.5 / speed_b_ms[:-1] + 0.5 / speed_b_ms[1:])
    delta_b_ahead = np.concatenate(([0.0], np.cumsum(time_a - time_b)))

    return {
        "distance_m": distance,
        "speed_a_kmh": speed_a,
        "speed_b_kmh": speed_b,
        "delta_b_ahead_s": delta_b_ahead,
        "lap_a": float(laps_a[0]),
        "lap_b": float(laps_b[0]),
        "lap_a_seconds": lap_time_seconds(
            frame_a[frame_a["Lap Number"] == laps_a[0]]
        ),
        "lap_b_seconds": lap_time_seconds(
            frame_b[frame_b["Lap Number"] == laps_b[0]]
        ),
        "track": (
            str(track_a[0])
            if len(track_a)
            else str(track_b[0]) if len(track_b) else None
        ),
    }


def optimal_lap_analysis(frame, micro_sector_meters=50.0):
    if micro_sector_meters <= 0:
        raise ValueError("micro_sector_meters must be greater than zero.")

    valid_laps = valid_lap_numbers(frame)
    if not valid_laps:
        raise ValueError("No complete laps are available for optimal-lap analysis.")

    spatial_laps = {
        lap_number: (
            frame[frame["Lap Number"] == lap_number]
            .groupby("Normalized_Dist", as_index=False)["Ground Speed"]
            .mean()
            .sort_values("Normalized_Dist")
        )
        for lap_number in valid_laps
    }
    common_distance = min(
        float(lap["Normalized_Dist"].max()) for lap in spatial_laps.values()
    )
    if common_distance <= 0:
        raise ValueError("Complete laps do not contain enough distance for analysis.")

    edges = np.arange(0, common_distance, micro_sector_meters)
    edges = np.append(edges, common_distance)
    sector_count = len(edges) - 1
    times_by_lap = {}
    for lap_number, lap in spatial_laps.items():
        lap_distance = lap["Normalized_Dist"].to_numpy(dtype=float)
        lap_speed = np.maximum(lap["Ground Speed"].to_numpy(dtype=float) / 3.6, 1.0)
        interior = (lap_distance > 0) & (lap_distance < common_distance)
        distances = np.concatenate(([0.0], lap_distance[interior], [common_distance]))
        speeds = np.interp(distances, lap_distance, lap_speed)
        intervals = np.diff(distances)
        interval_times = intervals * (0.5 / speeds[:-1] + 0.5 / speeds[1:])
        midpoints = (distances[:-1] + distances[1:]) / 2
        sector_indices = np.minimum(
            np.searchsorted(edges[1:], midpoints, side="right"),
            sector_count - 1,
        )
        times_by_lap[lap_number] = np.bincount(
            sector_indices,
            weights=interval_times,
            minlength=sector_count,
        )

    lap_totals = {
        lap_number: float(times.sum())
        for lap_number, times in times_by_lap.items()
    }
    actual_best_lap = min(lap_totals, key=lap_totals.__getitem__)
    sector_results = []
    for index in range(sector_count):
        best_lap = min(
            valid_laps,
            key=lambda lap_number: times_by_lap[lap_number][index],
        )
        sector_results.append(
            {
                "start_m": float(edges[index]),
                "end_m": float(edges[index + 1]),
                "best_time_seconds": float(times_by_lap[best_lap][index]),
                "best_lap": float(best_lap),
            }
        )
    theoretical_best = sum(
        sector["best_time_seconds"] for sector in sector_results
    )
    return {
        "valid_laps": [float(lap_number) for lap_number in valid_laps],
        "common_distance_m": common_distance,
        "micro_sector_meters": float(micro_sector_meters),
        "actual_best_lap": float(actual_best_lap),
        "actual_best_seconds": lap_totals[actual_best_lap],
        "theoretical_best_seconds": theoretical_best,
        "potential_gain_seconds": lap_totals[actual_best_lap] - theoretical_best,
        "sectors": sector_results,
    }


def sector_results(frame, lap_a, lap_b, track_config):
    laps = {
        lap_number: frame[frame["Lap Number"] == lap_number].sort_values("Normalized_Dist")
        for lap_number in (lap_a, lap_b)
    }
    results = []
    for name, bounds in track_config.get("sectors", {}).items():
        start, end = bounds
        metrics = {}
        for lap_number, lap in laps.items():
            in_sector = lap["Normalized_Dist"].between(start, end, inclusive="both")
            if not in_sector.any():
                break
            duration = _sector_time(lap, start, end)
            if duration is None:
                break
            metrics[lap_number] = {
                "v_min_kmh": float(lap.loc[in_sector, "Ground Speed"].min()),
                "duration_seconds": duration,
            }
        if len(metrics) != 2:
            results.append(
                {
                    "sector": name,
                    "start_m": start,
                    "end_m": end,
                    "available": False,
                }
            )
            continue
        first = metrics[lap_a]
        second = metrics[lap_b]
        results.append(
            {
                "sector": name,
                "start_m": start,
                "end_m": end,
                "available": True,
                "lap_a_v_min_kmh": first["v_min_kmh"],
                "lap_b_v_min_kmh": second["v_min_kmh"],
                "v_min_delta_kmh": second["v_min_kmh"] - first["v_min_kmh"],
                "lap_a_seconds": first["duration_seconds"],
                "lap_b_seconds": second["duration_seconds"],
                "delta_b_ahead_seconds": first["duration_seconds"] - second["duration_seconds"],
            }
        )
    return results


def load_track_config(frame, tracks_dir=None):
    if "Track" not in frame.columns or frame["Track"].nunique() != 1:
        return None
    track = frame["Track"].iloc[0]
    directory = (
        Path(tracks_dir)
        if tracks_dir
        else Path(__file__).resolve().parent.parent / "tracks"
    )
    config_path = directory / f"{track}.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read track configuration '{config_path}': {exc}") from exc


def analysis_report(frame, quality_report, lap_a, lap_b, grid_step=1.0):
    aligned = align_laps(frame, lap_a, lap_b, grid_step)
    config = load_track_config(frame)
    sectors = sector_results(frame, lap_a, lap_b, config) if config else []
    return {
        "schema_version": 1,
        "source_file": quality_report.get("source_file"),
        "track": quality_report.get("track"),
        "lap_comparison": {
            "reference_lap": float(lap_a),
            "compared_lap": float(lap_b),
            "reference_lap_seconds": round(aligned["lap_a_seconds"], 3),
            "compared_lap_seconds": round(aligned["lap_b_seconds"], 3),
            "final_delta_compared_lap_ahead_seconds": round(
                float(aligned["delta_b_ahead_s"][-1]), 3
            ),
            "delta_convention": "positive means the compared lap is ahead",
            "delta_method": "trapezoidal distance/speed integration",
        },
        "sector_results": sectors,
        "optimal_lap": optimal_lap_analysis(frame),
        "data_quality": quality_report,
    }
