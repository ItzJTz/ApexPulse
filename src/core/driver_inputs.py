"""Driver-input event detection and spatial corner mapping."""

import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


DEFAULT_SAMPLE_RATE_HZ = 50.0
EVENT_CLUSTER_DISTANCE_M = 40.0


def cluster_distances(distances, max_gap_m=EVENT_CLUSTER_DISTANCE_M):
    """Group sorted trigger distances whose consecutive gaps are within a limit."""
    if not math.isfinite(max_gap_m) or max_gap_m < 0:
        raise ValueError("max_gap_m must be finite and non-negative.")
    ordered = []
    for distance in distances:
        if distance is None:
            continue
        try:
            value = float(distance)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            ordered.append(value)
    ordered.sort()
    if not ordered:
        return []

    clusters = []
    current = [ordered[0]]
    for distance in ordered[1:]:
        if distance - current[-1] <= max_gap_m:
            current.append(distance)
        else:
            clusters.append(_distance_cluster(current))
            current = [distance]
    clusters.append(_distance_cluster(current))
    return clusters


def _distance_cluster(distances):
    return {
        "start_m": distances[0],
        "end_m": distances[-1],
        "distance_m": float(np.mean(distances)),
        "sample_count": len(distances),
    }


def _corner_label(corner, is_sepang):
    corner_id = str(corner.get("id", ""))
    name = str(corner.get("name", corner_id))
    if is_sepang and corner_id == "T1":
        name = "Pangkor Laut"
    elif is_sepang and corner_id == "T15":
        name = "Petronas Hairpin"
    return f"{corner_id} {name}".strip()


def map_distance_to_corner(distance_m, track_config=None):
    """Map a distance to the nearest configured apex and return its corner label."""
    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        return None
    corners = track_config.get("corner_metadata", []) if track_config else []
    if not math.isfinite(distance) or not corners:
        return None
    circuit_name = str(track_config.get("circuit_name", "")).casefold()
    is_sepang = "sepang" in circuit_name or str(
        track_config.get("ac_track_id", "")
    ).casefold() == "sepang"
    corner = min(
        corners,
        key=lambda item: abs(float(item["apex_m"]) - distance),
    )
    return _corner_label(corner, is_sepang)


def auto_detect_apexes(
    lap,
    min_corner_separation_m=120.0,
    speed_prominence_kmh=8.0,
    grid_step_m=10.0,
):
    """Find spatially separated, smoothed local speed minima as apex estimates."""
    for name, value in (
        ("min_corner_separation_m", min_corner_separation_m),
        ("speed_prominence_kmh", speed_prominence_kmh),
        ("grid_step_m", grid_step_m),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero.")
    if not {"Normalized_Dist", "Ground Speed"}.issubset(lap.columns):
        return []

    samples = pd.DataFrame(
        {
            "distance": pd.to_numeric(lap["Normalized_Dist"], errors="coerce"),
            "speed": pd.to_numeric(lap["Ground Speed"], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(samples) < 3:
        return []
    samples = (
        samples.groupby("distance", as_index=False)["speed"]
        .mean()
        .sort_values("distance")
    )
    distance = samples["distance"].to_numpy(dtype=float)
    if distance[-1] <= distance[0]:
        return []

    grid = np.arange(distance[0], distance[-1] + grid_step_m, grid_step_m)
    grid = grid[grid <= distance[-1]]
    if len(grid) < 3:
        return []
    speed = np.interp(grid, distance, samples["speed"].to_numpy(dtype=float))
    smooth_width = max(3, int(round(50.0 / grid_step_m)))
    if smooth_width % 2 == 0:
        smooth_width += 1
    smooth_speed = (
        pd.Series(speed)
        .rolling(smooth_width, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    peak_indices, _ = find_peaks(
        -smooth_speed,
        distance=max(1, int(math.ceil(min_corner_separation_m / grid_step_m))),
        prominence=speed_prominence_kmh,
    )
    if not len(peak_indices):
        return []

    apex_distances = grid[peak_indices]
    boundaries = [distance[0]]
    boundaries.extend(
        float((first + second) / 2)
        for first, second in zip(apex_distances, apex_distances[1:])
    )
    boundaries.append(distance[-1])
    return [
        {
            "id": f"T{index}",
            "name": f"Auto-detected corner {index}",
            "start_m": float(boundaries[index - 1]),
            "apex_m": float(grid[peak_index]),
            "end_m": float(boundaries[index]),
            "speed_kmh": float(smooth_speed[peak_index]),
        }
        for index, peak_index in enumerate(peak_indices, start=1)
    ]


def _lap_samples(frame, lap_number, sample_rate_hz):
    lap = frame[frame["Lap Number"] == lap_number].copy()
    lap["_distance"] = pd.to_numeric(lap["Normalized_Dist"], errors="coerce")
    lap = (
        lap.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["_distance"])
        .sort_values("_distance", kind="stable")
        .reset_index(drop=True)
    )
    if "Time" in lap:
        times = pd.to_numeric(lap["Time"], errors="coerce").to_numpy(dtype=float)
    else:
        times = np.arange(len(lap), dtype=float) / sample_rate_hz
    if len(times) > 1:
        valid_deltas = np.diff(times)
        valid_deltas = valid_deltas[np.isfinite(valid_deltas) & (valid_deltas > 0)]
        fallback_delta = (
            float(np.median(valid_deltas))
            if len(valid_deltas)
            else 1.0 / sample_rate_hz
        )
        times = np.where(np.isfinite(times), times, np.nan)
        times = pd.Series(times).interpolate(limit_direction="both").to_numpy()
        deltas = np.diff(times)
        deltas[~np.isfinite(deltas) | (deltas <= 0)] = fallback_delta
    else:
        deltas = np.array([], dtype=float)
    channels = {}
    for name in (
        "Brake Pos",
        "Throttle Pos",
        "Steering",
        "Ground Speed",
        "Gear",
    ):
        if name in lap:
            channels[name] = pd.to_numeric(
                lap[name], errors="coerce"
            ).to_numpy(dtype=float)
        else:
            channels[name] = np.full(len(lap), np.nan)
    channels["Time"] = times
    channels["Distance"] = lap["_distance"].to_numpy(dtype=float)
    channels["Delta Time"] = deltas
    return channels


def _event(distance, sample_index, **details):
    return {
        "distance_m": float(distance),
        "sample_index": int(sample_index),
        **details,
    }


def _double_braking_events(channels, corners):
    brake = channels["Brake Pos"]
    distance = channels["Distance"]
    events = []
    for corner in corners:
        apex = float(corner["apex_m"])
        in_window = np.flatnonzero(
            (distance >= apex - 150.0)
            & (distance <= apex)
            & np.isfinite(brake)
        )
        if len(in_window) < 3:
            continue
        for first_position, first_index in enumerate(in_window[:-2]):
            if brake[first_index] <= 20:
                continue
            lows = in_window[first_position + 1 :]
            lows = lows[brake[lows] < 10]
            if not len(lows):
                continue
            low_index = lows[0]
            second = in_window[in_window > low_index]
            second = second[brake[second] > 20]
            if len(second):
                index = int(second[0])
                events.append(
                    _event(
                        distance[index],
                        index,
                        apex_m=apex,
                        corner=str(corner.get("name", corner.get("id", ""))),
                    )
                )
                break
    return events


def _throttle_hesitation_events(channels):
    throttle = channels["Throttle Pos"]
    brake = channels["Brake Pos"]
    distance = channels["Distance"]
    events = []
    index = 0
    while index < len(throttle) - 1:
        if not (
            np.isfinite(throttle[index])
            and np.isfinite(brake[index])
            and throttle[index] > 70
            and brake[index] == 0
        ):
            index += 1
            continue
        start_throttle = throttle[index]
        cursor = index + 1
        trough = start_throttle
        trough_index = index
        while cursor < len(throttle):
            if not np.isfinite(brake[cursor]) or brake[cursor] != 0:
                break
            if np.isfinite(throttle[cursor]) and throttle[cursor] > 70:
                if start_throttle - trough > 20:
                    events.append(
                        _event(
                            distance[cursor],
                            cursor,
                            dip_percent=float(start_throttle - trough),
                            trough_distance_m=float(distance[trough_index]),
                        )
                    )
                index = cursor
                break
            if np.isfinite(throttle[cursor]) and throttle[cursor] < trough:
                trough = throttle[cursor]
                trough_index = cursor
            cursor += 1
        else:
            index = len(throttle)
            continue
        if cursor >= len(throttle):
            index = len(throttle)
        else:
            index += 1
    return events


def analyze_driver_inputs(
    frame,
    lap_number,
    track_config=None,
    sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ,
    cluster_distance_m=EVENT_CLUSTER_DISTANCE_M,
):
    """Calculate driver-input mistake indicators and cluster their locations."""
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and greater than zero.")
    if not math.isfinite(cluster_distance_m) or cluster_distance_m < 0:
        raise ValueError("cluster_distance_m must be finite and non-negative.")
    channels = _lap_samples(frame, lap_number, sample_rate_hz)
    distance = channels["Distance"]
    brake = channels["Brake Pos"]
    throttle = channels["Throttle Pos"]
    steering = channels["Steering"]
    speed = channels["Ground Speed"]
    gear = channels["Gear"]
    count = len(distance)

    corners = (
        track_config.get("corner_metadata", [])
        if track_config and track_config.get("corner_metadata")
        else auto_detect_apexes(
            pd.DataFrame(
                {
                    "Normalized_Dist": distance,
                    "Ground Speed": speed,
                }
            )
        )
    )
    events_by_detector = {
        "double_braking_m": _double_braking_events(channels, corners),
        "abrupt_brake_release_m": [],
        "loaded_throttle_snaps_m": [],
        "throttle_hesitations_m": _throttle_hesitation_events(channels),
        "loaded_downshifts_m": [],
    }

    if count > 1:
        delta_time = channels["Delta Time"]
        brake_slope = np.diff(brake) / delta_time
        throttle_slope = np.diff(throttle) / delta_time
        speed_at_change = speed[1:]
        steering_at_change = steering[1:]
        for offset in np.flatnonzero(
            (brake_slope < -600)
            & (speed_at_change > 60)
            & np.isfinite(brake_slope)
        ):
            index = int(offset + 1)
            events_by_detector["abrupt_brake_release_m"].append(
                _event(distance[index], index, slope_percent_s=float(brake_slope[offset]))
            )
        for offset in np.flatnonzero(
            (throttle_slope > 800)
            & (np.abs(steering_at_change) > 20)
            & np.isfinite(throttle_slope)
        ):
            index = int(offset + 1)
            events_by_detector["loaded_throttle_snaps_m"].append(
                _event(distance[index], index, slope_percent_s=float(throttle_slope[offset]))
            )
        downshifts = (
            (gear[1:] < gear[:-1])
            & (np.abs(steering_at_change) > 20)
            & np.isfinite(gear[1:])
            & np.isfinite(gear[:-1])
        )
        for offset in np.flatnonzero(downshifts):
            index = int(offset + 1)
            events_by_detector["loaded_downshifts_m"].append(
                _event(
                    distance[index],
                    index,
                    from_gear=float(gear[index - 1]),
                    to_gear=float(gear[index]),
                )
            )

    if count > 1 and all(
        np.isfinite(channels[name]).any()
        for name in ("Brake Pos", "Throttle Pos", "Ground Speed")
    ):
        coasting = (
            (brake[:-1] < 3)
            & (throttle[:-1] < 5)
            & (speed[:-1] > 50)
            & np.isfinite(brake[:-1])
            & np.isfinite(throttle[:-1])
            & np.isfinite(speed[:-1])
        )
        dead_coasting_seconds = float(
            np.sum(channels["Delta Time"][coasting])
        )
    else:
        dead_coasting_seconds = None

    braking_zone = np.isfinite(brake) & np.isfinite(steering) & (brake >= 5)
    if braking_zone.any() and np.isfinite(steering[braking_zone]).any():
        overlap = (
            braking_zone
            & (brake <= 40)
            & (np.abs(steering) > 15)
        )
        trail_brake_overlap_pct = float(overlap.sum() / braking_zone.sum() * 100)
    else:
        trail_brake_overlap_pct = None

    required_channels = {
        "double_braking_m": ("Brake Pos", "Ground Speed"),
        "abrupt_brake_release_m": ("Brake Pos", "Ground Speed"),
        "loaded_throttle_snaps_m": ("Throttle Pos", "Steering"),
        "throttle_hesitations_m": ("Throttle Pos", "Brake Pos"),
        "loaded_downshifts_m": ("Gear", "Steering"),
    }
    available = {
        detector: all(
            np.isfinite(channels[channel]).any()
            for channel in required
        )
        for detector, required in required_channels.items()
    }
    flattened_events = []
    for detector, events in events_by_detector.items():
        if not available[detector]:
            continue
        for event in events:
            flattened_events.append(
                {
                    "detector": detector,
                    **event,
                    "corner": map_distance_to_corner(
                        event["distance_m"], track_config
                    )
                    or map_distance_to_corner(
                        event["distance_m"],
                        {"corner_metadata": corners},
                    ),
                }
            )

    clusters = cluster_distances(
        (event["distance_m"] for event in flattened_events),
        max_gap_m=cluster_distance_m,
    )
    for cluster in clusters:
        grouped = [
            event
            for event in flattened_events
            if cluster["start_m"] <= event["distance_m"] <= cluster["end_m"]
        ]
        cluster["detectors"] = sorted({event["detector"] for event in grouped})
        cluster["events"] = grouped
        cluster["corner"] = (
            map_distance_to_corner(cluster["distance_m"], track_config)
            or map_distance_to_corner(
                cluster["distance_m"],
                {"corner_metadata": corners},
            )
        )

    return {
        "lap": float(lap_number),
        "sample_rate_hz": (
            float(1.0 / np.median(channels["Delta Time"]))
            if len(channels["Delta Time"])
            else float(sample_rate_hz)
        ),
        "metrics": {
            detector: (
                [event["distance_m"] for event in events]
                if available[detector]
                else None
            )
            for detector, events in events_by_detector.items()
        }
        | {
            "trail_brake_overlap_pct": trail_brake_overlap_pct,
            "dead_coasting_sec": dead_coasting_seconds,
        },
        "events": flattened_events,
        "clusters": clusters,
        "auto_detected_corners": corners,
    }
