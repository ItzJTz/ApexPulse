"""Driver-input event detection and spatial corner mapping."""

import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.config import config_value

from src.core.signal_rates import (
    native_sample_rate_hz,
    smoothed_derivative,
)
from src.core.session_filters import (
    classify_session_laps,
    lap_environment_summary,
    lap_is_excluded,
)
from src.core.sector_analysis import (
    _sector_time,
    scale_track_config_to_distance,
)
from src.pipeline.telemetry_io import lap_time_seconds, valid_lap_numbers


DEFAULT_SAMPLE_RATE_HZ = 50.0
EVENT_CLUSTER_DISTANCE_M = 40.0


def driver_input_profile():
    """Return validated device-specific pedal and steering thresholds."""
    profile_name = config_value("driver_inputs", "active_profile")
    profiles = config_value("driver_inputs", "profiles")
    if not isinstance(profile_name, str) or not isinstance(profiles, dict):
        raise ValueError(
            "driver_inputs.active_profile and driver_inputs.profiles "
            "must be a name and a mapping."
        )
    if profile_name not in profiles:
        raise ValueError(
            f"driver_inputs.active_profile '{profile_name}' is not configured."
        )
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(
            f"driver_inputs.profiles.{profile_name} must be a mapping."
        )
    required = (
        "brake_release_slope_pct_s",
        "brake_release_min_speed_kmh",
        "throttle_snap_slope_pct_s",
        "loaded_steering_angle_deg",
        "trail_brake_steering_angle_deg",
    )
    values = {"name": profile_name}
    for key in required:
        try:
            value = float(profile[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"driver_inputs.profiles.{profile_name}.{key} "
                "must be a finite number."
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"driver_inputs.profiles.{profile_name}.{key} "
                "must be finite and non-negative."
            )
        values[key] = value
    for key in (
        "brake_release_slope_pct_s",
        "throttle_snap_slope_pct_s",
        "loaded_steering_angle_deg",
        "trail_brake_steering_angle_deg",
    ):
        if values[key] == 0:
            raise ValueError(
                f"driver_inputs.profiles.{profile_name}.{key} must be positive."
            )
    return values


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


def cluster_events(events, max_gap_m=EVENT_CLUSTER_DISTANCE_M):
    """Collapse repeat trigger samples within each detector into corner events."""
    if not math.isfinite(max_gap_m) or max_gap_m < 0:
        raise ValueError("max_gap_m must be finite and non-negative.")
    ordered = sorted(events, key=lambda event: float(event["distance_m"]))
    groups = []
    current = []
    for event in ordered:
        if current and (
            float(event["distance_m"]) - float(current[-1]["distance_m"])
            > max_gap_m
        ):
            groups.append(current)
            current = []
        current.append(event)
    if current:
        groups.append(current)

    clustered = []
    for samples in groups:
        cluster = _distance_cluster(
            [float(event["distance_m"]) for event in samples]
        )
        representative = min(
            samples,
            key=lambda event: abs(event["distance_m"] - cluster["distance_m"]),
        )
        clustered.append(
            {
                **representative,
                "distance_m": cluster["distance_m"],
                "start_m": cluster["start_m"],
                "end_m": cluster["end_m"],
                "sample_count": cluster["sample_count"],
                "trigger_distances_m": [
                    float(event["distance_m"]) for event in samples
                ],
            }
        )
    return clustered


def _corner_label(corner, is_sepang):
    corner_id = str(corner.get("id", ""))
    name = str(corner.get("name", corner_id))
    if is_sepang and corner_id == "T1":
        name = "Pangkor Laut"
    elif is_sepang and corner_id == "T15":
        name = "Petronas Hairpin"
    return f"{corner_id} {name}".strip()


def map_distance_to_corner(distance_m, track_config=None):
    """Map an in-corner distance to its configured label without extrapolation."""
    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        return None
    corners = track_config.get("corner_metadata", []) if track_config else []
    corners = [
        corner
        for corner in corners
        if isinstance(corner.get("start_m"), (int, float))
        and isinstance(corner.get("end_m"), (int, float))
    ]
    if not math.isfinite(distance) or not corners:
        return None
    circuit_name = str(track_config.get("circuit_name", "")).casefold()
    is_sepang = "sepang" in circuit_name or str(
        track_config.get("ac_track_id", "")
    ).casefold() == "sepang"
    containing = [
        corner
        for corner in corners
        if float(corner.get("start_m", -math.inf))
        <= distance
        <= float(corner.get("end_m", math.inf))
    ]
    candidates = containing
    if not candidates:
        return None
    corner = min(
        candidates,
        key=lambda item: abs(
            float(
                item["apex_m"]
                if isinstance(item.get("apex_m"), (int, float))
                else (item["start_m"] + item["end_m"]) / 2
            )
            - distance
        ),
    )
    return _corner_label(corner, is_sepang)


def _map_to_detected_corner(distance_m, corners):
    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        return None
    if not corners or not math.isfinite(distance):
        return None
    corner = min(
        corners,
        key=lambda item: abs(float(item["apex_m"]) - distance),
    )
    return f"{corner.get('id', '')} {corner.get('name', '')}".strip()


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
    if "Lap Number" in lap and lap_is_excluded(lap, lap["Lap Number"].iloc[0]):
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
    time_is_ordered = False
    if "Time" in lap:
        times = pd.to_numeric(lap["Time"], errors="coerce").to_numpy(dtype=float)
        finite_times = pd.Series(times).dropna()
        time_is_ordered = finite_times.is_monotonic_increasing
        if time_is_ordered:
            lap["_time"] = times
            order_columns = ["_time"]
        else:
            times = np.arange(len(lap), dtype=float) / sample_rate_hz
            order_columns = []
    else:
        times = np.arange(len(lap), dtype=float) / sample_rate_hz
        order_columns = ["_distance"]
    lap = lap.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["_distance"]
    )
    if order_columns:
        lap = lap.sort_values(order_columns, kind="stable")
    lap = lap.reset_index(drop=True)
    if "Time" in lap and time_is_ordered:
        times = pd.to_numeric(lap["Time"], errors="coerce").to_numpy(dtype=float)
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
            ).to_numpy(dtype=float, copy=True)
        else:
            channels[name] = np.full(len(lap), np.nan)
    if "Race Flag" in lap:
        channels["Race Flag"] = lap["Race Flag"].astype(str).to_numpy()
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
        if not isinstance(corner.get("apex_m"), (int, float)):
            continue
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


def _distance_runs(mask, distance):
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end > start and np.isfinite(distance[start : end].astype(float)).any()
    ]


def _neutralized_intervals(frame, lap_number, channels, track_config):
    distance = channels["Distance"]
    speed = channels["Ground Speed"]
    steering = channels["Steering"]
    intervals = []
    tags = classify_session_laps(frame)["lap_tags"]
    comparison_laps = [
        lap
        for lap in valid_lap_numbers(frame)
        if lap != lap_number
        and not tags.get(float(lap), {}).get("excluded_from_baselines", False)
    ]
    straight_slow = (
        np.isfinite(speed)
        & (speed < 120.0)
        & np.isfinite(steering)
        & (np.abs(steering) < 10.0)
    )
    for start, end in _distance_runs(straight_slow, distance):
        run_distance = distance[start:end]
        start_m = float(np.nanmin(run_distance)) if len(run_distance) else 0.0
        end_m = float(np.nanmax(run_distance)) if len(run_distance) else 0.0
        reference_speed = []
        for other_lap in comparison_laps:
            other = frame[
                (frame["Lap Number"] == other_lap)
                & frame["Normalized_Dist"].between(start_m, end_m)
            ]
            if "Steering" in other:
                other_steering = pd.to_numeric(
                    other["Steering"],
                    errors="coerce",
                )
                other = other.loc[other_steering.abs() < 10.0]
            other_speed = pd.to_numeric(
                other.get("Ground Speed", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            if len(other_speed):
                reference_speed.append(float(other_speed.median()))
        target_median_speed = (
            float(np.nanmedian(speed[start:end])) if end > start else 0.0
        )
        reference_median_speed = (
            float(np.median(reference_speed)) if reference_speed else None
        )
        if (
            end_m - start_m > 300
            and reference_median_speed is not None
            and (
                reference_median_speed >= 120.0
                or target_median_speed < 0.9 * reference_median_speed
            )
        ):
            intervals.append(
                {
                    "start_m": start_m,
                    "end_m": end_m,
                    "reason": "slow speed on a straight for over 300 m",
                }
            )

    if "Race Flag" in channels:
        lap_flags = pd.Series(channels["Race Flag"]).astype(str).str.casefold()
        flag_mask = lap_flags.str.contains(
            r"yellow|caution|safety|slow",
            regex=True,
            na=False,
        ).to_numpy()
        for start, end in _distance_runs(flag_mask, distance):
            intervals.append(
                {
                    "start_m": float(np.nanmin(distance[start:end])),
                    "end_m": float(np.nanmax(distance[start:end])),
                    "reason": "race flag indicates caution or a slow zone",
                }
            )

    sectors = {}
    if track_config:
        lap_distance = float(np.nanmax(distance)) if len(distance) else 0.0
        scaled_config = scale_track_config_to_distance(
            track_config,
            lap_distance,
        )
        sectors = scaled_config.get("official_sectors") or scaled_config.get(
            "sectors",
            {},
        )
    if sectors:
        target_lap = frame[frame["Lap Number"] == lap_number]
        for bounds in sectors.values():
            if len(bounds) != 2:
                continue
            start_m, end_m = (float(bound) for bound in bounds)
            target_time = _sector_time(target_lap, start_m, end_m)
            reference_times = [
                _sector_time(
                    frame[frame["Lap Number"] == other_lap],
                    start_m,
                    end_m,
                )
                for other_lap in comparison_laps
            ]
            reference_times = [
                value
                for value in reference_times
                if value is not None and math.isfinite(value) and value > 0
            ]
            if (
                target_time is not None
                and math.isfinite(target_time)
                and reference_times
                and target_time > 1.1 * float(np.median(reference_times))
            ):
                intervals.append(
                    {
                        "start_m": start_m,
                        "end_m": end_m,
                        "reason": "sector elapsed time exceeds reference by over 10%",
                    }
                )

    return sorted(intervals, key=lambda interval: interval["start_m"])


def _traffic_lift_events(channels, corners, reference_channels):
    if reference_channels is None:
        return []
    events = []
    for corner in corners:
        if not isinstance(corner.get("apex_m"), (int, float)):
            continue
        apex = float(corner["apex_m"])
        target_mask = (
            (channels["Distance"] >= apex - 150)
            & (channels["Distance"] <= apex)
            & np.isfinite(channels["Brake Pos"])
        )
        reference_mask = (
            (reference_channels["Distance"] >= apex - 150)
            & (reference_channels["Distance"] <= apex)
            & np.isfinite(reference_channels["Brake Pos"])
        )
        target_indices = np.flatnonzero(target_mask)
        reference_brake = reference_channels["Brake Pos"][reference_mask]
        if not len(target_indices) or not len(reference_brake):
            continue
        target_index = int(
            target_indices[np.argmax(channels["Brake Pos"][target_indices])]
        )
        peak_brake = float(channels["Brake Pos"][target_index])
        reference_peak = float(np.max(reference_brake))
        if (
            20 <= peak_brake <= 45
            and reference_peak > 0
            and peak_brake < 0.6 * reference_peak
        ):
            events.append(
                _event(
                    channels["Distance"][target_index],
                    target_index,
                    corner=str(corner.get("name", corner.get("id", ""))),
                    classification="Traffic / Lift",
                    mistake=False,
                    peak_brake_pct=peak_brake,
                    reference_peak_brake_pct=reference_peak,
                )
            )
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
    input_profile = driver_input_profile()
    lap_tags = classify_session_laps(frame)["lap_tags"]
    environment_by_lap = lap_environment_summary(frame)
    lap_tag = lap_tags.get(float(lap_number), {})
    channels = _lap_samples(frame, lap_number, sample_rate_hz)
    detector_names = (
        "double_braking_m",
        "abrupt_brake_release_m",
        "trail_brake_overlap_pct",
        "loaded_throttle_snaps_m",
        "throttle_hesitations_m",
        "dead_coasting_sec",
        "loaded_downshifts_m",
        "traffic_lifts_m",
    )
    if lap_tag.get("excluded_from_driver_inputs"):
        reason = ", ".join(lap_tag["reasons"])
        return {
            "lap": float(lap_number),
            "input_profile": input_profile["name"],
            "sample_rate_hz": float(sample_rate_hz),
            "metrics": {name: None for name in detector_names},
            "events": [],
            "raw_events": [],
            "clusters": [],
            "events_by_detector": {
                name: [] for name in detector_names if name.endswith("_m")
            },
            "available": {
                name: False for name in detector_names if name.endswith("_m")
            },
            "auto_detected_corners": [],
            "excluded_reason": reason,
            "neutralized_zones": [],
            "reference_lap": None,
        }
    distance = channels["Distance"]
    brake = channels["Brake Pos"]
    throttle = channels["Throttle Pos"]
    steering = channels["Steering"]
    speed = channels["Ground Speed"]
    gear = channels["Gear"]
    count = len(distance)

    corners = auto_detect_apexes(
        pd.DataFrame(
            {
                "Normalized_Dist": distance,
                "Ground Speed": speed,
            }
        )
    )
    scaled_track_config = scale_track_config_to_distance(
        track_config,
        float(np.nanmax(distance)) if len(distance) else 0.0,
    )
    if not corners and track_config:
        corners = (
            scaled_track_config.get("corner_metadata", [])
            if scaled_track_config
            else []
        )
    neutralized = _neutralized_intervals(
        frame,
        lap_number,
        channels,
        scaled_track_config,
    )
    neutralized_mask = np.zeros(count, dtype=bool)
    for interval in neutralized:
        neutralized_mask |= (
            (distance >= interval["start_m"])
            & (distance <= interval["end_m"])
        )
    for name in (
        "Brake Pos",
        "Throttle Pos",
        "Steering",
        "Ground Speed",
        "Gear",
    ):
        channels[name][neutralized_mask] = np.nan

    reference_candidates = [
        candidate
        for candidate in valid_lap_numbers(frame)
        if candidate != lap_number
        and not lap_tags.get(float(candidate), {}).get(
            "excluded_from_baselines",
            False,
        )
        and (
            environment_by_lap.get(float(lap_number), {}).get("fuel_bucket")
            is None
            or environment_by_lap.get(float(candidate), {}).get("fuel_bucket")
            == environment_by_lap[float(lap_number)]["fuel_bucket"]
        )
    ]
    reference_lap = None
    if reference_candidates:
        reference_lap = min(
            reference_candidates,
            key=lambda candidate: lap_time_seconds(
                frame[frame["Lap Number"] == candidate]
            ),
        )
    reference_channels = (
        _lap_samples(frame, reference_lap, sample_rate_hz)
        if reference_lap is not None
        else None
    )
    if reference_channels is not None:
        reference_zones = _neutralized_intervals(
            frame,
            reference_lap,
            reference_channels,
            scaled_track_config,
        )
        reference_mask = np.zeros(len(reference_channels["Distance"]), dtype=bool)
        for interval in reference_zones:
            reference_mask |= (
                (reference_channels["Distance"] >= interval["start_m"])
                & (reference_channels["Distance"] <= interval["end_m"])
            )
        reference_channels["Brake Pos"][reference_mask] = np.nan

    events_by_detector = {
        "double_braking_m": _double_braking_events(channels, corners),
        "abrupt_brake_release_m": [],
        "loaded_throttle_snaps_m": [],
        "throttle_hesitations_m": _throttle_hesitation_events(channels),
        "loaded_downshifts_m": [],
        "traffic_lifts_m": _traffic_lift_events(
            channels,
            corners,
            reference_channels,
        ),
    }
    traffic_corners = {
        event["corner"] for event in events_by_detector["traffic_lifts_m"]
    }
    if traffic_corners:
        events_by_detector["double_braking_m"] = [
            event
            for event in events_by_detector["double_braking_m"]
            if event["corner"] not in traffic_corners
        ]

    if count > 1:
        delta_time = channels["Delta Time"]
        brake_slope = smoothed_derivative(
            brake,
            delta_time,
            native_sample_rate_hz(frame, "Brake Pos"),
        )
        throttle_slope = np.diff(throttle) / delta_time
        downshift_indices = np.flatnonzero(
            (gear[1:] < gear[:-1])
            & np.isfinite(gear[1:])
            & np.isfinite(gear[:-1])
        ) + 1
        downshift_times = channels["Time"][downshift_indices]
        for slope_index in np.flatnonzero(
            (brake_slope < -input_profile["brake_release_slope_pct_s"])
            & np.isfinite(brake_slope)
        ):
            index = int(slope_index)
            if (
                not np.isfinite(speed[index])
                or speed[index] <= input_profile["brake_release_min_speed_kmh"]
            ):
                continue
            events_by_detector["abrupt_brake_release_m"].append(
                _event(
                    distance[index],
                    index,
                    slope_percent_s=float(brake_slope[slope_index]),
                )
            )
        steering_at_change = steering[1:]
        for offset in np.flatnonzero(
            (throttle_slope > input_profile["throttle_snap_slope_pct_s"])
            & (
                np.abs(steering_at_change)
                > input_profile["loaded_steering_angle_deg"]
            )
            & np.isfinite(throttle_slope)
        ):
            index = int(offset + 1)
            if len(downshift_times):
                previous = np.searchsorted(
                    downshift_times,
                    channels["Time"][index],
                    side="right",
                ) - 1
                if (
                    previous >= 0
                    and channels["Time"][index] - downshift_times[previous]
                    <= 0.2 + 1e-9
                ):
                    continue
            events_by_detector["loaded_throttle_snaps_m"].append(
                _event(distance[index], index, slope_percent_s=float(throttle_slope[offset]))
            )
        downshifts = (
            (gear[1:] < gear[:-1])
            & (
                np.abs(steering_at_change)
                > input_profile["loaded_steering_angle_deg"]
            )
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
            & (
                np.abs(steering)
                > input_profile["trail_brake_steering_angle_deg"]
            )
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
        "traffic_lifts_m": ("Brake Pos",),
    }
    available = {
        detector: all(
            np.isfinite(channels[channel]).any()
            for channel in required
        )
        for detector, required in required_channels.items()
    }
    available["traffic_lifts_m"] &= reference_channels is not None
    raw_events = []
    for detector, events in events_by_detector.items():
        if not available[detector]:
            continue
        for event in events:
            mapped_corner = map_distance_to_corner(
                event["distance_m"],
                scaled_track_config,
            )
            raw_events.append(
                {
                    "detector": detector,
                    **event,
                    "corner": mapped_corner
                    or _map_to_detected_corner(event["distance_m"], corners),
                }
            )

    clustered_by_detector = {}
    for detector in events_by_detector:
        detector_events = [
            event for event in raw_events if event["detector"] == detector
        ]
        clustered_by_detector[detector] = cluster_events(
            detector_events,
            max_gap_m=cluster_distance_m,
        )
    flattened_events = [
        event
        for detector_events in clustered_by_detector.values()
        for event in detector_events
    ]

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
        cluster["corner"] = _map_to_detected_corner(
            cluster["distance_m"], corners
        )

    return {
        "lap": float(lap_number),
        "input_profile": input_profile["name"],
        "sample_rate_hz": (
            float(1.0 / np.median(channels["Delta Time"]))
            if len(channels["Delta Time"])
            else float(sample_rate_hz)
        ),
        "metrics": {
            detector: (
                [
                    event["distance_m"]
                    for event in clustered_by_detector[detector]
                ]
                if available[detector]
                else None
            )
            for detector in events_by_detector
        }
        | {
            "trail_brake_overlap_pct": trail_brake_overlap_pct,
            "dead_coasting_sec": dead_coasting_seconds,
        },
        "events": flattened_events,
        "raw_events": raw_events,
        "clusters": clusters,
        "auto_detected_corners": corners,
        "neutralized_zones": neutralized,
        "excluded_reason": None,
        "reference_lap": (
            float(reference_lap) if reference_lap is not None else None
        ),
    }
