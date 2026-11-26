import re

import numpy as np
import pandas as pd

from src.core.signal_rates import native_sample_rate_hz
from src.pipeline.telemetry_io import lap_time_seconds, valid_lap_numbers


PIT_SPEED_MIN_KMH = 60.0
PIT_SPEED_MAX_KMH = 80.0
PIT_SPEED_RANGE_KMH = 2.5
PIT_LIMITER_MINIMUM_SECONDS = 3.0


def _numeric(lap, column):
    if column not in lap:
        return np.full(len(lap), np.nan)
    return pd.to_numeric(lap[column], errors="coerce").to_numpy(dtype=float)


def _elapsed_time(lap):
    if "Time" in lap:
        elapsed = _numeric(lap, "Time")
        finite = elapsed[np.isfinite(elapsed)]
        if len(finite) > 1 and np.all(np.diff(finite) >= 0):
            return elapsed - finite[0]
    rate = native_sample_rate_hz(lap, "Ground Speed") or 50.0
    return np.arange(len(lap), dtype=float) / rate


def _is_pit_limiter_lap(lap):
    speed = _numeric(lap, "Ground Speed")
    throttle = _numeric(lap, "Throttle Pos")
    elapsed = _elapsed_time(lap)
    speed_band = (
        np.isfinite(speed)
        & (speed >= PIT_SPEED_MIN_KMH)
        & (speed <= PIT_SPEED_MAX_KMH)
    )
    transitions = np.diff(np.r_[False, speed_band, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    ignition_columns = [
        column
        for column in lap.columns
        if "ignition" in re.sub(r"[^a-z]", "", str(column).casefold())
        and "cut" in re.sub(r"[^a-z]", "", str(column).casefold())
    ]

    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        segment_time = elapsed[end - 1] - elapsed[start]
        segment_speed = speed[start:end]
        if (
            segment_time < PIT_LIMITER_MINIMUM_SECONDS
            or np.ptp(segment_speed) > PIT_SPEED_RANGE_KMH
        ):
            continue
        throttle_cuts = 0
        if np.isfinite(throttle[start:end]).any():
            throttle_delta = np.diff(throttle[start:end])
            sample_delta = np.diff(elapsed[start:end])
            throttle_cuts = int(
                np.sum(
                    (throttle_delta <= -15)
                    & (sample_delta <= 0.25)
                    & np.isfinite(throttle_delta)
                )
            )
        ignition_cuts = 0
        for column in ignition_columns:
            ignition = _numeric(lap, column)[start:end]
            ignition_cuts += int(np.sum(np.diff(ignition) > 0))
        if throttle_cuts >= 2 or ignition_cuts >= 2:
            return True
    return False


def _has_incident(lap):
    invalidated = _numeric(lap, "Lap Invalidated")
    if np.any(np.isfinite(invalidated) & (invalidated != 0)):
        return True
    if "Lap Invalidated" in lap:
        values = lap["Lap Invalidated"].astype(str).str.strip().str.casefold()
        if values.isin({"true", "yes", "invalid", "invalidated"}).any():
            return True

    yaw_columns = [
        column
        for column in lap.columns
        if any(
            token in re.sub(r"[^a-z]", "", str(column).casefold())
            for token in ("attitudevelocity", "chassisyawrate", "yawrate")
        )
    ]
    for column in yaw_columns:
        yaw = _numeric(lap, column)
        units = lap.attrs.get("quality", {}).get("channel_units", {})
        unit = str(units.get(column, "")).casefold().replace(" ", "")
        limit = 450.0
        if unit in {"rad/s", "radians/s", "rads-1"}:
            limit = np.deg2rad(450.0)
        if np.any(np.isfinite(yaw) & (np.abs(yaw) > limit)):
            return True

    speed = _numeric(lap, "Ground Speed")
    steering = _numeric(lap, "Steering")
    lateral = _numeric(lap, "G Force Lat")
    off_track = _numeric(lap, "Num Tires Off Track")
    mid_corner = (
        np.isfinite(speed)
        & (speed < 35.0)
        & np.isfinite(off_track)
        & (off_track >= 2)
        & (
            (np.isfinite(steering) & (np.abs(steering) > 10.0))
            | (np.isfinite(lateral) & (np.abs(lateral) > 0.5))
        )
    )
    return bool(mid_corner.any())


def classify_session_laps(frame):
    """Tag non-representative complete laps without removing them from the session."""
    cached = frame.attrs.get("quality", {}).get("race_state")
    if isinstance(cached, dict) and isinstance(cached.get("lap_tags"), dict):
        return cached
    valid_laps = valid_lap_numbers(frame)
    durations = {
        float(lap_number): lap_time_seconds(
            frame[frame["Lap Number"] == lap_number]
        )
        for lap_number in valid_laps
    }
    finite_durations = [
        duration
        for duration in durations.values()
        if np.isfinite(duration) and duration > 0
    ]
    median_duration = (
        float(np.median(finite_durations)) if finite_durations else None
    )
    first_lap = valid_laps[0] if valid_laps else None
    classifications = {}

    for lap_number in valid_laps:
        lap = frame[frame["Lap Number"] == lap_number]
        speed = _numeric(lap, "Ground Speed")
        start_speed = speed[0] if len(speed) and np.isfinite(speed[0]) else None
        launch_lap = bool(
            lap_number == first_lap
            and start_speed is not None
            and start_speed <= 20.0
        )
        duration = durations[float(lap_number)]
        slow_lap = bool(
            median_duration is not None
            and np.isfinite(duration)
            and duration > median_duration * 1.15
        )
        pit_limiter = _is_pit_limiter_lap(lap)
        incident = _has_incident(lap)
        reasons = []
        if launch_lap:
            reasons.append("launch_lap")
        if slow_lap:
            reasons.append("lap_time_above_115_percent_of_stint_median")
        if pit_limiter:
            reasons.append("pit_limiter_speed_band_with_throttle_or_ignition_cuts")
        if incident:
            reasons.append("incident_spin_or_off_track")
        exclude = bool(launch_lap or slow_lap or pit_limiter or incident)
        classifications[float(lap_number)] = {
            "duration_seconds": float(duration) if np.isfinite(duration) else None,
            "launch_lap": launch_lap,
            "slow_lap": slow_lap,
            "pit_limiter": pit_limiter,
            "incident": incident,
            "excluded_from_baselines": exclude,
            "excluded_from_degradation": exclude,
            "excluded_from_driver_inputs": exclude,
            "reasons": reasons,
        }

    return {
        "median_complete_lap_seconds": median_duration,
        "lap_tags": classifications,
    }


def lap_is_excluded(frame, lap_number):
    """Return whether a complete lap should be omitted from baseline analysis."""
    tags = classify_session_laps(frame)["lap_tags"]
    classification = tags.get(float(lap_number))
    return bool(
        classification
        and classification["excluded_from_baselines"]
    )


def lap_environment_summary(frame):
    """Return per-lap fuel buckets and available track-condition summaries."""
    cached = frame.attrs.get("quality", {}).get("environment_by_lap")
    if isinstance(cached, dict):
        return cached
    result = {}
    for lap_number in valid_lap_numbers(frame):
        lap = frame[frame["Lap Number"] == lap_number]
        row = {}
        if "Fuel Level" in lap:
            fuel = pd.to_numeric(lap["Fuel Level"], errors="coerce")
            fuel = fuel.replace([np.inf, -np.inf], np.nan).dropna()
            if len(fuel):
                fuel_mean = float(fuel.mean())
                row["fuel_level_mean_l"] = fuel_mean
                row["fuel_bucket"] = (
                    "low"
                    if fuel_mean < 15
                    else "medium"
                    if fuel_mean <= 35
                    else "high"
                )
        for column, key in (
            ("Surface Grip", "surface_grip_mean_pct"),
            ("Road Temp", "road_temperature_mean_c"),
        ):
            if column not in lap:
                continue
            values = pd.to_numeric(lap[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                row[key] = float(values.mean())
        result[float(lap_number)] = row
    return result
