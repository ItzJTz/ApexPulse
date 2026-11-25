"""Optional vehicle-dynamics estimates from logged telemetry channels."""

import math
import re

import numpy as np
import pandas as pd

from src.config import config_value
from src.core.driver_inputs import auto_detect_apexes
from src.core.signal_rates import (
    PEDAL_SLOPE_SAMPLES,
    smoothed_derivative,
)
from src.pipeline.telemetry_io import lap_time_seconds, valid_lap_numbers


WHEELS = ("FL", "FR", "RL", "RR")


def _numeric(frame, column):
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _wheel_channel(frame, prefixes, wheel):
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).casefold()): str(column)
        for column in frame.columns
    }
    for prefix in prefixes:
        key = re.sub(r"[^a-z0-9]", "", f"{prefix}{wheel}".casefold())
        if key in normalized:
            return normalized[key]
    return None


def corner_radius_trace(lap):
    """Return the kinematic radius estimate v²/|a_lat| for each sample."""
    speed = _numeric(lap, "Ground Speed") / 3.6
    lateral_g = _numeric(lap, "G Force Lat")
    minimum_g = float(config_value("vehicle", "minimum_cornering_lateral_g"))
    if not math.isfinite(minimum_g) or minimum_g <= 0:
        raise ValueError("vehicle.minimum_cornering_lateral_g must be positive.")
    valid = (
        np.isfinite(speed)
        & (speed > 0)
        & np.isfinite(lateral_g)
        & (np.abs(lateral_g) >= minimum_g)
    )
    radius = np.full(len(lap), np.nan)
    radius[valid] = speed[valid] ** 2 / (
        np.abs(lateral_g[valid]) * 9.80665
    )
    return pd.DataFrame(
        {
            "distance_m": _numeric(lap, "Normalized_Dist"),
            "radius_m": radius,
        },
        index=lap.index,
    )


def wheel_slip_ratio_trace(lap):
    """Estimate per-wheel longitudinal slip against ground speed in percent."""
    ground_speed = _numeric(lap, "Ground Speed")
    minimum_speed = float(config_value("vehicle", "minimum_slip_speed_kmh"))
    if not math.isfinite(minimum_speed) or minimum_speed <= 0:
        raise ValueError("vehicle.minimum_slip_speed_kmh must be positive.")
    valid_speed = np.isfinite(ground_speed) & (ground_speed >= minimum_speed)
    result = pd.DataFrame({"distance_m": _numeric(lap, "Normalized_Dist")})
    unit_map = lap.attrs.get("quality", {}).get("channel_units", {})
    unavailable = {}

    for wheel in WHEELS:
        speed_column = _wheel_channel(
            lap,
            ("Wheel Rot Speed ", "Wheel Angular Speed ", "Wheel Speed "),
            wheel,
        )
        if speed_column is None:
            unavailable[wheel] = "Wheel rotational-speed channel is missing."
            continue
        wheel_speed = _numeric(lap, speed_column)
        unit = str(unit_map.get(speed_column, "")).casefold().replace(" ", "")
        is_angular_speed = (
            any(
                token in speed_column.casefold()
                for token in ("angular", "rot speed")
            )
            or any(token in unit for token in ("rad/s", "rpm", "deg/s", "degree/s"))
        )
        if is_angular_speed:
            radius_column = _wheel_channel(
                lap,
                (
                    "Tire Loaded Radius ",
                    "Tyre Loaded Radius ",
                    "Wheel Loaded Radius ",
                ),
                wheel,
            )
            if radius_column is None:
                unavailable[wheel] = (
                    "Angular wheel speed requires a measured loaded tire radius."
                )
                continue
            tire_radius = _numeric(lap, radius_column)
            radius_unit = str(unit_map.get(radius_column, "")).casefold()
            finite_radius = tire_radius[np.isfinite(tire_radius)]
            inferred_millimeters = (
                not radius_unit
                and len(finite_radius) > 0
                and float(np.median(finite_radius)) > 2.0
            )
            if "mm" in radius_unit or inferred_millimeters:
                tire_radius = tire_radius / 1000.0
            elif "cm" in radius_unit:
                tire_radius = tire_radius / 100.0
            elif radius_unit in ("in", "inch", "inches"):
                tire_radius = tire_radius * 0.0254
            valid_radius = np.isfinite(tire_radius) & (tire_radius > 0)
            if not valid_radius.any():
                unavailable[wheel] = (
                    "Loaded tire-radius channel has no positive finite samples."
                )
                continue
            if "rpm" in unit:
                wheel_speed = wheel_speed * (2 * np.pi / 60)
            elif "deg/s" in unit or "degree/s" in unit:
                wheel_speed = np.deg2rad(wheel_speed)
            tire_speed_kmh = wheel_speed * tire_radius * 3.6
            tire_speed_kmh[~valid_radius] = np.nan
        else:
            if "m/s" in unit:
                wheel_speed = wheel_speed * 3.6
            elif unit in ("mph", "mi/h"):
                wheel_speed = wheel_speed * 1.609344
            tire_speed_kmh = wheel_speed

        valid = (
            valid_speed
            & np.isfinite(tire_speed_kmh)
            & np.isfinite(ground_speed)
        )
        ratio = np.full(len(lap), np.nan)
        ratio[valid] = (
            (tire_speed_kmh[valid] - ground_speed[valid])
            / ground_speed[valid]
            * 100.0
        )
        result[f"slip_ratio_{wheel}_pct"] = ratio
        unavailable[wheel] = None
    result.attrs["unavailable"] = unavailable
    return result


def _axle_displacement(lap, axle):
    explicit = (
        ("Front Ride Height",)
        if axle == "front"
        else ("Rear Ride Height",)
    )
    for column in explicit:
        values = _numeric(lap, column)
        if np.isfinite(values).sum() >= 3 and np.nanstd(values) > 1e-6:
            unit = lap.attrs.get("quality", {}).get("channel_units", {}).get(
                column, ""
            )
            return values, column, unit

    wheels = ("FL", "FR") if axle == "front" else ("RL", "RR")
    patterns = ("Ride Height ", "Susp Pos ", "Suspension Travel ")
    for prefix in patterns:
        columns = [
            _wheel_channel(lap, (prefix,), wheel)
            for wheel in wheels
        ]
        columns = [column for column in columns if column is not None]
        if not columns:
            continue
        values = np.column_stack([_numeric(lap, column) for column in columns])
        finite = np.isfinite(values)
        counts = finite.sum(axis=1)
        axle_values = np.divide(
            np.nansum(values, axis=1),
            counts,
            out=np.full(len(values), np.nan),
            where=counts > 0,
        )
        if np.isfinite(axle_values).sum() >= 3 and np.nanstd(axle_values) > 1e-6:
            units = lap.attrs.get("quality", {}).get("channel_units", {})
            unit = units.get(columns[0], "") if columns else ""
            return axle_values, prefix.rstrip(), unit
    return None, None, None


def aero_pitch_trace(lap):
    """Compare front/rear suspension position against a low-speed baseline."""
    speed = _numeric(lap, "Ground Speed")
    threshold = float(config_value("vehicle", "high_speed_aero_threshold_kmh"))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("vehicle.high_speed_aero_threshold_kmh must be positive.")
    result = pd.DataFrame({"distance_m": _numeric(lap, "Normalized_Dist")})
    low_speed = np.isfinite(speed) & (speed <= 80)
    high_speed = np.isfinite(speed) & (speed >= threshold)
    available = {}
    for axle in ("front", "rear"):
        values, channel_source, unit = _axle_displacement(lap, axle)
        if values is None:
            available[axle] = None
            continue
        baseline_samples = values[low_speed & np.isfinite(values)]
        if not len(baseline_samples):
            baseline_samples = values[np.isfinite(values)]
        baseline = float(np.nanmedian(baseline_samples))
        delta = values - baseline
        result[f"{axle}_relative"] = delta
        high_speed_values = delta[high_speed & np.isfinite(delta)]
        available[axle] = {
            "source": channel_source,
            "unit": unit or None,
            "baseline": baseline,
            "high_speed_relative_median": (
                float(np.median(high_speed_values))
                if len(high_speed_values)
                else None
            ),
        }
    if "front_relative" in result and "rear_relative" in result:
        front_unit = available["front"]["unit"]
        rear_unit = available["rear"]["unit"]
        if front_unit == rear_unit:
            result["front_minus_rear"] = (
                result["front_relative"] - result["rear_relative"]
            )
    result.attrs["available_axles"] = available
    result.attrs["high_speed_threshold_kmh"] = threshold
    return result


def brake_input_analysis(lap, sample_rate_hz=50.0):
    """Summarize brake application ramps and pressure decay through turn-in."""
    brake = _numeric(lap, "Brake Pos")
    steering = _numeric(lap, "Steering")
    speed = _numeric(lap, "Ground Speed")
    distance = _numeric(lap, "Normalized_Dist")
    threshold = float(config_value("vehicle", "brake_input_threshold_pct"))
    abrupt_limit = float(
        config_value("vehicle", "abrupt_brake_release_pct_s")
    )
    abrupt_minimum_speed = float(
        config_value("vehicle", "abrupt_brake_release_min_speed_kmh")
    )
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("vehicle.brake_input_threshold_pct must be non-negative.")
    if not math.isfinite(abrupt_limit) or abrupt_limit <= 0:
        raise ValueError("vehicle.abrupt_brake_release_pct_s must be positive.")
    if not math.isfinite(abrupt_minimum_speed) or abrupt_minimum_speed < 0:
        raise ValueError(
            "vehicle.abrupt_brake_release_min_speed_kmh must be non-negative."
        )
    if "Time" in lap:
        time = _numeric(lap, "Time")
    else:
        time = np.arange(len(lap), dtype=float) / sample_rate_hz
    time = pd.Series(time).interpolate(limit_direction="both").to_numpy()
    delta_time = np.diff(time)
    valid_dt = np.isfinite(delta_time) & (delta_time > 0)
    fallback_dt = (
        float(np.median(delta_time[valid_dt]))
        if valid_dt.any()
        else 1.0 / sample_rate_hz
    )
    delta_time[~valid_dt] = fallback_dt
    brake_slope = smoothed_derivative(brake, delta_time)
    steering_magnitude = np.abs(steering)
    braking = np.isfinite(brake) & (brake >= threshold)
    padded = np.concatenate(([False], braking, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    segments = list(zip(changes[::2], changes[1::2]))
    episodes = []
    for start, stop in segments:
        if stop - start < 2:
            continue
        local_brake = brake[start:stop]
        center_indices = np.arange(start, stop)
        slope_indices = center_indices + PEDAL_SLOPE_SAMPLES // 2
        valid_slope_indices = slope_indices < len(brake_slope)
        center_indices = center_indices[valid_slope_indices]
        slope_indices = slope_indices[valid_slope_indices]
        local_slopes = brake_slope[slope_indices]
        local_slopes = local_slopes[np.isfinite(local_slopes)]
        release_mask = (
            np.isfinite(brake_slope[slope_indices])
            & (brake_slope[slope_indices] < 0)
        )
        release_slope_indices = slope_indices[release_mask]
        release_center_indices = center_indices[release_mask]
        high_speed_release_mask = (
            np.isfinite(speed[release_center_indices])
            & (speed[release_center_indices] > abrupt_minimum_speed)
        )
        high_speed_release_indices = release_slope_indices[high_speed_release_mask]

        turning_release_mask = (
            (release_center_indices > 0)
            & np.isfinite(steering_magnitude[release_center_indices])
            & np.isfinite(steering_magnitude[np.maximum(release_center_indices - 1, 0)])
            & (
                steering_magnitude[release_center_indices]
                > steering_magnitude[np.maximum(release_center_indices - 1, 0)]
            )
        )
        turning_release_indices = release_center_indices[turning_release_mask]
        if len(turning_release_indices) >= 2:
            brake_values = brake[turning_release_indices]
            steering_values = steering_magnitude[turning_release_indices]
            correlation = (
                float(np.corrcoef(brake_values, steering_values)[0, 1])
                if np.std(brake_values) > 0 and np.std(steering_values) > 0
                else None
            )
            decay_per_degree = (
                float(np.polyfit(steering_values, brake_values, 1)[0])
                if np.ptp(steering_values) > 0
                else None
            )
        else:
            correlation = None
            decay_per_degree = None

        peak_release = (
            float(np.min(brake_slope[release_slope_indices]))
            if len(release_slope_indices)
            else None
        )
        peak_high_speed_release = (
            float(np.min(brake_slope[high_speed_release_indices]))
            if len(high_speed_release_indices)
            else None
        )
        ramp_slopes = local_slopes[local_slopes > 0]
        max_ramp_up = (
            float(np.max(ramp_slopes))
            if len(ramp_slopes)
            else None
        )
        episodes.append(
            {
                "start_m": float(distance[start]),
                "end_m": float(distance[min(stop - 1, len(distance) - 1)]),
                "peak_brake_pct": float(np.nanmax(local_brake)),
                "max_ramp_up_pct_s": max(0.0, max_ramp_up)
                if max_ramp_up is not None
                else None,
                "peak_release_pct_s": peak_release,
                "peak_high_speed_release_pct_s": peak_high_speed_release,
                "turn_in_brake_steering_correlation": correlation,
                "brake_decay_pct_per_steering_degree": decay_per_degree,
                "classification": (
                    "abrupt release"
                    if peak_high_speed_release is not None
                    and peak_high_speed_release < -abrupt_limit
                    else "abrupt release below speed threshold"
                    if peak_release is not None and peak_release < -abrupt_limit
                    else "progressive release"
                    if peak_release is not None
                    else "insufficient samples"
                ),
            }
        )
    releases = [
        episode["peak_release_pct_s"]
        for episode in episodes
        if episode["peak_release_pct_s"] is not None
    ]
    ramps = [
        episode["max_ramp_up_pct_s"]
        for episode in episodes
        if episode["max_ramp_up_pct_s"] is not None
    ]
    return {
        "available": bool(np.isfinite(brake).any()),
        "braking_episodes": episodes,
        "maximum_ramp_up_pct_s": max(ramps) if ramps else None,
        "steepest_release_pct_s": min(releases) if releases else None,
        "abrupt_release_count": sum(
            episode["classification"] == "abrupt release"
            for episode in episodes
        ),
        "low_speed_abrupt_release_count": sum(
            episode["classification"] == "abrupt release below speed threshold"
            for episode in episodes
        ),
        "abrupt_release_minimum_speed_kmh": abrupt_minimum_speed,
        "threshold_pct": threshold,
    }


def _sensor_columns(frame, tokens):
    return [
        column
        for column in frame.columns
        if any(token in str(column).casefold() for token in tokens)
    ]


def _lap_sensor_mean(lap, columns):
    if not columns:
        return None
    values = lap[columns].apply(pd.to_numeric, errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan)
    return float(finite.mean(axis=1).mean()) if finite.notna().any().any() else None


def predictive_tire_degradation(
    frame,
    minimum_laps=None,
    lap_time_drop_threshold_s=None,
):
    """Fit an interpretable linear stint trend and forecast a pace-loss threshold."""
    minimum_laps = (
        int(config_value("tire_analysis", "degradation_minimum_laps"))
        if minimum_laps is None
        else int(minimum_laps)
    )
    drop_threshold = (
        float(config_value("tire_analysis", "degradation_lap_time_drop_s"))
        if lap_time_drop_threshold_s is None
        else float(lap_time_drop_threshold_s)
    )
    if minimum_laps < 3:
        raise ValueError("minimum_laps must be at least 3.")
    if not math.isfinite(drop_threshold) or drop_threshold <= 0:
        raise ValueError("lap_time_drop_threshold_s must be positive and finite.")

    times = []
    valid_laps = valid_lap_numbers(frame)
    for lap_number in valid_laps:
        lap = frame[frame["Lap Number"] == lap_number]
        duration = lap_time_seconds(lap)
        if math.isfinite(duration) and duration > 0:
            times.append((float(lap_number), duration, lap))
    result = {
        "available": False,
        "minimum_laps_required": minimum_laps,
        "lap_time_drop_threshold_s": drop_threshold,
        "lap_count": len(times),
        "fit_method": "least-squares linear trend against stint lap number",
        "caveat": (
            "A lap-time trend is a correlation-based pace forecast, not proof "
            "that tire degradation caused the change."
        ),
    }
    if len(times) < minimum_laps:
        result["unavailable"] = (
            f"At least {minimum_laps} complete laps are required; "
            f"this stint has {len(times)}."
        )
        return result

    lap_numbers = np.array([item[0] for item in times], dtype=float)
    lap_seconds = np.array([item[1] for item in times], dtype=float)
    baseline_count = min(3, len(times))
    baseline_time = float(np.median(lap_seconds[:baseline_count]))
    slope, intercept = np.polyfit(lap_numbers, lap_seconds, 1)
    fitted = slope * lap_numbers + intercept
    residual_sum = float(np.sum((lap_seconds - fitted) ** 2))
    total_sum = float(np.sum((lap_seconds - lap_seconds.mean()) ** 2))
    r_squared = (
        float(1 - residual_sum / total_sum)
        if total_sum > 0
        else 1.0
    )

    core_columns = _sensor_columns(
        frame,
        ("tire temp core", "tyre temp core"),
    )
    if not core_columns:
        core_columns = _sensor_columns(
            frame,
            ("tire carcass temp", "tyre carcass temp"),
        )
    pressure_columns = _sensor_columns(
        frame,
        ("tire pressure", "tyre pressure"),
    )
    temperature_by_lap = [
        _lap_sensor_mean(item[2], core_columns) for item in times
    ]
    pressure_by_lap = [
        _lap_sensor_mean(item[2], pressure_columns) for item in times
    ]

    result.update(
        {
            "available": True,
            "lap_time_slope_s_per_lap": float(slope),
            "lap_time_trend_r_squared": r_squared,
            "baseline_lap_time_s": baseline_time,
            "target_lap_time_s": baseline_time + drop_threshold,
            "lap_times_seconds": {
                str(lap): float(seconds)
                for lap, seconds in zip(lap_numbers, lap_seconds)
            },
            "fitted_lap_times_seconds": {
                str(lap): float(value)
                for lap, value in zip(lap_numbers, fitted)
            },
            "tire_core_temperature_mean_by_lap": {
                str(lap): value
                for (lap, _, _), value in zip(times, temperature_by_lap)
                if value is not None
            },
            "tire_pressure_mean_by_lap": {
                str(lap): value
                for (lap, _, _), value in zip(times, pressure_by_lap)
                if value is not None
            },
            "sensor_trend_slopes_per_lap": {},
        }
    )
    for name, measurements in (
        ("tire_core_temperature", temperature_by_lap),
        ("tire_pressure", pressure_by_lap),
    ):
        valid = [
            (lap, measurement)
            for (lap, _, _), measurement in zip(times, measurements)
            if measurement is not None and math.isfinite(measurement)
        ]
        if len(valid) >= 3:
            sensor_slope, _ = np.polyfit(
                np.array([item[0] for item in valid], dtype=float),
                np.array([item[1] for item in valid], dtype=float),
                1,
            )
            result["sensor_trend_slopes_per_lap"][name] = float(sensor_slope)

    if slope <= 0:
        result["forecast_status"] = "no positive pace-degradation trend"
        result["predicted_threshold_lap"] = None
        return result

    crossing_lap = (baseline_time + drop_threshold - intercept) / slope
    predicted_lap = int(math.ceil(crossing_lap - 1e-9))
    result["predicted_threshold_lap"] = predicted_lap
    result["laps_until_threshold"] = max(
        0,
        predicted_lap - int(lap_numbers[-1]),
    )
    result["forecast_status"] = (
        "threshold already reached"
        if crossing_lap <= lap_numbers[-1]
        else "projected"
    )
    return result


def vehicle_dynamics_analysis(lap):
    """Return JSON-safe selected-lap summaries for optional vehicle signals."""
    radius = corner_radius_trace(lap)
    finite_radius = radius["radius_m"].replace([np.inf, -np.inf], np.nan).dropna()
    detected_corners = auto_detect_apexes(lap)
    radius_by_corner = []
    for corner in detected_corners:
        samples = radius.loc[
            radius["distance_m"].between(
                corner["apex_m"] - 30,
                corner["apex_m"] + 30,
            ),
            "radius_m",
        ].dropna()
        radius_by_corner.append(
            {
                "corner": corner["id"],
                "apex_m": corner["apex_m"],
                "median_radius_m": (
                    float(samples.median()) if len(samples) else None
                ),
                "p10_radius_m": (
                    float(samples.quantile(0.1)) if len(samples) else None
                ),
            }
        )

    slip = wheel_slip_ratio_trace(lap)
    slip_summary = {}
    for wheel in WHEELS:
        channel = f"slip_ratio_{wheel}_pct"
        if channel not in slip:
            slip_summary[wheel] = {
                "available": False,
                "unavailable": slip.attrs["unavailable"][wheel],
            }
            continue
        values = slip[channel].replace([np.inf, -np.inf], np.nan).dropna()
        slip_summary[wheel] = {
            "available": bool(len(values)),
            "sample_count": int(len(values)),
            "median_pct": float(values.median()) if len(values) else None,
            "p95_positive_pct": (
                float(values.clip(lower=0).quantile(0.95))
                if len(values)
                else None
            ),
            "p05_negative_pct": (
                float(values.clip(upper=0).quantile(0.05))
                if len(values)
                else None
            ),
            "unavailable": None if len(values) else "No samples above minimum speed.",
        }
    pitch = aero_pitch_trace(lap)
    return {
        "corner_radius": {
            "available": bool(len(finite_radius)),
            "sample_count": int(len(finite_radius)),
            "median_m": float(finite_radius.median()) if len(finite_radius) else None,
            "p10_m": float(finite_radius.quantile(0.1)) if len(finite_radius) else None,
            "p90_m": float(finite_radius.quantile(0.9)) if len(finite_radius) else None,
            "by_detected_corner": radius_by_corner,
            "unavailable": None
            if len(finite_radius)
            else "Ground speed and lateral acceleration are required.",
        },
        "wheel_slip_ratio": {
            "by_wheel": slip_summary,
            "reference_speed": "Ground Speed",
            "ratio_definition": "(wheel surface speed - ground speed) / ground speed × 100",
        },
        "aero_pitch_proxy": {
            "high_speed_threshold_kmh": pitch.attrs["high_speed_threshold_kmh"],
            "by_axle": pitch.attrs["available_axles"],
            "interpretation": (
                "Relative suspension/ride-height movement from the low-speed "
                "baseline. Sensor sign and calibration are required to infer "
                "compression or aerodynamic center-of-pressure movement."
            ),
        },
        "brake_input": brake_input_analysis(lap),
    }
