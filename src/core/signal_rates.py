"""Rate-of-change helpers for quantized telemetry channels."""

import numpy as np
import pandas as pd


PEDAL_SMOOTHING_SAMPLES = 3
PEDAL_SLOPE_SAMPLES = 4


def smoothed_derivative(values, delta_time, sample_rate_hz=None):
    """Measure a centered four-interval rate at the source's native cadence."""
    values = np.asarray(values, dtype=float)
    delta_time = np.asarray(delta_time, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if len(delta_time) != max(0, len(values) - 1):
        raise ValueError("delta_time must contain one interval per value pair.")
    if sample_rate_hz is not None and (
        not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be finite and greater than zero.")

    elapsed = np.concatenate(([0.0], np.cumsum(delta_time)))
    valid_intervals = delta_time[np.isfinite(delta_time) & (delta_time > 0)]
    if sample_rate_hz is not None and len(valid_intervals):
        observed_rate_hz = 1.0 / float(np.median(valid_intervals))
        if sample_rate_hz < observed_rate_hz * 0.95 and elapsed[-1] > 0:
            source_interval = 1.0 / sample_rate_hz
            source_times = np.arange(
                0.0,
                elapsed[-1] + source_interval * 1e-9,
                source_interval,
            )
            if len(source_times) > 1:
                source_values = np.interp(source_times, elapsed, values)
                source_rates = _centered_smoothed_derivative(
                    source_values,
                    source_times,
                )
                return np.interp(
                    elapsed,
                    source_times,
                    source_rates,
                    left=0.0,
                    right=0.0,
                )

    return _centered_smoothed_derivative(values, elapsed)


def native_sample_rate_hz(frame, channel):
    """Return a channel's recorded rate when the source provides it."""
    sample_rates = frame.attrs.get("quality", {}).get(
        "channel_sample_rates_hz", {}
    )
    try:
        rate = float(sample_rates.get(channel))
    except (TypeError, ValueError):
        return None
    return rate if np.isfinite(rate) and rate > 0 else None


def _centered_smoothed_derivative(values, elapsed):
    smoothed = (
        pd.Series(values)
        .rolling(
            window=PEDAL_SMOOTHING_SAMPLES,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(dtype=float)
    )
    rates = np.zeros(len(values), dtype=float)
    if len(values) <= PEDAL_SLOPE_SAMPLES:
        return rates

    half_window = PEDAL_SLOPE_SAMPLES // 2
    centers = np.arange(half_window, len(values) - half_window)
    spans = elapsed[centers + half_window] - elapsed[centers - half_window]
    changes = (
        smoothed[centers + half_window] - smoothed[centers - half_window]
    )
    valid = np.isfinite(spans) & (spans > 0) & np.isfinite(changes)
    rates[centers] = np.divide(
        changes,
        spans,
        out=np.zeros_like(changes),
        where=valid,
    )
    return rates
