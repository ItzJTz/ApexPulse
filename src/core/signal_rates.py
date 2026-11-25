"""Rate-of-change helpers for quantized telemetry channels."""

import numpy as np
import pandas as pd


PEDAL_SMOOTHING_SAMPLES = 3
PEDAL_SLOPE_SAMPLES = 4


def smoothed_derivative(values, delta_time):
    """Measure a pedal rate over four intervals after three-sample smoothing."""
    values = np.asarray(values, dtype=float)
    delta_time = np.asarray(delta_time, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if len(delta_time) != max(0, len(values) - 1):
        raise ValueError("delta_time must contain one interval per value pair.")

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

    elapsed = np.concatenate(([0.0], np.cumsum(delta_time)))
    spans = elapsed[PEDAL_SLOPE_SAMPLES:] - elapsed[:-PEDAL_SLOPE_SAMPLES]
    changes = smoothed[PEDAL_SLOPE_SAMPLES:] - smoothed[:-PEDAL_SLOPE_SAMPLES]
    valid = np.isfinite(spans) & (spans > 0) & np.isfinite(changes)
    rates[PEDAL_SLOPE_SAMPLES:] = np.divide(
        changes,
        spans,
        out=np.zeros_like(changes),
        where=valid,
    )
    return rates
