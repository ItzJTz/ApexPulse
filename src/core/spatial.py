"""Shared interpolation helpers for distance-aligned telemetry."""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def interpolate_spatial(distance, values, grid):
    """Interpolate finite samples without extrapolating beyond their coverage."""
    distance = np.asarray(distance, dtype=float)
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if not np.isfinite(grid).all():
        raise ValueError("Spatial interpolation grid must contain only finite values.")
    finite = np.isfinite(distance) & np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError("Spatial interpolation requires at least one finite sample.")

    samples = (
        pd.DataFrame({"distance": distance[finite], "value": values[finite]})
        .groupby("distance", as_index=False)["value"]
        .mean()
        .sort_values("distance")
    )
    sample_distance = samples["distance"].to_numpy(dtype=float)
    sample_values = samples["value"].to_numpy(dtype=float)
    if grid.size and (
        grid.min() < sample_distance[0] or grid.max() > sample_distance[-1]
    ):
        raise ValueError("Spatial interpolation grid must remain within sample coverage.")
    if len(sample_distance) == 1:
        if grid.size and np.any(grid != sample_distance[0]):
            raise ValueError("A single spatial sample cannot cover a wider interpolation grid.")
        return np.full(grid.shape, sample_values[0], dtype=float)
    if len(sample_distance) < 3:
        return np.interp(grid, sample_distance, sample_values)
    return CubicSpline(sample_distance, sample_values, extrapolate=False)(grid)
