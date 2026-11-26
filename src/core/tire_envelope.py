import re

import numpy as np
import pandas as pd

from src.core.signal_rates import native_sample_rate_hz

POLAR_ENVELOPE_WEDGES = 36
POLAR_ENVELOPE_PERCENTILE = 95.0


def polar_envelope_area(lateral_g, longitudinal_g):
    """Return a 95th-percentile polar G-G area and populated wedge count."""
    lateral = np.asarray(lateral_g, dtype=float)
    longitudinal = np.asarray(longitudinal_g, dtype=float)
    finite = np.isfinite(lateral) & np.isfinite(longitudinal)
    lateral = lateral[finite]
    longitudinal = longitudinal[finite]
    if len(lateral) < 3:
        return None, 0

    radius = np.hypot(lateral, longitudinal)
    angle = np.arctan2(lateral, longitudinal)
    wedge_width = 2 * np.pi / POLAR_ENVELOPE_WEDGES
    wedge_indexes = np.floor((angle + np.pi) / wedge_width).astype(int)
    wedge_indexes = np.clip(wedge_indexes, 0, POLAR_ENVELOPE_WEDGES - 1)

    radii = np.zeros(POLAR_ENVELOPE_WEDGES, dtype=float)
    populated = np.zeros(POLAR_ENVELOPE_WEDGES, dtype=bool)
    for wedge in range(POLAR_ENVELOPE_WEDGES):
        samples = radius[wedge_indexes == wedge]
        if len(samples):
            radii[wedge] = float(
                np.percentile(samples, POLAR_ENVELOPE_PERCENTILE)
            )
            populated[wedge] = True
    if not populated.any():
        return None, 0

    angles = -np.pi + (np.arange(POLAR_ENVELOPE_WEDGES) + 0.5) * wedge_width
    x = radii[populated] * np.cos(angles[populated])
    y = radii[populated] * np.sin(angles[populated])
    area = 0.5 * abs(
        float(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    )
    return area, int(populated.sum())


def off_track_spike_mask(lap):
    """Mark brief >1.9 G samples recorded while at least two tires are off track."""
    off_track_column = next(
        (
            column
            for column in lap.columns
            if re.sub(r"[^a-z0-9]", "", str(column).casefold())
            in {"numtiresofftrack", "numbertiresofftrack"}
        ),
        None,
    )
    if (
        off_track_column is None
        or "G Force Lat" not in lap
        or not any(
            re.sub(r"[^a-z0-9]", "", str(column).casefold())
            in {"cgaccelvertical", "verticalg", "gforcevertical"}
            for column in lap.columns
        )
    ):
        return np.zeros(len(lap), dtype=bool)

    vertical_column = next(
        column
        for column in lap.columns
        if re.sub(r"[^a-z0-9]", "", str(column).casefold())
        in {"cgaccelvertical", "verticalg", "gforcevertical"}
    )
    off_track = pd.to_numeric(lap[off_track_column], errors="coerce").to_numpy(
        dtype=float
    )
    lateral = pd.to_numeric(lap["G Force Lat"], errors="coerce").to_numpy(
        dtype=float
    )
    vertical = pd.to_numeric(lap[vertical_column], errors="coerce").to_numpy(
        dtype=float
    )
    spikes = (
        np.isfinite(off_track)
        & (off_track >= 2)
        & (
            (np.isfinite(lateral) & (np.abs(lateral) > 1.9))
            | (np.isfinite(vertical) & (np.abs(vertical) > 1.9))
        )
    )
    transitions = np.diff(np.r_[False, spikes, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    rejected = np.zeros(len(lap), dtype=bool)
    if "Time" in lap:
        elapsed = pd.to_numeric(lap["Time"], errors="coerce").to_numpy(
            dtype=float
        )
    else:
        sample_rate = native_sample_rate_hz(lap, "G Force Lat") or 50.0
        elapsed = np.arange(len(lap), dtype=float) / sample_rate
    valid_steps = np.diff(elapsed)
    valid_steps = valid_steps[np.isfinite(valid_steps) & (valid_steps > 0)]
    step = float(np.median(valid_steps)) if len(valid_steps) else 0.02
    for start, end in zip(starts, ends):
        run_time = elapsed[start:end]
        if (
            len(run_time) > 1
            and np.isfinite(run_time).all()
            and np.all(np.diff(run_time) > 0)
        ):
            duration = run_time[-1] - run_time[0] + step
        else:
            duration = (end - start) * step
        if duration < 0.06:
            rejected[start:end] = True
    return rejected


def sensor_unit_conversion(frame, column, sensor_kind):
    """Convert supported tire-temperature or pressure units to C or psi."""
    quality = frame.attrs.get("quality", {})
    units = quality.get("channel_units", {})
    unit = str(units.get(column, "") or "").strip().casefold()
    assumed = not unit
    if not unit:
        name = str(column).casefold()
        unit = next(
            (
                token
                for token in (
                    "degf",
                    "°f",
                    "fahrenheit",
                    "degc",
                    "°c",
                    "celsius",
                    "kpa",
                    "mpa",
                    "mbar",
                    "bar",
                    "psi",
                    "pa",
                )
                if token in name
            ),
            "",
        )
        assumed = not unit
    normalized = re.sub(r"[^a-z0-9]", "", unit)
    if sensor_kind == "temperature":
        conversions = {
            "c": (1.0, 0.0),
            "degc": (1.0, 0.0),
            "celsius": (1.0, 0.0),
            "f": (5 / 9, -32 * 5 / 9),
            "degf": (5 / 9, -32 * 5 / 9),
            "fahrenheit": (5 / 9, -32 * 5 / 9),
            "k": (1.0, -273.15),
            "kelvin": (1.0, -273.15),
        }
    elif sensor_kind == "pressure":
        conversions = {
            "psi": (1.0, 0.0),
            "psig": (1.0, 0.0),
            "psia": (1.0, 0.0),
            "kpa": (0.1450377377, 0.0),
            "bar": (14.50377377, 0.0),
            "barg": (14.50377377, 0.0),
            "bara": (14.50377377, 0.0),
            "pa": (0.0001450377377, 0.0),
            "mpa": (145.0377377, 0.0),
            "mbar": (0.01450377377, 0.0),
        }
    else:
        raise ValueError(f"Unsupported tire sensor kind: {sensor_kind}")
    conversion = conversions.get(normalized)
    if conversion is None:
        return None, None, False
    scale, offset = conversion
    return scale, offset, assumed
