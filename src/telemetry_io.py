import csv
import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_CHANNELS = ("Lap Number", "Lap Distance", "Ground Speed")


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _resolve_track(path, metadata, tracks_dir=None):
    tracks_path = Path(tracks_dir) if tracks_dir else Path(__file__).resolve().parent.parent / "tracks"
    if not tracks_path.is_dir():
        return None

    candidates = [Path(path).stem, metadata.get("venue", "")]
    normalized_candidates = [_key(candidate) for candidate in candidates if candidate]
    matches = []
    for config_path in tracks_path.glob("*.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read track configuration '{config_path}': {exc}") from exc
        aliases = [config_path.stem, config.get("circuit_name", "")]
        for alias in aliases:
            normalized_alias = _key(alias)
            if normalized_alias and any(
                normalized_alias in candidate or candidate in normalized_alias
                for candidate in normalized_candidates
            ):
                matches.append((len(normalized_alias), config_path.stem))
    return max(matches)[1] if matches else None


def prepare_telemetry_frame(frame, source_path=None, metadata=None, tracks_dir=None):
    """Canonicalize core channel names, validate samples, and sort by lap distance."""
    input_rows = len(frame)
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]
    normalized_columns = {_key(column): column for column in df.columns}

    def find_column(predicate, preferred=()):
        for name in preferred:
            if _key(name) in normalized_columns:
                return normalized_columns[_key(name)]
        return next(
            (column for column in df.columns if predicate(_key(column))),
            None,
        )

    lap_column = find_column(
        lambda key: "lap" in key and ("number" in key or key.endswith("num")),
        ("Lap Number", "Lap Num"),
    )
    if lap_column is None:
        lap_column = find_column(
            lambda key: "lap" in key and "dist" not in key and "time" not in key,
        )

    lap_distance_column = find_column(
        lambda key: "lap" in key and "dist" in key,
        ("Lap Distance",),
    )
    distance_column = find_column(
        lambda key: key in ("distance", "dist"),
        ("Distance", "Dist"),
    )
    speed_column = find_column(
        lambda key: "speed" in key and not any(
            component in key for component in ("wheel", "tyre", "tire")
        ),
        ("Ground Speed", "Speed"),
    )

    selected = {
        "Lap Number": lap_column,
        "Lap Distance": lap_distance_column or distance_column,
        "Ground Speed": speed_column,
    }
    source_columns = dict(selected)
    missing = [name for name, column in selected.items() if column is None]
    if missing:
        raise ValueError(
            "Telemetry is missing required channel(s): "
            + ", ".join(missing)
            + f". Available channels: {list(df.columns)}"
        )

    rename = {
        column: canonical
        for canonical, column in selected.items()
        if column != canonical
    }
    df = df.rename(columns=rename)

    time_column = find_column(
        lambda key: key == "time" or key == "laptime" or key == "sessionelapsedtime",
        ("Time", "Lap Time", "Session Elapsed Time"),
    )
    if time_column is not None and time_column != "Time" and "Time" not in df.columns:
        df = df.rename(columns={time_column: "Time"})

    numeric_columns = ["Lap Number", "Lap Distance", "Ground Speed"]
    for column in ("Distance", "Time"):
        if column in df.columns:
            numeric_columns.append(column)
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    optional_channels = (
        "Throttle Pos",
        "Brake Pos",
        "Steering",
        "G Force Lat",
        "G Force Long",
        "GPS Latitude",
        "GPS Longitude",
    )
    missing_percent = {}
    for channel in optional_channels:
        source_column = normalized_columns.get(_key(channel))
        if source_column is not None:
            missing_percent[channel] = round(
                float(pd.to_numeric(df[source_column], errors="coerce").isna().mean() * 100),
                2,
            )

    df = df.replace([np.inf, -np.inf], np.nan)
    rows_with_valid_required_channels = int(df.dropna(subset=list(REQUIRED_CHANNELS)).shape[0])
    df = df.dropna(subset=list(REQUIRED_CHANNELS)).copy()
    if df.empty:
        raise ValueError("Telemetry contains no rows with valid lap number, lap distance, and speed.")

    df = df.sort_values(["Lap Number", "Lap Distance"], kind="stable")
    repeated_distance_samples = int(
        df.duplicated(["Lap Number", "Lap Distance"]).sum()
    )
    df["Normalized_Dist"] = df.groupby("Lap Number")["Lap Distance"].transform(
        lambda distance: distance - distance.min()
    )

    track_metadata = dict(metadata or {})
    if "venue" not in track_metadata and "Track" in df.columns:
        existing_tracks = df["Track"].dropna().astype(str).unique()
        if len(existing_tracks) == 1:
            track_metadata["venue"] = existing_tracks[0]
    track = _resolve_track(source_path, track_metadata, tracks_dir) if source_path else None
    if track:
        df["Track"] = track
    df.attrs["quality"] = {
        "source_rows": input_rows,
        "invalid_required_rows": input_rows - rows_with_valid_required_channels,
        "repeated_distance_samples": repeated_distance_samples,
        "source_channels": list(frame.columns),
        "required_channel_mapping": source_columns,
        "optional_channel_missing_percent": missing_percent,
        "detected_track": track,
    }
    return df.reset_index(drop=True)


def _load_telemetry_text(text, source_name, tracks_dir):
    stream = io.StringIO(text)
    rows = list(csv.reader(stream))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any("dist" in _key(column) for column in row)
            and any("speed" in _key(column) for column in row)
            and any("lap" in _key(column) for column in row)
        ),
        None,
    )
    if header_index is None:
        raise ValueError(
            f"Could not find a telemetry header with distance, speed, and lap channels in '{source_name}'."
        )

    metadata = {
        str(row[0]).strip().lower(): str(row[1]).strip()
        for row in rows[:header_index]
        if len(row) >= 2
    }
    frame = pd.read_csv(
        io.StringIO(text),
        skiprows=header_index,
        low_memory=False,
    )
    prepared = prepare_telemetry_frame(frame, source_name, metadata, tracks_dir)
    prepared.attrs["quality"]["source_file"] = source_name
    return prepared


def load_telemetry_text(content, source_name="uploaded.csv", tracks_dir=None):
    """Read raw or cleaned CSV text/bytes into the canonical telemetry frame."""
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig", errors="replace")
    if not isinstance(content, str):
        raise TypeError("Telemetry content must be text or bytes.")
    return _load_telemetry_text(content, source_name, tracks_dir)


def load_telemetry(path, tracks_dir=None):
    """Read raw MoTeC-style or cleaned CSV files into the canonical frame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry file not found: '{path}'.")
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise OSError(f"Could not read telemetry file '{path}': {exc}") from exc
    return _load_telemetry_text(text, str(path), tracks_dir)


def process_session(frame, minimum_fraction=0.9):
    """Filter incomplete laps, smooth speed, and build a reproducible QC report."""
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be greater than 0 and at most 1.")
    quality = dict(frame.attrs.get("quality", {}))
    track_name = frame["Track"].iloc[0] if "Track" in frame.columns else None
    track_length = None
    if track_name:
        config_path = Path(__file__).resolve().parent.parent / "tracks" / f"{track_name}.json"
        if config_path.is_file():
            try:
                track_length = json.loads(config_path.read_text(encoding="utf-8")).get(
                    "total_length_m"
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read track configuration '{config_path}': {exc}") from exc

    lap_lengths = frame.groupby("Lap Number")["Normalized_Dist"].max()
    reference_length = track_length or (float(lap_lengths.max()) if not lap_lengths.empty else 0)
    threshold = reference_length * minimum_fraction
    valid_laps = lap_lengths[lap_lengths >= threshold].index.tolist()
    invalid_laps = lap_lengths[~lap_lengths.index.isin(valid_laps)].index.tolist()
    result = frame[frame["Lap Number"].isin(valid_laps)].copy()
    if result.empty:
        raise ValueError(
            f"No laps cover at least {minimum_fraction:.0%} of the reference distance "
            f"({reference_length:.1f} m)."
        )

    result["Ground Speed"] = result.groupby("Lap Number")["Ground Speed"].transform(
        lambda speed: speed.rolling(window=3, center=True, min_periods=1).median()
    )
    result.attrs["quality"] = quality
    fastest_laps = fastest_lap_numbers(result, count=2)
    report = {
        "schema_version": 1,
        "source_file": quality.get("source_file"),
        "track": track_name,
        "reference_track_length_m": round(float(reference_length), 2),
        "minimum_lap_fraction": minimum_fraction,
        "minimum_lap_distance_m": round(float(threshold), 2),
        "rows": {
            "source": quality.get("source_rows", len(frame)),
            "invalid_required_data": quality.get("invalid_required_rows", 0),
            "repeated_distance_samples": quality.get("repeated_distance_samples", 0),
            "canonical": len(frame),
            "after_lap_filter": len(result),
        },
        "channels": {
            "required_mapping": quality.get("required_channel_mapping", {}),
            "available": quality.get("source_channels", list(frame.columns)),
            "optional_missing_percent": quality.get("optional_channel_missing_percent", {}),
        },
        "laps": {
            "total": int(len(lap_lengths)),
            "complete": [float(lap) for lap in valid_laps],
            "excluded_incomplete": [float(lap) for lap in invalid_laps],
            "distance_m": {
                str(lap): round(float(distance), 2)
                for lap, distance in lap_lengths.items()
            },
        },
        "fastest_complete_laps": [
            {
                "lap": float(lap),
                "duration_seconds": round(lap_time_seconds(result[result["Lap Number"] == lap]), 3),
                "duration_source": lap_time_source(result[result["Lap Number"] == lap]),
            }
            for lap in fastest_laps
        ],
    }
    report["source_file"] = report["source_file"] or "in-memory telemetry"
    return result.reset_index(drop=True), report


def valid_lap_numbers(frame, minimum_fraction=0.9, track_length=None):
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be greater than 0 and at most 1.")
    lap_lengths = frame.groupby("Lap Number")["Normalized_Dist"].max()
    reference_length = track_length or lap_lengths.max()
    return lap_lengths[lap_lengths >= minimum_fraction * reference_length].index.tolist()


def lap_time_seconds(lap):
    """Use recorded elapsed time where possible, otherwise integrate ds / speed."""
    if lap_time_source(lap) == "recorded_time":
        recorded = pd.to_numeric(lap["Time"], errors="coerce").dropna()
        return float(recorded.max() - recorded.min())

    ordered = (
        lap.groupby("Normalized_Dist", as_index=False)["Ground Speed"]
        .mean()
        .sort_values("Normalized_Dist")
    )
    distance = ordered["Normalized_Dist"].to_numpy(dtype=float)
    speed_ms = np.maximum(ordered["Ground Speed"].to_numpy(dtype=float) / 3.6, 1.0)
    intervals = np.diff(distance)
    if len(intervals) == 0:
        return 0.0
    return float(np.sum(intervals * (0.5 / speed_ms[:-1] + 0.5 / speed_ms[1:])))


def lap_time_source(lap):
    if "Time" in lap.columns:
        recorded = pd.to_numeric(lap["Time"], errors="coerce").dropna()
        if len(recorded) > 1:
            duration = float(recorded.max() - recorded.min())
            if math.isfinite(duration) and duration > 0:
                return "recorded_time"
    return "estimated_from_distance_and_speed"


def fastest_lap_numbers(frame, count=2, minimum_fraction=0.9, track_length=None):
    valid = valid_lap_numbers(frame, minimum_fraction, track_length)
    times = {
        lap_number: lap_time_seconds(frame[frame["Lap Number"] == lap_number])
        for lap_number in valid
    }
    return sorted(times, key=times.__getitem__)[:count]
