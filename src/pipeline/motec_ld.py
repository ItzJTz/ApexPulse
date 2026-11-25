"""Read standard MoTeC i2 .ld files into aligned telemetry channels."""

import math
import struct

import numpy as np
import pandas as pd


_HEADER_SIZE = 0x6E4
_CHANNEL_SIZE = 124
_MAX_CHANNELS = 2048
_MAX_SAMPLE_RATE_HZ = 1000
_MAX_OUTPUT_SAMPLES = 2_000_000
_MAX_OUTPUT_CELLS = 20_000_000
_CHANNEL_HEADER_FORMAT = "<IIIIHHHHhhhh"
_DISCRETE_CHANNELS = {
    "gear",
    "gearnumber",
    "gearposition",
    "lap",
    "lapnumber",
    "pitstatus",
}


def _fixed_text(data):
    return data.split(b"\0", 1)[0].decode("latin-1", errors="replace").strip()


def _read_header_text(data, offset, length):
    return _fixed_text(data[offset : offset + length])


def _channel_dtype(family, width):
    if family == 0x07:
        return {2: "<f2", 4: "<f4"}.get(width)
    if family == 0x08 and width == 8:
        return "<f8"
    if family in (0x00, 0x03, 0x05):
        return {2: "<i2", 4: "<i4"}.get(width)
    return None


def _read_channel_headers(data):
    if len(data) < _HEADER_SIZE:
        raise ValueError("The MoTeC LD file is truncated or has an unsupported header.")
    if struct.unpack_from("<I", data, 0)[0] != 0x40:
        raise ValueError("This is not a supported MoTeC i2 .ld file.")

    metadata_pointer, first_data_pointer = struct.unpack_from("<II", data, 8)
    if not _HEADER_SIZE <= metadata_pointer <= len(data):
        raise ValueError("The MoTeC LD channel table pointer is invalid.")
    if not metadata_pointer <= first_data_pointer <= len(data):
        raise ValueError("The MoTeC LD data pointer is outside the file.")

    channels = []
    seen = set()
    pointer = metadata_pointer
    while pointer:
        if pointer in seen:
            raise ValueError("The MoTeC LD channel table contains a pointer loop.")
        if (
            pointer < _HEADER_SIZE
            or pointer + _CHANNEL_SIZE > first_data_pointer
        ):
            raise ValueError("The MoTeC LD channel table is truncated.")
        if len(channels) >= _MAX_CHANNELS:
            raise ValueError(
                f"The MoTeC LD file exceeds the {_MAX_CHANNELS}-channel safety limit."
            )
        seen.add(pointer)

        (
            _,
            next_pointer,
            data_pointer,
            sample_count,
            _,
            type_family,
            sample_width,
            sample_rate,
            shift,
            multiplier,
            scale,
            decimal_places,
        ) = struct.unpack_from(_CHANNEL_HEADER_FORMAT, data, pointer)
        name = _read_header_text(data, pointer + 32, 32)
        short_name = _read_header_text(data, pointer + 64, 8)
        unit = _read_header_text(data, pointer + 72, 12)
        dtype = _channel_dtype(type_family, sample_width)

        if name and dtype is not None and sample_count and sample_rate:
            if sample_rate > _MAX_SAMPLE_RATE_HZ:
                raise ValueError(
                    f"Channel '{name}' samples at {sample_rate} Hz, above the "
                    f"{_MAX_SAMPLE_RATE_HZ} Hz supported limit."
                )
            if abs(decimal_places) > 12:
                raise ValueError(
                    f"Channel '{name}' has an invalid decimal scaling value."
                )
            byte_count = sample_count * np.dtype(dtype).itemsize
            if (
                data_pointer < first_data_pointer
                or data_pointer + byte_count > len(data)
            ):
                raise ValueError(f"Channel '{name}' data is truncated or out of bounds.")
            channels.append(
                {
                    "name": name,
                    "short_name": short_name,
                    "unit": unit,
                    "data_pointer": data_pointer,
                    "sample_count": sample_count,
                    "sample_rate": sample_rate,
                    "dtype": dtype,
                    "float_data": type_family in (0x07, 0x08),
                    "shift": shift,
                    "multiplier": multiplier,
                    "scale": scale,
                    "decimal_places": decimal_places,
                }
            )
        pointer = next_pointer

    if not channels:
        raise ValueError("The MoTeC LD file contains no supported numeric channels.")
    return channels


def _read_channel_values(data, channel):
    values = np.frombuffer(
        data,
        dtype=channel["dtype"],
        count=channel["sample_count"],
        offset=channel["data_pointer"],
    ).astype(np.float64)
    if channel["float_data"]:
        return values

    scale = channel["scale"] or 1
    multiplier = channel["multiplier"] or 1
    return (
        values
        * multiplier
        / scale
        * (10.0 ** -channel["decimal_places"])
        + channel["shift"]
    )


def _sample_to_grid(values, source_rate, target_times, discrete):
    source_times = np.arange(len(values), dtype=np.float64) / source_rate
    valid = np.isfinite(values)
    if not valid.any():
        return np.full(len(target_times), np.nan)

    source_times = source_times[valid]
    source_values = values[valid]
    if discrete:
        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)
        previous = np.maximum(indices - 1, 0)
        choose_previous = (
            np.abs(target_times - source_times[previous])
            <= np.abs(source_times[indices] - target_times)
        )
        indices[choose_previous] = previous[choose_previous]
        result = source_values[indices].copy()
    else:
        result = np.interp(
            target_times,
            source_times,
            source_values,
            left=np.nan,
            right=np.nan,
        )
    result[target_times < source_times[0]] = np.nan
    result[target_times > source_times[-1]] = np.nan
    return result


def parse_motec_ld(content):
    """Return channel data, header metadata, and channel units from an LD file."""
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("MoTeC LD content must be binary data.")
    data = bytes(content)
    channels = _read_channel_headers(data)

    sample_rate = max(channel["sample_rate"] for channel in channels)
    duration = max(
        (channel["sample_count"] - 1) / channel["sample_rate"]
        for channel in channels
    )
    sample_count = int(math.floor(duration * sample_rate + 1e-9)) + 1
    if sample_count > _MAX_OUTPUT_SAMPLES:
        raise ValueError(
            f"The MoTeC LD session would expand to {sample_count:,} samples; "
            f"the supported limit is {_MAX_OUTPUT_SAMPLES:,}."
        )
    if sample_count * len(channels) > _MAX_OUTPUT_CELLS:
        raise ValueError(
            "The MoTeC LD channels would exceed the conversion memory limit. "
            "Export fewer channels or split the session in MoTeC i2."
        )
    target_times = np.arange(sample_count, dtype=np.float64) / sample_rate

    columns = {}
    units = {}
    duplicate_names = {}
    channel_sample_rates = {}
    for channel in channels:
        original_name = channel["name"]
        duplicate_names[original_name] = duplicate_names.get(original_name, 0) + 1
        occurrence = duplicate_names[original_name]
        name = original_name if occurrence == 1 else f"{original_name} ({occurrence})"
        normalized_name = "".join(
            character
            for character in name.casefold()
            if character.isalnum()
        )
        values = _read_channel_values(data, channel)
        columns[name] = _sample_to_grid(
            values,
            channel["sample_rate"],
            target_times,
            discrete=normalized_name in _DISCRETE_CHANNELS,
        )
        units[name] = channel["unit"]
        channel_sample_rates[name] = channel["sample_rate"]

    columns["Time"] = target_times
    units["Time"] = "s"
    frame = pd.DataFrame(columns)
    metadata = {
        "venue": _read_header_text(data, 0x15E, 64),
        "driver": _read_header_text(data, 0x9E, 64),
        "vehicle": _read_header_text(data, 0xDE, 64),
        "date": _read_header_text(data, 0x5E, 16),
        "time": _read_header_text(data, 0x7E, 16),
    }
    return frame, metadata, units, channel_sample_rates
