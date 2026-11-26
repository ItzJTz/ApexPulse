import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.motec_ld import _HEADER_SIZE, _CHANNEL_SIZE
from src.pipeline.telemetry_io import (
    fastest_lap_numbers,
    lap_time_seconds,
    load_telemetry,
    load_telemetry_content,
    load_telemetry_text,
    prepare_telemetry_frame,
    process_session,
    valid_lap_numbers,
)


def _build_motec_ld(channels, venue="Test Circuit GP"):
    metadata_pointer = _HEADER_SIZE
    data_pointer = metadata_pointer + len(channels) * _CHANNEL_SIZE
    content = bytearray(data_pointer)
    struct.pack_into("<I", content, 0, 0x40)
    struct.pack_into("<II", content, 8, metadata_pointer, data_pointer)
    for offset, value, length in (
        (0x5E, "01/01/2025", 16),
        (0x7E, "12:00:00", 16),
        (0x9E, "Test Driver", 64),
        (0xDE, "Test Vehicle", 64),
        (0x15E, venue, 64),
    ):
        encoded = value.encode("ascii")
        content[offset : offset + min(len(encoded), length)] = encoded[:length]

    channel_data = []
    current_data_pointer = data_pointer
    for index, channel in enumerate(channels):
        channel_pointer = metadata_pointer + index * _CHANNEL_SIZE
        next_pointer = (
            channel_pointer + _CHANNEL_SIZE
            if index < len(channels) - 1
            else 0
        )
        values = channel["values"]
        dtype = channel.get("dtype", "<f4")
        raw = np.asarray(values, dtype=dtype).tobytes()
        family = channel.get("family", 0x07 if dtype.startswith("<f") else 0x03)
        struct.pack_into(
            "<IIIIHHHHhhhh",
            content,
            channel_pointer,
            channel_pointer - _CHANNEL_SIZE if index else 0,
            next_pointer,
            current_data_pointer,
            len(values),
            0,
            family,
            np.dtype(dtype).itemsize,
            channel["rate"],
            channel.get("shift", 0),
            channel.get("multiplier", 1),
            channel.get("scale", 1),
            channel.get("decimal_places", 0),
        )
        for field_offset, field_value, field_length in (
            (32, channel["name"], 32),
            (64, channel.get("short_name", channel["name"]), 8),
            (72, channel.get("unit", ""), 12),
        ):
            encoded = field_value.encode("ascii")
            start = channel_pointer + field_offset
            content[start : start + min(len(encoded), field_length)] = encoded[:field_length]
        channel_data.append(raw)
        current_data_pointer += len(raw)
    for raw in channel_data:
        content.extend(raw)
    return bytes(content)


class TelemetryIoTests(unittest.TestCase):
    def test_loads_motec_ld_and_converts_required_units(self):
        channels = [
            {
                "name": "Lap Number",
                "values": [1, 1, 1, 2, 2, 2],
                "dtype": "<i2",
                "rate": 2,
            },
            {
                "name": "Lap Distance",
                "values": [0, 100, 200, 0, 100, 200],
                "unit": "m",
                "rate": 2,
            },
            {
                "name": "Ground Speed",
                "values": [10, 10, 10, 10, 10, 10],
                "dtype": "<i2",
                "unit": "m/s",
                "rate": 2,
            },
            {
                "name": "Throttle Pos",
                "values": [-25000, 0, 25000, -25000, 0, 25000],
                "dtype": "<i2",
                "unit": "%",
                "rate": 2,
                "shift": 50,
                "multiplier": 2,
                "decimal_places": 3,
            },
            {
                "name": "Gear",
                "values": [1, 2, 3, 1, 2, 3],
                "dtype": "<i2",
                "rate": 2,
                "shift": 2,
                "multiplier": 2,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            tracks = Path(temporary_directory)
            (tracks / "TestCircuit.json").write_text(
                json.dumps(
                    {
                        "circuit_name": "Test Circuit GP",
                        "total_length_m": 200,
                        "sectors": {},
                    }
                ),
                encoding="utf-8",
            )

            frame = load_telemetry_content(
                _build_motec_ld(channels),
                "test_session.ld",
                tracks,
            )
            clean, report = process_session(frame, rolling_median_window=1)
            csv_roundtrip = load_telemetry_text(
                clean.to_csv(index=False),
                "converted_session.csv",
            )

        self.assertEqual(frame["Lap Number"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(frame["Lap Distance"].tolist(), [0, 100, 200, 0, 100, 200])
        self.assertEqual(frame["Ground Speed"].tolist(), [36] * 6)
        self.assertEqual(frame["Gear"].tolist(), [4, 6, 8, 4, 6, 8])
        self.assertEqual(frame["Throttle Pos"].tolist(), [0, 50, 100, 0, 50, 100])
        self.assertEqual(frame["Track"].unique().tolist(), ["TestCircuit"])
        self.assertEqual(frame.attrs["quality"]["source_format"], "MoTeC LD")
        self.assertEqual(
            frame.attrs["quality"]["channel_units"]["Ground Speed"],
            "km/h",
        )
        self.assertEqual(
            frame.attrs["quality"]["channel_sample_rates_hz"]["Throttle Pos"],
            2,
        )
        self.assertEqual(
            frame.attrs["quality"]["conversion_warnings"],
            ["Ground Speed in m/s was converted to km/h."],
        )
        self.assertEqual(report["source_format"], "MoTeC LD")
        self.assertEqual(report["laps"]["complete"], [0.0, 1.0])
        self.assertEqual(csv_roundtrip["Ground Speed"].tolist(), [36] * 6)

    def test_resamples_mixed_rate_ld_channels_and_keeps_lap_numbers_discrete(self):
        channels = [
            {
                "name": "Lap Number",
                "values": [1, 2],
                "dtype": "<i2",
                "rate": 1,
            },
            {
                "name": "Lap Distance",
                "values": [0, 10, 20],
                "unit": "m",
                "rate": 2,
            },
            {
                "name": "Ground Speed",
                "values": [100, 100, 100],
                "unit": "km/h",
                "rate": 2,
            },
        ]

        frame = load_telemetry_content(
            _build_motec_ld(channels),
            "mixed_rate.ld",
        )

        self.assertEqual(frame["Lap Number"].tolist(), [1, 1, 2])
        self.assertEqual(frame["Time"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(frame["Lap Distance"].tolist(), [0, 10, 20])

    def test_rejects_unsupported_and_invalid_motec_files(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            load_telemetry_content(b"not an LD", "session.ldx")
        with self.assertRaisesRegex(ValueError, "not a supported MoTeC"):
            load_telemetry_content(b"\0" * _HEADER_SIZE, "session.ld")

    def test_rejects_ld_channel_pointer_loops_and_overlapping_data(self):
        channels = [
            {"name": "Lap Number", "values": [1], "dtype": "<i2", "rate": 1},
        ]
        looped = bytearray(_build_motec_ld(channels))
        struct.pack_into("<I", looped, _HEADER_SIZE + 4, _HEADER_SIZE)
        with self.assertRaisesRegex(ValueError, "pointer loop"):
            load_telemetry_content(looped, "looped.ld")

        overlapping = bytearray(_build_motec_ld(channels))
        struct.pack_into("<I", overlapping, _HEADER_SIZE + 8, _HEADER_SIZE)
        with self.assertRaisesRegex(ValueError, "truncated or out of bounds"):
            load_telemetry_content(overlapping, "overlap.ld")

    def test_load_telemetry_reads_ld_paths_and_csv_dispatch_remains_supported(self):
        channels = [
            {"name": "Lap Number", "values": [1, 1], "dtype": "<i2", "rate": 1},
            {"name": "Lap Distance", "values": [0, 10], "unit": "m", "rate": 1},
            {"name": "Ground Speed", "values": [36, 36], "unit": "km/h", "rate": 1},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            ld_path = Path(temporary_directory) / "session.ld"
            ld_path.write_bytes(_build_motec_ld(channels))
            ld_frame = load_telemetry(ld_path)

        csv_frame = load_telemetry_content(
            b"Time,Distance,Ground Speed,Lap Number,Lap Distance\n"
            b"0,0,100,1,0\n"
            b"1,10,100,1,10\n",
            "session.csv",
        )

        self.assertEqual(ld_frame.attrs["quality"]["source_format"], "MoTeC LD")
        self.assertEqual(csv_frame["Ground Speed"].tolist(), [100, 100])

    def test_preserves_unit_row_for_vehicle_dynamics_and_qc_report(self):
        telemetry = (
            '"Venue","Test Circuit GP Event"\n'
            "Time,Distance,Ground Speed,Lap Number,Lap Distance,"
            "Wheel Rot Speed FL,Tire Loaded Radius FL\n"
            "s,m,km/h,,m,rad/s,m\n"
            "0,0,36,1,0,20,0.5\n"
            "1,10,36,1,10,20,0.5\n"
        )

        frame = load_telemetry_text(telemetry, "session.csv")
        _, report = process_session(frame, rolling_median_window=1)

        self.assertEqual(frame.attrs["quality"]["channel_units"]["Wheel Rot Speed FL"], "rad/s")
        self.assertEqual(
            report["channels"]["units"]["Tire Loaded Radius FL"],
            "m",
        )

    def test_loads_metadata_csv_and_resolves_track(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tracks = root / "tracks"
            tracks.mkdir()
            (tracks / "TestCircuit.json").write_text(
                json.dumps(
                    {
                        "circuit_name": "Test Circuit GP",
                        "total_length_m": 100,
                        "sectors": {},
                    }
                ),
                encoding="utf-8",
            )
            telemetry = root / "session.csv"
            telemetry.write_text(
                '"Venue","Test Circuit GP Event"\n'
                "Sample,Metadata\n"
                "Time,Distance,Ground Speed,Lap Number,Lap Distance\n"
                "0,0,100,1,0\n"
                "1,50,100,1,50\n"
                "2,100,100,1,100\n"
                "3,0,120,2,0\n"
                "3.5,50,120,2,50\n"
                "4,100,120,2,100\n",
                encoding="utf-8",
            )

            frame = load_telemetry(telemetry, tracks)

        self.assertEqual(frame["Track"].unique().tolist(), ["TestCircuit"])
        self.assertEqual(frame["Normalized_Dist"].tolist(), [0, 50, 100, 0, 50, 100])
        self.assertEqual(frame.attrs["quality"]["channel_units"], {})
        self.assertEqual(fastest_lap_numbers(frame), [1, 0])

    def test_splits_hotlap_on_distance_resets_and_discards_transition_glitches(self):
        distances = [
            800,
            900,
            *range(0, 1001),
            700,
            200,
            0,
            *range(1, 1001),
            0,
            1,
            2,
        ]
        frame = pd.DataFrame(
            {
                "Lap Number": [0] * len(distances),
                "Lap Distance": distances,
                "Ground Speed": [100] * len(distances),
                "Time": np.arange(len(distances), dtype=float) / 50,
                "Lap Time": [999] * len(distances),
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            tracks = Path(temporary_directory)
            (tracks / "TestCircuit.json").write_text(
                json.dumps(
                    {
                        "circuit_name": "Test Circuit GP",
                        "total_length_m": 1000,
                    }
                ),
                encoding="utf-8",
            )
            prepared = prepare_telemetry_frame(
                frame,
                "hotlap.csv",
                {"venue": "Test Circuit GP"},
                tracks,
            )
            clean, report = process_session(
                prepared,
                rolling_median_window=1,
            )

        self.assertEqual(
            prepared["Lap Number"].drop_duplicates().tolist(),
            [0, 1, 2, 3],
        )
        self.assertTrue(prepared["Time"].is_monotonic_increasing)
        self.assertEqual(
            report["lap_segmentation"]["method"],
            "distance_reset",
        )
        self.assertEqual(report["lap_segmentation"]["reset_count"], 3)
        self.assertEqual(
            report["lap_segmentation"]["transition_rows_removed"],
            2,
        )
        self.assertEqual(report["laps"]["complete"], [1.0, 2.0])
        self.assertEqual(report["laps"]["excluded_bad_start"], [0.0])
        self.assertEqual(report["laps"]["excluded_incomplete"], [3.0])
        self.assertAlmostEqual(float(clean["Lap Time"].max()), 20.0)
        self.assertAlmostEqual(lap_time_seconds(clean), 20.0)

    def test_valid_laps_must_start_near_zero(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tracks = Path(temporary_directory)
            (tracks / "TestCircuit.json").write_text(
                json.dumps(
                    {
                        "circuit_name": "Test Circuit GP",
                        "total_length_m": 1000,
                    }
                ),
                encoding="utf-8",
            )
            frame = prepare_telemetry_frame(
                pd.DataFrame(
                    {
                        "Time": list(range(10)),
                        "Lap Number": [1] * 10,
                        "Lap Distance": list(range(150, 1051, 100)),
                        "Ground Speed": [100] * 10,
                    }
                ),
                "test.csv",
                {"venue": "Test Circuit GP"},
                tracks,
            )

        self.assertEqual(
            valid_lap_numbers(frame, minimum_fraction=0.9, track_length=1000),
            [],
        )

    def test_normalizes_channels_sorts_and_deduplicates(self):
        frame = pd.DataFrame(
            {
                " lap number ": [1, 1, 1, 2],
                "Lap Dist": [10, 0, 10, 0],
                "Speed": [100, 90, 101, 110],
            }
        )

        prepared = prepare_telemetry_frame(frame)

        self.assertEqual(prepared["Lap Number"].tolist(), [1, 1, 1, 2])
        self.assertEqual(prepared["Lap Distance"].tolist(), [0, 10, 10, 0])
        self.assertEqual(prepared["Ground Speed"].tolist(), [90, 100, 101, 110])
        self.assertEqual(prepared.attrs["quality"]["repeated_distance_samples"], 1)

    def test_normalizes_assetto_corsa_optional_channel_names(self):
        frame = pd.DataFrame(
            {
                "Lap Number": [1, 1, 1],
                "Lap Distance": [0, 10, 20],
                "Ground Speed": [36, 72, 36],
                "Steering Angle": [0, 5, -5],
                "CG Accel Lateral": [0.1, 0.8, -0.7],
                "CG Accel Longitudinal": [-0.2, 0.3, -0.4],
                "Tire Temp Inner FL": [100, 102, 104],
                "Tire Temp Inner FR": [110, 112, 114],
                "Tire Temp Outer FL": [90, 92, 94],
                "Tire Temp Outer FR": [100, 102, 104],
            }
        )

        prepared = prepare_telemetry_frame(frame)

        self.assertEqual(prepared["Steering"].tolist(), [0, 5, -5])
        self.assertEqual(prepared["G Force Lat"].tolist(), [0.1, 0.8, -0.7])
        self.assertEqual(prepared["G Force Long"].tolist(), [-0.2, 0.3, -0.4])
        self.assertEqual(prepared["Tire Temp Inner"].tolist(), [105, 107, 109])
        self.assertEqual(prepared["Tire Temp Outer"].tolist(), [95, 97, 99])
        self.assertEqual(
            prepared.attrs["quality"]["optional_channel_missing_percent"][
                "G Force Lat"
            ],
            0.0,
        )

    def test_rejects_missing_required_channel(self):
        with self.assertRaisesRegex(ValueError, "Ground Speed"):
            prepare_telemetry_frame(
                pd.DataFrame({"Lap Number": [1], "Lap Distance": [0]})
            )

    def test_valid_laps_and_integrated_fallback_time(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 2, 2],
                    "Lap Distance": [0, 100, 0, 80],
                    "Ground Speed": [36, 36, 36, 36],
                }
            )
        )

        self.assertEqual(valid_lap_numbers(frame), [1])
        self.assertAlmostEqual(
            lap_time_seconds(frame[frame["Lap Number"] == 1]),
            10.0,
        )

    def test_prefers_lap_timer_over_session_elapsed_time(self):
        lap = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1],
                    "Lap Distance": [0, 50, 100],
                    "Ground Speed": [36, 36, 36],
                    "Time": [10, 15, 20],
                    "Lap Time": [0, 2.4, 4.8],
                }
            )
        )

        self.assertAlmostEqual(lap_time_seconds(lap), 4.8)

    def test_track_length_excludes_overlong_and_incomplete_laps(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Time": [0, 1, 2, 3, 4, 5],
                    "Lap Number": [0, 0, 1, 1, 2, 2],
                    "Lap Distance": [0, 8167, 0, 5500, 0, 200],
                    "Ground Speed": [100] * 6,
                    "Track": ["Sepang"] * 6,
                }
            )
        )

        clean, report = process_session(frame, rolling_median_window=1)

        self.assertEqual(report["laps"]["complete"], [0.0])
        self.assertEqual(report["laps"]["excluded_overlength"], [])
        self.assertEqual(report["laps"]["excluded_incomplete"], [1.0])
        self.assertEqual(report["rows"]["distance_outlier_quarantined"], 1)
        self.assertEqual(valid_lap_numbers(clean), [0])

    def test_session_report_accounts_for_bad_rows_duplicates_and_partial_laps(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 1, 2, 2, 3],
                    "Lap Distance": [0, 50, 100, 50, 0, 50, 0],
                    "Ground Speed": [36, 36, 36, 36, 36, 36, None],
                    "Throttle Pos": [0, 50, 100, 50, 0, 50, 0],
                    "Brake Temp FL": [26] * 7,
                }
            )
        )

        clean, report = process_session(frame)

        self.assertEqual(len(clean), 4)
        self.assertEqual(report["rows"]["invalid_required_data"], 1)
        self.assertEqual(report["rows"]["repeated_distance_samples"], 1)
        self.assertEqual(report["laps"]["complete"], [1.0])
        self.assertEqual(report["laps"]["excluded_incomplete"], [2.0])
        self.assertEqual(report["fastest_complete_laps"][0]["duration_source"], "estimated_from_distance_and_speed")
        self.assertEqual(
            report["channels"]["optional_missing_percent"]["Throttle Pos"],
            0.0,
        )
        self.assertEqual(
            report["channels"]["constant_channels"]["Brake Temp FL"]["value"],
            26.0,
        )
        self.assertNotIn(
            "Lap Number",
            report["channels"]["constant_channels"],
        )

    def test_zero_phase_filter_and_outlier_quarantine_are_reported(self):
        count = 101
        lateral = [0.0] * count
        lateral[50] = 3.6
        lateral[51] = 1.0
        speed = [36.0] * count
        speed[20] = 351.0
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Time": [index / 100 for index in range(count)],
                    "Lap Number": [1] * count,
                    "Lap Distance": list(range(count)),
                    "Ground Speed": speed,
                    "G Force Lat": lateral,
                }
            )
        )

        clean, report = process_session(frame)

        self.assertEqual(report["rows"]["speed_outlier_quarantined"], 1)
        self.assertEqual(report["rows"]["lateral_g_outlier_quarantined"], 1)
        self.assertAlmostEqual(
            report["signal_filter"]["laps"]["1"]["sample_rate_hz"],
            100,
        )
        self.assertTrue(
            report["signal_filter"]["laps"]["1"]["channels"]["G Force Lat"][
                "filtered_segments"
            ]
        )
        self.assertTrue(clean["G Force Lat"].isna().any())

    def test_zero_phase_filter_keeps_a_centered_impulse_centered(self):
        count = 101
        lateral = [0.0] * count
        lateral[50] = 2.0
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Time": [index / 100 for index in range(count)],
                    "Lap Number": [1] * count,
                    "Lap Distance": list(range(count)),
                    "Ground Speed": [36.0] * count,
                    "G Force Lat": lateral,
                }
            )
        )

        clean, _ = process_session(frame)

        self.assertEqual(
            clean["G Force Lat"].to_numpy().argmax(),
            50,
        )

    def test_rolling_median_window_is_read_from_shared_config(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * 7,
                    "Lap Distance": list(range(7)),
                    "Ground Speed": [36, 36, 36, 100, 36, 36, 36],
                }
            )
        )

        clean, report = process_session(frame)

        self.assertEqual(
            report["signal_filter"]["rolling_median_window"],
            3,
        )
        self.assertEqual(clean["Ground Speed"].tolist(), [36.0] * 7)


if __name__ == "__main__":
    unittest.main()
