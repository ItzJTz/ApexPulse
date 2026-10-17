import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.telemetry_io import (
    fastest_lap_numbers,
    lap_time_seconds,
    load_telemetry,
    prepare_telemetry_frame,
    process_session,
    valid_lap_numbers,
)


class TelemetryIoTests(unittest.TestCase):
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
        self.assertEqual(fastest_lap_numbers(frame), [2, 1])

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

    def test_session_report_accounts_for_bad_rows_duplicates_and_partial_laps(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 1, 2, 2, 3],
                    "Lap Distance": [0, 50, 100, 50, 0, 50, 0],
                    "Ground Speed": [36, 36, 36, 36, 36, 36, None],
                    "Throttle Pos": [0, 50, 100, 50, 0, 50, 0],
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


if __name__ == "__main__":
    unittest.main()
