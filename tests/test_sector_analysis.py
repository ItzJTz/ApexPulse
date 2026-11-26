import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.core.sector_analysis import (
    _sector_time,
    analyze_sectors,
    scale_track_config_to_distance,
)
from src.pipeline.telemetry_io import prepare_telemetry_frame


class SectorAnalysisTests(unittest.TestCase):
    def test_all_track_configs_have_valid_approximate_corner_gates(self):
        track_directory = Path(__file__).resolve().parents[1] / "tracks"
        configs = sorted(track_directory.glob("*.json"))

        self.assertTrue(configs)
        for config_path in configs:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            corners = config.get("corner_metadata", [])

            with self.subTest(track=config_path.stem):
                self.assertTrue(corners)
                self.assertTrue(config.get("corner_distance_quality"))
                self.assertEqual(
                    len({corner["id"] for corner in corners}),
                    len(corners),
                )
                self.assertEqual(
                    [
                        corner["apex_m"]
                        for corner in corners
                        if isinstance(corner["apex_m"], (int, float))
                    ],
                    sorted(
                        corner["apex_m"]
                        for corner in corners
                        if isinstance(corner["apex_m"], (int, float))
                    ),
                )
                for corner in corners:
                    self.assertLessEqual(0, corner["start_m"])
                    if isinstance(corner["apex_m"], (int, float)):
                        self.assertLessEqual(
                            corner["start_m"],
                            corner["apex_m"],
                        )
                        self.assertLessEqual(
                            corner["apex_m"],
                            corner["end_m"],
                        )
                    else:
                        self.assertEqual(corner["apex_status"], "flat_out")
                    self.assertLessEqual(
                        corner["end_m"],
                        config["total_length_m"],
                    )

    def test_sector_time_integrates_interpolated_boundaries(self):
        lap = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1],
                    "Lap Distance": [0, 50, 100],
                    "Ground Speed": [36, 36, 36],
                }
            )
        )

        self.assertAlmostEqual(_sector_time(lap, 25, 75), 5.0)

    def test_sector_requires_full_gate_coverage(self):
        lap = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1],
                    "Lap Distance": [0, 50],
                    "Ground Speed": [36, 36],
                }
            )
        )

        self.assertIsNone(_sector_time(lap, 25, 75))

    def test_sector_uses_elapsed_time_and_track_gates_scale_to_measured_distance(self):
        lap = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1],
                    "Lap Distance": [0, 50, 100],
                    "Ground Speed": [180, 180, 180],
                    "Time": [10, 15, 20],
                }
            )
        )
        config = {
            "total_length_m": 200,
            "sectors": {"First": [0, 100]},
            "corner_metadata": [
                {"id": "T1", "start_m": 20, "apex_m": 50, "end_m": 80}
            ],
        }

        scaled = scale_track_config_to_distance(config, 100)

        self.assertAlmostEqual(_sector_time(lap, 25, 75), 5.0)
        self.assertEqual(scaled["sectors"]["First"], [0.0, 50.0])
        self.assertEqual(
            scaled["corner_metadata"][0],
            {"id": "T1", "start_m": 10.0, "apex_m": 25.0, "end_m": 40.0},
        )
        self.assertEqual(config["sectors"]["First"], [0, 100])

    def test_unidentified_track_error_names_the_input_file(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1],
                    "Lap Distance": [0, 50],
                    "Ground Speed": [36, 36],
                }
            )
        )

        with patch("src.core.sector_analysis.load_telemetry", return_value=frame):
            with self.assertRaisesRegex(ValueError, "clean_data.csv"):
                analyze_sectors("clean_data.csv")


if __name__ == "__main__":
    unittest.main()
