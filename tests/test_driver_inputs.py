import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.analysis import corner_delta_breakdown
from src.core.driver_inputs import (
    analyze_driver_inputs,
    auto_detect_apexes,
    cluster_distances,
    map_distance_to_corner,
)
from src.pipeline.telemetry_io import prepare_telemetry_frame


class DriverInputTests(unittest.TestCase):
    def test_detects_driver_input_events_and_clusters_nearby_triggers(self):
        count = 500
        distance = np.arange(count, dtype=float) * 2
        brake = np.zeros(count)
        throttle = np.full(count, 50.0)
        steering = np.zeros(count)
        gear = np.full(count, 4.0)
        speed = np.full(count, 100.0)

        brake[40] = 30
        brake[42] = 30
        brake[180:184] = 20
        steering[180:184] = 20
        throttle[150:152] = [0, 20]
        steering[151] = 25
        throttle[200:203] = [90, 60, 95]
        throttle[300:331] = 0
        gear[260] = 3
        steering[260] = 25
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * count,
                    "Lap Distance": distance,
                    "Ground Speed": speed,
                    "Time": np.arange(count, dtype=float) / 50,
                    "Throttle Pos": throttle,
                    "Brake Pos": brake,
                    "Steering": steering,
                    "Gear": gear,
                }
            )
        )
        track_config = {
            "circuit_name": "Test Circuit",
            "corner_metadata": [
                {
                    "id": "T1",
                    "name": "Test corner",
                    "start_m": 100,
                    "apex_m": 200,
                    "end_m": 300,
                }
            ],
        }

        result = analyze_driver_inputs(frame, 1, track_config)

        self.assertEqual(len(result["metrics"]["double_braking_m"]), 1)
        self.assertEqual(len(result["metrics"]["abrupt_brake_release_m"]), 3)
        self.assertAlmostEqual(result["metrics"]["trail_brake_overlap_pct"], 200 / 3)
        self.assertEqual(len(result["metrics"]["loaded_throttle_snaps_m"]), 1)
        self.assertEqual(len(result["metrics"]["throttle_hesitations_m"]), 1)
        self.assertAlmostEqual(result["metrics"]["dead_coasting_sec"], 0.64)
        self.assertEqual(len(result["metrics"]["loaded_downshifts_m"]), 1)
        self.assertTrue(result["clusters"])
        self.assertEqual(result["clusters"][0]["corner"], "T1 Test corner")

    def test_cluster_distances_uses_consecutive_gap_threshold(self):
        clusters = cluster_distances([81, 40, 0, 200], max_gap_m=40)

        self.assertEqual(
            [(cluster["start_m"], cluster["end_m"]) for cluster in clusters],
            [(0, 40), (81, 81), (200, 200)],
        )

    def test_sepang_distance_mapper_names_turn_one_and_turn_fifteen(self):
        config_path = Path(__file__).resolve().parents[1] / "tracks" / "Sepang.json"
        track_config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(map_distance_to_corner(670, track_config), "T1 Pangkor Laut")
        self.assertEqual(
            map_distance_to_corner(5190, track_config),
            "T15 Petronas Hairpin",
        )
        self.assertEqual(
            [
                map_distance_to_corner(corner["apex_m"], track_config).split()[0]
                for corner in track_config["corner_metadata"]
            ],
            [f"T{index}" for index in range(1, 16)],
        )

    def test_track_agnostic_apex_detection_finds_local_speed_minima(self):
        distance = np.arange(0, 1000, 10, dtype=float)
        speed = (
            180
            - 55 * np.exp(-((distance - 300) / 45) ** 2)
            - 45 * np.exp(-((distance - 700) / 50) ** 2)
        )
        lap = pd.DataFrame(
            {"Normalized_Dist": distance, "Ground Speed": speed}
        )

        corners = auto_detect_apexes(lap)

        self.assertEqual(len(corners), 2)
        self.assertAlmostEqual(corners[0]["apex_m"], 300, delta=30)
        self.assertAlmostEqual(corners[1]["apex_m"], 700, delta=30)

    def test_corner_delta_breakdown_ranks_time_loss_against_reference_trace(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * 5 + [2] * 5,
                    "Lap Distance": [0, 25, 50, 75, 100] * 2,
                    "Ground Speed": [36] * 5 + [72] * 5,
                    "Time": [0, 2.5, 5, 7.5, 10, 0, 1.25, 2.5, 3.75, 5],
                }
            )
        )
        track_config = {
            "corner_metadata": [
                {
                    "id": "T1",
                    "name": "Test corner",
                    "start_m": 25,
                    "apex_m": 50,
                    "end_m": 75,
                }
            ]
        }

        result = corner_delta_breakdown(frame, 1, 2, track_config, 144.638)

        self.assertFalse(result["reference_matches_benchmark"])
        self.assertEqual(result["reference_benchmark_seconds"], 144.638)
        corner = result["ranked_by_time_loss"][0]
        self.assertAlmostEqual(corner["time_loss_seconds"], -2.5)
        self.assertAlmostEqual(corner["cumulative_time_loss_seconds"], -2.5)

    def test_reference_match_tolerance_respects_50_hz_sampling(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * 4 + [2] * 4,
                    "Lap Distance": [0, 10, 20, 30] * 2,
                    "Ground Speed": [72] * 8,
                    "Time": [0, 0.02, 0.04, 0.06] * 2,
                    "Lap Time": [
                        144.59,
                        144.61,
                        144.63,
                        144.65,
                        144.59,
                        144.61,
                        144.63,
                        144.65,
                    ],
                }
            )
        )
        track_config = {
            "corner_metadata": [
                {"id": "T1", "name": "Turn", "start_m": 0, "apex_m": 15, "end_m": 30}
            ]
        }

        result = corner_delta_breakdown(frame, 1, 2, track_config, 144.638)

        self.assertAlmostEqual(result["reference_lap_seconds"], 144.65)
        self.assertTrue(result["reference_matches_benchmark"])

    def test_missing_optional_input_channels_are_reported_unavailable(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1],
                    "Lap Distance": [0, 10, 20],
                    "Ground Speed": [80, 90, 100],
                }
            )
        )

        result = analyze_driver_inputs(frame, 1)

        self.assertIsNone(result["metrics"]["abrupt_brake_release_m"])
        self.assertIsNone(result["metrics"]["trail_brake_overlap_pct"])
        self.assertIsNone(result["metrics"]["dead_coasting_sec"])
        self.assertIsNone(result["metrics"]["loaded_downshifts_m"])


if __name__ == "__main__":
    unittest.main()
