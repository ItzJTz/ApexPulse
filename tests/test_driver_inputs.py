import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.analysis import corner_delta_breakdown
from src.core.driver_inputs import (
    analyze_driver_inputs,
    auto_detect_apexes,
    cluster_events,
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
        self.assertEqual(len(result["metrics"]["abrupt_brake_release_m"]), 0)
        self.assertAlmostEqual(result["metrics"]["trail_brake_overlap_pct"], 200 / 3)
        self.assertEqual(len(result["metrics"]["loaded_throttle_snaps_m"]), 1)
        self.assertEqual(len(result["metrics"]["throttle_hesitations_m"]), 1)
        self.assertAlmostEqual(result["metrics"]["dead_coasting_sec"], 0.64)
        self.assertEqual(len(result["metrics"]["loaded_downshifts_m"]), 1)
        self.assertTrue(result["clusters"])
        self.assertEqual(result["clusters"][0]["corner"], "T1 Test corner")
        self.assertFalse(
            any(
                event["detector"] == "abrupt_brake_release_m"
                for event in result["events"]
            )
        )

    def test_abrupt_brake_detector_uses_smoothed_release_rates(self):
        count = 500
        brake = np.zeros(count)
        brake[40:45] = [100, 75, 50, 25, 0]
        brake[180:185] = [100, 75, 50, 25, 0]
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * count,
                    "Lap Distance": np.arange(count, dtype=float) * 2,
                    "Ground Speed": np.full(count, 100.0),
                    "Time": np.arange(count, dtype=float) / 50,
                    "Throttle Pos": np.zeros(count),
                    "Brake Pos": brake,
                    "Steering": np.zeros(count),
                    "Gear": np.full(count, 4.0),
                }
            )
        )

        result = analyze_driver_inputs(frame, 1)

        self.assertEqual(len(result["metrics"]["abrupt_brake_release_m"]), 2)
        abrupt_events = [
            event
            for event in result["events"]
            if event["detector"] == "abrupt_brake_release_m"
        ]
        self.assertEqual(len(abrupt_events), 2)
        self.assertTrue(
            all(event["slope_percent_s"] <= -600 for event in abrupt_events)
        )

        low_rate_frame = frame.copy()
        low_rate_frame.attrs["quality"] = {
            **frame.attrs["quality"],
            "channel_sample_rates_hz": {"Brake Pos": 10.0},
        }
        low_rate_result = analyze_driver_inputs(low_rate_frame, 1)

        self.assertEqual(
            low_rate_result["metrics"]["abrupt_brake_release_m"],
            [],
        )

    def test_driver_brake_events_follow_time_when_distance_runs_backward(self):
        count = 30
        brake = np.r_[np.full(8, 100.0), np.zeros(count - 8)]
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * count,
                    "Lap Distance": np.arange(count, dtype=float)[::-1],
                    "Ground Speed": np.full(count, 100.0),
                    "Time": np.arange(count, dtype=float) / 50,
                    "Throttle Pos": np.zeros(count),
                    "Brake Pos": brake,
                    "Steering": np.zeros(count),
                    "Gear": np.full(count, 4.0),
                }
            )
        )

        result = analyze_driver_inputs(frame, 1)

        self.assertEqual(len(result["metrics"]["abrupt_brake_release_m"]), 1)

    def test_cluster_distances_uses_consecutive_gap_threshold(self):
        clusters = cluster_distances([81, 40, 0, 200], max_gap_m=40)

        self.assertEqual(
            [(cluster["start_m"], cluster["end_m"]) for cluster in clusters],
            [(0, 40), (81, 81), (200, 200)],
        )

    def test_ten_trigger_samples_collapse_into_four_loaded_throttle_events(self):
        distances = [
            3103.5,
            3104.3,
            3105.8,
            3274.6,
            3275.7,
            3488.6,
            3490.7,
            4066.1,
            4069.0,
            4097.5,
        ]
        triggers = [
            {"detector": "loaded_throttle_snaps_m", "distance_m": distance}
            for distance in distances
        ]

        events = cluster_events(triggers)

        self.assertEqual(len(events), 4)
        self.assertEqual([event["sample_count"] for event in events], [3, 2, 2, 3])
        self.assertEqual(
            [(event["start_m"], event["end_m"]) for event in events],
            [
                (3103.5, 3105.8),
                (3274.6, 3275.7),
                (3488.6, 3490.7),
                (4066.1, 4097.5),
            ],
        )

    def test_detector_count_and_event_table_use_clustered_trigger_samples(self):
        distance = np.arange(4100, dtype=float)
        throttle = np.zeros(len(distance))
        steering = np.zeros(len(distance))
        trigger_groups = (
            (3103, 3104, 3105),
            (3274, 3275),
            (3488, 3490),
            (4066, 4069, 4097),
        )
        for group in trigger_groups:
            throttle[group[0] - 1] = 0
            for level, index in enumerate(group, start=1):
                throttle[index] = level * 20
                steering[index] = 25
        speed = (
            180
            - 55 * np.exp(-((distance - 1000) / 45) ** 2)
            - 45 * np.exp(-((distance - 2500) / 50) ** 2)
            - 50 * np.exp(-((distance - 3600) / 50) ** 2)
        )
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * len(distance),
                    "Lap Distance": distance,
                    "Ground Speed": speed,
                    "Time": np.arange(len(distance), dtype=float) / 50,
                    "Throttle Pos": throttle,
                    "Brake Pos": np.zeros(len(distance)),
                    "Steering": steering,
                    "Gear": np.full(len(distance), 4),
                }
            )
        )

        result = analyze_driver_inputs(frame, 1)

        self.assertEqual(len(result["raw_events"]), 10)
        self.assertEqual(len(result["metrics"]["loaded_throttle_snaps_m"]), 4)
        throttle_events = [
            event
            for event in result["events"]
            if event["detector"] == "loaded_throttle_snaps_m"
        ]
        self.assertEqual(len(throttle_events), 4)
        self.assertEqual(
            [event["sample_count"] for event in throttle_events],
            [3, 2, 2, 3],
        )

    def test_sepang_distance_mapper_names_turn_one_and_turn_fifteen(self):
        config_path = Path(__file__).resolve().parents[1] / "tracks" / "Sepang.json"
        track_config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(map_distance_to_corner(670, track_config), "T1 Pangkor Laut")
        self.assertEqual(
            map_distance_to_corner(5190, track_config),
            "T15 Petronas Hairpin",
        )
        flat_out_corners = [
            corner
            for corner in track_config["corner_metadata"]
            if corner.get("apex_status") == "flat_out"
        ]
        self.assertEqual(
            [corner["id"] for corner in flat_out_corners],
            ["T3", "T6", "T8", "T11", "T14"],
        )
        self.assertEqual(
            [
                map_distance_to_corner(corner["apex_m"], track_config).split()[0]
                for corner in track_config["corner_metadata"]
                if isinstance(corner["apex_m"], (int, float))
            ],
            ["T1", "T2", "T4", "T5", "T7", "T9", "T10", "T12", "T13", "T15"],
        )
        for corner in flat_out_corners:
            midpoint = (corner["start_m"] + corner["end_m"]) / 2
            self.assertTrue(
                map_distance_to_corner(midpoint, track_config).startswith(
                    corner["id"]
                )
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

    def test_event_corner_uses_speed_minima_instead_of_sepang_distance_gates(self):
        distance = np.arange(1000, dtype=float)
        speed = (
            180
            - 55 * np.exp(-((distance - 300) / 45) ** 2)
            - 45 * np.exp(-((distance - 700) / 50) ** 2)
        )
        throttle = np.zeros(len(distance))
        throttle[295:297] = [0, 20]
        steering = np.zeros(len(distance))
        steering[296] = 25
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * len(distance),
                    "Lap Distance": distance,
                    "Ground Speed": speed,
                    "Time": np.arange(len(distance), dtype=float) / 50,
                    "Throttle Pos": throttle,
                    "Brake Pos": np.zeros(len(distance)),
                    "Steering": steering,
                    "Gear": np.full(len(distance), 4),
                }
            )
        )
        misleading_track_gates = {
            "circuit_name": "Sepang International Circuit",
            "corner_metadata": [
                {
                    "id": "T9",
                    "name": "Static wrong corner",
                    "start_m": 250,
                    "apex_m": 260,
                    "end_m": 280,
                }
            ],
        }

        result = analyze_driver_inputs(frame, 1, misleading_track_gates)

        event = result["events"][0]
        self.assertEqual(event["detector"], "loaded_throttle_snaps_m")
        self.assertEqual(event["corner"], "T1 Auto-detected corner 1")
        self.assertEqual(result["auto_detected_corners"][0]["id"], "T1")

    def test_auto_blip_after_downshift_is_ignored_but_later_snap_is_kept(self):
        count = 15
        throttle = np.zeros(count)
        throttle[3] = 30
        throttle[14] = 25
        gear = np.full(count, 4)
        gear[2:] = 3
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1] * count,
                    "Lap Distance": np.arange(count, dtype=float),
                    "Ground Speed": [100] * count,
                    "Time": np.arange(count, dtype=float) / 50,
                    "Throttle Pos": throttle,
                    "Brake Pos": np.zeros(count),
                    "Steering": np.full(count, 25),
                    "Gear": gear,
                }
            )
        )

        result = analyze_driver_inputs(frame, 1)

        snap_events = [
            event
            for event in result["raw_events"]
            if event["detector"] == "loaded_throttle_snaps_m"
        ]
        self.assertEqual(len(snap_events), 1)
        self.assertEqual(snap_events[0]["sample_index"], 14)

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

        result = corner_delta_breakdown(frame, 1, 2, track_config, 123.456)

        self.assertFalse(result["reference_matches_benchmark"])
        self.assertEqual(result["reference_benchmark_seconds"], 123.456)
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

        result = corner_delta_breakdown(frame, 1, 2, track_config, 144.650)

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
