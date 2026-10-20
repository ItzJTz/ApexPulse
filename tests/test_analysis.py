import unittest

import pandas as pd

from src.analysis import align_driver_sessions, align_laps, optimal_lap_analysis
from src.telemetry_io import prepare_telemetry_frame


class LapAlignmentTests(unittest.TestCase):
    def test_delta_is_positive_when_compared_lap_is_faster_and_includes_endpoint(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 2, 2],
                    "Lap Distance": [0, 10, 0, 10],
                    "Ground Speed": [36, 36, 72, 72],
                }
            )
        )

        result = align_laps(frame, 1, 2, grid_step=3)

        self.assertEqual(result["distance_m"][-1], 10)
        self.assertAlmostEqual(result["delta_b_ahead_s"][-1], 0.5)

    def test_rejects_comparing_a_lap_to_itself(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1],
                    "Lap Distance": [0, 10],
                    "Ground Speed": [36, 36],
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "different laps"):
            align_laps(frame, 1, 1)

    def test_repeated_distance_samples_are_averaged_for_spatial_comparison(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 1, 2, 2, 2],
                    "Lap Distance": [0, 5, 5, 10, 0, 5, 10],
                    "Ground Speed": [36, 36, 72, 36, 72, 72, 72],
                }
            )
        )

        result = align_laps(frame, 1, 2, grid_step=5)

        self.assertEqual(result["distance_m"].tolist(), [0, 5, 10])
        self.assertEqual(result["speed_a_kmh"].tolist(), [36, 54, 36])
        self.assertTrue(all(value == value for value in result["delta_b_ahead_s"]))

    def test_cross_session_alignment_uses_fastest_complete_laps(self):
        frame_a = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1],
                    "Lap Distance": [0, 50, 100],
                    "Ground Speed": [36, 36, 36],
                }
            )
        )
        frame_b = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 2, 2, 2],
                    "Lap Distance": [0, 50, 100, 0, 50, 100],
                    "Ground Speed": [36, 36, 36, 72, 72, 72],
                }
            )
        )

        result = align_driver_sessions(frame_a, frame_b, grid_step=50)

        self.assertEqual(result["lap_a"], 1)
        self.assertEqual(result["lap_b"], 2)
        self.assertEqual(result["distance_m"].tolist(), [0, 50, 100])
        self.assertGreater(result["delta_b_ahead_s"][-1], 0)

    def test_lap_comparison_keeps_recorded_elapsed_times(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 2, 2],
                    "Lap Distance": [0, 100, 0, 100],
                    "Ground Speed": [36, 36, 72, 72],
                    "Time": [0, 10, 0, 1],
                }
            )
        )

        result = align_laps(frame, 1, 2)

        self.assertEqual(result["lap_a_seconds"], 10)
        self.assertEqual(result["lap_b_seconds"], 1)

    def test_cross_session_alignment_rejects_known_different_tracks(self):
        frame_a = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1],
                    "Lap Distance": [0, 100],
                    "Ground Speed": [36, 36],
                }
            )
        )
        frame_b = frame_a.copy()
        frame_a["Track"] = "Silverstone"
        frame_b["Track"] = "Monza"

        with self.assertRaisesRegex(ValueError, "different tracks"):
            align_driver_sessions(frame_a, frame_b)

    def test_optimal_lap_combines_fastest_micro_sectors(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 2, 2, 2],
                    "Lap Distance": [0, 50, 100, 0, 50, 100],
                    "Ground Speed": [72, 36, 36, 36, 36, 72],
                }
            )
        )

        result = optimal_lap_analysis(frame, micro_sector_meters=50)

        self.assertEqual(len(result["sectors"]), 2)
        self.assertLess(
            result["theoretical_best_seconds"],
            result["actual_best_seconds"],
        )
        self.assertAlmostEqual(
            result["potential_gain_seconds"],
            result["actual_best_seconds"] - result["theoretical_best_seconds"],
        )

    def test_optimal_lap_reports_recorded_best_without_changing_estimated_gain(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 2, 2, 2],
                    "Lap Distance": [0, 50, 100, 0, 50, 100],
                    "Ground Speed": [36, 36, 36, 72, 72, 72],
                    "Time": [0, 4.9, 9.8, 0, 5, 10],
                }
            )
        )

        result = optimal_lap_analysis(frame, micro_sector_meters=50)

        self.assertEqual(result["actual_best_lap"], 1)
        self.assertAlmostEqual(result["actual_best_seconds"], 9.8)
        self.assertEqual(result["spatial_best_lap"], 2)
        self.assertAlmostEqual(result["spatial_best_seconds"], 5.0)
        self.assertAlmostEqual(
            result["potential_gain_seconds"],
            result["spatial_best_seconds"] - result["theoretical_best_seconds"],
        )


if __name__ == "__main__":
    unittest.main()
