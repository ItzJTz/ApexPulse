import unittest

import pandas as pd

from src.core.analysis import align_driver_sessions, align_laps, optimal_lap_analysis
from src.core.spatial import interpolate_spatial
from src.pipeline.telemetry_io import prepare_telemetry_frame


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
        self.assertEqual(result["speed_a_kmh"][:2].tolist(), [36, 54])
        self.assertAlmostEqual(result["speed_a_kmh"][2], 36)
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

    def test_optional_dtw_does_not_change_distance_delta(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 2, 2, 2],
                    "Lap Distance": [0, 50, 100, 0, 50, 100],
                    "Ground Speed": [36, 72, 36, 72, 36, 72],
                }
            )
        )

        standard = align_laps(frame, 1, 2, grid_step=25)
        diagnostic = align_laps(frame, 1, 2, grid_step=25, include_dtw=True)

        self.assertEqual(standard["delta_b_ahead_s"].tolist(), diagnostic["delta_b_ahead_s"].tolist())
        self.assertIn("path_cost", diagnostic["dtw"])
        self.assertGreater(len(diagnostic["dtw"]["distance_a_m"]), 0)

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

    def test_optimal_lap_keeps_recorded_and_theoretical_times_consistent(self):
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
        self.assertLessEqual(
            result["theoretical_best_seconds"],
            result["actual_best_seconds"],
        )
        self.assertAlmostEqual(
            result["potential_gain_seconds"],
            result["actual_best_seconds"] - result["theoretical_best_seconds"],
        )

    def test_optimal_lap_rejects_discontinuous_sector_splices(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 2, 2, 2],
                    "Lap Distance": [0, 50, 100, 0, 50, 100],
                    "Ground Speed": [100, 10, 10, 10, 30, 100],
                }
            )
        )

        result = optimal_lap_analysis(frame, micro_sector_meters=50)

        self.assertEqual(len(set(result["realizable_sector_laps"])), 1)
        for previous, following in zip(
            result["sectors"],
            result["sectors"][1:],
        ):
            self.assertLessEqual(
                abs(previous["end_speed_mps"] - following["start_speed_mps"]),
                result["max_boundary_speed_delta_mps"],
            )

    def test_spatial_interpolation_uses_cubic_spline_for_cubic_data(self):
        distances = [0, 1, 2, 3]
        values = [0, 1, 8, 27]

        result = interpolate_spatial(distances, values, [0.5, 1.5, 2.5])

        self.assertTrue(
            ((result - [0.125, 3.375, 15.625]) ** 2 < 1e-20).all()
        )


if __name__ == "__main__":
    unittest.main()
