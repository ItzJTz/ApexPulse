import unittest

import numpy as np
import pandas as pd

from src.core.session_filters import (
    classify_session_laps,
    lap_environment_summary,
)


class SessionFilterTests(unittest.TestCase):
    def test_tags_launch_slow_and_incident_laps(self):
        rows = []
        for lap_number, duration in ((1, 100), (2, 100), (3, 120)):
            for index, distance in enumerate((0, 500, 1000)):
                rows.append(
                    {
                        "Lap Number": lap_number,
                        "Lap Distance": distance,
                        "Lap Start Distance": 0,
                        "Normalized_Dist": distance,
                        "Ground Speed": (
                            (0, 25, 30)[index] if lap_number == 1 else 100
                        ),
                        "Lap Time": duration,
                        "Lap Invalidated": 1 if lap_number == 2 and index == 1 else 0,
                    }
                )
        frame = pd.DataFrame(rows)

        result = classify_session_laps(frame)["lap_tags"]

        self.assertTrue(result[1.0]["launch_lap"])
        self.assertTrue(result[1.0]["excluded_from_baselines"])
        self.assertTrue(result[2.0]["incident"])
        self.assertTrue(result[3.0]["slow_lap"])
        self.assertTrue(result[3.0]["excluded_from_degradation"])

    def test_tire_off_track_channel_alone_does_not_exclude_a_lap(self):
        frame = pd.DataFrame(
            {
                "Lap Number": [1] * 4 + [2] * 4,
                "Lap Distance": [0, 10, 20, 30] * 2,
                "Lap Start Distance": [0] * 8,
                "Normalized_Dist": [0, 10, 20, 30] * 2,
                "Ground Speed": [100] * 8,
                "Lap Time": [100] * 8,
                "Num Tires Off Track": [0] * 4 + [2] * 4,
            }
        )

        result = classify_session_laps(frame)["lap_tags"]

        self.assertFalse(result[2.0]["incident"])
        self.assertFalse(result[2.0]["excluded_from_baselines"])

    def test_slow_hairpin_requires_off_track_evidence_to_be_an_incident(self):
        frame = pd.DataFrame(
            {
                "Lap Number": [1] * 3,
                "Lap Distance": [0, 500, 1000],
                "Lap Start Distance": [0] * 3,
                "Normalized_Dist": [0, 500, 1000],
                "Ground Speed": [100, 30, 100],
                "Steering": [0, 20, 0],
                "Num Tires Off Track": [0, 0, 0],
                "Lap Time": [100] * 3,
            }
        )

        self.assertFalse(classify_session_laps(frame)["lap_tags"][1.0]["incident"])

        frame.loc[1, "Num Tires Off Track"] = 2
        self.assertTrue(classify_session_laps(frame)["lap_tags"][1.0]["incident"])

    def test_tags_a_flat_pit_speed_run_with_repeated_throttle_cuts(self):
        count = 101
        frame = pd.DataFrame(
            {
                "Lap Number": np.ones(count),
                "Lap Distance": np.linspace(0, 1000, count),
                "Lap Start Distance": np.zeros(count),
                "Normalized_Dist": np.linspace(0, 1000, count),
                "Ground Speed": np.full(count, 70.0),
                "Time": np.arange(count, dtype=float) / 10,
                "Throttle Pos": np.tile([60.0, 60.0, 0.0, 0.0], 26)[:count],
                "Lap Time": np.full(count, 100.0),
            }
        )

        result = classify_session_laps(frame)["lap_tags"][1.0]

        self.assertTrue(result["pit_limiter"])
        self.assertIn(
            "pit_limiter_speed_band_with_throttle_or_ignition_cuts",
            result["reasons"],
        )

    def test_reports_fuel_buckets_and_track_conditions_by_lap(self):
        frame = pd.DataFrame(
            {
                "Lap Number": [1, 1, 1],
                "Lap Distance": [0, 500, 1000],
                "Lap Start Distance": [0, 0, 0],
                "Normalized_Dist": [0, 500, 1000],
                "Ground Speed": [100, 100, 100],
                "Fuel Level": [10, 12, 14],
                "Surface Grip": [96, 96, 97],
                "Road Temp": [28, 29, 30],
            }
        )

        summary = lap_environment_summary(frame)[1.0]

        self.assertEqual(summary["fuel_bucket"], "low")
        self.assertEqual(summary["fuel_level_mean_l"], 12.0)
        self.assertAlmostEqual(summary["surface_grip_mean_pct"], 96.33333333333333)
        self.assertEqual(summary["road_temperature_mean_c"], 29.0)


if __name__ == "__main__":
    unittest.main()
