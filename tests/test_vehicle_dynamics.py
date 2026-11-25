import json
import unittest

import numpy as np
import pandas as pd

from src.core.vehicle_dynamics import (
    aero_pitch_trace,
    brake_input_analysis,
    corner_radius_trace,
    predictive_tire_degradation,
    vehicle_dynamics_analysis,
    wheel_slip_ratio_trace,
)


class VehicleDynamicsTests(unittest.TestCase):
    def test_corner_radius_converts_speed_and_masks_low_lateral_acceleration(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": [0, 10, 20],
                "Ground Speed": [36, 36, 36],
                "G Force Lat": [1.0, -1.0, 0.01],
            }
        )

        result = corner_radius_trace(lap)

        self.assertAlmostEqual(result.loc[0, "radius_m"], 100 / 9.80665)
        self.assertAlmostEqual(result.loc[1, "radius_m"], 100 / 9.80665)
        self.assertTrue(np.isnan(result.loc[2, "radius_m"]))

    def test_wheel_slip_uses_rad_per_second_and_requires_loaded_radius(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": [0, 10, 20],
                "Ground Speed": [36, 36, 36],
                "Wheel Rot Speed FL": [20, 22, 18],
                "Tire Loaded Radius FL": [0.5, 0.5, 0.5],
            }
        )
        lap.attrs["quality"] = {
            "channel_units": {
                "Wheel Rot Speed FL": "rad/s",
                "Tire Loaded Radius FL": "m",
            }
        }

        result = wheel_slip_ratio_trace(lap)

        self.assertAlmostEqual(result.loc[0, "slip_ratio_FL_pct"], 0)
        self.assertAlmostEqual(result.loc[1, "slip_ratio_FL_pct"], 10)
        self.assertIn("channel is missing", result.attrs["unavailable"]["FR"])

        without_radius = lap.drop(columns="Tire Loaded Radius FL")
        unavailable = wheel_slip_ratio_trace(without_radius)
        self.assertNotIn("slip_ratio_FL_pct", unavailable.columns)
        self.assertIn(
            "requires a measured loaded tire radius",
            unavailable.attrs["unavailable"]["FL"],
        )

    def test_aero_trace_reports_relative_sensor_motion_without_claiming_downforce(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": [0, 10, 20, 30],
                "Ground Speed": [60, 70, 210, 220],
                "Ride Height FL": [10, 10, 8, 6],
            }
        )

        result = aero_pitch_trace(lap)

        self.assertEqual(result["front_relative"].tolist(), [0, 0, -2, -4])
        self.assertAlmostEqual(
            result.attrs["available_axles"]["front"]["high_speed_relative_median"],
            -3,
        )
        self.assertIsNone(result.attrs["available_axles"]["rear"])

    def test_brake_input_classifies_abrupt_release_and_reports_ramp(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": np.arange(8) * 10,
                "Time": np.arange(8) / 50,
                "Ground Speed": [100] * 8,
                "Brake Pos": [0, 0, 50, 100, 100, 80, 40, 0],
                "Steering": [0, 0, 0, 5, 10, 20, 30, 40],
            }
        )

        result = brake_input_analysis(lap)

        self.assertTrue(result["available"])
        self.assertEqual(result["abrupt_release_count"], 1)
        self.assertEqual(result["braking_episodes"][0]["classification"], "abrupt release")
        self.assertAlmostEqual(result["maximum_ramp_up_pct_s"], 2500)
        self.assertLess(result["steepest_release_pct_s"], -600)
        self.assertIsNotNone(
            result["braking_episodes"][0]["brake_decay_pct_per_steering_degree"]
        )

        slow_lap = lap.copy()
        slow_lap["Ground Speed"] = 40
        slow_result = brake_input_analysis(slow_lap)
        self.assertEqual(slow_result["abrupt_release_count"], 0)
        self.assertEqual(slow_result["low_speed_abrupt_release_count"], 1)

    def test_degradation_forecast_requires_fifteen_laps_and_fits_trend(self):
        rows = []
        for lap_number in range(1, 16):
            for distance in (0, 500, 1000):
                rows.append(
                    {
                        "Lap Number": lap_number,
                        "Normalized_Dist": distance,
                        "Ground Speed": 100,
                        "Lap Time": 100 + lap_number / 10,
                        "Tire Temp Core FL": 80 + lap_number,
                        "Tire Pressure FL": 25 + lap_number / 10,
                    }
                )
        frame = pd.DataFrame(rows)

        result = predictive_tire_degradation(frame)
        short_result = predictive_tire_degradation(
            frame[frame["Lap Number"] < 15]
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["lap_count"], 15)
        self.assertAlmostEqual(result["lap_time_slope_s_per_lap"], 0.1)
        self.assertEqual(result["predicted_threshold_lap"], 7)
        self.assertIn("tire_core_temperature", result["sensor_trend_slopes_per_lap"])
        self.assertFalse(short_result["available"])
        self.assertIn("At least 15 complete laps", short_result["unavailable"])

    def test_vehicle_summary_is_json_friendly_when_optional_channels_are_missing(self):
        lap = pd.DataFrame(
            {
                "Lap Number": [1, 1, 1],
                "Normalized_Dist": [0, 10, 20],
                "Ground Speed": [36, 50, 36],
                "G Force Lat": [0, 0.5, 0],
            }
        )

        result = vehicle_dynamics_analysis(lap)

        self.assertIn("corner_radius", result)
        self.assertFalse(
            result["wheel_slip_ratio"]["by_wheel"]["FL"]["available"]
        )
        self.assertEqual(
            result["aero_pitch_proxy"]["by_axle"],
            {"front": None, "rear": None},
        )
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
