import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core import math_channels
from src.core.math_channels import tire_metrics
from src.pipeline.telemetry_io import prepare_telemetry_frame


class TireMetricsTests(unittest.TestCase):
    def test_math_channels_uses_configured_vehicle_parameters(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1],
                    "Lap Distance": [0, 10, 20],
                    "Ground Speed": [36, 72, 108],
                    "Brake Pos": [0.049, 0.05, 0.051],
                }
            )
        )
        values = {
            "brake_active_threshold": 0.05,
            "default_brake_bias": 61.0,
            "aero_downforce_mult": 0.9,
            "friction_circle_exponent": 1.5,
        }

        with (
            patch.object(math_channels, "load_telemetry", return_value=frame),
            patch.object(math_channels, "process_session", return_value=(frame, {})),
            patch.object(math_channels, "fastest_lap_numbers", return_value=[1]),
            patch.object(
                math_channels,
                "config_value",
                side_effect=lambda section, key: values[key],
            ),
            patch.object(pd.DataFrame, "to_csv", autospec=True) as write_csv,
            patch.object(math_channels.plt, "savefig"),
            patch.object(math_channels.plt, "show"),
        ):
            math_channels.generate_math_channels("unused.csv")
            result = write_csv.call_args.args[0]

        np.testing.assert_allclose(result["Math_BrakeBias"], [0.0, 0.0, 61.0])
        np.testing.assert_allclose(
            result["Math_AeroDownforce_N"],
            [0.9 * 10**2, 0.9 * 20**2, 0.9 * 30**2],
        )

    def test_computes_conditional_thermal_envelope_and_work_metrics(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [2] * 4 + [25] * 4,
                    "Lap Distance": [0, 10, 20, 30] * 2,
                    "Ground Speed": [36] * 8,
                    "Steering": [0, 1, -1, 0] * 2,
                    "G Force Lat": [0, 1, -1, 0, 0, 0.8, -0.8, 0],
                    "G Force Long": [
                        -1,
                        0.2,
                        0.2,
                        -1,
                        -0.8,
                        0.16,
                        0.16,
                        -0.8,
                    ],
                    "Tyre Temp FL Inner": [100] * 8,
                    "Tyre Temp FL Outer": [90] * 8,
                }
            )
        )

        result = tire_metrics(frame)

        self.assertIsNotNone(result["lap_2_to_25_area_loss_percent"])
        self.assertGreater(result["lap_2_to_25_area_loss_percent"], 0)
        self.assertTrue(result["tire_work_integral_by_lap"][2.0] > 0)
        self.assertTrue(result["thermal_gradient"]["gradient_c"].tolist() == [10] * 8)
        self.assertIn(2.0, result["friction_circle_area_g2_by_lap"])

    def test_reports_missing_channels_instead_of_inventing_tire_metrics(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1],
                    "Lap Distance": [0, 10],
                    "Ground Speed": [36, 36],
                }
            )
        )

        result = tire_metrics(frame)

        self.assertIsNone(result["lap_2_to_25_area_loss_percent"])
        self.assertIn("thermal_gradient", result["unavailable"])
        self.assertIn("friction_circle_area", result["unavailable"])
        self.assertIn("tire_work_integral", result["unavailable"])


if __name__ == "__main__":
    unittest.main()
