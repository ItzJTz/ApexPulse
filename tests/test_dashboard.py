import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src import config
from src.core.driver_inputs import driver_input_profile
from src.visualization.dashboard import (
    _aero_sensor_summary,
    _gg_figure,
    _track_figure,
)


class DashboardTests(unittest.TestCase):
    def test_aero_sensor_summary_is_compact_and_labels_units(self):
        summary = _aero_sensor_summary(
            {
                "front": {
                    "source": "Suspension Travel",
                    "unit": "mm",
                    "baseline": 46.92345,
                    "high_speed_relative_median": -10.2499,
                },
                "rear": None,
            }
        )

        self.assertEqual(
            summary.columns.tolist(),
            [
                "Axle",
                "Sensor",
                "Unit",
                "Low-speed baseline",
                "High-speed median change",
            ],
        )
        self.assertEqual(summary.to_dict("records"), [{
            "Axle": "Front",
            "Sensor": "Suspension Travel",
            "Unit": "mm",
            "Low-speed baseline": 46.92345,
            "High-speed median change": -10.2499,
        }])

    def test_gg_chart_handles_string_and_invalid_acceleration_samples(self):
        lap = pd.DataFrame(
            {
                "G Force Lat": ["0.1", "0.8", "invalid", "inf"],
                "G Force Long": ["0.2", "-0.5", "0", "0"],
            }
        )

        figure = _gg_figure(lap)

        self.assertIsNotNone(figure)
        self.assertEqual(len(figure.data), 1)
        density = np.asarray(figure.data[0].z, dtype=float)
        self.assertTrue(np.isnan(density).any())
        self.assertTrue(np.isfinite(density).any())
        positive_acceleration_bin = (
            np.abs(np.asarray(figure.data[0].x) - 0.1).argmin(),
            np.abs(np.asarray(figure.data[0].y) - 0.2).argmin(),
        )
        negative_braking_bin = (
            np.abs(np.asarray(figure.data[0].x) - 0.8).argmin(),
            np.abs(np.asarray(figure.data[0].y) + 0.5).argmin(),
        )
        self.assertTrue(
            np.isfinite(
                density[
                    positive_acceleration_bin[1],
                    positive_acceleration_bin[0],
                ]
            )
        )
        self.assertTrue(
            np.isfinite(
                density[
                    negative_braking_bin[1],
                    negative_braking_bin[0],
                ]
            )
        )

    def test_track_map_labels_reconstructed_path_as_approximate(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": [0, 10, 20, 30],
                "Ground Speed": [100, 110, 90, 120],
                "G Force Lat": [0.0, 0.5, -0.4, 0.0],
            }
        )

        figure, method = _track_figure(lap)

        self.assertIsNotNone(figure)
        self.assertIn("Approximate", method)
        self.assertIn("not surveyed geometry", figure.layout.title.text)

    def test_track_map_uses_logged_vehicle_coordinates(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": [0, 10, 20, 30],
                "Ground Speed": [100, 110, 90, 120],
                "Car Coord X": [0, 10, 12, 0],
                "Car Coord Y": [0, 1, 12, 10],
                "G Force Lat": [0.0, 0.5, -0.4, 0.0],
            }
        )

        figure, method = _track_figure(lap)

        self.assertIsNotNone(figure)
        self.assertEqual(method, "Logged vehicle coordinates")
        self.assertEqual(list(figure.data[0].x), [0, 10, 12, 0])
        self.assertEqual(figure.data[0].mode, "markers")
        self.assertIn("not surveyed geometry", figure.layout.title.text)

    def test_track_map_keeps_gps_track_markers_without_connector_lines(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": [0, 10, 20, 30],
                "Ground Speed": [100, 110, 90, 120],
                "GPS Latitude": [3.0, 3.001, 3.002, 3.003],
                "GPS Longitude": [101.0, 101.001, 101.002, 101.003],
            }
        )

        figure, method = _track_figure(lap)

        self.assertEqual(method, "Logged GPS coordinates")
        self.assertEqual(figure.data[0].mode, "markers")
        self.assertNotIn("lines", figure.data[0].mode)

    def test_track_map_marks_approximate_corner_apex(self):
        lap = pd.DataFrame(
            {
                "Normalized_Dist": [0, 10, 20, 30],
                "Ground Speed": [100, 110, 90, 120],
                "Car Coord X": [0, 10, 12, 0],
                "Car Coord Y": [0, 1, 12, 10],
            }
        )
        corners = [
            {
                "id": "T1",
                "name": "First turn",
                "start_m": 8,
                "apex_m": 15,
                "end_m": 20,
            }
        ]

        figure, _ = _track_figure(lap, corners)

        self.assertEqual(figure.data[-1].name, "Corners")
        self.assertEqual(list(figure.data[-1].text), ["T1"])
        self.assertEqual(list(figure.data[-1].x), [11])

    def test_dashboard_renders_all_tabs_without_runtime_exceptions(self):
        app_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "visualization"
            / "dashboard.py"
        )
        app = AppTest.from_file(str(app_path)).run(timeout=120)

        self.assertEqual(len(app.exception), 0)
        self.assertIn(
            "Input device profile",
            [selectbox.label for selectbox in app.selectbox],
        )
        self.assertIn(
            "Abrupt brake-release threshold (%/s)",
            [widget.label for widget in app.number_input],
        )
        self.assertEqual(
            [tab.label for tab in app.tabs],
            [
                "Overview",
                "Driver inputs",
                "Vehicle dynamics",
                "G-G",
                "Track map",
                "Sectors",
                "Data quality",
                "Driver comparison",
                "Optimal lap",
                "Tire analytics",
            ],
        )

    def test_dashboard_saves_and_applies_input_profile_edits(self):
        app_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "visualization"
            / "dashboard.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "config.local.yaml"
            with patch.object(config, "LOCAL_CONFIG_PATH", local_path):
                config.load_config.cache_clear()
                app = AppTest.from_file(str(app_path)).run(timeout=120)
                app.selectbox[0].select("LOAD_CELL_WHEEL").run(timeout=120)
                next(
                    widget
                    for widget in app.number_input
                    if widget.label == "Loaded throttle-snap threshold (%/s)"
                ).set_value(1250.0)
                next(
                    button
                    for button in app.button
                    if button.label == "Save and apply profile"
                ).click()
                app.run(timeout=120)

                self.assertEqual(len(app.exception), 0)
                self.assertTrue(local_path.is_file())
                self.assertEqual(
                    config.config_value("driver_inputs", "active_profile"),
                    "LOAD_CELL_WHEEL",
                )
                self.assertEqual(
                    driver_input_profile()["throttle_snap_slope_pct_s"],
                    1250.0,
                )
            config.load_config.cache_clear()


if __name__ == "__main__":
    unittest.main()
