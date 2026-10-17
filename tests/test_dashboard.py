import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.dashboard import _gg_figure, _track_figure


class DashboardTests(unittest.TestCase):
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

    def test_dashboard_renders_all_tabs_without_runtime_exceptions(self):
        app_path = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"
        app = AppTest.from_file(str(app_path)).run(timeout=120)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            [
                "Overview",
                "Driver inputs",
                "G-G",
                "Track map",
                "Sectors",
                "Data quality",
                "Driver comparison",
                "Optimal lap",
            ],
        )


if __name__ == "__main__":
    unittest.main()
