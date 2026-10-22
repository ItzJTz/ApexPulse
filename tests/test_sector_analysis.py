import unittest
from unittest.mock import patch

import pandas as pd

from src.core.sector_analysis import _sector_time, analyze_sectors
from src.pipeline.telemetry_io import prepare_telemetry_frame


class SectorAnalysisTests(unittest.TestCase):
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
