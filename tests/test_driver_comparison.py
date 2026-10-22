import unittest
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization import driver_comparison
from src.pipeline.telemetry_io import prepare_telemetry_frame


def _telemetry_frame(speed):
    return prepare_telemetry_frame(
        pd.DataFrame(
            {
                "Lap Number": [1, 1, 1, 1],
                "Lap Distance": [0, 0, 50, 100],
                "Ground Speed": [speed, speed, speed, speed],
            }
        )
    )


class DriverComparisonTests(unittest.TestCase):
    def test_extract_fastest_lap_selects_fastest_complete_lap(self):
        frame = prepare_telemetry_frame(
            pd.DataFrame(
                {
                    "Lap Number": [1, 1, 1, 2, 2, 2],
                    "Lap Distance": [0, 50, 100, 0, 50, 100],
                    "Ground Speed": [36, 36, 36, 72, 72, 72],
                }
            )
        )

        lap, distance_column, speed_column, lap_number = (
            driver_comparison.extract_fastest_lap(frame)
        )

        self.assertEqual(lap_number, 2)
        self.assertEqual(distance_column, "Normalized_Dist")
        self.assertEqual(speed_column, "Ground Speed")
        self.assertEqual(lap["Normalized_Dist"].iloc[0], 0)
        self.assertTrue((lap["Ground Speed"] == 72).all())

    def test_compare_two_drivers_handles_repeated_distance_samples(self):
        frame_a = _telemetry_frame(36)
        frame_b = _telemetry_frame(72)

        with tempfile.TemporaryDirectory() as output_directory:
            output_path = f"{output_directory}/comparison.png"
            with (
                patch.object(driver_comparison, "load_file", side_effect=[frame_a, frame_b]),
                patch.object(driver_comparison.plt, "savefig") as savefig,
                patch.object(driver_comparison.plt, "show"),
            ):
                driver_comparison.compare_two_drivers(
                    "driver_a.csv",
                    "driver_b.csv",
                    output_path=output_path,
                )
                figure = plt.gcf()

            self.assertEqual(figure.axes[0].lines[0].get_ydata()[0], 36)
            self.assertEqual(figure.axes[0].lines[1].get_ydata()[0], 72)
            self.assertEqual(len(figure.axes[0].lines[0].get_xdata()), 101)
            savefig.assert_called_once()
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
