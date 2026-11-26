import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import config
from src.core.driver_inputs import driver_input_profile


class ConfigTests(unittest.TestCase):
    def setUp(self):
        config.load_config.cache_clear()

    def tearDown(self):
        config.load_config.cache_clear()

    def test_driver_input_settings_are_persisted_as_a_local_override(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "config.local.yaml"
            with patch.object(config, "LOCAL_CONFIG_PATH", local_path):
                profiles = config.config_value("driver_inputs", "profiles")
                profiles["LOAD_CELL_WHEEL"][
                    "throttle_snap_slope_pct_s"
                ] = 1250

                saved_path = config.save_driver_input_settings(
                    "LOAD_CELL_WHEEL",
                    profiles,
                )

                self.assertEqual(saved_path, local_path)
                self.assertTrue(local_path.is_file())
                self.assertEqual(
                    config.config_value("driver_inputs", "active_profile"),
                    "LOAD_CELL_WHEEL",
                )
                self.assertEqual(
                    config.config_value("driver_inputs", "profiles")[
                        "LOAD_CELL_WHEEL"
                    ]["throttle_snap_slope_pct_s"],
                    1250,
                )
                self.assertEqual(
                    driver_input_profile()["name"],
                    "LOAD_CELL_WHEEL",
                )
                self.assertEqual(
                    config.config_value("vehicle", "brake_input_threshold_pct"),
                    5.0,
                )

    def test_invalid_driver_input_settings_are_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "config.local.yaml"
            with patch.object(config, "LOCAL_CONFIG_PATH", local_path):
                profiles = config.config_value("driver_inputs", "profiles")
                profiles["G7_PRO_CONTROLLER"][
                    "brake_release_slope_pct_s"
                ] = 0

                with self.assertRaisesRegex(ValueError, "must be positive"):
                    config.save_driver_input_settings(
                        "G7_PRO_CONTROLLER",
                        profiles,
                    )

                self.assertFalse(local_path.exists())


if __name__ == "__main__":
    unittest.main()
