import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import _dashboard, main


class CliTests(unittest.TestCase):
    @patch("src.cli.subprocess.call", return_value=0)
    def test_dashboard_launcher_disables_first_run_prompts(self, subprocess_call):
        self.assertEqual(_dashboard(None), 0)
        command = subprocess_call.call_args.args[0]
        self.assertIn("--server.headless=true", command)
        self.assertIn("--browser.gatherUsageStats=false", command)

    def test_clean_command_writes_clean_csv_and_both_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "session.csv"
            source.write_text(
                "Time,Distance,Ground Speed,Lap Number,Lap Distance\n"
                "0,0,36,1,0\n"
                "5,50,36,1,50\n"
                "10,100,36,1,100\n"
                "20,0,72,2,0\n"
                "22.5,50,72,2,50\n"
                "25,100,72,2,100\n",
                encoding="utf-8",
            )
            output = root / "reports"

            result = main(["clean", str(source), "--output-dir", str(output)])

            self.assertEqual(result, 0)
            self.assertTrue((output / "cleaned_session.csv").is_file())
            quality = json.loads((output / "session_quality.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["laps"]["complete"], [1.0, 2.0])
            report = json.loads(
                (output / "session_session_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["lap_comparison"]["reference_lap"], 2.0)
            self.assertEqual(report["lap_comparison"]["compared_lap"], 1.0)


if __name__ == "__main__":
    unittest.main()
