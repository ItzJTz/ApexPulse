import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__:
    from .analysis import analysis_report
    from .clean_telemetry import clean_telemetry
    from .telemetry_io import fastest_lap_numbers, load_telemetry
else:
    from analysis import analysis_report
    from clean_telemetry import clean_telemetry
    from telemetry_io import fastest_lap_numbers, load_telemetry


def _clean(args):
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    cleaned_path = output_directory / f"cleaned_{input_path.stem}.csv"
    quality_path = output_directory / f"{input_path.stem}_quality.json"
    clean_telemetry(
        str(input_path),
        str(cleaned_path),
        str(quality_path),
        minimum_lap_fraction=args.minimum_lap_fraction,
    )

    report_path = Path(args.report_json) if args.report_json else output_directory / f"{input_path.stem}_session_report.json"
    cleaned_frame = load_telemetry(cleaned_path)
    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    selected_laps = fastest_lap_numbers(cleaned_frame)
    if len(selected_laps) >= 2:
        report = analysis_report(
            cleaned_frame,
            quality_report,
            selected_laps[0],
            selected_laps[1],
        )
    else:
        report = {
            "schema_version": 1,
            "source_file": str(input_path),
            "track": quality_report.get("track"),
            "data_quality": quality_report,
            "lap_comparison": None,
            "note": "At least two complete laps are required for comparative results.",
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Session report saved to '{report_path}'.")
    return 0


def _dashboard(_):
    app_path = Path(__file__).with_name("dashboard.py")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="apexpulse",
        description="Clean and explore motorsport telemetry sessions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    clean_parser = commands.add_parser(
        "clean",
        help="Clean a telemetry CSV and create a quality/session report.",
    )
    clean_parser.add_argument("input", help="Raw or already-clean telemetry CSV path.")
    clean_parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for cleaned CSV and JSON reports (default: reports).",
    )
    clean_parser.add_argument(
        "--report-json",
        help="Optional explicit session-report JSON path.",
    )
    clean_parser.add_argument(
        "--minimum-lap-fraction",
        type=float,
        default=0.9,
        help="Minimum fraction of track distance required for a valid lap (default: 0.9).",
    )
    clean_parser.set_defaults(handler=_clean)

    dashboard_parser = commands.add_parser(
        "dashboard",
        help="Start the interactive telemetry dashboard.",
    )
    dashboard_parser.set_defaults(handler=_dashboard)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"apexpulse: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
