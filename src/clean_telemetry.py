import json
import os
import argparse
from pathlib import Path

if __package__:
    from .telemetry_io import load_telemetry, process_session
else:
    from telemetry_io import load_telemetry, process_session


def clean_telemetry(
    raw_csv_path,
    output_csv_path=None,
    report_json_path=None,
    minimum_lap_fraction=0.9,
):
    """Normalize telemetry, filter incomplete laps, and export CSV plus QC JSON."""
    if output_csv_path is None:
        base_name = os.path.basename(raw_csv_path)
        output_csv_path = os.path.join("data", f"clean_{base_name}")

    output_csv_path = Path(output_csv_path)
    report_json_path = (
        Path(report_json_path)
        if report_json_path
        else output_csv_path.with_name(f"{output_csv_path.stem}_report.json")
    )
    telemetry, report = process_session(
        load_telemetry(raw_csv_path),
        minimum_fraction=minimum_lap_fraction,
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(output_csv_path, index=False)
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Successfully cleaned '{raw_csv_path}' -> Saved to '{output_csv_path}'")
    print(f"Data quality report saved to '{report_json_path}'")
    return str(output_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw telemetry and write a data-quality report.")
    parser.add_argument(
        "input",
        nargs="?",
        default="data/raw_telemetrySilverstone.csv",
        help="Raw telemetry CSV (default: data/raw_telemetrySilverstone.csv).",
    )
    parser.add_argument(
        "--output",
        default="data/clean_data.csv",
        help="Cleaned CSV output path (default: data/clean_data.csv).",
    )
    parser.add_argument("--report-json", help="Optional path for the quality report JSON.")
    parser.add_argument(
        "--minimum-lap-fraction",
        type=float,
        default=0.9,
        help="Minimum fraction of track distance required for a valid lap.",
    )
    arguments = parser.parse_args()
    clean_telemetry(
        arguments.input,
        arguments.output,
        arguments.report_json,
        arguments.minimum_lap_fraction,
    )
