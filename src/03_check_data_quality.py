from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT / "data" / "processed" / "combined_cleaned_data.csv"
OUTPUT_FILE = PROJECT / "data" / "processed" / "data_quality_report.csv"

SENSOR_COLUMNS = [
    "Xg", "Yg", "Zg",
    "Xdeg", "Ydeg", "Zdeg",
]


# ------------------------------------------------------------
# LONGEST REPEATED SENSOR RUN
# ------------------------------------------------------------

def longest_repeat_run(data):
    same_as_previous = (
        data[SENSOR_COLUMNS]
        .eq(data[SENSOR_COLUMNS].shift())
        .all(axis=1)
    )

    longest = 1
    current = 1

    for repeated in same_as_previous.iloc[1:]:
        if repeated:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


# ------------------------------------------------------------
# CHECK ONE RECORDING
# ------------------------------------------------------------

def check_recording(data):
    data = data.sort_values("elapsed_seconds").copy()

    rows = len(data)

    duration = (
        data["elapsed_seconds"].max()
        - data["elapsed_seconds"].min()
    )

    # Best rate for this BLE dataset:
    # total samples divided by total recording time.
    if duration > 0:
        effective_rate = (rows - 1) / duration
    else:
        effective_rate = 0

    data["interval_ms"] = (
        data["elapsed_seconds"].diff() * 1000
    )

    large_gaps = int(
        (data["interval_ms"] > 120).sum()
    )

    gap_percent = (
        large_gaps / max(rows - 1, 1) * 100
    )

    missing_values = int(
        data[
            ["elapsed_seconds"] + SENSOR_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    longest_repeat = longest_repeat_run(data)

    warnings = []

    if missing_values > 0:
        warnings.append("missing values found")

    if effective_rate < 20 or effective_rate > 30:
        warnings.append("effective rate far from 25 Hz")

    if longest_repeat >= 25:
        warnings.append(
            "sensor values repeated for at least 1 second"
        )

    quality = "PASS" if len(warnings) == 0 else "WARNING"

    return {
        "rows": rows,
        "duration_seconds": round(duration, 2),
        "effective_rate_hz": round(effective_rate, 2),
        "large_gap_count": large_gaps,
        "large_gap_percent": round(gap_percent, 2),
        "timing_jitter_found": large_gaps > 0,
        "missing_value_count": missing_values,
        "longest_identical_run_rows": longest_repeat,
        "max_acceleration_g": round(
            data["acceleration_magnitude_g"].max(),
            3,
        ),
        "max_gyroscope_dps": round(
            data["gyroscope_magnitude_dps"].max(),
            3,
        ),
        "quality": quality,
        "warning_reason": " | ".join(warnings),
    }


# ------------------------------------------------------------
# RUN ALL RECORDINGS
# ------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        print("ERROR: combined_cleaned_data.csv was not found.")
        print("Run 02_clean_all_data.py first.")
        return

    data = pd.read_csv(INPUT_FILE)

    needed = [
        "worker",
        "recording_type",
        "source_file",
        "elapsed_seconds",
        "acceleration_magnitude_g",
        "gyroscope_magnitude_dps",
        *SENSOR_COLUMNS,
    ]

    missing = [
        column
        for column in needed
        if column not in data.columns
    ]

    if missing:
        print("ERROR: Missing columns:", missing)
        return

    report_rows = []

    print("=" * 60)
    print("GROUP 2 DATA QUALITY CHECK")
    print("=" * 60)

    groups = data.groupby(
        ["worker", "recording_type", "source_file"],
        sort=True,
    )

    for group_name, recording in groups:
        worker, recording_type, source_file = group_name

        result = check_recording(recording)

        result["worker"] = worker
        result["recording_type"] = recording_type
        result["source_file"] = source_file

        report_rows.append(result)

        print()
        print("File:", source_file)
        print("Worker:", worker)
        print("Recording:", recording_type)
        print("Rate:", result["effective_rate_hz"], "Hz")
        print("Large gaps:", result["large_gap_count"])
        print(
            "Longest identical run:",
            result["longest_identical_run_rows"],
            "rows",
        )
        print("Quality:", result["quality"])

    report = pd.DataFrame(report_rows)

    first_columns = [
        "source_file",
        "worker",
        "recording_type",
    ]

    other_columns = [
        column
        for column in report.columns
        if column not in first_columns
    ]

    report = report[
        first_columns + other_columns
    ]

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    passed = int(
        (report["quality"] == "PASS").sum()
    )

    warnings = int(
        (report["quality"] == "WARNING").sum()
    )

    print()
    print("=" * 60)
    print("QUALITY CHECK FINISHED")
    print("Total recordings:", len(report))
    print("Passed:", passed)
    print("Warnings:", warnings)
    print("Report saved to:")
    print(OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()