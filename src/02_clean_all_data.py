from pathlib import Path
import re
import shutil
import pandas as pd

SAMPLE_RATE = 25
PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data" / "processed"
WORKERS_OUT = OUT / "cleaned_workers"

SENSORS = ["Xg", "Yg", "Zg", "Xdeg", "Ydeg", "Zdeg"]

COLUMN_MAP = {
    "timestamp": "timestamp", "time": "timestamp", "datetime": "timestamp",
    "elapsedseconds": "elapsed_seconds", "elapsedtime": "elapsed_seconds",
    "elapsed": "elapsed_seconds",
    "xg": "Xg", "accx": "Xg", "accelx": "Xg", "ax": "Xg",
    "yg": "Yg", "accy": "Yg", "accely": "Yg", "ay": "Yg",
    "zg": "Zg", "accz": "Zg", "accelz": "Zg", "az": "Zg",
    "xdeg": "Xdeg", "xdps": "Xdeg", "gyrox": "Xdeg",
    "ydeg": "Ydeg", "ydps": "Ydeg", "gyroy": "Ydeg",
    "zdeg": "Zdeg", "zdps": "Zdeg", "gyroz": "Zdeg",
}


def simple(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def get_action(file):
    name = simple(file.stem)

    if "nearmiss" in name:
        return "near_miss"

    if "fall" in name or "die" in name:
        return "fall_like"

    return "normal_work"


def fix_columns(data):
    rename = {}

    for old_name in data.columns:
        name = simple(old_name)

        if name in COLUMN_MAP:
            rename[old_name] = COLUMN_MAP[name]

    data = data.rename(columns=rename)
    return data.loc[:, ~data.columns.duplicated()]


def clean_file(file):
    worker = file.parent.name.lower().replace(" ", "_")
    action = get_action(file)

    data = pd.read_csv(file)
    rows_before = len(data)

    data = data.dropna(how="all")
    data = fix_columns(data)

    missing = [
        column for column in SENSORS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    time_rebuilt = False

    if "elapsed_seconds" not in data.columns:
        data["elapsed_seconds"] = [
            row / SAMPLE_RATE
            for row in range(len(data))
        ]
        time_rebuilt = True

    if "timestamp" not in data.columns:
        data["timestamp"] = ""

    number_columns = ["elapsed_seconds"] + SENSORS

    for column in number_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    bad_rows = data[number_columns].isna().any(axis=1)
    missing_rows = int(bad_rows.sum())
    data = data.loc[~bad_rows].copy()

    data = data.sort_values("elapsed_seconds")

    duplicate_rows = int(
        data.duplicated(
            subset=number_columns
        ).sum()
    )

    data = data.drop_duplicates(
        subset=number_columns
    ).copy()

    if data.empty:
        raise ValueError("No usable rows left")

    data["acceleration_magnitude_g"] = (
        data["Xg"] ** 2
        + data["Yg"] ** 2
        + data["Zg"] ** 2
    ) ** 0.5

    data["gyroscope_magnitude_dps"] = (
        data["Xdeg"] ** 2
        + data["Ydeg"] ** 2
        + data["Zdeg"] ** 2
    ) ** 0.5

    data["interval_ms"] = (
        data["elapsed_seconds"].diff() * 1000
    )

    large_gaps = int(
        (data["interval_ms"] > 120).sum()
    )

    data.insert(0, "worker", worker)
    data.insert(1, "recording_type", action)
    data.insert(2, "source_file", file.name)
    data = data.reset_index(drop=True)

    gaps = data["interval_ms"].dropna()
    gaps = gaps[gaps > 0]

    actual_rate = (
        1000 / gaps.median()
        if len(gaps) > 0
        else 0
    )

    warning = (
        missing_rows > 0
        or duplicate_rows > 0
        or large_gaps > 0
        or time_rebuilt
    )

    summary = {
        "source_file": file.name,
        "worker": worker,
        "recording_type": action,
        "rows_before": rows_before,
        "rows_after": len(data),
        "rows_removed": rows_before - len(data),
        "missing_rows_removed": missing_rows,
        "duplicate_rows_removed": duplicate_rows,
        "time_rebuilt": time_rebuilt,
        "actual_rate_hz": round(actual_rate, 2),
        "large_gap_count": large_gaps,
        "quality": "WARNING" if warning else "PASS",
        "status": "accepted",
        "error": "",
    }

    return data, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if WORKERS_OUT.exists():
        shutil.rmtree(WORKERS_OUT)

    WORKERS_OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW.rglob("*.csv"))
    cleaned_files = []
    summaries = []

    print("GROUP 5 DATA CLEANING")
    print("Files found:", len(files))

    for file in files:
        print("\nCleaning:", file.name)

        try:
            data, summary = clean_file(file)

            worker_folder = WORKERS_OUT / summary["worker"]
            worker_folder.mkdir(parents=True, exist_ok=True)

            data.to_csv(
                worker_folder / f"cleaned_{file.name}",
                index=False,
            )

            cleaned_files.append(data)
            summaries.append(summary)

            print("Accepted:", summary["rows_after"], "rows")
            print("Quality:", summary["quality"])

        except Exception as error:
            summaries.append({
                "source_file": file.name,
                "worker": file.parent.name.lower().replace(" ", "_"),
                "recording_type": get_action(file),
                "rows_before": 0,
                "rows_after": 0,
                "rows_removed": 0,
                "missing_rows_removed": 0,
                "duplicate_rows_removed": 0,
                "time_rebuilt": False,
                "actual_rate_hz": 0,
                "large_gap_count": 0,
                "quality": "REJECTED",
                "status": "rejected",
                "error": str(error),
            })

            print("Rejected:", error)

    if cleaned_files:
        pd.concat(
            cleaned_files,
            ignore_index=True,
        ).to_csv(
            OUT / "combined_cleaned_data.csv",
            index=False,
        )

    pd.DataFrame(summaries).to_csv(
        OUT / "cleaning_summary.csv",
        index=False,
    )

    accepted = sum(
        row["status"] == "accepted"
        for row in summaries
    )

    rejected = sum(
        row["status"] == "rejected"
        for row in summaries
    )

    warnings = sum(
        row["quality"] == "WARNING"
        for row in summaries
    )

    print("\nFINISHED")
    print("Total files:", len(files))
    print("Accepted files:", accepted)
    print("Files with warnings:", warnings)
    print("Rejected files:", rejected)


if __name__ == "__main__":
    main()