from pathlib import Path
import re
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

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
    names = {
        "timestamp": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "elapsedseconds": "elapsed_seconds",
        "elapsedtime": "elapsed_seconds",
        "elapsed": "elapsed_seconds",
        "xg": "Xg",
        "accx": "Xg",
        "accelx": "Xg",
        "ax": "Xg",
        "yg": "Yg",
        "accy": "Yg",
        "accely": "Yg",
        "ay": "Yg",
        "zg": "Zg",
        "accz": "Zg",
        "accelz": "Zg",
        "az": "Zg",
        "xdeg": "Xdeg",
        "xdps": "Xdeg",
        "gyrox": "Xdeg",
        "ydeg": "Ydeg",
        "ydps": "Ydeg",
        "gyroy": "Ydeg",
        "zdeg": "Zdeg",
        "zdps": "Zdeg",
        "gyroz": "Zdeg",
    }
    change = {}
    for old_name in data.columns:
        clean_name = simple(old_name)
        if clean_name in names:
            change[old_name] = names[clean_name]
    data = data.rename(columns=change)
    return data.loc[:, ~data.columns.duplicated()]

files = sorted(RAW.rglob("*.csv"))
needed = ["Xg", "Yg", "Zg", "Xdeg", "Ydeg", "Zdeg"]
all_data = []
summary = []

print("CLEANING ALL WORKERS")
print("Files found:", len(files))

for file in files:
    worker = file.parent.name.lower().replace(" ", "_")
    action = get_action(file)
    print("\nCleaning:", file.name)

    try:
        data = pd.read_csv(file)
        rows_before = len(data)
        data = fix_columns(data)

        missing = [
            name for name in needed
            if name not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if "elapsed_seconds" not in data.columns:
            data["elapsed_seconds"] = [
                row / 25 for row in range(len(data))
            ]
        if "timestamp" not in data.columns:
            data["timestamp"] = ""

        number_columns = ["elapsed_seconds"] + needed
        for name in number_columns:
            data[name] = pd.to_numeric(
                data[name],
                errors="coerce"
            )

        data = data.dropna(subset=number_columns)
        data = data.drop_duplicates(subset=number_columns)

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

        data["worker"] = worker
        data["work_state"] = action
        data["source_file"] = file.name

        worker_folder = OUT / worker
        worker_folder.mkdir(parents=True, exist_ok=True)
        save_file = worker_folder / f"cleaned_{file.name}"
        data.to_csv(save_file, index=False)

        all_data.append(data)
        summary.append({
            "file": file.name,
            "worker": worker,
            "work_state": action,
            "rows_before": rows_before,
            "rows_after": len(data),
            "rows_removed": rows_before - len(data),
            "status": "accepted",
            "error": "",
        })
        print("Accepted:", len(data), "rows")

    except Exception as error:
        summary.append({
            "file": file.name,
            "worker": worker,
            "work_state": action,
            "rows_before": 0,
            "rows_after": 0,
            "rows_removed": 0,
            "status": "rejected",
            "error": str(error),
        })
        print("Rejected:", error)

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(
        OUT / "combined_cleaned_data.csv",
        index=False
    )

report = pd.DataFrame(summary)
report.to_csv(
    OUT / "cleaning_summary.csv",
    index=False
)

accepted = sum(
    item["status"] == "accepted"
    for item in summary
)
rejected = len(summary) - accepted

print("\nFINISHED")
print("Total files:", len(files))
print("Accepted files:", accepted)
print("Rejected files:", rejected)
print("Output folder:", OUT)