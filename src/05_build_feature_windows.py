from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 WINDOW + FEATURE EXTRACTION
# Statistical/rule-based analysis only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "combined_cleaned_data.csv"
LABEL_FILE = PROCESSED / "event_labels.csv"

ALL_WINDOWS_OUTPUT = PROCESSED / "window_features_all.csv"
ANALYSIS_WINDOWS_OUTPUT = PROCESSED / "window_features_analysis.csv"
SUMMARY_OUTPUT = PROCESSED / "window_feature_summary.csv"

TARGET_RATE_HZ = 25
DT = 1.0 / TARGET_RATE_HZ

WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.5
MIN_WINDOW_COVERAGE = 0.80

SENSOR_COLUMNS = [
    "Xg", "Yg", "Zg",
    "Xdeg", "Ydeg", "Zdeg",
    "acceleration_magnitude_g",
    "gyroscope_magnitude_dps",
]

REQUIRED_COLUMNS = [
    "worker",
    "recording_type",
    "source_file",
    "elapsed_seconds",
    *SENSOR_COLUMNS,
]

REQUIRED_LABEL_COLUMNS = [
    "worker",
    "source_file",
    "event_start_seconds",
    "event_end_seconds",
    "post_event_end_seconds",
    "final_event_type",
    "confirmed",
]


def vector_angle_degrees(vector_a, vector_b):
    """Return the angle between two 3D vectors in degrees."""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return np.nan

    cosine = np.dot(a, b) / (norm_a * norm_b)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def resample_recording(recording):
    """
    Interpolate one recording onto a regular 25 Hz timeline.

    This reduces BLE timestamp irregularity before jerk and window
    features are calculated. Raw/cleaned CSV files are not overwritten.
    """
    recording = (
        recording
        .sort_values("elapsed_seconds")
        .drop_duplicates(subset=["elapsed_seconds"], keep="first")
        .copy()
    )

    if len(recording) < 2:
        return pd.DataFrame()

    time_values = pd.to_numeric(
        recording["elapsed_seconds"],
        errors="coerce",
    )

    valid_time = time_values.notna()
    recording = recording.loc[valid_time].copy()
    time_values = time_values.loc[valid_time]

    if len(recording) < 2:
        return pd.DataFrame()

    start = float(time_values.min())
    end = float(time_values.max())

    if end <= start:
        return pd.DataFrame()

    regular_time = np.arange(
        start,
        end + DT / 2,
        DT,
    )

    output = pd.DataFrame({
        "elapsed_seconds": regular_time,
    })

    for column in SENSOR_COLUMNS:
        values = pd.to_numeric(
            recording[column],
            errors="coerce",
        )

        valid = values.notna()

        if valid.sum() < 2:
            output[column] = np.nan
            continue

        output[column] = np.interp(
            regular_time,
            time_values.loc[valid].to_numpy(dtype=float),
            values.loc[valid].to_numpy(dtype=float),
        )

    output["jerk_g_per_second"] = (
        output["acceleration_magnitude_g"]
        .diff()
        .abs()
        / DT
    ).fillna(0.0)

    return output


def extract_window_features(window):
    """Calculate explainable sensor features for one 1-second window."""
    acceleration = window["acceleration_magnitude_g"]
    gyroscope = window["gyroscope_magnitude_dps"]
    jerk = window["jerk_g_per_second"]

    quarter_rows = max(
        1,
        int(round(len(window) * 0.25)),
    )

    start_vector = (
        window[["Xg", "Yg", "Zg"]]
        .iloc[:quarter_rows]
        .median()
        .to_numpy(dtype=float)
    )

    end_vector = (
        window[["Xg", "Yg", "Zg"]]
        .iloc[-quarter_rows:]
        .median()
        .to_numpy(dtype=float)
    )

    orientation_change = vector_angle_degrees(
        start_vector,
        end_vector,
    )

    acceleration_sma = (
        window["Xg"].abs()
        + window["Yg"].abs()
        + window["Zg"].abs()
    ).mean()

    gyroscope_sma = (
        window["Xdeg"].abs()
        + window["Ydeg"].abs()
        + window["Zdeg"].abs()
    ).mean()

    axis_accel_variance = (
        window["Xg"].var(ddof=0)
        + window["Yg"].var(ddof=0)
        + window["Zg"].var(ddof=0)
    )

    return {
        "accel_mean_g": float(acceleration.mean()),
        "accel_std_g": float(acceleration.std(ddof=0)),
        "accel_min_g": float(acceleration.min()),
        "accel_max_g": float(acceleration.max()),
        "accel_range_g": float(
            acceleration.max() - acceleration.min()
        ),
        "gyro_mean_dps": float(gyroscope.mean()),
        "gyro_std_dps": float(gyroscope.std(ddof=0)),
        "gyro_max_dps": float(gyroscope.max()),
        "jerk_mean_g_per_second": float(jerk.mean()),
        "jerk_std_g_per_second": float(jerk.std(ddof=0)),
        "jerk_max_g_per_second": float(jerk.max()),
        "sma_g": float(acceleration_sma),
        "gyro_sma_dps": float(gyroscope_sma),
        "orientation_change_deg": orientation_change,
        "axis_accel_std_total_g": float(
            math.sqrt(max(axis_accel_variance, 0.0))
        ),
    }


def build_label_lookup(labels):
    """
    Validate manually confirmed incident labels and return a lookup.

    Normal-work recordings do not require a row in event_labels.csv.
    """
    labels = labels.copy()

    labels["confirmed"] = (
        labels["confirmed"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    labels["final_event_type"] = (
        labels["final_event_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    incomplete = labels[
        (labels["confirmed"] != "YES")
        | (labels["final_event_type"] == "")
    ]

    if not incomplete.empty:
        print()
        print("ERROR: incident labels are not fully confirmed yet.")
        print(
            incomplete[
                [
                    "worker",
                    "source_file",
                    "final_event_type",
                    "confirmed",
                ]
            ].to_string(index=False)
        )
        raise SystemExit(
            "\nFinish Step 6 first, then run Code 05 again."
        )

    lookup = {}

    for _, row in labels.iterrows():
        key = (
            str(row["worker"]),
            str(row["source_file"]),
        )

        lookup[key] = {
            "event_start_seconds": float(
                row["event_start_seconds"]
            ),
            "event_end_seconds": float(
                row["event_end_seconds"]
            ),
            "post_event_end_seconds": float(
                row["post_event_end_seconds"]
            ),
            "final_event_type": str(
                row["final_event_type"]
            ),
        }

    return lookup


def label_window(
    recording_type,
    worker,
    source_file,
    window_center,
    label_lookup,
):
    """
    Assign an analysis label using the window centre.

    Normal recordings:
      every valid window -> normal

    Incident recordings:
      confirmed event interval -> near_miss or fall
      post-fall interval -> post_fall_recovery/post_fall_inactive
      remaining incident-file windows -> excluded_background

    Excluding non-event background from incident files prevents setup,
    recovery, and intentional test movement from contaminating the
    normal baseline.
    """
    recording_type_lower = str(recording_type).lower()

    if "normal" in recording_type_lower:
        return "normal", "normal_work"

    key = (
        str(worker),
        str(source_file),
    )

    if key not in label_lookup:
        raise ValueError(
            "Missing confirmed incident label for "
            f"{worker} | {source_file}"
        )

    label = label_lookup[key]

    event_start = label["event_start_seconds"]
    event_end = label["event_end_seconds"]
    post_end = label["post_event_end_seconds"]
    event_type = label["final_event_type"]

    if event_start <= window_center <= event_end:
        if event_type == "near_miss":
            return "near_miss", "near_miss"

        if event_type in {
            "fall_recovery",
            "fall_inactive",
        }:
            return "fall", event_type

        raise ValueError(
            f"Unsupported final_event_type: {event_type}"
        )

    if event_end < window_center <= post_end:
        if event_type == "fall_recovery":
            return "post_fall_recovery", event_type

        if event_type == "fall_inactive":
            return "post_fall_inactive", event_type

        if event_type == "near_miss":
            return "post_near_miss", event_type

    return "excluded_background", event_type


def main():
    print("=" * 72)
    print("GROUP 5 - 1 SECOND WINDOW + FEATURE EXTRACTION")
    print("25 Hz | 1.0 s windows | 50% overlap | NO AI / NO ML")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"ERROR: missing cleaned data:\n{INPUT_FILE}"
        )

    if not LABEL_FILE.exists():
        raise SystemExit(
            f"ERROR: missing event labels:\n{LABEL_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)
    labels = pd.read_csv(LABEL_FILE)

    missing_data_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_data_columns:
        raise SystemExit(
            "ERROR: combined cleaned data is missing columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_data_columns
            )
        )

    missing_label_columns = [
        column
        for column in REQUIRED_LABEL_COLUMNS
        if column not in labels.columns
    ]

    if missing_label_columns:
        raise SystemExit(
            "ERROR: event_labels.csv is missing columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_label_columns
            )
        )

    label_lookup = build_label_lookup(labels)

    all_rows = []

    grouped = data.groupby(
        ["worker", "source_file", "recording_type"],
        sort=False,
        dropna=False,
    )

    for (
        worker,
        source_file,
        recording_type,
    ), recording in grouped:

        regular = resample_recording(recording)

        if regular.empty:
            print(
                "WARNING: skipped recording that could not "
                f"be resampled: {worker} | {source_file}"
            )
            continue

        recording_start = float(
            regular["elapsed_seconds"].min()
        )

        recording_end = float(
            regular["elapsed_seconds"].max()
        )

        latest_window_start = (
            recording_end - WINDOW_SECONDS
        )

        if latest_window_start < recording_start:
            print(
                "WARNING: recording shorter than 1 second: "
                f"{worker} | {source_file}"
            )
            continue

        window_starts = np.arange(
            recording_start,
            latest_window_start + STEP_SECONDS / 2,
            STEP_SECONDS,
        )

        recording_window_count = 0

        for window_number, window_start in enumerate(
            window_starts,
            start=1,
        ):
            window_end = (
                window_start + WINDOW_SECONDS
            )

            window = regular[
                (
                    regular["elapsed_seconds"]
                    >= window_start
                )
                & (
                    regular["elapsed_seconds"]
                    < window_end
                )
            ].copy()

            expected_rows = int(
                round(
                    WINDOW_SECONDS
                    * TARGET_RATE_HZ
                )
            )

            minimum_rows = int(
                math.ceil(
                    expected_rows
                    * MIN_WINDOW_COVERAGE
                )
            )

            if len(window) < minimum_rows:
                continue

            if window[SENSOR_COLUMNS].isna().any().any():
                continue

            window_center = (
                window_start
                + WINDOW_SECONDS / 2
            )

            primary_label, event_subtype = label_window(
                recording_type=recording_type,
                worker=worker,
                source_file=source_file,
                window_center=window_center,
                label_lookup=label_lookup,
            )

            features = extract_window_features(
                window
            )

            all_rows.append({
                "worker": worker,
                "source_file": source_file,
                "recording_type": recording_type,
                "window_number": window_number,
                "window_start_seconds": round(
                    float(window_start),
                    3,
                ),
                "window_end_seconds": round(
                    float(window_end),
                    3,
                ),
                "window_center_seconds": round(
                    float(window_center),
                    3,
                ),
                "samples_in_window": int(
                    len(window)
                ),
                "primary_label": primary_label,
                "event_subtype": event_subtype,
                **features,
            })

            recording_window_count += 1

        print(
            f"{worker:12s} | "
            f"{str(recording_type):10s} | "
            f"{recording_window_count:4d} windows | "
            f"{source_file}"
        )

    if not all_rows:
        raise SystemExit(
            "ERROR: no feature windows were created."
        )

    all_windows = pd.DataFrame(all_rows)

    numeric_columns = all_windows.select_dtypes(
        include=[np.number]
    ).columns

    all_windows[numeric_columns] = (
        all_windows[numeric_columns]
        .round(5)
    )

    ALL_WINDOWS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_windows.to_csv(
        ALL_WINDOWS_OUTPUT,
        index=False,
    )

    analysis_windows = all_windows[
        all_windows["primary_label"].isin(
            ["normal", "near_miss", "fall"]
        )
    ].copy()

    analysis_windows.to_csv(
        ANALYSIS_WINDOWS_OUTPUT,
        index=False,
    )

    summary = (
        all_windows
        .groupby(
            ["primary_label", "worker"],
            dropna=False,
        )
        .size()
        .reset_index(name="window_count")
        .sort_values(
            ["primary_label", "worker"]
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("=" * 72)
    print("WINDOW FEATURE EXTRACTION FINISHED")
    print("=" * 72)

    print()
    print("Main analysis label counts:")
    print(
        analysis_windows[
            "primary_label"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("All label counts:")
    print(
        all_windows[
            "primary_label"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Files created:")
    print(ALL_WINDOWS_OUTPUT)
    print(ANALYSIS_WINDOWS_OUTPUT)
    print(SUMMARY_OUTPUT)

    print()
    print("NEXT STEP:")
    print(
        "Rank features and compare normal vs "
        "near_miss vs fall."
    )


if __name__ == "__main__":
    main()
