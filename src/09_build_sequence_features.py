from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 SEQUENCE FEATURE EXTRACTION
# Temporal/rule-based analysis only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "combined_cleaned_data.csv"
LABEL_FILE = PROCESSED / "event_labels.csv"
OUTPUT_FILE = PROCESSED / "sequence_features.csv"
SUMMARY_FILE = PROCESSED / "sequence_feature_summary.csv"

TARGET_RATE_HZ = 25
DT = 1.0 / TARGET_RATE_HZ
NORMAL_STEP_SECONDS = 1.0

SENSOR_COLUMNS = [
    "Xg", "Yg", "Zg",
    "Xdeg", "Ydeg", "Zdeg",
    "acceleration_magnitude_g",
    "gyroscope_magnitude_dps",
]

REQUIRED_COLUMNS = [
    "worker",
    "source_file",
    "recording_type",
    "elapsed_seconds",
    *SENSOR_COLUMNS,
]


def vector_angle_degrees(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na <= 1e-9 or nb <= 1e-9:
        return np.nan

    cosine = np.dot(a, b) / (na * nb)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def resample_recording(recording):
    recording = (
        recording
        .sort_values("elapsed_seconds")
        .drop_duplicates(subset=["elapsed_seconds"], keep="first")
        .copy()
    )

    time_values = pd.to_numeric(recording["elapsed_seconds"], errors="coerce")
    valid_time = time_values.notna()
    recording = recording.loc[valid_time].copy()
    time_values = time_values.loc[valid_time]

    if len(recording) < 2:
        return pd.DataFrame()

    start = float(time_values.min())
    end = float(time_values.max())

    if end <= start:
        return pd.DataFrame()

    regular_time = np.arange(start, end + DT / 2, DT)
    output = pd.DataFrame({"elapsed_seconds": regular_time})

    for column in SENSOR_COLUMNS:
        values = pd.to_numeric(recording[column], errors="coerce")
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
        output["acceleration_magnitude_g"].diff().abs() / DT
    ).fillna(0.0)

    return output


def get_window(data, center, start_offset, end_offset):
    start = center + start_offset
    end = center + end_offset

    return data[
        (data["elapsed_seconds"] >= start)
        & (data["elapsed_seconds"] <= end)
    ].copy()


def median_accel_vector(window):
    return (
        window[["Xg", "Yg", "Zg"]]
        .median()
        .to_numpy(dtype=float)
    )


def extract_sequence_features(data, center):
    # Same physical sequence used during candidate verification:
    # pre posture -> incident -> post movement -> late movement/posture.
    pre = get_window(data, center, -3.0, -1.0)
    event = get_window(data, center, -0.8, 1.2)
    post = get_window(data, center, 1.5, 4.5)
    late = get_window(data, center, 4.5, 8.0)

    if min(len(pre), len(event), len(post), len(late)) < 5:
        return None

    pre_vector = median_accel_vector(pre)
    post_vector = median_accel_vector(post)
    late_vector = median_accel_vector(late)

    posture_change_post = vector_angle_degrees(pre_vector, post_vector)
    posture_change_late = vector_angle_degrees(pre_vector, late_vector)

    event_peak_accel = float(event["acceleration_magnitude_g"].max())
    event_peak_gyro = float(event["gyroscope_magnitude_dps"].max())
    event_peak_jerk = float(event["jerk_g_per_second"].max())

    # Approximate total rotational activity during the event window.
    event_rotation_integral = float(
        np.trapz(
            event["gyroscope_magnitude_dps"].to_numpy(dtype=float),
            event["elapsed_seconds"].to_numpy(dtype=float),
        )
    )

    return {
        "event_peak_acceleration_g": event_peak_accel,
        "event_peak_gyroscope_dps": event_peak_gyro,
        "event_peak_jerk_g_per_second": event_peak_jerk,
        "event_rotation_integral_deg": event_rotation_integral,
        "posture_change_post_deg": posture_change_post,
        "posture_change_late_deg": posture_change_late,
        "post_accel_std_g": float(post["acceleration_magnitude_g"].std(ddof=0)),
        "post_gyro_mean_dps": float(post["gyroscope_magnitude_dps"].mean()),
        "late_accel_std_g": float(late["acceleration_magnitude_g"].std(ddof=0)),
        "late_gyro_mean_dps": float(late["gyroscope_magnitude_dps"].mean()),
    }


def main():
    print("=" * 76)
    print("GROUP 5 - SEQUENCE FEATURE EXTRACTION")
    print("Pre-event + event + recovery/inactivity behaviour | NO AI / NO ML")
    print("=" * 76)

    if not INPUT_FILE.exists():
        raise SystemExit(f"ERROR: missing cleaned data:\n{INPUT_FILE}")

    if not LABEL_FILE.exists():
        raise SystemExit(f"ERROR: missing labels:\n{LABEL_FILE}")

    data = pd.read_csv(INPUT_FILE)
    labels = pd.read_csv(LABEL_FILE)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise SystemExit(
            "ERROR: cleaned data is missing columns:\n"
            + "\n".join(f" - {c}" for c in missing)
        )

    labels["confirmed"] = labels["confirmed"].astype(str).str.upper().str.strip()

    confirmed = labels[
        (labels["confirmed"] == "YES")
        & labels["final_event_type"].notna()
        & labels["suggested_event_time_seconds"].notna()
    ].copy()

    if len(confirmed) != 8:
        raise SystemExit(
            f"ERROR: expected 8 confirmed incidents, found {len(confirmed)}."
        )

    label_lookup = {
        (str(r.worker), str(r.source_file)): r
        for r in confirmed.itertuples(index=False)
    }

    rows = []

    grouped = data.groupby(
        ["worker", "source_file", "recording_type"],
        sort=False,
        dropna=False,
    )

    for (worker, source_file, recording_type), recording in grouped:
        regular = resample_recording(recording)

        if regular.empty:
            print(f"WARNING: skipped {worker} | {source_file}")
            continue

        is_normal = "normal" in str(recording_type).lower()

        if is_normal:
            start = float(regular["elapsed_seconds"].min()) + 3.0
            end = float(regular["elapsed_seconds"].max()) - 8.0

            if end <= start:
                continue

            centers = np.arange(start, end + NORMAL_STEP_SECONDS / 2, NORMAL_STEP_SECONDS)

            added = 0
            for center in centers:
                features = extract_sequence_features(regular, float(center))
                if features is None:
                    continue

                rows.append({
                    "worker": worker,
                    "source_file": source_file,
                    "recording_type": recording_type,
                    "center_seconds": round(float(center), 3),
                    "sequence_label": "normal",
                    "event_subtype": "normal_work",
                    **features,
                })
                added += 1

            print(f"{worker:12s} | normal    | {added:4d} sequence positions | {source_file}")
            continue

        key = (str(worker), str(source_file))
        if key not in label_lookup:
            raise SystemExit(
                f"ERROR: missing confirmed event label for {worker} | {source_file}"
            )

        label_row = label_lookup[key]
        center = float(label_row.suggested_event_time_seconds)
        features = extract_sequence_features(regular, center)

        if features is None:
            print(f"WARNING: insufficient sequence data for {worker} | {source_file}")
            continue

        subtype = str(label_row.final_event_type)
        sequence_label = "near_miss" if subtype == "near_miss" else "fall"

        rows.append({
            "worker": worker,
            "source_file": source_file,
            "recording_type": recording_type,
            "center_seconds": round(center, 3),
            "sequence_label": sequence_label,
            "event_subtype": subtype,
            **features,
        })

        print(f"{worker:12s} | {sequence_label:9s} | event @ {center:7.2f}s | {source_file}")

    result = pd.DataFrame(rows)

    if result.empty:
        raise SystemExit("ERROR: no sequence features created.")

    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(5)
    result.to_csv(OUTPUT_FILE, index=False)

    feature_columns = [
        "event_peak_acceleration_g",
        "event_peak_gyroscope_dps",
        "event_peak_jerk_g_per_second",
        "event_rotation_integral_deg",
        "posture_change_post_deg",
        "posture_change_late_deg",
        "post_accel_std_g",
        "post_gyro_mean_dps",
        "late_accel_std_g",
        "late_gyro_mean_dps",
    ]

    summary_rows = []
    for label in ["normal", "near_miss", "fall"]:
        subset = result[result["sequence_label"] == label]
        for feature in feature_columns:
            series = subset[feature]
            summary_rows.append({
                "sequence_label": label,
                "feature": feature,
                "count": int(series.count()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "q95": float(series.quantile(0.95)),
                "max": float(series.max()),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print("Sequence row counts:")
    print(result["sequence_label"].value_counts().to_string())

    print()
    print("Confirmed incident sequence features:")
    print(
        result[result["sequence_label"].isin(["near_miss", "fall"])][
            [
                "worker",
                "sequence_label",
                "event_subtype",
                "center_seconds",
                "event_peak_acceleration_g",
                "event_peak_gyroscope_dps",
                "posture_change_late_deg",
                "late_accel_std_g",
                "late_gyro_mean_dps",
            ]
        ].to_string(index=False)
    )

    print()
    print("Files created:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print()
    print("WHY THIS STEP EXISTS:")
    print("The 1-second snapshot rules did not generalise well enough across workers.")
    print("This adds the physical sequence after an event: posture change, recovery, or inactivity.")
    print("We will validate the improved state-machine rules only after reviewing these features.")


if __name__ == "__main__":
    main()
