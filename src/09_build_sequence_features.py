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

REQUIRED_LABEL_COLUMNS = [
    "worker",
    "source_file",
    "suggested_event_time_seconds",
    "final_event_type",
    "confirmed",
]


# ------------------------------------------------------------
# NUMPY-VERSION-SAFE TRAPEZOID INTEGRATION
# ------------------------------------------------------------
def trapezoid_integral(y_values, x_values):
    """
    Integrate y with respect to x using the trapezoidal rule.

    Implemented directly instead of np.trapz/np.trapezoid so the
    project works across old and new NumPy versions.
    """
    y = np.asarray(y_values, dtype=float)
    x = np.asarray(x_values, dtype=float)

    if len(y) != len(x):
        raise ValueError("Integration arrays must have the same length.")

    if len(y) < 2:
        return 0.0

    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]

    if len(y) < 2:
        return 0.0

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    dx = np.diff(x)
    areas = 0.5 * (y[:-1] + y[1:]) * dx

    return float(np.sum(areas))


def vector_angle_degrees(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != (3,) or b.shape != (3,):
        return np.nan

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na <= 1e-9 or nb <= 1e-9:
        return np.nan

    cosine = np.dot(a, b) / (na * nb)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def resample_recording(recording):
    recording = recording.copy()

    recording["elapsed_seconds"] = pd.to_numeric(
        recording["elapsed_seconds"],
        errors="coerce",
    )

    recording = (
        recording
        .dropna(subset=["elapsed_seconds"])
        .sort_values("elapsed_seconds")
        .drop_duplicates(subset=["elapsed_seconds"], keep="first")
        .copy()
    )

    if len(recording) < 2:
        return pd.DataFrame()

    time_values = recording["elapsed_seconds"].to_numpy(dtype=float)

    start = float(time_values.min())
    end = float(time_values.max())

    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return pd.DataFrame()

    regular_time = np.arange(start, end + DT / 2, DT)

    if len(regular_time) < 2:
        return pd.DataFrame()

    output = pd.DataFrame({"elapsed_seconds": regular_time})

    for column in SENSOR_COLUMNS:
        values = pd.to_numeric(recording[column], errors="coerce")
        valid = values.notna().to_numpy()

        if valid.sum() < 2:
            output[column] = np.nan
            continue

        source_time = time_values[valid]
        source_values = values.to_numpy(dtype=float)[valid]

        output[column] = np.interp(
            regular_time,
            source_time,
            source_values,
        )

    # Jerk is calculated only after resampling to a regular 25 Hz grid.
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
    """
    Describe the physical sequence around a possible incident:

    pre   = posture before the event
    event = impact / rotation period
    post  = early recovery or early inactivity
    late  = later recovery / continued inactivity / final posture
    """
    pre = get_window(data, center, -3.0, -1.0)
    event = get_window(data, center, -0.8, 1.2)
    post = get_window(data, center, 1.5, 4.5)
    late = get_window(data, center, 4.5, 8.0)

    windows = [pre, event, post, late]

    if min(len(window) for window in windows) < 5:
        return None

    required_numeric = [*SENSOR_COLUMNS, "jerk_g_per_second"]

    for window in windows:
        if window[required_numeric].isna().any().any():
            return None

    pre_vector = median_accel_vector(pre)
    post_vector = median_accel_vector(post)
    late_vector = median_accel_vector(late)

    posture_change_post = vector_angle_degrees(pre_vector, post_vector)
    posture_change_late = vector_angle_degrees(pre_vector, late_vector)

    event_peak_accel = float(event["acceleration_magnitude_g"].max())
    event_peak_gyro = float(event["gyroscope_magnitude_dps"].max())
    event_peak_jerk = float(event["jerk_g_per_second"].max())

    event_rotation_integral = trapezoid_integral(
        event["gyroscope_magnitude_dps"].to_numpy(dtype=float),
        event["elapsed_seconds"].to_numpy(dtype=float),
    )

    result = {
        "event_peak_acceleration_g": event_peak_accel,
        "event_peak_gyroscope_dps": event_peak_gyro,
        "event_peak_jerk_g_per_second": event_peak_jerk,
        "event_rotation_integral_deg": event_rotation_integral,
        "posture_change_post_deg": posture_change_post,
        "posture_change_late_deg": posture_change_late,
        "post_accel_std_g": float(
            post["acceleration_magnitude_g"].std(ddof=0)
        ),
        "post_gyro_mean_dps": float(
            post["gyroscope_magnitude_dps"].mean()
        ),
        "late_accel_std_g": float(
            late["acceleration_magnitude_g"].std(ddof=0)
        ),
        "late_gyro_mean_dps": float(
            late["gyroscope_magnitude_dps"].mean()
        ),
    }

    # Reject incomplete/non-finite feature rows instead of silently
    # writing invalid values into the next analysis stage.
    if not all(np.isfinite(value) for value in result.values()):
        return None

    return result


def validate_inputs(data, labels):
    missing_data = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_data:
        raise SystemExit(
            "ERROR: cleaned data is missing columns:\n"
            + "\n".join(f" - {column}" for column in missing_data)
        )

    missing_labels = [
        column for column in REQUIRED_LABEL_COLUMNS
        if column not in labels.columns
    ]

    if missing_labels:
        raise SystemExit(
            "ERROR: event_labels.csv is missing columns:\n"
            + "\n".join(f" - {column}" for column in missing_labels)
        )


def main():
    print("=" * 76)
    print("GROUP 5 - SEQUENCE FEATURE EXTRACTION")
    print("Pre-event + event + recovery/inactivity behaviour | NO AI / NO ML")
    print("=" * 76)
    print(f"NumPy version:  {np.__version__}")
    print(f"Pandas version: {pd.__version__}")

    if not INPUT_FILE.exists():
        raise SystemExit(f"ERROR: missing cleaned data:\n{INPUT_FILE}")

    if not LABEL_FILE.exists():
        raise SystemExit(f"ERROR: missing labels:\n{LABEL_FILE}")

    data = pd.read_csv(INPUT_FILE)
    labels = pd.read_csv(LABEL_FILE)

    validate_inputs(data, labels)

    labels["confirmed"] = (
        labels["confirmed"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    labels["suggested_event_time_seconds"] = pd.to_numeric(
        labels["suggested_event_time_seconds"],
        errors="coerce",
    )

    labels["final_event_type"] = (
        labels["final_event_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    allowed_event_types = {
        "near_miss",
        "fall_recovery",
        "fall_inactive",
    }

    confirmed = labels[
        (labels["confirmed"] == "YES")
        & labels["suggested_event_time_seconds"].notna()
        & labels["final_event_type"].isin(allowed_event_types)
    ].copy()

    if len(confirmed) != 8:
        print()
        print("Confirmed label rows found:")
        print(
            confirmed[
                [
                    "worker",
                    "source_file",
                    "suggested_event_time_seconds",
                    "final_event_type",
                    "confirmed",
                ]
            ].to_string(index=False)
        )
        raise SystemExit(
            f"\nERROR: expected exactly 8 valid confirmed incidents, "
            f"found {len(confirmed)}."
        )

    duplicate_labels = confirmed.duplicated(
        subset=["worker", "source_file"],
        keep=False,
    )

    if duplicate_labels.any():
        raise SystemExit(
            "ERROR: duplicate confirmed labels found for a worker/source file."
        )

    label_lookup = {
        (str(row.worker), str(row.source_file)): row
        for row in confirmed.itertuples(index=False)
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
            print(f"WARNING: skipped {worker} | {source_file}: resampling failed")
            continue

        if regular[[*SENSOR_COLUMNS, "jerk_g_per_second"]].isna().any().any():
            print(f"WARNING: skipped {worker} | {source_file}: missing sensor values")
            continue

        is_normal = "normal" in str(recording_type).lower()

        if is_normal:
            # Need 3 s of history and 8 s of future data around each center.
            start = float(regular["elapsed_seconds"].min()) + 3.0
            end = float(regular["elapsed_seconds"].max()) - 8.0

            if end <= start:
                print(f"WARNING: normal recording too short: {worker} | {source_file}")
                continue

            centers = np.arange(
                start,
                end + NORMAL_STEP_SECONDS / 2,
                NORMAL_STEP_SECONDS,
            )

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

            print(
                f"{worker:12s} | normal    | "
                f"{added:4d} sequence positions | {source_file}"
            )
            continue

        key = (str(worker), str(source_file))

        if key not in label_lookup:
            raise SystemExit(
                f"ERROR: missing confirmed event label for "
                f"{worker} | {source_file}"
            )

        label_row = label_lookup[key]
        center = float(label_row.suggested_event_time_seconds)
        features = extract_sequence_features(regular, center)

        if features is None:
            raise SystemExit(
                f"ERROR: insufficient/invalid sequence data around confirmed "
                f"event for {worker} | {source_file} | {center:.2f}s"
            )

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

        print(
            f"{worker:12s} | {sequence_label:9s} | "
            f"event @ {center:7.2f}s | {source_file}"
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise SystemExit("ERROR: no sequence features created.")

    incident_rows = result[
        result["sequence_label"].isin(["near_miss", "fall"])
    ]

    if len(incident_rows) != 8:
        raise SystemExit(
            f"ERROR: expected 8 incident sequence rows, created {len(incident_rows)}."
        )

    incident_counts = incident_rows["sequence_label"].value_counts()

    if incident_counts.get("near_miss", 0) != 4 or incident_counts.get("fall", 0) != 4:
        raise SystemExit(
            "ERROR: expected 4 near-miss and 4 fall sequence rows."
        )

    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(5)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
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

        if subset.empty:
            raise SystemExit(
                f"ERROR: no sequence rows created for label '{label}'."
            )

        for feature in feature_columns:
            series = pd.to_numeric(subset[feature], errors="coerce").dropna()

            if series.empty:
                raise SystemExit(
                    f"ERROR: feature '{feature}' has no valid values for '{label}'."
                )

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
    summary_numeric = summary.select_dtypes(include=[np.number]).columns
    summary[summary_numeric] = summary[summary_numeric].round(5)
    summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print("Sequence row counts:")
    print(result["sequence_label"].value_counts().to_string())

    print()
    print("Confirmed incident sequence features:")
    print(
        incident_rows[
            [
                "worker",
                "sequence_label",
                "event_subtype",
                "center_seconds",
                "event_peak_acceleration_g",
                "event_peak_gyroscope_dps",
                "event_peak_jerk_g_per_second",
                "event_rotation_integral_deg",
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
    print("CHECKS PASSED:")
    print("- input columns validated")
    print("- exactly 8 confirmed incidents validated")
    print("- exactly 4 near-miss + 4 fall sequences created")
    print("- NumPy-version-safe integration used")
    print("- invalid/non-finite feature rows rejected")

    print()
    print("WHY THIS STEP EXISTS:")
    print("The 1-second snapshot rules did not generalise well enough across workers.")
    print("This adds the physical sequence after an event: posture change, recovery, or inactivity.")
    print("We will validate the improved state-machine rules only after reviewing these features.")


if __name__ == "__main__":
    main()
