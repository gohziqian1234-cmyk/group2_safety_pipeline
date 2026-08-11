from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 HISTORICAL REPLAY OF FIXED EVENT DETECTOR
# Continuous scan of recorded data | NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "combined_cleaned_data.csv"
LABEL_FILE = PROCESSED / "event_labels.csv"
OUTPUT_FILE = PROCESSED / "historical_event_replay_log.csv"
SUMMARY_FILE = PROCESSED / "historical_event_replay_summary.csv"

TARGET_RATE_HZ = 25
DT = 1.0 / TARGET_RATE_HZ
SCAN_STEP_SECONDS = 0.5
COOLDOWN_SECONDS = 8.0
MATCH_TOLERANCE_SECONDS = 3.0

THRESHOLDS = {
    "event_peak_acceleration_g": 3.28154,
    "event_peak_gyroscope_dps": 641.28650,
    "event_peak_jerk_g_per_second": 67.40781,
    "event_rotation_integral_deg": 524.51258,
}
REQUIRED_VOTES = 2

UNCERTAIN_WORKER = "ziqian"
UNCERTAIN_SOURCE = "ziqian_die_data.csv"

REQUIRED_COLUMNS = [
    "worker",
    "source_file",
    "recording_type",
    "elapsed_seconds",
    "acceleration_magnitude_g",
    "gyroscope_magnitude_dps",
]

REQUIRED_LABEL_COLUMNS = [
    "worker",
    "source_file",
    "suggested_event_time_seconds",
    "final_event_type",
    "confirmed",
]


def trapezoid_integral(y_values, x_values):
    y = np.asarray(y_values, dtype=float)
    x = np.asarray(x_values, dtype=float)

    if len(y) != len(x):
        raise ValueError("Integration arrays must have equal length.")

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

    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def resample_recording(recording):
    frame = recording.copy()

    for column in [
        "elapsed_seconds",
        "acceleration_magnitude_g",
        "gyroscope_magnitude_dps",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame
        .dropna(
            subset=[
                "elapsed_seconds",
                "acceleration_magnitude_g",
                "gyroscope_magnitude_dps",
            ]
        )
        .sort_values("elapsed_seconds")
        .drop_duplicates(subset=["elapsed_seconds"], keep="first")
        .copy()
    )

    if len(frame) < 2:
        return pd.DataFrame()

    time = frame["elapsed_seconds"].to_numpy(dtype=float)
    start = float(time.min())
    end = float(time.max())

    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return pd.DataFrame()

    regular_time = np.arange(start, end + DT / 2.0, DT)

    output = pd.DataFrame({"elapsed_seconds": regular_time})
    output["acceleration_magnitude_g"] = np.interp(
        regular_time,
        time,
        frame["acceleration_magnitude_g"].to_numpy(dtype=float),
    )
    output["gyroscope_magnitude_dps"] = np.interp(
        regular_time,
        time,
        frame["gyroscope_magnitude_dps"].to_numpy(dtype=float),
    )

    output["jerk_g_per_second"] = (
        output["acceleration_magnitude_g"].diff().abs() / DT
    ).fillna(0.0)

    return output


def extract_event_features(data, center):
    # Same event window definition used during sequence feature development.
    start = center - 0.8
    end = center + 1.2

    window = data[
        (data["elapsed_seconds"] >= start)
        & (data["elapsed_seconds"] <= end)
    ]

    if len(window) < 20:
        return None

    accel = window["acceleration_magnitude_g"].to_numpy(dtype=float)
    gyro = window["gyroscope_magnitude_dps"].to_numpy(dtype=float)
    jerk = window["jerk_g_per_second"].to_numpy(dtype=float)
    time = window["elapsed_seconds"].to_numpy(dtype=float)

    if not (
        np.isfinite(accel).all()
        and np.isfinite(gyro).all()
        and np.isfinite(jerk).all()
        and np.isfinite(time).all()
    ):
        return None

    return {
        "event_peak_acceleration_g": float(np.max(accel)),
        "event_peak_gyroscope_dps": float(np.max(gyro)),
        "event_peak_jerk_g_per_second": float(np.max(jerk)),
        "event_rotation_integral_deg": trapezoid_integral(gyro, time),
    }


def vote_count(features):
    votes = 0
    for feature, threshold in THRESHOLDS.items():
        votes += int(features[feature] >= threshold)
    return votes


def scan_recording(data):
    start = float(data["elapsed_seconds"].min()) + 0.8
    end = float(data["elapsed_seconds"].max()) - 1.2

    if end <= start:
        return []

    centers = np.arange(start, end + SCAN_STEP_SECONDS / 2.0, SCAN_STEP_SECONDS)
    detections = []
    cooldown_until = -np.inf

    for center in centers:
        center = float(center)

        if center < cooldown_until:
            continue

        features = extract_event_features(data, center)
        if features is None:
            continue

        votes = vote_count(features)

        if votes >= REQUIRED_VOTES:
            detections.append({
                "detected_time_seconds": round(center, 3),
                "event_votes": int(votes),
                **features,
            })
            cooldown_until = center + COOLDOWN_SECONDS

    return detections


def build_confirmed_lookup(labels):
    labels = labels.copy()
    labels["confirmed"] = labels["confirmed"].astype(str).str.upper().str.strip()
    labels["suggested_event_time_seconds"] = pd.to_numeric(
        labels["suggested_event_time_seconds"], errors="coerce"
    )
    labels["final_event_type"] = (
        labels["final_event_type"].fillna("").astype(str).str.strip()
    )

    valid_types = {"near_miss", "fall_recovery", "fall_inactive"}
    confirmed = labels[
        (labels["confirmed"] == "YES")
        & labels["suggested_event_time_seconds"].notna()
        & labels["final_event_type"].isin(valid_types)
    ].copy()

    if len(confirmed) != 8:
        raise SystemExit(
            f"ERROR: expected 8 confirmed incident rows, found {len(confirmed)}."
        )

    return {
        (str(row.worker), str(row.source_file)): row
        for row in confirmed.itertuples(index=False)
    }


def main():
    print("=" * 88)
    print("GROUP 5 - HISTORICAL REPLAY OF FIXED EVENT DETECTOR")
    print("Continuous scan + cooldown | NO AI / NO ML")
    print("=" * 88)

    if not INPUT_FILE.exists():
        raise SystemExit(f"ERROR: missing cleaned data:\n{INPUT_FILE}")

    if not LABEL_FILE.exists():
        raise SystemExit(f"ERROR: missing event labels:\n{LABEL_FILE}")

    data = pd.read_csv(INPUT_FILE)
    labels = pd.read_csv(LABEL_FILE)

    missing_data = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_data:
        raise SystemExit(
            "ERROR: cleaned data missing columns:\n"
            + "\n".join(f" - {c}" for c in missing_data)
        )

    missing_labels = [c for c in REQUIRED_LABEL_COLUMNS if c not in labels.columns]
    if missing_labels:
        raise SystemExit(
            "ERROR: event_labels.csv missing columns:\n"
            + "\n".join(f" - {c}" for c in missing_labels)
        )

    confirmed_lookup = build_confirmed_lookup(labels)
    replay_rows = []
    summary_rows = []

    grouped = data.groupby(
        ["worker", "source_file", "recording_type"],
        sort=False,
        dropna=False,
    )

    for (worker, source_file, recording_type), recording in grouped:
        regular = resample_recording(recording)

        if regular.empty:
            raise SystemExit(
                f"ERROR: resampling failed for {worker} | {source_file}"
            )

        detections = scan_recording(regular)
        is_normal = "normal" in str(recording_type).lower()

        expected_time = np.nan
        expected_type = "normal"
        reliable_for_score = False
        matched = False
        nearest_error = np.nan

        if not is_normal:
            key = (str(worker), str(source_file))
            if key not in confirmed_lookup:
                raise SystemExit(
                    f"ERROR: missing confirmed incident for {worker} | {source_file}"
                )

            label = confirmed_lookup[key]
            expected_time = float(label.suggested_event_time_seconds)
            expected_type = str(label.final_event_type)

            reliable_for_score = not (
                str(worker) == UNCERTAIN_WORKER
                and str(source_file) == UNCERTAIN_SOURCE
                and expected_type.startswith("fall")
            )

            if detections:
                errors = [
                    abs(float(d["detected_time_seconds"]) - expected_time)
                    for d in detections
                ]
                nearest_error = float(min(errors))
                matched = bool(nearest_error <= MATCH_TOLERANCE_SECONDS)

        for detection in detections:
            replay_rows.append({
                "worker": worker,
                "source_file": source_file,
                "recording_type": recording_type,
                "detected_time_seconds": detection["detected_time_seconds"],
                "event_votes": detection["event_votes"],
                "event_peak_acceleration_g": detection["event_peak_acceleration_g"],
                "event_peak_gyroscope_dps": detection["event_peak_gyroscope_dps"],
                "event_peak_jerk_g_per_second": detection[
                    "event_peak_jerk_g_per_second"
                ],
                "event_rotation_integral_deg": detection[
                    "event_rotation_integral_deg"
                ],
                "expected_event_time_seconds": expected_time,
                "expected_event_type": expected_type,
                "reliable_for_score": reliable_for_score,
                "within_match_tolerance": (
                    False
                    if is_normal or not np.isfinite(expected_time)
                    else abs(detection["detected_time_seconds"] - expected_time)
                    <= MATCH_TOLERANCE_SECONDS
                ),
            })

        summary_rows.append({
            "worker": worker,
            "source_file": source_file,
            "recording_type": recording_type,
            "is_normal_recording": is_normal,
            "expected_event_time_seconds": expected_time,
            "expected_event_type": expected_type,
            "reliable_for_score": reliable_for_score,
            "number_of_detector_alerts": int(len(detections)),
            "confirmed_event_matched": bool(matched),
            "nearest_detection_error_seconds": nearest_error,
        })

        print(
            f"{worker:12s} | {str(recording_type):12s} | "
            f"alerts={len(detections):2d} | "
            + (
                "normal"
                if is_normal
                else f"matched={'YES' if matched else 'NO'}"
            )
            + f" | {source_file}"
        )

    replay = pd.DataFrame(replay_rows)
    summary = pd.DataFrame(summary_rows)

    if not replay.empty:
        numeric = replay.select_dtypes(include=[np.number]).columns
        replay[numeric] = replay[numeric].round(5)
    replay.to_csv(OUTPUT_FILE, index=False)

    numeric_summary = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_summary] = summary[numeric_summary].round(5)
    summary.to_csv(SUMMARY_FILE, index=False)

    reliable_incidents = summary[
        (~summary["is_normal_recording"])
        & summary["reliable_for_score"]
    ]
    normal_recordings = summary[summary["is_normal_recording"]]
    uncertain = summary[
        (~summary["is_normal_recording"])
        & (~summary["reliable_for_score"])
    ]

    reliable_matched = int(reliable_incidents["confirmed_event_matched"].sum())
    reliable_total = int(len(reliable_incidents))
    normal_with_alert = int((normal_recordings["number_of_detector_alerts"] > 0).sum())
    normal_alert_count = int(normal_recordings["number_of_detector_alerts"].sum())

    print()
    print("=" * 88)
    print("HISTORICAL REPLAY SUMMARY")
    print("=" * 88)
    print(f"Reliable confirmed incidents matched: {reliable_matched}/{reliable_total}")
    print(f"Normal recordings with >=1 alert: {normal_with_alert}/{len(normal_recordings)}")
    print(f"Total alerts across normal recordings: {normal_alert_count}")

    if len(uncertain) == 1:
        row = uncertain.iloc[0]
        print(
            "Uncertain Ziqian fall stress test matched: "
            f"{'YES' if bool(row['confirmed_event_matched']) else 'NO'}"
        )

    print()
    print("Fixed detector used:")
    print(f"  Trigger when at least {REQUIRED_VOTES}/4 conditions are true")
    for feature, threshold in THRESHOLDS.items():
        print(f"  {feature:38s} >= {threshold:.5f}")
    print(f"  Scan step: {SCAN_STEP_SECONDS:.1f} s")
    print(f"  Cooldown: {COOLDOWN_SECONDS:.1f} s")
    print(f"  Confirmed-event matching tolerance: +/- {MATCH_TOLERANCE_SECONDS:.1f} s")

    print()
    print("Files created:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print()
    print("IMPORTANT:")
    print("- This is the final deployment-like offline test of the fixed Stage-1 event trigger.")
    print("- It scans whole recordings instead of evaluating only the manually selected event row.")
    print("- Stage-2 fall-vs-near-miss classification is NOT considered validated and is not used here.")
    print("- A successful Stage-1 alert still allows the system to warn locally and notify the dashboard.")


if __name__ == "__main__":
    main()
