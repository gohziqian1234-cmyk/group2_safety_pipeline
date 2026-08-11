from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 FINAL STAGE-1 DEPLOYMENT TUNING
# Fixed thresholds + simple startup grace + cooldown
# Safety priority: ZERO FN on reliable incidents | NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "combined_cleaned_data.csv"
LABEL_FILE = PROCESSED / "event_labels.csv"
GRID_OUTPUT = PROCESSED / "final_stage1_tuning_grid.csv"
DETAIL_OUTPUT = PROCESSED / "final_stage1_selected_details.csv"
FINAL_OUTPUT = PROCESSED / "final_stage1_detector_config.csv"

TARGET_RATE_HZ = 25
DT = 1.0 / TARGET_RATE_HZ
SCAN_STEP_SECONDS = 0.5
MATCH_TOLERANCE_SECONDS = 3.0

THRESHOLDS = {
    "event_peak_acceleration_g": 3.28154,
    "event_peak_gyroscope_dps": 641.28650,
    "event_peak_jerk_g_per_second": 67.40781,
    "event_rotation_integral_deg": 524.51258,
}
REQUIRED_VOTES = 2

STARTUP_GRACE_OPTIONS = [0.0, 2.0, 3.0, 5.0]
COOLDOWN_OPTIONS = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]

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
    return sum(
        int(features[feature] >= threshold)
        for feature, threshold in THRESHOLDS.items()
    )


def scan_recording(data, startup_grace, cooldown):
    start = float(data["elapsed_seconds"].min()) + max(0.8, startup_grace)
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
            cooldown_until = center + cooldown

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


def score_config(recordings, confirmed_lookup, startup_grace, cooldown):
    detail_rows = []

    for record in recordings:
        worker = record["worker"]
        source_file = record["source_file"]
        recording_type = record["recording_type"]
        regular = record["regular"]

        alerts = scan_recording(regular, startup_grace, cooldown)
        is_normal = "normal" in str(recording_type).lower()

        expected_time = np.nan
        expected_type = "normal"
        reliable_for_score = False
        matched = False
        nearest_error = np.nan
        extra_alerts = len(alerts)

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

            if alerts:
                errors = np.array(
                    [abs(float(a["detected_time_seconds"]) - expected_time) for a in alerts],
                    dtype=float,
                )
                nearest_error = float(errors.min())
                matched = bool(nearest_error <= MATCH_TOLERANCE_SECONDS)
                extra_alerts = len(alerts) - (1 if matched else 0)

        detail_rows.append({
            "worker": worker,
            "source_file": source_file,
            "recording_type": recording_type,
            "is_normal_recording": is_normal,
            "expected_event_time_seconds": expected_time,
            "expected_event_type": expected_type,
            "reliable_for_score": reliable_for_score,
            "number_of_alerts": int(len(alerts)),
            "confirmed_event_matched": bool(matched),
            "nearest_detection_error_seconds": nearest_error,
            "extra_alert_count": int(extra_alerts),
            "alert_times_seconds": ",".join(
                f"{float(a['detected_time_seconds']):.2f}" for a in alerts
            ),
        })

    details = pd.DataFrame(detail_rows)

    reliable = details[
        (~details["is_normal_recording"])
        & details["reliable_for_score"]
    ]
    normal = details[details["is_normal_recording"]]
    uncertain = details[
        (~details["is_normal_recording"])
        & (~details["reliable_for_score"])
    ]

    reliable_matched = int(reliable["confirmed_event_matched"].sum())
    reliable_total = int(len(reliable))
    fn = reliable_total - reliable_matched

    uncertain_match = 0
    if len(uncertain) == 1:
        uncertain_match = int(bool(uncertain.iloc[0]["confirmed_event_matched"]))

    summary = {
        "startup_grace_seconds": float(startup_grace),
        "cooldown_seconds": float(cooldown),
        "reliable_incidents_matched": reliable_matched,
        "reliable_incidents_tested": reliable_total,
        "false_negatives": int(fn),
        "reliable_detection_rate": (
            reliable_matched / reliable_total if reliable_total else np.nan
        ),
        "total_extra_alerts_all_recordings": int(details["extra_alert_count"].sum()),
        "normal_recordings_with_alert": int((normal["number_of_alerts"] > 0).sum()),
        "normal_recordings_tested": int(len(normal)),
        "total_normal_alerts": int(normal["number_of_alerts"].sum()),
        "uncertain_ziqian_fall_matched": uncertain_match,
    }

    return summary, details


def choose_best(grid):
    # Absolute priority: FN = 0.
    zero_fn = grid[grid["false_negatives"] == 0].copy()

    if zero_fn.empty:
        best_fn = int(grid["false_negatives"].min())
        zero_fn = grid[grid["false_negatives"] == best_fn].copy()

    # Once FN is protected, reduce duplicate and normal alerts.
    ordered = zero_fn.sort_values(
        [
            "total_extra_alerts_all_recordings",
            "normal_recordings_with_alert",
            "total_normal_alerts",
            "cooldown_seconds",
            "startup_grace_seconds",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    return ordered.iloc[0].to_dict()


def main():
    print("=" * 92)
    print("GROUP 5 - FINAL STAGE-1 DEPLOYMENT TUNING")
    print("ZERO FN first | fixed thresholds | startup grace + simple cooldown | NO AI / NO ML")
    print("=" * 92)

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

    recordings = []
    grouped = data.groupby(
        ["worker", "source_file", "recording_type"],
        sort=False,
        dropna=False,
    )

    print("Preparing 12 recordings...")

    for (worker, source_file, recording_type), recording in grouped:
        regular = resample_recording(recording)

        if regular.empty:
            raise SystemExit(
                f"ERROR: resampling failed for {worker} | {source_file}"
            )

        recordings.append({
            "worker": worker,
            "source_file": source_file,
            "recording_type": recording_type,
            "regular": regular,
        })

    if len(recordings) != 12:
        raise SystemExit(f"ERROR: expected 12 recordings, found {len(recordings)}.")

    grid_rows = []
    details_by_config = {}

    for startup_grace in STARTUP_GRACE_OPTIONS:
        for cooldown in COOLDOWN_OPTIONS:
            summary, details = score_config(
                recordings,
                confirmed_lookup,
                startup_grace,
                cooldown,
            )
            grid_rows.append(summary)
            details_by_config[(float(startup_grace), float(cooldown))] = details

    grid = pd.DataFrame(grid_rows)
    best = choose_best(grid)

    selected_key = (
        float(best["startup_grace_seconds"]),
        float(best["cooldown_seconds"]),
    )
    selected_details = details_by_config[selected_key].copy()

    numeric_grid = grid.select_dtypes(include=[np.number]).columns
    grid[numeric_grid] = grid[numeric_grid].round(6)
    grid.to_csv(GRID_OUTPUT, index=False)

    numeric_details = selected_details.select_dtypes(include=[np.number]).columns
    selected_details[numeric_details] = selected_details[numeric_details].round(6)
    selected_details.to_csv(DETAIL_OUTPUT, index=False)

    final_row = {
        **best,
        "required_votes_out_of_4": REQUIRED_VOTES,
        **{f"threshold_{feature}": threshold for feature, threshold in THRESHOLDS.items()},
        "scan_step_seconds": SCAN_STEP_SECONDS,
        "match_tolerance_seconds": MATCH_TOLERANCE_SECONDS,
    }

    final = pd.DataFrame([final_row])
    numeric_final = final.select_dtypes(include=[np.number]).columns
    final[numeric_final] = final[numeric_final].round(6)
    final.to_csv(FINAL_OUTPUT, index=False)

    print()
    print("=" * 92)
    print("TOP ZERO-FN CONFIGURATIONS")
    print("=" * 92)

    zero_fn_table = grid[grid["false_negatives"] == 0].sort_values(
        [
            "total_extra_alerts_all_recordings",
            "normal_recordings_with_alert",
            "total_normal_alerts",
            "cooldown_seconds",
            "startup_grace_seconds",
        ],
        ascending=[True, True, True, True, True],
    )

    if zero_fn_table.empty:
        print("WARNING: no zero-FN configuration found.")
        print(
            grid.sort_values(
                ["false_negatives", "total_extra_alerts_all_recordings"]
            ).head(10).to_string(index=False)
        )
    else:
        print(zero_fn_table.head(10).to_string(index=False))

    print()
    print("=" * 92)
    print("FINAL STAGE-1 DETECTOR CONFIGURATION")
    print("=" * 92)
    print(f"Startup grace: {best['startup_grace_seconds']:.1f} s")
    print(f"Cooldown after alert: {best['cooldown_seconds']:.1f} s")
    print(f"Reliable incidents detected: {int(best['reliable_incidents_matched'])}/{int(best['reliable_incidents_tested'])}")
    print(f"FALSE NEGATIVES: {int(best['false_negatives'])}")
    print(f"Extra alerts across all 12 recordings: {int(best['total_extra_alerts_all_recordings'])}")
    print(f"Normal recordings with >=1 alert: {int(best['normal_recordings_with_alert'])}/{int(best['normal_recordings_tested'])}")
    print(f"Total normal alerts: {int(best['total_normal_alerts'])}")
    print(
        "Uncertain Ziqian fall stress test matched: "
        f"{'YES' if int(best['uncertain_ziqian_fall_matched']) else 'NO'}"
    )

    print()
    print("Fixed threshold rule:")
    print(f"  Trigger when >= {REQUIRED_VOTES}/4 conditions are true")
    for feature, threshold in THRESHOLDS.items():
        print(f"  {feature:38s} >= {threshold:.5f}")

    print()
    print("SELECTED CONFIGURATION BY RECORDING:")
    print(
        selected_details[
            [
                "worker",
                "recording_type",
                "number_of_alerts",
                "confirmed_event_matched",
                "extra_alert_count",
                "alert_times_seconds",
                "source_file",
            ]
        ].to_string(index=False)
    )

    print()
    print("Files created:")
    print(GRID_OUTPUT)
    print(DETAIL_OUTPUT)
    print(FINAL_OUTPUT)

    print()
    print("DECISION RULE:")
    print("- FN = 0 is mandatory on the reliable validation dataset.")
    print("- Among zero-FN configurations, choose the one with the fewest extra/normal alerts.")
    print("- The uncertain Ziqian fall remains excluded from scoring and is reported separately.")
    print("- Stage-2 fall classification remains unvalidated; deploy Stage 1 as SAFETY_EVENT only.")


if __name__ == "__main__":
    main()
