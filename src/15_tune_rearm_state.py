from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 EVENT-REARM STATE TUNING
# Fixed Stage-1 thresholds + stateful rearm | NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "combined_cleaned_data.csv"
LABEL_FILE = PROCESSED / "event_labels.csv"
GRID_OUTPUT = PROCESSED / "rearm_tuning_grid.csv"
DETAIL_OUTPUT = PROCESSED / "rearm_tuning_selected_details.csv"
FINAL_OUTPUT = PROCESSED / "rearm_tuning_final.csv"

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

MIN_COOLDOWN_OPTIONS = [4.0, 8.0, 12.0, 15.0, 20.0]
QUIET_REARM_OPTIONS = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0]

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


def build_scan_table(data):
    start = float(data["elapsed_seconds"].min()) + 0.8
    end = float(data["elapsed_seconds"].max()) - 1.2

    if end <= start:
        return pd.DataFrame()

    rows = []
    centers = np.arange(start, end + SCAN_STEP_SECONDS / 2.0, SCAN_STEP_SECONDS)

    for center in centers:
        center = float(center)
        features = extract_event_features(data, center)

        if features is None:
            continue

        votes = sum(
            int(features[feature] >= threshold)
            for feature, threshold in THRESHOLDS.items()
        )

        rows.append({
            "center_seconds": round(center, 3),
            "event_votes": int(votes),
            "condition_true": bool(votes >= REQUIRED_VOTES),
            **features,
        })

    return pd.DataFrame(rows)


def simulate_rearm(scan_table, min_cooldown, quiet_rearm):
    alerts = []
    armed = True
    last_alert_time = -np.inf
    quiet_start = None

    for row in scan_table.itertuples(index=False):
        time = float(row.center_seconds)
        condition_true = bool(row.condition_true)

        if armed:
            if condition_true:
                alerts.append({
                    "detected_time_seconds": time,
                    "event_votes": int(row.event_votes),
                    "event_peak_acceleration_g": float(row.event_peak_acceleration_g),
                    "event_peak_gyroscope_dps": float(row.event_peak_gyroscope_dps),
                    "event_peak_jerk_g_per_second": float(row.event_peak_jerk_g_per_second),
                    "event_rotation_integral_deg": float(row.event_rotation_integral_deg),
                })
                armed = False
                last_alert_time = time
                quiet_start = None
            continue

        # Detector is disarmed after an alert.
        if condition_true:
            quiet_start = None
            continue

        # No event condition in this scan position.
        if quiet_start is None:
            quiet_start = time

        enough_cooldown = (time - last_alert_time) >= min_cooldown
        enough_quiet = (time - quiet_start) >= quiet_rearm

        if enough_cooldown and enough_quiet:
            armed = True
            quiet_start = None

    return alerts


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


def score_configuration(recordings, confirmed_lookup, min_cooldown, quiet_rearm):
    details = []

    for record in recordings:
        worker = record["worker"]
        source_file = record["source_file"]
        recording_type = record["recording_type"]
        scan_table = record["scan_table"]

        alerts = simulate_rearm(scan_table, min_cooldown, quiet_rearm)
        is_normal = "normal" in str(recording_type).lower()

        expected_time = np.nan
        expected_type = "normal"
        reliable_for_score = False
        matched = False
        matched_alert_count = 0
        extra_alert_count = len(alerts)
        nearest_error = np.nan

        if not is_normal:
            key = (str(worker), str(source_file))
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
                matched_mask = errors <= MATCH_TOLERANCE_SECONDS
                matched_alert_count = int(matched_mask.sum())
                matched = bool(matched_alert_count > 0)

                # One alert is enough to represent the real incident.
                extra_alert_count = len(alerts) - (1 if matched else 0)

        details.append({
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
            "matched_alert_count": matched_alert_count,
            "extra_alert_count": int(extra_alert_count),
            "alert_times_seconds": ",".join(
                f"{float(a['detected_time_seconds']):.2f}" for a in alerts
            ),
        })

    details = pd.DataFrame(details)

    reliable_incidents = details[
        (~details["is_normal_recording"])
        & details["reliable_for_score"]
    ]
    uncertain = details[
        (~details["is_normal_recording"])
        & (~details["reliable_for_score"])
    ]
    normal = details[details["is_normal_recording"]]

    reliable_matched = int(reliable_incidents["confirmed_event_matched"].sum())
    reliable_total = int(len(reliable_incidents))
    extra_alerts_all = int(details["extra_alert_count"].sum())
    normal_with_alert = int((normal["number_of_alerts"] > 0).sum())
    normal_alerts = int(normal["number_of_alerts"].sum())

    uncertain_matched = False
    if len(uncertain) == 1:
        uncertain_matched = bool(uncertain.iloc[0]["confirmed_event_matched"])

    summary = {
        "min_cooldown_seconds": float(min_cooldown),
        "quiet_rearm_seconds": float(quiet_rearm),
        "reliable_incidents_matched": reliable_matched,
        "reliable_incidents_tested": reliable_total,
        "reliable_detection_rate": (
            reliable_matched / reliable_total if reliable_total else np.nan
        ),
        "total_extra_alerts_all_recordings": extra_alerts_all,
        "normal_recordings_with_alert": normal_with_alert,
        "normal_recordings_tested": int(len(normal)),
        "total_normal_alerts": normal_alerts,
        "uncertain_ziqian_fall_matched": int(uncertain_matched),
    }

    return summary, details


def choose_best(grid):
    # Safety first: preserve all reliable incidents.
    # Then minimise extra alerts, then normal-recording alerts.
    # Finally prefer the shortest suppression settings so a later real event
    # can re-arm as soon as reasonably possible.
    ordered = grid.sort_values(
        [
            "reliable_detection_rate",
            "total_extra_alerts_all_recordings",
            "normal_recordings_with_alert",
            "total_normal_alerts",
            "min_cooldown_seconds",
            "quiet_rearm_seconds",
        ],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)

    return ordered.iloc[0].to_dict()


def main():
    print("=" * 90)
    print("GROUP 5 - EVENT-REARM STATE TUNING")
    print("Fixed Stage-1 thresholds + cooldown + quiet rearm | NO AI / NO ML")
    print("=" * 90)

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

    print("Building scan tables once...")

    for (worker, source_file, recording_type), recording in grouped:
        regular = resample_recording(recording)
        if regular.empty:
            raise SystemExit(
                f"ERROR: resampling failed for {worker} | {source_file}"
            )

        scan_table = build_scan_table(regular)
        if scan_table.empty:
            raise SystemExit(
                f"ERROR: scan table empty for {worker} | {source_file}"
            )

        recordings.append({
            "worker": worker,
            "source_file": source_file,
            "recording_type": recording_type,
            "scan_table": scan_table,
        })

        print(
            f"  {worker:12s} | {str(recording_type):12s} | "
            f"scan_positions={len(scan_table):3d} | {source_file}"
        )

    if len(recordings) != 12:
        raise SystemExit(f"ERROR: expected 12 recordings, found {len(recordings)}.")

    grid_rows = []
    detail_by_config = {}

    for min_cooldown in MIN_COOLDOWN_OPTIONS:
        for quiet_rearm in QUIET_REARM_OPTIONS:
            summary, details = score_configuration(
                recordings,
                confirmed_lookup,
                min_cooldown,
                quiet_rearm,
            )
            grid_rows.append(summary)
            detail_by_config[(float(min_cooldown), float(quiet_rearm))] = details

    grid = pd.DataFrame(grid_rows)
    best = choose_best(grid)

    best_key = (
        float(best["min_cooldown_seconds"]),
        float(best["quiet_rearm_seconds"]),
    )
    selected_details = detail_by_config[best_key].copy()

    numeric_grid = grid.select_dtypes(include=[np.number]).columns
    grid[numeric_grid] = grid[numeric_grid].round(6)
    grid.to_csv(GRID_OUTPUT, index=False)

    numeric_detail = selected_details.select_dtypes(include=[np.number]).columns
    selected_details[numeric_detail] = selected_details[numeric_detail].round(6)
    selected_details.to_csv(DETAIL_OUTPUT, index=False)

    final = pd.DataFrame([best])
    numeric_final = final.select_dtypes(include=[np.number]).columns
    final[numeric_final] = final[numeric_final].round(6)
    final.to_csv(FINAL_OUTPUT, index=False)

    print()
    print("=" * 90)
    print("TOP REARM CONFIGURATIONS")
    print("=" * 90)
    print(
        grid.sort_values(
            [
                "reliable_detection_rate",
                "total_extra_alerts_all_recordings",
                "normal_recordings_with_alert",
                "total_normal_alerts",
                "min_cooldown_seconds",
                "quiet_rearm_seconds",
            ],
            ascending=[False, True, True, True, True, True],
        ).head(10).to_string(index=False)
    )

    print()
    print("=" * 90)
    print("SELECTED REARM STATE")
    print("=" * 90)
    print(f"Minimum cooldown after alert: {best['min_cooldown_seconds']:.1f} s")
    print(f"Required continuous quiet before rearm: {best['quiet_rearm_seconds']:.1f} s")
    print(
        f"Reliable incidents matched: "
        f"{int(best['reliable_incidents_matched'])}/{int(best['reliable_incidents_tested'])}"
    )
    print(
        f"Extra alerts across all 12 recordings: "
        f"{int(best['total_extra_alerts_all_recordings'])}"
    )
    print(
        f"Normal recordings with >=1 alert: "
        f"{int(best['normal_recordings_with_alert'])}/{int(best['normal_recordings_tested'])}"
    )
    print(f"Total normal alerts: {int(best['total_normal_alerts'])}")
    print(
        "Uncertain Ziqian fall stress test matched: "
        f"{'YES' if int(best['uncertain_ziqian_fall_matched']) else 'NO'}"
    )

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
    print("IMPORTANT:")
    print("- The Stage-1 thresholds remain unchanged.")
    print("- This step only tunes when the detector is allowed to re-arm after an alert.")
    print("- Extra alerts outside the one confirmed incident are counted explicitly.")
    print("- Reliable incident detection is always prioritised over reducing nuisance alerts.")
    print("- The uncertain Ziqian fall remains a separate stress test and is not scored.")


if __name__ == "__main__":
    main()
