from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 TWO-STAGE STATE-MACHINE VALIDATION
# Stage 1: abnormal safety-event trigger
# Stage 2: possible-fall escalation
# Explainable rules only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "sequence_features.csv"
FOLD_OUTPUT = PROCESSED / "two_stage_state_machine_folds.csv"
FINAL_OUTPUT = PROCESSED / "two_stage_state_machine_final.csv"

EVENT_FEATURES = [
    "event_peak_acceleration_g",
    "event_peak_gyroscope_dps",
    "event_peak_jerk_g_per_second",
    "event_rotation_integral_deg",
]

FALL_FEATURE_POSTURE = "posture_change_late_deg"
FALL_FEATURE_LATE_GYRO = "late_gyro_mean_dps"

# Selected by Code 12 cross-validation.
EVENT_NORMAL_QUANTILE = 0.995
EVENT_REQUIRED_VOTES = 2

UNCERTAIN_WORKER = "ziqian"
UNCERTAIN_SOURCE = "ziqian_die_data.csv"

REQUIRED_COLUMNS = [
    "worker",
    "source_file",
    "sequence_label",
    *EVENT_FEATURES,
    FALL_FEATURE_POSTURE,
    FALL_FEATURE_LATE_GYRO,
]


def derive_event_thresholds(normal_training):
    thresholds = {}

    for feature in EVENT_FEATURES:
        series = pd.to_numeric(normal_training[feature], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"No normal data for {feature}")

        value = float(series.quantile(EVENT_NORMAL_QUANTILE))
        if not np.isfinite(value):
            raise ValueError(f"Invalid event threshold for {feature}")

        thresholds[feature] = value

    return thresholds


def event_trigger(data, thresholds):
    votes = np.zeros(len(data), dtype=int)

    for feature in EVENT_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
        votes += (values >= thresholds[feature]).astype(int)

    return votes >= EVENT_REQUIRED_VOTES, votes


def reliable_incident_mask(data):
    incident = data["sequence_label"].isin(["near_miss", "fall"])
    uncertain = (
        (data["worker"].astype(str) == UNCERTAIN_WORKER)
        & (data["source_file"].astype(str) == UNCERTAIN_SOURCE)
        & (data["sequence_label"] == "fall")
    )
    return incident & (~uncertain)


def threshold_candidates(values, direction):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.array([], dtype=float)

    unique = np.unique(np.round(arr, 8))
    candidates = list(unique)

    if len(unique) > 1:
        candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())

    spread = max(float(np.ptp(unique)), 1.0)

    if direction == "high":
        # +infinity equivalent lets the low-gyro branch act alone.
        candidates.append(float(unique.max() + spread + 1.0))
    elif direction == "low":
        # negative threshold lets the high-posture branch act alone.
        candidates.append(float(unique.min() - spread - 1.0))
    else:
        raise ValueError("direction must be 'high' or 'low'")

    return np.unique(np.asarray(candidates, dtype=float))


def fall_prediction(data, posture_threshold, late_gyro_threshold):
    posture = pd.to_numeric(
        data[FALL_FEATURE_POSTURE], errors="coerce"
    ).to_numpy(dtype=float)
    late_gyro = pd.to_numeric(
        data[FALL_FEATURE_LATE_GYRO], errors="coerce"
    ).to_numpy(dtype=float)

    # A possible fall is escalated when EITHER:
    # 1) posture remains substantially changed, OR
    # 2) later rotational movement is unusually low.
    return (posture >= posture_threshold) | (late_gyro <= late_gyro_threshold)


def choose_fall_rule(training, event_thresholds):
    reliable = training[reliable_incident_mask(training)].copy()
    falls = reliable[reliable["sequence_label"] == "fall"].copy()
    near = reliable[reliable["sequence_label"] == "near_miss"].copy()

    if falls.empty or near.empty:
        raise ValueError("Training fold needs both fall and near-miss incidents.")

    normal = training[training["sequence_label"] == "normal"].copy()
    normal_triggered, _ = event_trigger(normal, event_thresholds)
    triggered_normal = normal.loc[normal_triggered].copy()

    # Negatives for fall escalation = near misses + any normal false event triggers.
    negatives = pd.concat([near, triggered_normal], ignore_index=True)

    posture_values = pd.concat(
        [falls[FALL_FEATURE_POSTURE], negatives[FALL_FEATURE_POSTURE]],
        ignore_index=True,
    )
    gyro_values = pd.concat(
        [falls[FALL_FEATURE_LATE_GYRO], negatives[FALL_FEATURE_LATE_GYRO]],
        ignore_index=True,
    )

    posture_candidates = threshold_candidates(posture_values, "high")
    gyro_candidates = threshold_candidates(gyro_values, "low")

    rows = []

    for posture_threshold in posture_candidates:
        for gyro_threshold in gyro_candidates:
            fall_pred = fall_prediction(falls, posture_threshold, gyro_threshold)
            negative_pred = fall_prediction(
                negatives, posture_threshold, gyro_threshold
            ) if not negatives.empty else np.array([], dtype=bool)

            fall_detected = int(fall_pred.sum())
            fall_total = int(len(falls))
            false_fall = int(negative_pred.sum())
            negative_total = int(len(negatives))

            near_pred = fall_prediction(near, posture_threshold, gyro_threshold)
            near_false_fall = int(near_pred.sum())

            normal_false_fall = 0
            if not triggered_normal.empty:
                normal_false_fall = int(
                    fall_prediction(
                        triggered_normal,
                        posture_threshold,
                        gyro_threshold,
                    ).sum()
                )

            rows.append({
                "posture_threshold": float(posture_threshold),
                "late_gyro_threshold": float(gyro_threshold),
                "fall_detection_rate": fall_detected / fall_total,
                "falls_detected": fall_detected,
                "falls_total": fall_total,
                "negative_false_fall_rate": (
                    false_fall / negative_total if negative_total else 0.0
                ),
                "near_miss_false_falls": near_false_fall,
                "near_miss_total": int(len(near)),
                "triggered_normal_false_falls": normal_false_fall,
                "triggered_normal_total": int(len(triggered_normal)),
            })

    search = pd.DataFrame(rows)

    # First protect fall recall, then minimise false escalation.
    full_fall = search[np.isclose(search["fall_detection_rate"], 1.0)]
    if not full_fall.empty:
        search = full_fall
    else:
        best_rate = search["fall_detection_rate"].max()
        search = search[np.isclose(search["fall_detection_rate"], best_rate)]

    search = search.sort_values(
        [
            "negative_false_fall_rate",
            "near_miss_false_falls",
            "triggered_normal_false_falls",
            "posture_threshold",
            "late_gyro_threshold",
        ],
        ascending=[True, True, True, False, True],
    )

    return search.iloc[0].to_dict()


def classify_rows(data, event_thresholds, fall_rule):
    triggered, votes = event_trigger(data, event_thresholds)
    possible_fall = np.zeros(len(data), dtype=bool)

    if triggered.any():
        triggered_frame = data.loc[triggered].copy()
        possible_fall[triggered] = fall_prediction(
            triggered_frame,
            float(fall_rule["posture_threshold"]),
            float(fall_rule["late_gyro_threshold"]),
        )

    state = np.full(len(data), "SAFE", dtype=object)
    state[triggered] = "SAFETY_EVENT"
    state[possible_fall] = "POSSIBLE_FALL"

    return state, votes


def main():
    print("=" * 88)
    print("GROUP 5 - TWO-STAGE STATE-MACHINE VALIDATION")
    print("Event trigger -> possible-fall escalation | NO AI / NO ML")
    print("=" * 88)

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"ERROR: missing input file:\n{INPUT_FILE}\n"
            "Run Code 09 and Code 12 first."
        )

    data = pd.read_csv(INPUT_FILE)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise SystemExit(
            "ERROR: sequence_features.csv is missing columns:\n"
            + "\n".join(f" - {c}" for c in missing)
        )

    numeric_columns = [
        *EVENT_FEATURES,
        FALL_FEATURE_POSTURE,
        FALL_FEATURE_LATE_GYRO,
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[numeric_columns].isna().any().any():
        raise SystemExit("ERROR: missing/non-numeric values found in required features.")

    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise SystemExit("ERROR: non-finite values found in required features.")

    workers = sorted(data["worker"].astype(str).unique())
    if len(workers) != 4:
        raise SystemExit(f"ERROR: expected 4 workers, found {len(workers)}.")

    reliable_all = data[reliable_incident_mask(data)].copy()
    uncertain = data[
        (data["worker"].astype(str) == UNCERTAIN_WORKER)
        & (data["source_file"].astype(str) == UNCERTAIN_SOURCE)
        & (data["sequence_label"] == "fall")
    ].copy()

    if len(reliable_all) != 7 or len(uncertain) != 1:
        raise SystemExit(
            "ERROR: expected 7 reliable incidents and 1 uncertain Ziqian fall."
        )

    fold_rows = []

    for held_out_worker in workers:
        training = data[data["worker"].astype(str) != held_out_worker].copy()
        testing = data[data["worker"].astype(str) == held_out_worker].copy()

        training_normal = training[training["sequence_label"] == "normal"].copy()
        event_thresholds = derive_event_thresholds(training_normal)
        fall_rule = choose_fall_rule(training, event_thresholds)

        reliable_test = testing[reliable_incident_mask(testing)].copy()
        test_normal = testing[testing["sequence_label"] == "normal"].copy()

        incident_states, _ = classify_rows(
            reliable_test, event_thresholds, fall_rule
        ) if not reliable_test.empty else (np.array([], dtype=object), np.array([], dtype=int))

        normal_states, _ = classify_rows(test_normal, event_thresholds, fall_rule)

        incident_frame = reliable_test.copy()
        incident_frame["predicted_state"] = incident_states

        near = incident_frame[incident_frame["sequence_label"] == "near_miss"]
        falls = incident_frame[incident_frame["sequence_label"] == "fall"]

        near_tested = int(len(near))
        fall_tested = int(len(falls))

        near_event_detected = int(
            near["predicted_state"].isin(["SAFETY_EVENT", "POSSIBLE_FALL"]).sum()
        )
        near_correct = int((near["predicted_state"] == "SAFETY_EVENT").sum())
        near_false_fall = int((near["predicted_state"] == "POSSIBLE_FALL").sum())

        fall_event_detected = int(
            falls["predicted_state"].isin(["SAFETY_EVENT", "POSSIBLE_FALL"]).sum()
        )
        fall_correct = int((falls["predicted_state"] == "POSSIBLE_FALL").sum())

        normal_event_false_positions = int(
            np.isin(normal_states, ["SAFETY_EVENT", "POSSIBLE_FALL"]).sum()
        )
        normal_fall_false_positions = int((normal_states == "POSSIBLE_FALL").sum())

        normal_recording_false_event = int(normal_event_false_positions > 0)
        normal_recording_false_fall = int(normal_fall_false_positions > 0)

        fold_rows.append({
            "held_out_worker": held_out_worker,
            "near_miss_tested": near_tested,
            "near_miss_event_detected": near_event_detected,
            "near_miss_correct_safety_event": near_correct,
            "near_miss_false_possible_fall": near_false_fall,
            "fall_tested": fall_tested,
            "fall_event_detected": fall_event_detected,
            "fall_correct_possible_fall": fall_correct,
            "normal_positions_tested": int(len(test_normal)),
            "normal_false_event_positions": normal_event_false_positions,
            "normal_false_fall_positions": normal_fall_false_positions,
            "normal_recording_false_event": normal_recording_false_event,
            "normal_recording_false_fall": normal_recording_false_fall,
            "posture_threshold": fall_rule["posture_threshold"],
            "late_gyro_threshold": fall_rule["late_gyro_threshold"],
            **{f"event_threshold_{f}": event_thresholds[f] for f in EVENT_FEATURES},
        })

        print()
        print(f"HOLD OUT: {held_out_worker}")
        print(
            f"  Fall escalation rule: posture >= {fall_rule['posture_threshold']:.3f} deg "
            f"OR late gyro <= {fall_rule['late_gyro_threshold']:.3f} dps"
        )
        print(
            f"  Near miss: tested={near_tested}, event_detected={near_event_detected}, "
            f"correct_event={near_correct}, false_fall={near_false_fall}"
        )
        print(
            f"  Fall: tested={fall_tested}, event_detected={fall_event_detected}, "
            f"possible_fall={fall_correct}"
        )
        print(
            f"  Normal false event positions={normal_event_false_positions}, "
            f"false possible-fall positions={normal_fall_false_positions}"
        )

    folds = pd.DataFrame(fold_rows)

    summary = {
        "reliable_near_miss_correct": int(
            folds["near_miss_correct_safety_event"].sum()
        ),
        "reliable_near_miss_tested": int(folds["near_miss_tested"].sum()),
        "near_miss_false_possible_fall": int(
            folds["near_miss_false_possible_fall"].sum()
        ),
        "reliable_falls_correct_possible_fall": int(
            folds["fall_correct_possible_fall"].sum()
        ),
        "reliable_falls_tested": int(folds["fall_tested"].sum()),
        "reliable_incident_event_detection": int(
            folds["near_miss_event_detected"].sum()
            + folds["fall_event_detected"].sum()
        ),
        "reliable_incidents_tested": int(
            folds["near_miss_tested"].sum() + folds["fall_tested"].sum()
        ),
        "normal_false_event_positions": int(
            folds["normal_false_event_positions"].sum()
        ),
        "normal_positions_tested": int(folds["normal_positions_tested"].sum()),
        "workers_with_normal_false_event": int(
            folds["normal_recording_false_event"].sum()
        ),
        "normal_false_possible_fall_positions": int(
            folds["normal_false_fall_positions"].sum()
        ),
        "workers_with_normal_false_possible_fall": int(
            folds["normal_recording_false_fall"].sum()
        ),
    }

    # Derive the final offline configuration from all reliable development data.
    normal_all = data[data["sequence_label"] == "normal"].copy()
    final_event_thresholds = derive_event_thresholds(normal_all)
    final_fall_rule = choose_fall_rule(data, final_event_thresholds)

    reliable_states, _ = classify_rows(
        reliable_all,
        final_event_thresholds,
        final_fall_rule,
    )
    uncertain_state, uncertain_votes = classify_rows(
        uncertain,
        final_event_thresholds,
        final_fall_rule,
    )
    normal_states, _ = classify_rows(
        normal_all,
        final_event_thresholds,
        final_fall_rule,
    )

    reliable_eval = reliable_all.copy()
    reliable_eval["predicted_state"] = reliable_states

    final_row = {
        **summary,
        "normal_event_position_false_positive_rate": (
            summary["normal_false_event_positions"] / summary["normal_positions_tested"]
            if summary["normal_positions_tested"] else np.nan
        ),
        "normal_possible_fall_position_false_positive_rate": (
            summary["normal_false_possible_fall_positions"] / summary["normal_positions_tested"]
            if summary["normal_positions_tested"] else np.nan
        ),
        "final_posture_threshold_deg": final_fall_rule["posture_threshold"],
        "final_late_gyro_threshold_dps": final_fall_rule["late_gyro_threshold"],
        "final_reliable_near_miss_correct": int(
            (
                reliable_eval.loc[
                    reliable_eval["sequence_label"] == "near_miss",
                    "predicted_state",
                ] == "SAFETY_EVENT"
            ).sum()
        ),
        "final_reliable_falls_correct": int(
            (
                reliable_eval.loc[
                    reliable_eval["sequence_label"] == "fall",
                    "predicted_state",
                ] == "POSSIBLE_FALL"
            ).sum()
        ),
        "final_normal_false_event_positions": int(
            np.isin(normal_states, ["SAFETY_EVENT", "POSSIBLE_FALL"]).sum()
        ),
        "final_normal_false_possible_fall_positions": int(
            (normal_states == "POSSIBLE_FALL").sum()
        ),
        "uncertain_ziqian_fall_state": str(uncertain_state[0]),
        "uncertain_ziqian_fall_event_votes": int(uncertain_votes[0]),
        **{f"final_event_threshold_{f}": final_event_thresholds[f] for f in EVENT_FEATURES},
    }

    numeric_folds = folds.select_dtypes(include=[np.number]).columns
    folds[numeric_folds] = folds[numeric_folds].round(6)
    folds.to_csv(FOLD_OUTPUT, index=False)

    final = pd.DataFrame([final_row])
    numeric_final = final.select_dtypes(include=[np.number]).columns
    final[numeric_final] = final[numeric_final].round(6)
    final.to_csv(FINAL_OUTPUT, index=False)

    print()
    print("=" * 88)
    print("TWO-STAGE CROSS-VALIDATION SUMMARY")
    print("=" * 88)
    print(f"Reliable incident event detection: {summary['reliable_incident_event_detection']}/{summary['reliable_incidents_tested']}")
    print(f"Near misses correctly kept as SAFETY_EVENT: {summary['reliable_near_miss_correct']}/{summary['reliable_near_miss_tested']}")
    print(f"Reliable falls escalated to POSSIBLE_FALL: {summary['reliable_falls_correct_possible_fall']}/{summary['reliable_falls_tested']}")
    print(f"Near misses falsely escalated to POSSIBLE_FALL: {summary['near_miss_false_possible_fall']}")
    print(f"Normal false event positions: {summary['normal_false_event_positions']}/{summary['normal_positions_tested']}")
    print(f"Normal false POSSIBLE_FALL positions: {summary['normal_false_possible_fall_positions']}/{summary['normal_positions_tested']}")
    print(f"Workers with any normal false event: {summary['workers_with_normal_false_event']}/4")
    print(f"Workers with any normal false POSSIBLE_FALL: {summary['workers_with_normal_false_possible_fall']}/4")

    print()
    print("FINAL OFFLINE STATE-MACHINE CONFIGURATION:")
    print(f"Stage 1: >= {EVENT_REQUIRED_VOTES}/4 event signals above normal {EVENT_NORMAL_QUANTILE * 100:.1f}th percentile")
    for feature in EVENT_FEATURES:
        print(f"  {feature:38s} >= {final_event_thresholds[feature]:.5f}")
    print(
        f"Stage 2 POSSIBLE_FALL if posture_change_late_deg >= "
        f"{final_fall_rule['posture_threshold']:.5f} OR late_gyro_mean_dps <= "
        f"{final_fall_rule['late_gyro_threshold']:.5f}"
    )

    print()
    print("FINAL OFFLINE CHECK:")
    print(f"Reliable near misses correct: {final_row['final_reliable_near_miss_correct']}/4")
    print(f"Reliable falls correct: {final_row['final_reliable_falls_correct']}/3")
    print(f"Normal false event positions: {final_row['final_normal_false_event_positions']}/{len(normal_all)}")
    print(f"Normal false POSSIBLE_FALL positions: {final_row['final_normal_false_possible_fall_positions']}/{len(normal_all)}")
    print(
        f"Uncertain Ziqian fall stress test: {final_row['uncertain_ziqian_fall_state']} "
        f"({final_row['uncertain_ziqian_fall_event_votes']}/4 event votes)"
    )

    print()
    print("Files created:")
    print(FOLD_OUTPUT)
    print(FINAL_OUTPUT)

    print()
    print("IMPORTANT:")
    print("- SAFETY_EVENT means an abnormal event was detected but a fall is not yet confirmed.")
    print("- POSSIBLE_FALL is an escalation based on the post-event sequence, not a medical certainty.")
    print("- The uncertain Ziqian fall remains excluded from rule derivation and scoring.")
    print("- This script uses only explainable thresholds and a state-machine structure; no AI/ML.")


if __name__ == "__main__":
    main()
