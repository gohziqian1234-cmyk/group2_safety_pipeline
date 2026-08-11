from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 FIXED EVENT-RULE CROSS VALIDATION
# Explainable threshold voting only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "sequence_features.csv"
GRID_OUTPUT = PROCESSED / "fixed_event_rule_cv_grid.csv"
FOLD_OUTPUT = PROCESSED / "fixed_event_rule_cv_folds.csv"
FINAL_OUTPUT = PROCESSED / "fixed_event_rule_final.csv"

FEATURES = [
    "event_peak_acceleration_g",
    "event_peak_gyroscope_dps",
    "event_peak_jerk_g_per_second",
    "event_rotation_integral_deg",
]

QUANTILES = [0.90, 0.925, 0.95, 0.975, 0.98, 0.99, 0.995]
VOTE_OPTIONS = [2, 3]

UNCERTAIN_WORKER = "ziqian"
UNCERTAIN_SOURCE = "ziqian_die_data.csv"

REQUIRED_COLUMNS = [
    "worker",
    "source_file",
    "sequence_label",
    *FEATURES,
]


def safe_divide(a, b):
    return np.nan if b == 0 else a / b


def derive_thresholds(normal_training, quantile):
    thresholds = {}

    for feature in FEATURES:
        series = pd.to_numeric(normal_training[feature], errors="coerce").dropna()

        if series.empty:
            raise ValueError(f"No normal training data for feature: {feature}")

        threshold = float(series.quantile(quantile))

        if not np.isfinite(threshold):
            raise ValueError(f"Invalid threshold for feature: {feature}")

        thresholds[feature] = threshold

    return thresholds


def predict(data, thresholds, required_votes):
    votes = np.zeros(len(data), dtype=int)

    for feature in FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
        votes += (values >= thresholds[feature]).astype(int)

    return votes >= required_votes


def normal_metrics(normal_data, predictions):
    frame = normal_data.copy()
    frame["predicted_event"] = np.asarray(predictions, dtype=bool)

    fp = int(frame["predicted_event"].sum())
    tn = int((~frame["predicted_event"]).sum())

    position_fpr = safe_divide(fp, fp + tn)

    if frame.empty:
        recording_far = np.nan
        recording_count = 0
    else:
        recording_alarm = (
            frame.groupby(["worker", "source_file"])["predicted_event"]
            .max()
        )
        recording_far = float(recording_alarm.mean())
        recording_count = int(len(recording_alarm))

    return {
        "normal_position_false_positive_rate": position_fpr,
        "normal_recording_false_alarm_rate": recording_far,
        "normal_recording_count": recording_count,
        "normal_false_positive_positions": fp,
        "normal_total_positions": int(len(frame)),
    }


def reliable_incident_mask(data):
    is_incident = data["sequence_label"].isin(["near_miss", "fall"])
    is_uncertain = (
        (data["worker"].astype(str) == UNCERTAIN_WORKER)
        & (data["source_file"].astype(str) == UNCERTAIN_SOURCE)
        & (data["sequence_label"] == "fall")
    )
    return is_incident & (~is_uncertain)


def evaluate_config(data, quantile, required_votes):
    workers = sorted(data["worker"].astype(str).unique())
    fold_rows = []

    for held_out_worker in workers:
        train = data[data["worker"].astype(str) != held_out_worker].copy()
        test = data[data["worker"].astype(str) == held_out_worker].copy()

        train_normal = train[train["sequence_label"] == "normal"].copy()
        test_normal = test[test["sequence_label"] == "normal"].copy()

        thresholds = derive_thresholds(train_normal, quantile)

        reliable_test = test[reliable_incident_mask(test)].copy()
        reliable_predictions = predict(
            reliable_test,
            thresholds,
            required_votes,
        ) if not reliable_test.empty else np.array([], dtype=bool)

        detected = int(np.asarray(reliable_predictions, dtype=bool).sum())
        tested = int(len(reliable_test))

        near_mask = reliable_test["sequence_label"] == "near_miss"
        fall_mask = reliable_test["sequence_label"] == "fall"

        near_tested = int(near_mask.sum())
        fall_tested = int(fall_mask.sum())

        near_detected = int(
            np.asarray(reliable_predictions, dtype=bool)[near_mask.to_numpy()].sum()
        ) if near_tested else 0

        fall_detected = int(
            np.asarray(reliable_predictions, dtype=bool)[fall_mask.to_numpy()].sum()
        ) if fall_tested else 0

        normal_predictions = predict(test_normal, thresholds, required_votes)
        nmetrics = normal_metrics(test_normal, normal_predictions)

        fold_rows.append({
            "quantile": quantile,
            "required_votes": required_votes,
            "held_out_worker": held_out_worker,
            "reliable_incidents_tested": tested,
            "reliable_incidents_detected": detected,
            "near_miss_tested": near_tested,
            "near_miss_detected": near_detected,
            "fall_tested": fall_tested,
            "fall_detected": fall_detected,
            **nmetrics,
            **{f"threshold_{feature}": thresholds[feature] for feature in FEATURES},
        })

    folds = pd.DataFrame(fold_rows)

    total_incidents = int(folds["reliable_incidents_tested"].sum())
    total_detected = int(folds["reliable_incidents_detected"].sum())
    total_near = int(folds["near_miss_tested"].sum())
    detected_near = int(folds["near_miss_detected"].sum())
    total_fall = int(folds["fall_tested"].sum())
    detected_fall = int(folds["fall_detected"].sum())

    workers_with_false_alarm = int(
        (folds["normal_recording_false_alarm_rate"] > 0).sum()
    )

    total_normal_positions = int(folds["normal_total_positions"].sum())
    total_false_positions = int(folds["normal_false_positive_positions"].sum())

    summary = {
        "quantile": quantile,
        "required_votes": required_votes,
        "reliable_incidents_detected": total_detected,
        "reliable_incidents_tested": total_incidents,
        "reliable_incident_detection_rate": safe_divide(total_detected, total_incidents),
        "near_miss_detected": detected_near,
        "near_miss_tested": total_near,
        "fall_detected": detected_fall,
        "fall_tested": total_fall,
        "workers_with_normal_false_alarm": workers_with_false_alarm,
        "workers_tested": int(len(folds)),
        "normal_position_false_positive_rate": safe_divide(
            total_false_positions,
            total_normal_positions,
        ),
        "mean_normal_recording_false_alarm_rate": float(
            folds["normal_recording_false_alarm_rate"].mean()
        ),
    }

    return summary, folds


def choose_best(grid):
    # Safety first: highest reliable incident recall.
    # Then minimise workers with any normal false alarm.
    # Then minimise normal-position FPR.
    # Then prefer stricter percentile and fewer required votes only as tie-breaks.
    ordered = grid.sort_values(
        [
            "reliable_incident_detection_rate",
            "workers_with_normal_false_alarm",
            "normal_position_false_positive_rate",
            "quantile",
            "required_votes",
        ],
        ascending=[False, True, True, False, True],
    ).reset_index(drop=True)

    return ordered.iloc[0].to_dict()


def main():
    print("=" * 86)
    print("GROUP 5 - FIXED EVENT-RULE CROSS VALIDATION")
    print("Same rule structure across all workers | NO AI / NO ML")
    print("=" * 86)

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"ERROR: missing input file:\n{INPUT_FILE}\n"
            "Run src\\09_build_sequence_features.py first."
        )

    data = pd.read_csv(INPUT_FILE)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise SystemExit(
            "ERROR: sequence_features.csv is missing columns:\n"
            + "\n".join(f" - {column}" for column in missing)
        )

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    if data[FEATURES].isna().any().any():
        raise SystemExit("ERROR: missing/non-numeric feature values found.")

    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise SystemExit("ERROR: non-finite feature values found.")

    workers = sorted(data["worker"].astype(str).unique())

    if len(workers) != 4:
        raise SystemExit(f"ERROR: expected 4 workers, found {len(workers)}.")

    reliable_incidents = data[reliable_incident_mask(data)].copy()
    uncertain = data[
        (data["worker"].astype(str) == UNCERTAIN_WORKER)
        & (data["source_file"].astype(str) == UNCERTAIN_SOURCE)
        & (data["sequence_label"] == "fall")
    ].copy()

    if len(reliable_incidents) != 7:
        raise SystemExit(
            f"ERROR: expected 7 reliable incidents after excluding the uncertain fall; "
            f"found {len(reliable_incidents)}."
        )

    if len(uncertain) != 1:
        raise SystemExit(
            f"ERROR: expected exactly 1 uncertain Ziqian fall row; found {len(uncertain)}."
        )

    if int((reliable_incidents["sequence_label"] == "near_miss").sum()) != 4:
        raise SystemExit("ERROR: expected 4 reliable near misses.")

    if int((reliable_incidents["sequence_label"] == "fall").sum()) != 3:
        raise SystemExit("ERROR: expected 3 reliable falls.")

    print("Reliable development incidents: 7 (4 near miss + 3 fall)")
    print("Uncertain stress-test incident: Ziqian fall @ 64.05 s")
    print()

    grid_rows = []
    all_fold_rows = []

    for quantile in QUANTILES:
        for required_votes in VOTE_OPTIONS:
            summary, folds = evaluate_config(data, quantile, required_votes)
            grid_rows.append(summary)
            all_fold_rows.append(folds)

    grid = pd.DataFrame(grid_rows)
    all_folds = pd.concat(all_fold_rows, ignore_index=True)

    best = choose_best(grid)
    best_quantile = float(best["quantile"])
    best_votes = int(best["required_votes"])

    selected_folds = all_folds[
        np.isclose(all_folds["quantile"], best_quantile)
        & (all_folds["required_votes"] == best_votes)
    ].copy()

    normal_all = data[data["sequence_label"] == "normal"].copy()
    final_thresholds = derive_thresholds(normal_all, best_quantile)

    reliable_predictions = predict(
        reliable_incidents,
        final_thresholds,
        best_votes,
    )
    uncertain_prediction = bool(
        predict(uncertain, final_thresholds, best_votes)[0]
    )

    normal_predictions = predict(normal_all, final_thresholds, best_votes)
    final_normal_metrics = normal_metrics(normal_all, normal_predictions)

    final_row = {
        "selected_normal_quantile": best_quantile,
        "required_votes_out_of_4": best_votes,
        "crossvalidated_reliable_incidents_detected": int(
            best["reliable_incidents_detected"]
        ),
        "crossvalidated_reliable_incidents_tested": int(
            best["reliable_incidents_tested"]
        ),
        "crossvalidated_detection_rate": best[
            "reliable_incident_detection_rate"
        ],
        "crossvalidated_workers_with_normal_false_alarm": int(
            best["workers_with_normal_false_alarm"]
        ),
        "crossvalidated_normal_position_false_positive_rate": best[
            "normal_position_false_positive_rate"
        ],
        "all_reliable_incidents_triggered_using_final_thresholds": int(
            np.asarray(reliable_predictions, dtype=bool).sum()
        ),
        "uncertain_ziqian_fall_triggered_stress_test": int(uncertain_prediction),
        "all_normal_position_false_positive_rate_final_thresholds": final_normal_metrics[
            "normal_position_false_positive_rate"
        ],
        "all_normal_recording_false_alarm_rate_final_thresholds": final_normal_metrics[
            "normal_recording_false_alarm_rate"
        ],
        **{f"threshold_{feature}": final_thresholds[feature] for feature in FEATURES},
    }

    final = pd.DataFrame([final_row])

    numeric_grid = grid.select_dtypes(include=[np.number]).columns
    grid[numeric_grid] = grid[numeric_grid].round(6)

    numeric_folds = selected_folds.select_dtypes(include=[np.number]).columns
    selected_folds[numeric_folds] = selected_folds[numeric_folds].round(6)

    numeric_final = final.select_dtypes(include=[np.number]).columns
    final[numeric_final] = final[numeric_final].round(6)

    grid.to_csv(GRID_OUTPUT, index=False)
    selected_folds.to_csv(FOLD_OUTPUT, index=False)
    final.to_csv(FINAL_OUTPUT, index=False)

    print("TOP CROSS-VALIDATED CONFIGURATIONS:")
    print(
        grid.sort_values(
            [
                "reliable_incident_detection_rate",
                "workers_with_normal_false_alarm",
                "normal_position_false_positive_rate",
                "quantile",
            ],
            ascending=[False, True, True, False],
        )[
            [
                "quantile",
                "required_votes",
                "reliable_incidents_detected",
                "reliable_incidents_tested",
                "workers_with_normal_false_alarm",
                "normal_position_false_positive_rate",
            ]
        ].head(8).to_string(index=False)
    )

    print()
    print("=" * 86)
    print("SELECTED FIXED RULE")
    print("=" * 86)
    print(
        f"Trigger POSSIBLE SAFETY EVENT when at least {best_votes}/4 signals exceed "
        f"the training-normal {best_quantile * 100:.1f}th-percentile thresholds."
    )
    print()
    for feature in FEATURES:
        print(f"  {feature:38s} >= {final_thresholds[feature]:.5f}")

    print()
    print("LEAVE-ONE-WORKER-OUT RESULTS FOR SELECTED RULE:")
    print(
        selected_folds[
            [
                "held_out_worker",
                "reliable_incidents_detected",
                "reliable_incidents_tested",
                "near_miss_detected",
                "near_miss_tested",
                "fall_detected",
                "fall_tested",
                "normal_position_false_positive_rate",
                "normal_recording_false_alarm_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("FINAL-OFFLINE CHECK USING ALL NORMAL DATA FOR THRESHOLDS:")
    print(
        f"Reliable incidents triggered: "
        f"{int(np.asarray(reliable_predictions, dtype=bool).sum())}/7"
    )
    print(
        f"Uncertain Ziqian fall stress-test triggered: "
        f"{'YES' if uncertain_prediction else 'NO'}"
    )
    print(
        f"Normal position false-positive rate: "
        f"{final_normal_metrics['normal_position_false_positive_rate']:.4f}"
    )
    print(
        f"Normal recording false-alarm rate: "
        f"{final_normal_metrics['normal_recording_false_alarm_rate']:.4f}"
    )

    print()
    print("Files created:")
    print(GRID_OUTPUT)
    print(FOLD_OUTPUT)
    print(FINAL_OUTPUT)

    print()
    print("IMPORTANT:")
    print("- Ziqian fall is NOT used to choose or validate the event rule.")
    print("- It is reported separately as an uncertain stress test.")
    print("- This is a POSSIBLE SAFETY EVENT trigger, not yet fall-vs-near-miss classification.")
    print("- Thresholds are derived from normal movement only; no AI/ML is used.")


if __name__ == "__main__":
    main()
