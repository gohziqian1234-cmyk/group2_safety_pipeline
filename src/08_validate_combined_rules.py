from pathlib import Path
import itertools

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 LEAVE-ONE-WORKER-OUT RULE VALIDATION
# Explainable threshold rules only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "window_features_analysis.csv"
FOLD_OUTPUT = PROCESSED / "rule_validation_folds.csv"
SUMMARY_OUTPUT = PROCESSED / "rule_validation_summary.csv"

FEATURES = [
    "gyro_max_dps",
    "accel_max_g",
    "axis_accel_std_total_g",
]

TARGETS = ["near_miss", "fall"]

REQUIRED_COLUMNS = [
    "worker",
    "source_file",
    "primary_label",
    *FEATURES,
]


def safe_divide(a, b):
    return np.nan if b == 0 else a / b


def candidate_thresholds(normal_values, target_values):
    """Create a small deterministic threshold set from TRAINING data only."""
    normal = pd.to_numeric(normal_values, errors="coerce").dropna().to_numpy(float)
    target = pd.to_numeric(target_values, errors="coerce").dropna().to_numpy(float)

    if len(normal) == 0 or len(target) == 0:
        return np.array([], dtype=float)

    values = []

    for q in [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 1.0]:
        values.append(float(np.quantile(normal, q)))

    for q in [0.00, 0.10, 0.25, 0.50, 0.75, 0.90]:
        values.append(float(np.quantile(target, q)))

    normal_max = float(np.max(normal))
    target_above_normal = target[target > normal_max]
    if len(target_above_normal) > 0:
        values.append((normal_max + float(np.min(target_above_normal))) / 2.0)

    values = np.unique(np.round(values, 6))
    return values


def apply_rule(data, rule_type, feature_1, threshold_1, feature_2=None, threshold_2=None):
    first = data[feature_1] >= threshold_1

    if rule_type == "single":
        return first

    if rule_type == "and_pair":
        second = data[feature_2] >= threshold_2
        return first & second

    raise ValueError(f"Unknown rule_type: {rule_type}")


def evaluate_predictions(data, target_label, predictions):
    frame = data.copy()
    frame["predicted_positive"] = np.asarray(predictions, dtype=bool)

    target = frame[frame["primary_label"] == target_label]
    normal = frame[frame["primary_label"] == "normal"]

    tp = int(target["predicted_positive"].sum())
    fn = int((~target["predicted_positive"]).sum())
    fp = int(normal["predicted_positive"].sum())
    tn = int((~normal["predicted_positive"]).sum())

    if len(target) > 0:
        event_hits = (
            target.groupby(["worker", "source_file"])["predicted_positive"]
            .max()
        )
        event_detection_rate = float(event_hits.mean())
        event_count = int(len(event_hits))
    else:
        event_detection_rate = np.nan
        event_count = 0

    if len(normal) > 0:
        normal_alarm = (
            normal.groupby(["worker", "source_file"])["predicted_positive"]
            .max()
        )
        normal_recording_false_alarm_rate = float(normal_alarm.mean())
        normal_recording_count = int(len(normal_alarm))
    else:
        normal_recording_false_alarm_rate = np.nan
        normal_recording_count = 0

    return {
        "event_detection_rate": event_detection_rate,
        "event_count": event_count,
        "window_sensitivity": safe_divide(tp, tp + fn),
        "window_false_positive_rate": safe_divide(fp, fp + tn),
        "normal_recording_false_alarm_rate": normal_recording_false_alarm_rate,
        "normal_recording_count": normal_recording_count,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def search_best_training_rule(training, target_label):
    comparison = training[
        training["primary_label"].isin(["normal", target_label])
    ].copy()

    normal = comparison[comparison["primary_label"] == "normal"]
    target = comparison[comparison["primary_label"] == target_label]

    thresholds = {
        feature: candidate_thresholds(normal[feature], target[feature])
        for feature in FEATURES
    }

    rows = []

    # Single-feature rules.
    for feature in FEATURES:
        for threshold in thresholds[feature]:
            pred = apply_rule(
                comparison,
                "single",
                feature,
                threshold,
            )
            metrics = evaluate_predictions(comparison, target_label, pred)
            rows.append({
                "rule_type": "single",
                "feature_1": feature,
                "threshold_1": float(threshold),
                "feature_2": "",
                "threshold_2": np.nan,
                "feature_count": 1,
                **metrics,
            })

    # Pairwise AND rules.
    for feature_1, feature_2 in itertools.combinations(FEATURES, 2):
        for threshold_1 in thresholds[feature_1]:
            for threshold_2 in thresholds[feature_2]:
                pred = apply_rule(
                    comparison,
                    "and_pair",
                    feature_1,
                    threshold_1,
                    feature_2,
                    threshold_2,
                )
                metrics = evaluate_predictions(comparison, target_label, pred)
                rows.append({
                    "rule_type": "and_pair",
                    "feature_1": feature_1,
                    "threshold_1": float(threshold_1),
                    "feature_2": feature_2,
                    "threshold_2": float(threshold_2),
                    "feature_count": 2,
                    **metrics,
                })

    result = pd.DataFrame(rows)

    # Safety priority: first keep rules that detect every TRAINING event.
    full_detection = result[np.isclose(result["event_detection_rate"], 1.0)]

    if not full_detection.empty:
        result = full_detection
    else:
        best_event_rate = result["event_detection_rate"].max()
        result = result[np.isclose(result["event_detection_rate"], best_event_rate)]

    # Then reduce nuisance alarms, while preferring simpler rules if tied.
    result = result.sort_values(
        [
            "normal_recording_false_alarm_rate",
            "window_false_positive_rate",
            "window_sensitivity",
            "feature_count",
            "threshold_1",
            "threshold_2",
        ],
        ascending=[True, True, False, True, False, False],
        na_position="last",
    )

    return result.iloc[0].to_dict()


def main():
    print("=" * 78)
    print("GROUP 5 - LEAVE-ONE-WORKER-OUT COMBINED RULE VALIDATION")
    print("Threshold rules only | NO AI / NO ML")
    print("=" * 78)

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"ERROR: missing input file:\n{INPUT_FILE}\nRun Code 05 first."
        )

    data = pd.read_csv(INPUT_FILE)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise SystemExit(
            "ERROR: missing columns:\n" + "\n".join(f" - {c}" for c in missing)
        )

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data = data.dropna(subset=FEATURES).copy()

    workers = sorted(data["worker"].astype(str).unique())
    fold_rows = []

    for target_label in TARGETS:
        print()
        print(f"TARGET: {target_label}")

        for held_out_worker in workers:
            training = data[data["worker"].astype(str) != held_out_worker].copy()
            testing = data[data["worker"].astype(str) == held_out_worker].copy()

            best = search_best_training_rule(training, target_label)

            test_comparison = testing[
                testing["primary_label"].isin(["normal", target_label])
            ].copy()

            test_pred = apply_rule(
                test_comparison,
                best["rule_type"],
                best["feature_1"],
                best["threshold_1"],
                None if best["feature_2"] == "" else best["feature_2"],
                None if pd.isna(best["threshold_2"]) else best["threshold_2"],
            )

            test_metrics = evaluate_predictions(
                test_comparison,
                target_label,
                test_pred,
            )

            row = {
                "target_label": target_label,
                "held_out_worker": held_out_worker,
                "rule_type": best["rule_type"],
                "feature_1": best["feature_1"],
                "threshold_1": best["threshold_1"],
                "feature_2": best["feature_2"],
                "threshold_2": best["threshold_2"],
                "training_event_detection_rate": best["event_detection_rate"],
                "training_normal_recording_false_alarm_rate": best["normal_recording_false_alarm_rate"],
                "test_event_detection_rate": test_metrics["event_detection_rate"],
                "test_window_sensitivity": test_metrics["window_sensitivity"],
                "test_window_false_positive_rate": test_metrics["window_false_positive_rate"],
                "test_normal_recording_false_alarm_rate": test_metrics["normal_recording_false_alarm_rate"],
            }
            fold_rows.append(row)

            rule_text = (
                f"{best['feature_1']} >= {best['threshold_1']:.4f}"
                if best["rule_type"] == "single"
                else (
                    f"{best['feature_1']} >= {best['threshold_1']:.4f} AND "
                    f"{best['feature_2']} >= {best['threshold_2']:.4f}"
                )
            )

            print(
                f"  hold out {held_out_worker:12s} | "
                f"event_detected={test_metrics['event_detection_rate']:.0f} | "
                f"normal_false_alarm={test_metrics['normal_recording_false_alarm_rate']:.0f} | "
                f"{rule_text}"
            )

    folds = pd.DataFrame(fold_rows)

    numeric_columns = folds.select_dtypes(include=[np.number]).columns
    folds[numeric_columns] = folds[numeric_columns].round(5)
    folds.to_csv(FOLD_OUTPUT, index=False)

    summary = (
        folds.groupby("target_label")
        .agg(
            workers_tested=("held_out_worker", "count"),
            workers_event_detected=("test_event_detection_rate", "sum"),
            mean_event_detection_rate=("test_event_detection_rate", "mean"),
            workers_with_normal_false_alarm=("test_normal_recording_false_alarm_rate", "sum"),
            mean_normal_recording_false_alarm_rate=("test_normal_recording_false_alarm_rate", "mean"),
            mean_window_sensitivity=("test_window_sensitivity", "mean"),
            mean_window_false_positive_rate=("test_window_false_positive_rate", "mean"),
        )
        .reset_index()
    )

    summary_numeric = summary.select_dtypes(include=[np.number]).columns
    summary[summary_numeric] = summary[summary_numeric].round(5)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print()
    print("=" * 78)
    print("LEAVE-ONE-WORKER-OUT SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))

    print()
    print("Files created:")
    print(FOLD_OUTPUT)
    print(SUMMARY_OUTPUT)

    print()
    print("IMPORTANT:")
    print("Each held-out worker is never used to derive that fold's thresholds.")
    print("This checks whether the rule logic generalises across different workers.")
    print("Do not upload final NESSO thresholds until these results are reviewed.")


if __name__ == "__main__":
    main()
