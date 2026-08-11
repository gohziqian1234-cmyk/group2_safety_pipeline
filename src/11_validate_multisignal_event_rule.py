from pathlib import Path
import itertools

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 MULTI-SIGNAL EVENT RULE VALIDATION
# Explainable voting rule only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "sequence_features.csv"
FOLD_OUTPUT = PROCESSED / "multisignal_event_validation_folds.csv"
SUMMARY_OUTPUT = PROCESSED / "multisignal_event_validation_summary.csv"

FEATURES = [
    "event_peak_acceleration_g",
    "event_peak_gyroscope_dps",
    "event_peak_jerk_g_per_second",
    "event_rotation_integral_deg",
]

QUANTILES = [0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975]

REQUIRED_COLUMNS = [
    "worker",
    "source_file",
    "sequence_label",
    *FEATURES,
]


def safe_divide(a, b):
    return np.nan if b == 0 else a / b


def build_predictions(data, feature_subset, thresholds, required_votes):
    votes = np.zeros(len(data), dtype=int)

    for feature in feature_subset:
        votes += (
            pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
            >= thresholds[feature]
        ).astype(int)

    return votes >= required_votes


def evaluate(data, predictions):
    frame = data.copy()
    frame["predicted_event"] = np.asarray(predictions, dtype=bool)

    incident_mask = frame["sequence_label"].isin(["near_miss", "fall"])
    normal_mask = frame["sequence_label"] == "normal"

    incident = frame[incident_mask]
    normal = frame[normal_mask]

    tp = int(incident["predicted_event"].sum())
    fn = int((~incident["predicted_event"]).sum())
    fp = int(normal["predicted_event"].sum())
    tn = int((~normal["predicted_event"]).sum())

    incident_detection_rate = safe_divide(tp, tp + fn)
    normal_position_false_positive_rate = safe_divide(fp, fp + tn)

    if len(normal) > 0:
        normal_recording_alarm = (
            normal
            .groupby(["worker", "source_file"])["predicted_event"]
            .max()
        )
        normal_recording_false_alarm_rate = float(normal_recording_alarm.mean())
    else:
        normal_recording_false_alarm_rate = np.nan

    return {
        "incident_detection_rate": incident_detection_rate,
        "normal_position_false_positive_rate": normal_position_false_positive_rate,
        "normal_recording_false_alarm_rate": normal_recording_false_alarm_rate,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def search_best_rule(training):
    normal = training[training["sequence_label"] == "normal"]
    incident = training[training["sequence_label"].isin(["near_miss", "fall"])]

    if normal.empty or incident.empty:
        raise ValueError("Training fold is missing normal or incident rows.")

    rows = []

    for subset_size in [2, 3, 4]:
        for feature_subset in itertools.combinations(FEATURES, subset_size):
            for quantile in QUANTILES:
                thresholds = {
                    feature: float(
                        pd.to_numeric(normal[feature], errors="coerce")
                        .dropna()
                        .quantile(quantile)
                    )
                    for feature in feature_subset
                }

                if not all(np.isfinite(v) for v in thresholds.values()):
                    continue

                for required_votes in range(1, subset_size + 1):
                    predictions = build_predictions(
                        training,
                        feature_subset,
                        thresholds,
                        required_votes,
                    )

                    metrics = evaluate(training, predictions)

                    row = {
                        "feature_subset": "+".join(feature_subset),
                        "subset_size": subset_size,
                        "normal_quantile": quantile,
                        "required_votes": required_votes,
                        "threshold_event_peak_acceleration_g": np.nan,
                        "threshold_event_peak_gyroscope_dps": np.nan,
                        "threshold_event_peak_jerk_g_per_second": np.nan,
                        "threshold_event_rotation_integral_deg": np.nan,
                        **metrics,
                    }

                    for feature, threshold in thresholds.items():
                        row[f"threshold_{feature}"] = threshold

                    rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No candidate rules were generated.")

    # Safety priority: prefer rules that detect every training incident.
    perfect = result[np.isclose(result["incident_detection_rate"], 1.0)]

    if not perfect.empty:
        result = perfect
    else:
        best_recall = result["incident_detection_rate"].max()
        result = result[np.isclose(result["incident_detection_rate"], best_recall)]

    # Then reduce nuisance alarms. Prefer simpler rules only when performance ties.
    result = result.sort_values(
        [
            "normal_recording_false_alarm_rate",
            "normal_position_false_positive_rate",
            "subset_size",
            "required_votes",
            "normal_quantile",
        ],
        ascending=[True, True, True, True, False],
    )

    return result.iloc[0].to_dict()


def parse_feature_subset(text):
    return tuple(str(text).split("+"))


def thresholds_from_rule(rule):
    thresholds = {}
    for feature in parse_feature_subset(rule["feature_subset"]):
        value = rule[f"threshold_{feature}"]
        if pd.isna(value):
            raise ValueError(f"Missing threshold for {feature}")
        thresholds[feature] = float(value)
    return thresholds


def main():
    print("=" * 82)
    print("GROUP 5 - MULTI-SIGNAL SAFETY-EVENT VALIDATION")
    print("Voting thresholds derived from training-normal quantiles | NO AI / NO ML")
    print("=" * 82)

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

    allowed_labels = {"normal", "near_miss", "fall"}
    unexpected = set(data["sequence_label"].astype(str).unique()) - allowed_labels
    if unexpected:
        raise SystemExit(
            "ERROR: unexpected sequence labels: " + ", ".join(sorted(unexpected))
        )

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    if data[FEATURES].isna().any().any():
        raise SystemExit("ERROR: missing/non-numeric values found in sequence features.")

    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise SystemExit("ERROR: non-finite values found in sequence features.")

    workers = sorted(data["worker"].astype(str).unique())

    if len(workers) != 4:
        raise SystemExit(f"ERROR: expected 4 workers, found {len(workers)}.")

    fold_rows = []

    for held_out_worker in workers:
        training = data[data["worker"].astype(str) != held_out_worker].copy()
        testing = data[data["worker"].astype(str) == held_out_worker].copy()

        train_incident_count = int(
            training["sequence_label"].isin(["near_miss", "fall"]).sum()
        )
        test_incident_count = int(
            testing["sequence_label"].isin(["near_miss", "fall"]).sum()
        )

        if train_incident_count != 6 or test_incident_count != 2:
            raise SystemExit(
                f"ERROR: unexpected incident counts for holdout {held_out_worker}: "
                f"train={train_incident_count}, test={test_incident_count}"
            )

        best = search_best_rule(training)
        feature_subset = parse_feature_subset(best["feature_subset"])
        thresholds = thresholds_from_rule(best)
        required_votes = int(best["required_votes"])

        test_predictions = build_predictions(
            testing,
            feature_subset,
            thresholds,
            required_votes,
        )
        test_metrics = evaluate(testing, test_predictions)

        incident_test = testing[
            testing["sequence_label"].isin(["near_miss", "fall"])
        ].copy()
        incident_test["predicted_event"] = build_predictions(
            incident_test,
            feature_subset,
            thresholds,
            required_votes,
        )

        near_hit = int(
            incident_test.loc[
                incident_test["sequence_label"] == "near_miss",
                "predicted_event",
            ].max()
        )
        fall_hit = int(
            incident_test.loc[
                incident_test["sequence_label"] == "fall",
                "predicted_event",
            ].max()
        )

        fold_row = {
            "held_out_worker": held_out_worker,
            "feature_subset": best["feature_subset"],
            "normal_quantile": best["normal_quantile"],
            "required_votes": required_votes,
            "training_incident_detection_rate": best["incident_detection_rate"],
            "training_normal_position_false_positive_rate": best[
                "normal_position_false_positive_rate"
            ],
            "training_normal_recording_false_alarm_rate": best[
                "normal_recording_false_alarm_rate"
            ],
            "test_incident_detection_rate": test_metrics["incident_detection_rate"],
            "test_near_miss_detected": near_hit,
            "test_fall_detected": fall_hit,
            "test_normal_position_false_positive_rate": test_metrics[
                "normal_position_false_positive_rate"
            ],
            "test_normal_recording_false_alarm_rate": test_metrics[
                "normal_recording_false_alarm_rate"
            ],
            "threshold_event_peak_acceleration_g": best[
                "threshold_event_peak_acceleration_g"
            ],
            "threshold_event_peak_gyroscope_dps": best[
                "threshold_event_peak_gyroscope_dps"
            ],
            "threshold_event_peak_jerk_g_per_second": best[
                "threshold_event_peak_jerk_g_per_second"
            ],
            "threshold_event_rotation_integral_deg": best[
                "threshold_event_rotation_integral_deg"
            ],
        }
        fold_rows.append(fold_row)

        threshold_text = ", ".join(
            f"{feature}>={thresholds[feature]:.3f}"
            for feature in feature_subset
        )

        print()
        print(f"HOLD OUT: {held_out_worker}")
        print(
            f"  Rule: at least {required_votes}/{len(feature_subset)} conditions true"
        )
        print(f"  Thresholds: {threshold_text}")
        print(
            f"  TEST near miss detected: {near_hit} | fall detected: {fall_hit}"
        )
        print(
            "  TEST normal position false-positive rate: "
            f"{test_metrics['normal_position_false_positive_rate']:.4f}"
        )
        print(
            "  TEST normal recording false-alarm rate: "
            f"{test_metrics['normal_recording_false_alarm_rate']:.4f}"
        )

    folds = pd.DataFrame(fold_rows)
    numeric = folds.select_dtypes(include=[np.number]).columns
    folds[numeric] = folds[numeric].round(5)
    folds.to_csv(FOLD_OUTPUT, index=False)

    summary = pd.DataFrame(
        [{
            "workers_tested": int(len(folds)),
            "near_miss_workers_detected": int(folds["test_near_miss_detected"].sum()),
            "fall_workers_detected": int(folds["test_fall_detected"].sum()),
            "total_incidents_detected": int(
                folds["test_near_miss_detected"].sum()
                + folds["test_fall_detected"].sum()
            ),
            "total_incidents_tested": 8,
            "mean_incident_detection_rate": float(
                folds["test_incident_detection_rate"].mean()
            ),
            "mean_normal_position_false_positive_rate": float(
                folds["test_normal_position_false_positive_rate"].mean()
            ),
            "workers_with_normal_recording_false_alarm": int(
                (folds["test_normal_recording_false_alarm_rate"] > 0).sum()
            ),
            "mean_normal_recording_false_alarm_rate": float(
                folds["test_normal_recording_false_alarm_rate"].mean()
            ),
        }]
    )

    summary_numeric = summary.select_dtypes(include=[np.number]).columns
    summary[summary_numeric] = summary[summary_numeric].round(5)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print()
    print("=" * 82)
    print("MULTI-SIGNAL EVENT VALIDATION SUMMARY")
    print("=" * 82)
    print(summary.to_string(index=False))

    print()
    print("Files created:")
    print(FOLD_OUTPUT)
    print(SUMMARY_OUTPUT)

    print()
    print("IMPORTANT:")
    print("This validates SAFETY-EVENT detection first, not fall-vs-near-miss classification.")
    print("Each held-out worker is excluded from threshold derivation for that fold.")
    print("The next step is only justified if this event trigger generalises well enough.")


if __name__ == "__main__":
    main()
