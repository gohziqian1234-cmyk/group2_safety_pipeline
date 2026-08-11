from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 EXPLORATORY THRESHOLD SWEEP
# Rule-based analysis only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
REPORT_FOLDER = PROJECT / "reports" / "threshold_plots"

INPUT_FILE = PROCESSED / "window_features_analysis.csv"
SWEEP_OUTPUT = PROCESSED / "threshold_sweep_all.csv"
RECOMMENDATION_OUTPUT = PROCESSED / "threshold_recommendations.csv"

# Strong, simple, edge-friendly features from Code 06.
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


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def build_threshold_grid(normal_values, target_values):
    """Create deterministic candidate thresholds from observed values."""
    combined = np.concatenate([
        np.asarray(normal_values, dtype=float),
        np.asarray(target_values, dtype=float),
    ])

    combined = combined[np.isfinite(combined)]

    if len(combined) == 0:
        return np.array([], dtype=float)

    # Include observed values plus midpoints so a threshold can sit between
    # neighbouring measurements instead of only on top of a sample value.
    unique_values = np.unique(np.round(combined, 8))

    if len(unique_values) == 1:
        return unique_values

    midpoints = (
        unique_values[:-1] + unique_values[1:]
    ) / 2.0

    thresholds = np.unique(
        np.concatenate([unique_values, midpoints])
    )

    return thresholds


def evaluate_threshold(data, feature, target_label, threshold):
    """
    Evaluate one simple rule:
        feature >= threshold -> incident candidate

    Metrics are descriptive. Final rules are validated later with
    leave-one-worker-out testing.
    """
    comparison = data[
        data["primary_label"].isin(["normal", target_label])
    ].copy()

    comparison["predicted_positive"] = (
        comparison[feature] >= threshold
    )

    target_mask = comparison["primary_label"] == target_label
    normal_mask = comparison["primary_label"] == "normal"

    tp = int((target_mask & comparison["predicted_positive"]).sum())
    fn = int((target_mask & ~comparison["predicted_positive"]).sum())
    fp = int((normal_mask & comparison["predicted_positive"]).sum())
    tn = int((normal_mask & ~comparison["predicted_positive"]).sum())

    sensitivity = safe_divide(tp, tp + fn)
    false_positive_rate = safe_divide(fp, fp + tn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)

    # Event-level detection: did at least one labelled target window from
    # each incident recording cross the threshold?
    target_rows = comparison[target_mask]

    if len(target_rows) > 0:
        target_event_results = (
            target_rows
            .groupby(["worker", "source_file"])["predicted_positive"]
            .max()
        )
        event_detection_rate = float(target_event_results.mean())
        target_event_count = int(len(target_event_results))
    else:
        event_detection_rate = np.nan
        target_event_count = 0

    # Normal-recording false alarm: did any window in each normal recording
    # cross the threshold?
    normal_rows = comparison[normal_mask]

    if len(normal_rows) > 0:
        normal_recording_results = (
            normal_rows
            .groupby(["worker", "source_file"])["predicted_positive"]
            .max()
        )
        normal_recording_false_alarm_rate = float(
            normal_recording_results.mean()
        )
        normal_recording_count = int(len(normal_recording_results))
    else:
        normal_recording_false_alarm_rate = np.nan
        normal_recording_count = 0

    return {
        "feature": feature,
        "target_label": target_label,
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "window_sensitivity": sensitivity,
        "window_false_positive_rate": false_positive_rate,
        "window_specificity": specificity,
        "window_precision": precision,
        "event_detection_rate": event_detection_rate,
        "target_event_count": target_event_count,
        "normal_recording_false_alarm_rate": normal_recording_false_alarm_rate,
        "normal_recording_count": normal_recording_count,
    }


def choose_recommendation(group):
    """
    Choose an exploratory single-feature threshold.

    Priority:
    1. Detect every recorded incident if possible.
    2. Minimise normal-recording false alarms.
    3. Minimise normal-window false positives.
    4. Maximise incident-window sensitivity.

    This is NOT the final deployed rule. Multiple features and
    leave-one-worker-out validation come next.
    """
    candidates = group.copy()

    perfect_event = candidates[
        np.isclose(candidates["event_detection_rate"], 1.0)
    ]

    if not perfect_event.empty:
        candidates = perfect_event
        selection_reason = "100% recorded-event detection, then minimise false alarms"
    else:
        best_event_rate = candidates["event_detection_rate"].max()
        candidates = candidates[
            np.isclose(
                candidates["event_detection_rate"],
                best_event_rate,
            )
        ]
        selection_reason = "best available recorded-event detection, then minimise false alarms"

    candidates = candidates.sort_values(
        [
            "normal_recording_false_alarm_rate",
            "window_false_positive_rate",
            "window_sensitivity",
            "threshold",
        ],
        ascending=[True, True, False, False],
    )

    best = candidates.iloc[0].copy()
    best["selection_reason"] = selection_reason

    return best


def make_plot(group, feature, target_label):
    group = group.sort_values("threshold")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        group["threshold"],
        group["window_sensitivity"],
        label="Incident window sensitivity",
    )
    ax.plot(
        group["threshold"],
        group["window_false_positive_rate"],
        label="Normal window false positive rate",
    )
    ax.plot(
        group["threshold"],
        group["event_detection_rate"],
        label="Recorded-event detection rate",
    )
    ax.set_title(
        f"Threshold Sweep: {feature} | {target_label} vs normal"
    )
    ax.set_xlabel(f"Threshold for {feature}")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    output = REPORT_FOLDER / f"{target_label}_{feature}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 76)
    print("GROUP 5 - EXPLORATORY THRESHOLD SWEEP")
    print("Simple rule thresholds | NO AI / NO ML")
    print("=" * 76)

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"ERROR: missing input file:\n{INPUT_FILE}\n"
            "Run Code 05 first."
        )

    data = pd.read_csv(INPUT_FILE)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise SystemExit(
            "ERROR: input file is missing columns:\n"
            + "\n".join(f" - {column}" for column in missing)
        )

    for feature in FEATURES:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

    data = data.dropna(subset=FEATURES).copy()

    sweep_rows = []

    for target_label in TARGETS:
        target_values = data.loc[
            data["primary_label"] == target_label,
            FEATURES,
        ]

        normal_values = data.loc[
            data["primary_label"] == "normal",
            FEATURES,
        ]

        if target_values.empty or normal_values.empty:
            raise SystemExit(
                f"ERROR: missing normal or {target_label} windows."
            )

        for feature in FEATURES:
            thresholds = build_threshold_grid(
                normal_values[feature],
                target_values[feature],
            )

            for threshold in thresholds:
                sweep_rows.append(
                    evaluate_threshold(
                        data=data,
                        feature=feature,
                        target_label=target_label,
                        threshold=threshold,
                    )
                )

    sweep = pd.DataFrame(sweep_rows)

    recommendations = (
        sweep
        .groupby(["target_label", "feature"], group_keys=False)
        .apply(choose_recommendation, include_groups=False)
        .reset_index()
    )

    numeric_sweep = sweep.select_dtypes(include=[np.number]).columns
    sweep[numeric_sweep] = sweep[numeric_sweep].round(5)

    numeric_rec = recommendations.select_dtypes(include=[np.number]).columns
    recommendations[numeric_rec] = recommendations[numeric_rec].round(5)

    sweep.to_csv(SWEEP_OUTPUT, index=False)
    recommendations.to_csv(RECOMMENDATION_OUTPUT, index=False)

    for (target_label, feature), group in sweep.groupby(
        ["target_label", "feature"]
    ):
        make_plot(group, feature, target_label)

    print()
    print("Exploratory single-feature threshold recommendations:")
    print(
        recommendations[
            [
                "target_label",
                "feature",
                "threshold",
                "event_detection_rate",
                "window_sensitivity",
                "window_false_positive_rate",
                "normal_recording_false_alarm_rate",
            ]
        ]
        .sort_values(["target_label", "feature"])
        .to_string(index=False)
    )

    print()
    print("Files created:")
    print(SWEEP_OUTPUT)
    print(RECOMMENDATION_OUTPUT)
    print(REPORT_FOLDER)

    print()
    print("IMPORTANT:")
    print("These are exploratory SINGLE-FEATURE thresholds, not final NESSO rules.")
    print("Next we combine features and validate by leaving one worker out at a time.")


if __name__ == "__main__":
    main()
