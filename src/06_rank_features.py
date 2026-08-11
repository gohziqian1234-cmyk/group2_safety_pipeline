from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 FEATURE RANKING
# Statistical comparison only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
REPORT_FOLDER = PROJECT / "reports" / "feature_plots"

INPUT_FILE = PROCESSED / "window_features_analysis.csv"
RANKING_OUTPUT = PROCESSED / "feature_ranking.csv"
STATS_OUTPUT = PROCESSED / "feature_statistics_by_label.csv"
WORKER_OUTPUT = PROCESSED / "feature_worker_medians.csv"

FEATURE_COLUMNS = [
    "accel_mean_g",
    "accel_std_g",
    "accel_min_g",
    "accel_max_g",
    "accel_range_g",
    "gyro_mean_dps",
    "gyro_std_dps",
    "gyro_max_dps",
    "jerk_mean_g_per_second",
    "jerk_std_g_per_second",
    "jerk_max_g_per_second",
    "sma_g",
    "gyro_sma_dps",
    "orientation_change_deg",
    "axis_accel_std_total_g",
]

REQUIRED_COLUMNS = [
    "worker",
    "source_file",
    "primary_label",
    *FEATURE_COLUMNS,
]

EXPECTED_LABELS = ["normal", "near_miss", "fall"]


def cohens_d(group_a, group_b):
    """
    Cohen's d = difference in means divided by pooled standard deviation.

    Positive d means group_a is higher than group_b.
    Negative d means group_a is lower than group_b.
    """
    a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    variance_a = np.var(a, ddof=1)
    variance_b = np.var(b, ddof=1)

    pooled_denominator = len(a) + len(b) - 2

    if pooled_denominator <= 0:
        return np.nan

    pooled_variance = (
        ((len(a) - 1) * variance_a)
        + ((len(b) - 1) * variance_b)
    ) / pooled_denominator

    if pooled_variance <= 1e-15:
        return 0.0

    return float(
        (np.mean(a) - np.mean(b))
        / np.sqrt(pooled_variance)
    )


def common_language_effect(group_a, group_b):
    """
    Probability that a randomly selected value from A exceeds B.

    0.50 = no directional separation.
    Above 0.50 = A tends to be higher.
    Below 0.50 = A tends to be lower.

    This is a descriptive effect-size statistic, not AI.
    """
    a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)

    if len(a) == 0 or len(b) == 0:
        return np.nan

    # Memory-safe exact calculation for this project size.
    greater = 0.0
    comparisons = 0

    for value in a:
        greater += np.sum(value > b)
        greater += 0.5 * np.sum(value == b)
        comparisons += len(b)

    if comparisons == 0:
        return np.nan

    return float(greater / comparisons)


def effect_strength(abs_d):
    if pd.isna(abs_d):
        return "unknown"
    if abs_d >= 1.20:
        return "very_large"
    if abs_d >= 0.80:
        return "large"
    if abs_d >= 0.50:
        return "medium"
    if abs_d >= 0.20:
        return "small"
    return "very_small"


def direction_from_medians(target, normal):
    target_median = float(pd.to_numeric(target, errors="coerce").median())
    normal_median = float(pd.to_numeric(normal, errors="coerce").median())

    if np.isclose(target_median, normal_median, atol=1e-12):
        return "similar"

    return "higher" if target_median > normal_median else "lower"


def make_boxplot(data, feature):
    values = []
    labels = []

    for label in EXPECTED_LABELS:
        series = pd.to_numeric(
            data.loc[data["primary_label"] == label, feature],
            errors="coerce",
        ).dropna()

        values.append(series.to_numpy(dtype=float))
        labels.append(label.replace("_", " ").title())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_title(f"{feature} | Normal vs Near Miss vs Fall")
    ax.set_ylabel(feature)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    output = REPORT_FOLDER / f"{feature}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 74)
    print("GROUP 5 - FEATURE RANKING")
    print("Cohen's d + descriptive statistics | NO AI / NO ML")
    print("=" * 74)

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"ERROR: missing feature-window file:\n{INPUT_FILE}\n"
            "Run src\\05_build_feature_windows.py first."
        )

    data = pd.read_csv(INPUT_FILE)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise SystemExit(
            "ERROR: window feature file is missing columns:\n"
            + "\n".join(f" - {column}" for column in missing)
        )

    present_labels = set(
        data["primary_label"].astype(str).unique()
    )

    missing_labels = [
        label for label in EXPECTED_LABELS
        if label not in present_labels
    ]

    if missing_labels:
        raise SystemExit(
            "ERROR: expected analysis labels are missing:\n"
            + "\n".join(f" - {label}" for label in missing_labels)
        )

    print()
    print("Analysis window counts:")
    print(data["primary_label"].value_counts().to_string())

    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------
    stats_rows = []

    for feature in FEATURE_COLUMNS:
        for label in EXPECTED_LABELS:
            series = pd.to_numeric(
                data.loc[
                    data["primary_label"] == label,
                    feature,
                ],
                errors="coerce",
            ).dropna()

            stats_rows.append({
                "feature": feature,
                "primary_label": label,
                "count": int(len(series)),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=1)),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "min": float(series.min()),
                "max": float(series.max()),
            })

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(STATS_OUTPUT, index=False)

    # --------------------------------------------------------
    # WORKER-BALANCED MEDIANS
    # --------------------------------------------------------
    worker_medians = (
        data
        .groupby(["worker", "primary_label"], dropna=False)[FEATURE_COLUMNS]
        .median(numeric_only=True)
        .reset_index()
    )

    worker_medians.to_csv(WORKER_OUTPUT, index=False)

    # --------------------------------------------------------
    # EFFECT-SIZE RANKING
    # --------------------------------------------------------
    normal = data[data["primary_label"] == "normal"]
    near_miss = data[data["primary_label"] == "near_miss"]
    fall = data[data["primary_label"] == "fall"]

    ranking_rows = []

    for feature in FEATURE_COLUMNS:
        normal_values = normal[feature]
        near_values = near_miss[feature]
        fall_values = fall[feature]

        d_near = cohens_d(near_values, normal_values)
        d_fall = cohens_d(fall_values, normal_values)

        cl_near = common_language_effect(near_values, normal_values)
        cl_fall = common_language_effect(fall_values, normal_values)

        abs_d_near = abs(d_near) if not pd.isna(d_near) else np.nan
        abs_d_fall = abs(d_fall) if not pd.isna(d_fall) else np.nan

        # Conservative combined score: a feature ranks highly only when
        # it separates BOTH incident types from normal reasonably well.
        combined_score = (
            min(abs_d_near, abs_d_fall)
            if not pd.isna(abs_d_near) and not pd.isna(abs_d_fall)
            else np.nan
        )

        ranking_rows.append({
            "feature": feature,
            "near_miss_cohens_d": d_near,
            "near_miss_abs_d": abs_d_near,
            "near_miss_effect_strength": effect_strength(abs_d_near),
            "near_miss_direction_vs_normal": direction_from_medians(
                near_values,
                normal_values,
            ),
            "near_miss_common_language_effect": cl_near,
            "fall_cohens_d": d_fall,
            "fall_abs_d": abs_d_fall,
            "fall_effect_strength": effect_strength(abs_d_fall),
            "fall_direction_vs_normal": direction_from_medians(
                fall_values,
                normal_values,
            ),
            "fall_common_language_effect": cl_fall,
            "combined_conservative_score": combined_score,
        })

    ranking = pd.DataFrame(ranking_rows)

    ranking["near_miss_rank"] = (
        ranking["near_miss_abs_d"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    ranking["fall_rank"] = (
        ranking["fall_abs_d"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    ranking["combined_rank"] = (
        ranking["combined_conservative_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    ranking = ranking.sort_values(
        ["combined_rank", "fall_rank", "near_miss_rank"],
        ascending=True,
    ).reset_index(drop=True)

    numeric_ranking_columns = ranking.select_dtypes(
        include=[np.number]
    ).columns

    ranking[numeric_ranking_columns] = (
        ranking[numeric_ranking_columns]
        .round(4)
    )

    ranking.to_csv(RANKING_OUTPUT, index=False)

    # --------------------------------------------------------
    # BOXPLOTS
    # --------------------------------------------------------
    for feature in FEATURE_COLUMNS:
        make_boxplot(data, feature)

    print()
    print("Top features by conservative combined separation:")
    print(
        ranking[
            [
                "combined_rank",
                "feature",
                "near_miss_abs_d",
                "fall_abs_d",
                "near_miss_direction_vs_normal",
                "fall_direction_vs_normal",
                "combined_conservative_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Files created:")
    print(RANKING_OUTPUT)
    print(STATS_OUTPUT)
    print(WORKER_OUTPUT)
    print(REPORT_FOLDER)

    print()
    print("IMPORTANT:")
    print(
        "The 1-second windows overlap, so neighbouring windows are correlated."
    )
    print(
        "Use these effect sizes for feature selection, not as independent "
        "hypothesis-test samples."
    )
    print(
        "Final rule validation will use leave-one-worker-out testing to avoid "
        "window leakage between workers."
    )

    print()
    print("NEXT STEP:")
    print(
        "Inspect the strongest features, then perform threshold sweeps "
        "for fall and near-miss rules."
    )


if __name__ == "__main__":
    main()
