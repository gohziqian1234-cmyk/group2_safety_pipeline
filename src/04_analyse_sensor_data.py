from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
except ImportError:
    raise SystemExit(
        "\nERROR: scipy is not installed.\n"
        "Run this command in the VS Code terminal:\n"
        "py -m pip install scipy\n"
    )


# ============================================================
# GROUP 5 INCIDENT-CANDIDATE ANALYSIS
# Statistical signal processing only — NO AI / NO ML model
# ============================================================


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT
    / "data"
    / "processed"
    / "combined_cleaned_data.csv"
)

PROCESSED_FOLDER = PROJECT / "data" / "processed"
REPORT_FOLDER = PROJECT / "reports" / "event_plots"

CANDIDATE_OUTPUT = PROCESSED_FOLDER / "event_candidates.csv"
LABEL_OUTPUT = PROCESSED_FOLDER / "event_labels.csv"


# ------------------------------------------------------------
# ANALYSIS SETTINGS
# ------------------------------------------------------------

TARGET_RATE_HZ = 25
SAMPLE_INTERVAL_SECONDS = 1 / TARGET_RATE_HZ

NUMBER_OF_CANDIDATES = 3

# Candidates must be separated so one incident is not selected repeatedly.
MINIMUM_CANDIDATE_DISTANCE_SECONDS = 4.0
MERGE_CANDIDATE_DISTANCE_SECONDS = 0.50

# Candidate-peak prominence is not a final fall threshold.
# It only removes small local bumps during incident-time searching.
MINIMUM_PEAK_PROMINENCE = 0.75

# Light smoothing before derivative: 3 rows at 25 Hz = 0.12 seconds.
LIGHT_SIGNAL_SMOOTHING_ROWS = 3

# Smoothing is mainly applied to the final candidate scores.
SCORE_SMOOTHING_ROWS = 5

# Suggested windows for later human confirmation.
EVENT_BEFORE_SECONDS = 1.5
EVENT_AFTER_SECONDS = 1.5
POST_EVENT_SECONDS = 8.0


# ------------------------------------------------------------
# REQUIRED INPUT COLUMNS
# ------------------------------------------------------------

REQUIRED_COLUMNS = [
    "worker",
    "recording_type",
    "source_file",
    "elapsed_seconds",
    "acceleration_magnitude_g",
    "gyroscope_magnitude_dps",
]


# ------------------------------------------------------------
# GENERAL HELPERS
# ------------------------------------------------------------

def safe_filename(value):
    """Convert a name into a safe Windows file name."""
    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value),
    )
    return value.strip("._") or "recording"


def robust_z_score(series):
    """
    Calculate a robust z-score using median and MAD.

    This is statistical scaling only. It is not AI.
    """
    values = pd.Series(
        series,
        dtype="float64",
    )

    median = float(values.median())
    mad = float(
        (values - median)
        .abs()
        .median()
    )

    if mad > 0:
        return (
            values - median
        ) / (1.4826 * mad)

    standard_deviation = float(
        values.std(ddof=0)
    )

    if standard_deviation > 0:
        return (
            values - median
        ) / standard_deviation

    return pd.Series(
        np.zeros(len(values)),
        index=values.index,
        dtype="float64",
    )


# ------------------------------------------------------------
# RESAMPLING
# ------------------------------------------------------------

def resample_recording(recording):
    """
    Interpolate one recording onto a regular 25 Hz timeline.

    This compensates for uneven BLE packet-arrival timing.
    It does not overwrite raw or cleaned CSV files.
    """
    recording = (
        recording
        .sort_values("elapsed_seconds")
        .drop_duplicates(
            subset=["elapsed_seconds"],
            keep="first",
        )
        .copy()
    )

    start_time = float(
        recording["elapsed_seconds"].min()
    )
    end_time = float(
        recording["elapsed_seconds"].max()
    )

    if end_time <= start_time:
        return pd.DataFrame()

    regular_time = np.arange(
        start_time,
        end_time + SAMPLE_INTERVAL_SECONDS / 2,
        SAMPLE_INTERVAL_SECONDS,
    )

    output = pd.DataFrame({
        "elapsed_seconds": regular_time,
    })

    for column in [
        "acceleration_magnitude_g",
        "gyroscope_magnitude_dps",
    ]:
        output[column] = np.interp(
            regular_time,
            recording["elapsed_seconds"],
            recording[column],
        )

    return output


# ------------------------------------------------------------
# SIGNALS AND SCORES
# ------------------------------------------------------------

def calculate_candidate_signals(recording):
    """
    Calculate raw/lightly-smoothed jerk and two separate scores:

    1. Impact score:
       positive acceleration z-score + positive jerk z-score

    2. Balance-loss score:
       positive gyroscope z-score + positive jerk z-score

    Separate scores avoid assuming that falls and near misses
    have exactly the same sensor pattern.
    """
    data = recording.copy()

    # Light smoothing keeps sharp events while reducing tiny noise.
    data["accel_light_g"] = (
        data["acceleration_magnitude_g"]
        .rolling(
            window=LIGHT_SIGNAL_SMOOTHING_ROWS,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    data["gyro_light_dps"] = (
        data["gyroscope_magnitude_dps"]
        .rolling(
            window=LIGHT_SIGNAL_SMOOTHING_ROWS,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    # Jerk from resampled but otherwise raw acceleration magnitude.
    data["jerk_raw_g_per_second"] = (
        data["acceleration_magnitude_g"]
        .diff()
        .abs()
        / SAMPLE_INTERVAL_SECONDS
    ).fillna(0)

    # Jerk from only lightly-smoothed acceleration.
    data["jerk_light_g_per_second"] = (
        data["accel_light_g"]
        .diff()
        .abs()
        / SAMPLE_INTERVAL_SECONDS
    ).fillna(0)

    # Store the component robust z-scores for auditability.
    data["acceleration_z"] = robust_z_score(
        data["accel_light_g"]
    )

    data["gyroscope_z"] = robust_z_score(
        data["gyro_light_dps"]
    )

    data["jerk_z"] = robust_z_score(
        data["jerk_light_g_per_second"]
    )

    acceleration_positive = (
        data["acceleration_z"]
        .clip(lower=0)
    )

    gyroscope_positive = (
        data["gyroscope_z"]
        .clip(lower=0)
    )

    jerk_positive = (
        data["jerk_z"]
        .clip(lower=0)
    )

    # Equal weighting is used only for candidate-time searching.
    # Evidence-based final thresholds/weights come after manual labels.
    data["impact_score_raw"] = (
        acceleration_positive
        + jerk_positive
    )

    data["balance_loss_score_raw"] = (
        gyroscope_positive
        + jerk_positive
    )

    data["impact_score"] = (
        data["impact_score_raw"]
        .rolling(
            window=SCORE_SMOOTHING_ROWS,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    data["balance_loss_score"] = (
        data["balance_loss_score_raw"]
        .rolling(
            window=SCORE_SMOOTHING_ROWS,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    return data


# ------------------------------------------------------------
# PEAK DETECTION
# ------------------------------------------------------------

def get_peak_rows(data, score_column, source_name):
    """
    Use scipy.signal.find_peaks with distance and prominence.

    This is standard signal processing, not machine learning.
    """
    distance_rows = max(
        1,
        int(
            MINIMUM_CANDIDATE_DISTANCE_SECONDS
            * TARGET_RATE_HZ
        ),
    )

    score_values = (
        data[score_column]
        .fillna(0)
        .to_numpy()
    )

    peak_indexes, properties = find_peaks(
        score_values,
        distance=distance_rows,
        prominence=MINIMUM_PEAK_PROMINENCE,
    )

    peak_rows = []

    for position, index in enumerate(peak_indexes):
        peak_rows.append({
            "index": int(index),
            "candidate_source": source_name,
            "peak_prominence": float(
                properties["prominences"][position]
            ),
        })

    return peak_rows


def add_fallback_peak(data, score_column, source_name):
    """
    Return the global maximum when no prominent peak was found.

    It is clearly marked as a fallback, and the user may later
    select 'none_found' during manual confirmation.
    """
    index = int(
        data[score_column]
        .fillna(0)
        .idxmax()
    )

    return {
        "index": index,
        "candidate_source": source_name,
        "peak_prominence": np.nan,
        "fallback_used": True,
    }


def merge_nearby_peak_rows(data, peak_rows):
    """
    Merge impact and balance peaks that occur almost together.
    """
    if not peak_rows:
        return []

    ordered = sorted(
        peak_rows,
        key=lambda item: float(
            data.loc[
                item["index"],
                "elapsed_seconds",
            ]
        ),
    )

    merged = []

    for item in ordered:
        item_time = float(
            data.loc[
                item["index"],
                "elapsed_seconds",
            ]
        )

        nearby = None

        for existing in merged:
            if (
                abs(
                    item_time
                    - existing["candidate_time_seconds"]
                )
                <= MERGE_CANDIDATE_DISTANCE_SECONDS
            ):
                nearby = existing
                break

        if nearby is None:
            merged.append({
                "indexes": [item["index"]],
                "candidate_sources": {
                    item["candidate_source"]
                },
                "prominences": [
                    item["peak_prominence"]
                ],
                "candidate_time_seconds": item_time,
                "fallback_used": bool(
                    item.get(
                        "fallback_used",
                        False,
                    )
                ),
            })
        else:
            nearby["indexes"].append(
                item["index"]
            )
            nearby["candidate_sources"].add(
                item["candidate_source"]
            )
            nearby["prominences"].append(
                item["peak_prominence"]
            )
            nearby["fallback_used"] = (
                nearby["fallback_used"]
                or bool(
                    item.get(
                        "fallback_used",
                        False,
                    )
                )
            )

            # Keep the strongest point from the nearby peaks.
            strongest_index = max(
                nearby["indexes"],
                key=lambda index: max(
                    float(
                        data.loc[
                            index,
                            "impact_score",
                        ]
                    ),
                    float(
                        data.loc[
                            index,
                            "balance_loss_score",
                        ]
                    ),
                ),
            )

            nearby["candidate_time_seconds"] = float(
                data.loc[
                    strongest_index,
                    "elapsed_seconds",
                ]
            )

    return merged


def build_final_candidates(data):
    """
    Find candidates from both impact and balance-loss scores.
    Rank using the larger of the two scores.
    """
    impact_peaks = get_peak_rows(
        data,
        "impact_score",
        "impact",
    )

    balance_peaks = get_peak_rows(
        data,
        "balance_loss_score",
        "balance_loss",
    )

    peak_rows = impact_peaks + balance_peaks

    if not impact_peaks:
        peak_rows.append(
            add_fallback_peak(
                data,
                "impact_score",
                "impact",
            )
        )

    if not balance_peaks:
        peak_rows.append(
            add_fallback_peak(
                data,
                "balance_loss_score",
                "balance_loss",
            )
        )

    merged = merge_nearby_peak_rows(
        data,
        peak_rows,
    )

    detailed = []

    for item in merged:
        strongest_index = max(
            item["indexes"],
            key=lambda index: max(
                float(
                    data.loc[
                        index,
                        "impact_score",
                    ]
                ),
                float(
                    data.loc[
                        index,
                        "balance_loss_score",
                    ]
                ),
            ),
        )

        row = data.loc[strongest_index]

        valid_prominences = [
            value
            for value in item["prominences"]
            if not pd.isna(value)
        ]

        detailed.append({
            "candidate_time_seconds": round(
                float(row["elapsed_seconds"]),
                2,
            ),
            "candidate_source": "+".join(
                sorted(item["candidate_sources"])
            ),
            "peak_prominence": (
                round(
                    max(valid_prominences),
                    4,
                )
                if valid_prominences
                else np.nan
            ),
            "fallback_used": item["fallback_used"],
            "acceleration_g": round(
                float(row["accel_light_g"]),
                4,
            ),
            "gyroscope_dps": round(
                float(row["gyro_light_dps"]),
                4,
            ),
            "jerk_raw_g_per_second": round(
                float(
                    row[
                        "jerk_raw_g_per_second"
                    ]
                ),
                4,
            ),
            "jerk_light_g_per_second": round(
                float(
                    row[
                        "jerk_light_g_per_second"
                    ]
                ),
                4,
            ),
            "acceleration_z": round(
                float(row["acceleration_z"]),
                4,
            ),
            "gyroscope_z": round(
                float(row["gyroscope_z"]),
                4,
            ),
            "jerk_z": round(
                float(row["jerk_z"]),
                4,
            ),
            "impact_score": round(
                float(row["impact_score"]),
                4,
            ),
            "balance_loss_score": round(
                float(
                    row["balance_loss_score"]
                ),
                4,
            ),
            "ranking_score": round(
                max(
                    float(row["impact_score"]),
                    float(
                        row["balance_loss_score"]
                    ),
                ),
                4,
            ),
        })

    detailed = sorted(
        detailed,
        key=lambda item: item["ranking_score"],
        reverse=True,
    )

    selected = []

    for candidate in detailed:
        far_enough = all(
            abs(
                candidate[
                    "candidate_time_seconds"
                ]
                - selected_candidate[
                    "candidate_time_seconds"
                ]
            )
            >= MINIMUM_CANDIDATE_DISTANCE_SECONDS
            for selected_candidate in selected
        )

        if far_enough:
            selected.append(candidate)

        if len(selected) >= NUMBER_OF_CANDIDATES:
            break

    return selected


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

def add_candidate_lines(candidate_times):
    for rank, candidate_time in enumerate(
        candidate_times,
        start=1,
    ):
        plt.axvline(
            candidate_time,
            linestyle="--",
        )

        top_value = plt.ylim()[1]

        plt.text(
            candidate_time,
            top_value,
            f" C{rank}",
            verticalalignment="top",
        )


def save_single_plot(
    data,
    x_column,
    y_columns,
    title,
    y_label,
    output_file,
    candidate_times,
):
    plt.figure(figsize=(12, 5))

    for column, label in y_columns:
        plt.plot(
            data[x_column],
            data[column],
            label=label,
        )

    add_candidate_lines(candidate_times)

    plt.title(title)
    plt.xlabel("Elapsed time (seconds)")
    plt.ylabel(y_label)
    plt.grid(True)

    if len(y_columns) > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=160,
    )
    plt.close()


def save_recording_plots(
    data,
    worker,
    recording_type,
    source_file,
    candidates,
):
    safe_source = safe_filename(
        Path(source_file).stem
    )

    base_name = (
        f"{safe_filename(worker)}__"
        f"{safe_filename(recording_type)}__"
        f"{safe_source}"
    )

    candidate_times = [
        item["candidate_time_seconds"]
        for item in candidates
    ]

    common_title = (
        f"{worker} | {recording_type}\n"
        f"{source_file}"
    )

    save_single_plot(
        data=data,
        x_column="elapsed_seconds",
        y_columns=[
            (
                "acceleration_magnitude_g",
                "Resampled acceleration",
            ),
            (
                "accel_light_g",
                "Lightly smoothed acceleration",
            ),
        ],
        title=(
            common_title
            + " | Acceleration comparison"
        ),
        y_label="Acceleration magnitude (g)",
        output_file=(
            REPORT_FOLDER
            / f"{base_name}__acceleration.png"
        ),
        candidate_times=candidate_times,
    )

    save_single_plot(
        data=data,
        x_column="elapsed_seconds",
        y_columns=[
            (
                "gyroscope_magnitude_dps",
                "Resampled gyroscope",
            ),
            (
                "gyro_light_dps",
                "Lightly smoothed gyroscope",
            ),
        ],
        title=(
            common_title
            + " | Gyroscope comparison"
        ),
        y_label="Gyroscope magnitude (degrees/second)",
        output_file=(
            REPORT_FOLDER
            / f"{base_name}__gyroscope.png"
        ),
        candidate_times=candidate_times,
    )

    save_single_plot(
        data=data,
        x_column="elapsed_seconds",
        y_columns=[
            (
                "jerk_raw_g_per_second",
                "Jerk from resampled acceleration",
            ),
            (
                "jerk_light_g_per_second",
                "Jerk after light smoothing",
            ),
        ],
        title=(
            common_title
            + " | Jerk before/after light smoothing"
        ),
        y_label="Absolute jerk (g/second)",
        output_file=(
            REPORT_FOLDER
            / f"{base_name}__jerk_comparison.png"
        ),
        candidate_times=candidate_times,
    )

    save_single_plot(
        data=data,
        x_column="elapsed_seconds",
        y_columns=[
            (
                "impact_score",
                "Impact candidate score",
            ),
        ],
        title=(
            common_title
            + " | Impact candidate score"
        ),
        y_label="Statistical candidate score",
        output_file=(
            REPORT_FOLDER
            / f"{base_name}__impact_score.png"
        ),
        candidate_times=candidate_times,
    )

    save_single_plot(
        data=data,
        x_column="elapsed_seconds",
        y_columns=[
            (
                "balance_loss_score",
                "Balance-loss candidate score",
            ),
        ],
        title=(
            common_title
            + " | Balance-loss candidate score"
        ),
        y_label="Statistical candidate score",
        output_file=(
            REPORT_FOLDER
            / f"{base_name}__balance_score.png"
        ),
        candidate_times=candidate_times,
    )


# ------------------------------------------------------------
# LABEL-FILE MANAGEMENT
# ------------------------------------------------------------

def save_labels_without_overwriting_manual_work(
    new_labels,
):
    """
    Preserve existing manual confirmation values.

    Only new source files are appended.
    """
    if not LABEL_OUTPUT.exists():
        new_labels.to_csv(
            LABEL_OUTPUT,
            index=False,
        )
        return len(new_labels)

    existing = pd.read_csv(
        LABEL_OUTPUT,
        dtype=str,
    )

    if "source_file" not in existing.columns:
        backup_file = (
            PROCESSED_FOLDER
            / "event_labels_old_backup.csv"
        )

        existing.to_csv(
            backup_file,
            index=False,
        )

        new_labels.to_csv(
            LABEL_OUTPUT,
            index=False,
        )

        return len(new_labels)

    existing_sources = set(
        existing["source_file"]
        .dropna()
        .astype(str)
    )

    missing = new_labels[
        ~new_labels["source_file"]
        .astype(str)
        .isin(existing_sources)
    ]

    combined = pd.concat(
        [
            existing,
            missing,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        LABEL_OUTPUT,
        index=False,
    )

    return len(missing)


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def main():
    print("=" * 72)
    print("GROUP 5 INCIDENT-CANDIDATE ANALYSIS")
    print("STATISTICAL SIGNAL PROCESSING ONLY — NO AI / NO ML")
    print("=" * 72)

    if not INPUT_FILE.exists():
        print(
            "ERROR: combined_cleaned_data.csv was not found."
        )
        print(
            "Run 02_clean_all_data.py first."
        )
        return

    data = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        print(
            "ERROR: Missing required columns:"
        )
        print(missing_columns)
        return

    for column in [
        "elapsed_seconds",
        "acceleration_magnitude_g",
        "gyroscope_magnitude_dps",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    )

    PROCESSED_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_rows = []
    label_rows = []

    grouped = data.groupby(
        [
            "worker",
            "recording_type",
            "source_file",
        ],
        sort=True,
    )

    processed_count = 0
    skipped_normal_count = 0

    for group_name, recording in grouped:
        worker, recording_type, source_file = group_name

        recording_type_text = str(
            recording_type
        ).lower()

        if "normal" in recording_type_text:
            skipped_normal_count += 1
            print()
            print(
                "Skipped normal recording:",
                source_file,
            )
            continue

        regular_data = resample_recording(
            recording
        )

        if regular_data.empty:
            print()
            print(
                "WARNING: Could not resample:",
                source_file,
            )
            continue

        scored_data = calculate_candidate_signals(
            regular_data
        )

        candidates = build_final_candidates(
            scored_data
        )

        if not candidates:
            print()
            print(
                "WARNING: No candidate created:",
                source_file,
            )
            continue

        processed_count += 1

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            candidate_rows.append({
                "worker": worker,
                "recording_type": recording_type,
                "source_file": source_file,
                "candidate_rank": rank,
                **candidate,
            })

        best_time = candidates[0][
            "candidate_time_seconds"
        ]

        suggested_event_type = (
            "near_miss"
            if "near" in recording_type_text
            else "fall_event"
        )

        minimum_time = float(
            scored_data[
                "elapsed_seconds"
            ].min()
        )

        maximum_time = float(
            scored_data[
                "elapsed_seconds"
            ].max()
        )

        label_rows.append({
            "worker": worker,
            "source_file": source_file,
            "recording_type_from_file": recording_type,
            "candidate_1_seconds": (
                candidates[0][
                    "candidate_time_seconds"
                ]
                if len(candidates) >= 1
                else ""
            ),
            "candidate_2_seconds": (
                candidates[1][
                    "candidate_time_seconds"
                ]
                if len(candidates) >= 2
                else ""
            ),
            "candidate_3_seconds": (
                candidates[2][
                    "candidate_time_seconds"
                ]
                if len(candidates) >= 3
                else ""
            ),
            "suggested_event_type": suggested_event_type,
            "suggested_event_time_seconds": best_time,
            "event_start_seconds": round(
                max(
                    minimum_time,
                    best_time
                    - EVENT_BEFORE_SECONDS,
                ),
                2,
            ),
            "event_end_seconds": round(
                min(
                    maximum_time,
                    best_time
                    + EVENT_AFTER_SECONDS,
                ),
                2,
            ),
            "post_event_end_seconds": round(
                min(
                    maximum_time,
                    best_time
                    + POST_EVENT_SECONDS,
                ),
                2,
            ),
            "selected_candidate_rank": "",
            "final_event_type": "",
            "confirmed": "NO",
            "notes": (
                "Allowed final_event_type: "
                "near_miss, fall_recovery, "
                "fall_inactive, none_found"
            ),
        })

        save_recording_plots(
            scored_data,
            worker,
            recording_type,
            source_file,
            candidates,
        )

        print()
        print("File:", source_file)
        print("Worker:", worker)
        print(
            "Recording type:",
            recording_type,
        )
        print(
            "Candidate times:",
            [
                item[
                    "candidate_time_seconds"
                ]
                for item in candidates
            ],
        )
        print(
            "Candidate sources:",
            [
                item["candidate_source"]
                for item in candidates
            ],
        )

    if not candidate_rows:
        print()
        print(
            "ERROR: No incident candidates were created."
        )
        return

    candidate_report = pd.DataFrame(
        candidate_rows
    )

    candidate_report.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    new_labels = pd.DataFrame(
        label_rows
    )

    labels_added = (
        save_labels_without_overwriting_manual_work(
            new_labels
        )
    )

    print()
    print("=" * 72)
    print("CANDIDATE ANALYSIS FINISHED")
    print("Incident recordings processed:",processed_count,)
    print("Normal recordings skipped:",skipped_normal_count,)
    print( "Candidate rows created:", len(candidate_report),)
    print( "New label rows added:",labels_added, )
    print()
    print("Files created or updated:")
    print(CANDIDATE_OUTPUT)
    print(LABEL_OUTPUT)
    print(REPORT_FOLDER)
    print()
    print("NEXT STEP:")
    print(
        "Open the event plots and confirm "
        "the real incident time."
    )
    print(
        "Then edit event_labels.csv."
    )
    print(
        "Allowed final_event_type values:"
    )
    print(
        "near_miss, fall_recovery, "
        "fall_inactive, none_found"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()