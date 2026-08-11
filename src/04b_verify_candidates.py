from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 CANDIDATE VERIFICATION
# Rule-based statistical verification only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "combined_cleaned_data.csv"
LABEL_FILE = PROCESSED / "event_labels.csv"

DETAIL_OUTPUT = PROCESSED / "candidate_verification.csv"
SUMMARY_OUTPUT = PROCESSED / "candidate_verification_summary.csv"

TARGET_RATE_HZ = 25
DT = 1.0 / TARGET_RATE_HZ

# Windows relative to candidate time.
PRE_START = -3.0
PRE_END = -1.0
EVENT_START = -0.8
EVENT_END = 1.2
POST_START = 1.5
POST_END = 4.5
LATE_START = 4.5
LATE_END = 8.0

REQUIRED_COLUMNS = [
    "worker",
    "recording_type",
    "source_file",
    "elapsed_seconds",
    "Xg", "Yg", "Zg",
    "Xdeg", "Ydeg", "Zdeg",
    "acceleration_magnitude_g",
    "gyroscope_magnitude_dps",
]


def vector_angle_degrees(vector_a, vector_b):
    """Angle between two 3D vectors in degrees."""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return np.nan

    cosine = np.dot(a, b) / (norm_a * norm_b)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def safe_median_vector(frame):
    if frame.empty:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return frame[["Xg", "Yg", "Zg"]].median().to_numpy(dtype=float)


def select_window(frame, candidate_time, start_offset, end_offset):
    start = candidate_time + start_offset
    end = candidate_time + end_offset
    return frame[
        (frame["elapsed_seconds"] >= start)
        & (frame["elapsed_seconds"] <= end)
    ].copy()


def normalise_within_group(series):
    """0–1 scaling within the three candidates of one recording."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)

    if math.isclose(maximum, minimum, rel_tol=0.0, abs_tol=1e-12):
        return pd.Series(np.full(len(values), 0.5), index=values.index, dtype=float)

    return (values - minimum) / (maximum - minimum)


def resample_recording(recording):
    """
    Interpolate the six sensor axes and magnitudes to a regular 25 Hz grid.

    This reduces BLE timing irregularity before calculating jerk and
    integrated rotation. It does not overwrite the cleaned data.
    """
    recording = (
        recording
        .sort_values("elapsed_seconds")
        .drop_duplicates(subset=["elapsed_seconds"], keep="first")
        .copy()
    )

    if len(recording) < 2:
        return pd.DataFrame()

    start = float(recording["elapsed_seconds"].min())
    end = float(recording["elapsed_seconds"].max())

    if end <= start:
        return pd.DataFrame()

    regular_time = np.arange(start, end + DT / 2, DT)
    output = pd.DataFrame({"elapsed_seconds": regular_time})

    numeric_columns = [
        "Xg", "Yg", "Zg",
        "Xdeg", "Ydeg", "Zdeg",
        "acceleration_magnitude_g",
        "gyroscope_magnitude_dps",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(recording[column], errors="coerce")
        valid = recording["elapsed_seconds"].notna() & values.notna()

        if valid.sum() < 2:
            output[column] = np.nan
            continue

        output[column] = np.interp(
            regular_time,
            recording.loc[valid, "elapsed_seconds"].to_numpy(dtype=float),
            values.loc[valid].to_numpy(dtype=float),
        )

    output["jerk_g_per_second"] = (
        output["acceleration_magnitude_g"].diff().abs() / DT
    ).fillna(0.0)

    return output


def candidate_metrics(recording, candidate_time):
    pre = select_window(recording, candidate_time, PRE_START, PRE_END)
    event = select_window(recording, candidate_time, EVENT_START, EVENT_END)
    post = select_window(recording, candidate_time, POST_START, POST_END)
    late = select_window(recording, candidate_time, LATE_START, LATE_END)

    pre_vector = safe_median_vector(pre)
    post_vector = safe_median_vector(post)
    late_vector = safe_median_vector(late)

    posture_change_post_deg = vector_angle_degrees(pre_vector, post_vector)
    posture_change_late_deg = vector_angle_degrees(pre_vector, late_vector)

    event_peak_accel_g = (
        float(event["acceleration_magnitude_g"].max())
        if not event.empty else np.nan
    )
    event_peak_gyro_dps = (
        float(event["gyroscope_magnitude_dps"].max())
        if not event.empty else np.nan
    )
    event_peak_jerk = (
        float(event["jerk_g_per_second"].max())
        if not event.empty else np.nan
    )

    event_rotation_integral_deg = (
        float(np.trapezoid(
            event["gyroscope_magnitude_dps"].fillna(0).to_numpy(dtype=float),
            event["elapsed_seconds"].to_numpy(dtype=float),
        ))
        if len(event) >= 2 else np.nan
    )

    post_accel_std_g = (
        float(post["acceleration_magnitude_g"].std(ddof=0))
        if not post.empty else np.nan
    )
    post_gyro_mean_dps = (
        float(post["gyroscope_magnitude_dps"].mean())
        if not post.empty else np.nan
    )
    late_accel_std_g = (
        float(late["acceleration_magnitude_g"].std(ddof=0))
        if not late.empty else np.nan
    )
    late_gyro_mean_dps = (
        float(late["gyroscope_magnitude_dps"].mean())
        if not late.empty else np.nan
    )

    return {
        "candidate_time_seconds": round(float(candidate_time), 2),
        "event_peak_acceleration_g": event_peak_accel_g,
        "event_peak_gyroscope_dps": event_peak_gyro_dps,
        "event_peak_jerk_g_per_second": event_peak_jerk,
        "event_rotation_integral_deg": event_rotation_integral_deg,
        "posture_change_post_deg": posture_change_post_deg,
        "posture_change_late_deg": posture_change_late_deg,
        "post_accel_std_g": post_accel_std_g,
        "post_gyro_mean_dps": post_gyro_mean_dps,
        "late_accel_std_g": late_accel_std_g,
        "late_gyro_mean_dps": late_gyro_mean_dps,
    }


def score_candidates(group):
    """
    Rank the three candidates using rule-based, within-recording statistics.

    Fall-like:
      strong impact + rotation + jerk + sustained posture change

    Near-miss:
      strong rotation/jerk + continued post-event movement +
      return toward original posture

    These scores are ONLY for candidate verification.
    They are NOT the final live fall-detection thresholds.
    """
    scored = group.copy()

    metrics_to_scale = [
        "event_peak_acceleration_g",
        "event_peak_gyroscope_dps",
        "event_peak_jerk_g_per_second",
        "event_rotation_integral_deg",
        "posture_change_late_deg",
        "post_accel_std_g",
        "post_gyro_mean_dps",
    ]

    for metric in metrics_to_scale:
        scored[f"{metric}_n"] = normalise_within_group(scored[metric])

    recording_type = str(scored["recording_type"].iloc[0]).lower()

    if "near" in recording_type:
        posture_return_n = 1.0 - scored["posture_change_late_deg_n"]

        scored["verification_score"] = (
            0.20 * scored["event_peak_gyroscope_dps_n"]
            + 0.20 * scored["event_rotation_integral_deg_n"]
            + 0.20 * scored["event_peak_jerk_g_per_second_n"]
            + 0.15 * scored["event_peak_acceleration_g_n"]
            + 0.15 * scored["post_gyro_mean_dps_n"]
            + 0.05 * scored["post_accel_std_g_n"]
            + 0.05 * posture_return_n
        )
        scored["verification_logic"] = (
            "near_miss: disturbance + rotation + recovery movement + posture return"
        )

    else:
        scored["verification_score"] = (
            0.25 * scored["event_peak_acceleration_g_n"]
            + 0.20 * scored["event_peak_gyroscope_dps_n"]
            + 0.20 * scored["event_rotation_integral_deg_n"]
            + 0.20 * scored["event_peak_jerk_g_per_second_n"]
            + 0.15 * scored["posture_change_late_deg_n"]
        )
        scored["verification_logic"] = (
            "fall_like: impact + rotation + jerk + sustained posture change"
        )

    scored = scored.sort_values(
        ["verification_score", "candidate_rank"],
        ascending=[False, True],
    ).copy()
    scored["verification_rank"] = np.arange(1, len(scored) + 1)

    return scored


def main():
    print("=" * 68)
    print("GROUP 5 - CANDIDATE VERIFICATION")
    print("Rule-based statistics only - NO AI / NO ML")
    print("=" * 68)

    if not INPUT_FILE.exists():
        raise SystemExit(f"ERROR: missing input file:\n{INPUT_FILE}")

    if not LABEL_FILE.exists():
        raise SystemExit(f"ERROR: missing label file:\n{LABEL_FILE}")

    data = pd.read_csv(INPUT_FILE)
    labels = pd.read_csv(LABEL_FILE)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise SystemExit(
            "ERROR: combined cleaned data is missing columns:\n"
            + "\n".join(f" - {column}" for column in missing)
        )

    detail_rows = []

    for _, label_row in labels.iterrows():
        worker = str(label_row["worker"])
        source_file = str(label_row["source_file"])
        recording_type = str(label_row["recording_type_from_file"])

        recording = data[
            (data["worker"].astype(str) == worker)
            & (data["source_file"].astype(str) == source_file)
        ].copy()

        if recording.empty:
            print(f"WARNING: recording not found: {worker} | {source_file}")
            continue

        recording = resample_recording(recording)

        if recording.empty:
            print(f"WARNING: could not resample: {worker} | {source_file}")
            continue

        candidate_times = [
            label_row.get("candidate_1_seconds"),
            label_row.get("candidate_2_seconds"),
            label_row.get("candidate_3_seconds"),
        ]

        for rank, candidate_time in enumerate(candidate_times, start=1):
            if pd.isna(candidate_time):
                continue

            metrics = candidate_metrics(recording, float(candidate_time))

            detail_rows.append({
                "worker": worker,
                "source_file": source_file,
                "recording_type": recording_type,
                "candidate_rank": rank,
                **metrics,
            })

    if not detail_rows:
        raise SystemExit("ERROR: no candidate metrics were created.")

    detail = pd.DataFrame(detail_rows)

    scored_groups = []
    for _, group in detail.groupby(
        ["worker", "source_file"],
        sort=False,
        dropna=False,
    ):
        scored_groups.append(score_candidates(group))

    detail = pd.concat(scored_groups, ignore_index=True)

    numeric_columns = detail.select_dtypes(include=[np.number]).columns
    detail[numeric_columns] = detail[numeric_columns].round(4)

    detail = detail.sort_values(
        ["worker", "source_file", "verification_rank"],
        ascending=True,
    ).reset_index(drop=True)

    DETAIL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_OUTPUT, index=False)

    summary_rows = []

    for (worker, source_file), group in detail.groupby(
        ["worker", "source_file"],
        sort=False,
    ):
        ranked = group.sort_values("verification_rank")
        best = ranked.iloc[0]
        second_score = (
            float(ranked.iloc[1]["verification_score"])
            if len(ranked) > 1
            else np.nan
        )
        best_score = float(best["verification_score"])
        score_gap = (
            best_score - second_score
            if not pd.isna(second_score)
            else np.nan
        )

        if pd.isna(score_gap):
            confidence = "REVIEW"
        elif score_gap >= 0.25:
            confidence = "HIGH"
        elif score_gap >= 0.10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        summary_rows.append({
            "worker": worker,
            "source_file": source_file,
            "recording_type": best["recording_type"],
            "suggested_candidate_rank": int(best["candidate_rank"]),
            "suggested_event_time_seconds": round(
                float(best["candidate_time_seconds"]), 2
            ),
            "verification_score": round(best_score, 4),
            "score_gap_to_second": (
                round(score_gap, 4)
                if not pd.isna(score_gap)
                else np.nan
            ),
            "confidence": confidence,
            "posture_change_late_deg": round(
                float(best["posture_change_late_deg"]), 2
            ) if not pd.isna(best["posture_change_late_deg"]) else np.nan,
            "event_peak_acceleration_g": round(
                float(best["event_peak_acceleration_g"]), 3
            ) if not pd.isna(best["event_peak_acceleration_g"]) else np.nan,
            "event_peak_gyroscope_dps": round(
                float(best["event_peak_gyroscope_dps"]), 2
            ) if not pd.isna(best["event_peak_gyroscope_dps"]) else np.nan,
            "verification_logic": best["verification_logic"],
            "human_confirmation_required": "YES",
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print()
    print("Verification summary:")
    for _, row in summary.iterrows():
        print(
            f"{row['worker']} | {row['source_file']} | "
            f"C{int(row['suggested_candidate_rank'])} "
            f"@ {row['suggested_event_time_seconds']:.2f}s | "
            f"confidence={row['confidence']}"
        )

    print()
    print("Files created:")
    print(DETAIL_OUTPUT)
    print(SUMMARY_OUTPUT)
    print()
    print("IMPORTANT:")
    print("These are verification suggestions, not final detector thresholds.")
    print("Human confirmation is still required before final labels are written.")


if __name__ == "__main__":
    main()
