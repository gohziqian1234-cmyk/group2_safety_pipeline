from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GROUP 5 FALL-SIGNATURE CHECK
# Rule-based / physics-based diagnostics only — NO AI / NO ML
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

INPUT_FILE = PROCESSED / "combined_cleaned_data.csv"
LABEL_FILE = PROCESSED / "event_labels.csv"
OUTPUT_FILE = PROCESSED / "fall_signature_features.csv"
SUMMARY_FILE = PROCESSED / "fall_signature_summary.csv"

TARGET_RATE_HZ = 25
DT = 1.0 / TARGET_RATE_HZ
NORMAL_STEP_SECONDS = 1.0

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


def resample_recording(recording):
    recording = recording.copy()
    recording["elapsed_seconds"] = pd.to_numeric(
        recording["elapsed_seconds"], errors="coerce"
    )
    recording["acceleration_magnitude_g"] = pd.to_numeric(
        recording["acceleration_magnitude_g"], errors="coerce"
    )
    recording["gyroscope_magnitude_dps"] = pd.to_numeric(
        recording["gyroscope_magnitude_dps"], errors="coerce"
    )

    recording = (
        recording
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

    if len(recording) < 2:
        return pd.DataFrame()

    time_values = recording["elapsed_seconds"].to_numpy(dtype=float)
    start = float(time_values.min())
    end = float(time_values.max())

    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return pd.DataFrame()

    regular_time = np.arange(start, end + DT / 2, DT)

    if len(regular_time) < 2:
        return pd.DataFrame()

    output = pd.DataFrame({"elapsed_seconds": regular_time})

    for column in ["acceleration_magnitude_g", "gyroscope_magnitude_dps"]:
        output[column] = np.interp(
            regular_time,
            time_values,
            recording[column].to_numpy(dtype=float),
        )

    return output


def extract_signature(data, center):
    """
    Check whether a low-acceleration period is followed by an impact.

    Search interval: center - 1.5 s to center + 1.5 s.
    The strongest acceleration peak is treated as the impact candidate.
    Only acceleration BEFORE that peak is used for the low-g signature.
    """
    search_start = center - 1.5
    search_end = center + 1.5

    window = data[
        (data["elapsed_seconds"] >= search_start)
        & (data["elapsed_seconds"] <= search_end)
    ].copy()

    if len(window) < 10:
        return None

    accel = window["acceleration_magnitude_g"].to_numpy(dtype=float)
    gyro = window["gyroscope_magnitude_dps"].to_numpy(dtype=float)
    times = window["elapsed_seconds"].to_numpy(dtype=float)

    if not (
        np.all(np.isfinite(accel))
        and np.all(np.isfinite(gyro))
        and np.all(np.isfinite(times))
    ):
        return None

    peak_index = int(np.argmax(accel))
    peak_time = float(times[peak_index])
    peak_accel = float(accel[peak_index])
    peak_gyro = float(np.max(gyro))

    # Need at least several samples before the impact candidate.
    if peak_index < 4:
        return None

    pre_accel = accel[:peak_index]
    pre_times = times[:peak_index]

    min_index = int(np.argmin(pre_accel))
    min_accel = float(pre_accel[min_index])
    min_time = float(pre_times[min_index])

    time_min_to_peak = peak_time - min_time
    accel_rise = peak_accel - min_accel

    # Consecutive low-g evidence is useful for a real fall signature.
    low_08_samples = int(np.sum(pre_accel < 0.8))
    low_06_samples = int(np.sum(pre_accel < 0.6))
    low_04_samples = int(np.sum(pre_accel < 0.4))

    return {
        "pre_impact_min_accel_g": min_accel,
        "impact_peak_accel_g": peak_accel,
        "impact_peak_gyro_dps": peak_gyro,
        "accel_rise_min_to_impact_g": float(accel_rise),
        "time_min_to_impact_s": float(time_min_to_peak),
        "pre_impact_low_g_duration_lt_0_8_s": float(low_08_samples * DT),
        "pre_impact_low_g_duration_lt_0_6_s": float(low_06_samples * DT),
        "pre_impact_low_g_duration_lt_0_4_s": float(low_04_samples * DT),
    }


def main():
    print("=" * 78)
    print("GROUP 5 - FALL-SIGNATURE CHECK")
    print("Low-g -> impact temporal signature | NO AI / NO ML")
    print("=" * 78)

    if not INPUT_FILE.exists():
        raise SystemExit(f"ERROR: missing cleaned data:\n{INPUT_FILE}")

    if not LABEL_FILE.exists():
        raise SystemExit(f"ERROR: missing event labels:\n{LABEL_FILE}")

    data = pd.read_csv(INPUT_FILE)
    labels = pd.read_csv(LABEL_FILE)

    missing_data = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_data:
        raise SystemExit(
            "ERROR: cleaned data is missing columns:\n"
            + "\n".join(f" - {c}" for c in missing_data)
        )

    missing_labels = [c for c in REQUIRED_LABEL_COLUMNS if c not in labels.columns]
    if missing_labels:
        raise SystemExit(
            "ERROR: event_labels.csv is missing columns:\n"
            + "\n".join(f" - {c}" for c in missing_labels)
        )

    labels["confirmed"] = (
        labels["confirmed"].astype(str).str.upper().str.strip()
    )
    labels["suggested_event_time_seconds"] = pd.to_numeric(
        labels["suggested_event_time_seconds"], errors="coerce"
    )
    labels["final_event_type"] = (
        labels["final_event_type"].fillna("").astype(str).str.strip()
    )

    allowed = {"near_miss", "fall_recovery", "fall_inactive"}
    confirmed = labels[
        (labels["confirmed"] == "YES")
        & labels["suggested_event_time_seconds"].notna()
        & labels["final_event_type"].isin(allowed)
    ].copy()

    if len(confirmed) != 8:
        raise SystemExit(
            f"ERROR: expected 8 confirmed incidents, found {len(confirmed)}."
        )

    label_lookup = {
        (str(row.worker), str(row.source_file)): row
        for row in confirmed.itertuples(index=False)
    }

    rows = []

    grouped = data.groupby(
        ["worker", "source_file", "recording_type"],
        sort=False,
        dropna=False,
    )

    for (worker, source_file, recording_type), recording in grouped:
        regular = resample_recording(recording)

        if regular.empty:
            print(f"WARNING: skipped {worker} | {source_file}: resampling failed")
            continue

        is_normal = "normal" in str(recording_type).lower()

        if is_normal:
            start = float(regular["elapsed_seconds"].min()) + 2.0
            end = float(regular["elapsed_seconds"].max()) - 2.0

            if end <= start:
                continue

            centers = np.arange(
                start,
                end + NORMAL_STEP_SECONDS / 2,
                NORMAL_STEP_SECONDS,
            )

            added = 0
            for center in centers:
                features = extract_signature(regular, float(center))
                if features is None:
                    continue

                rows.append({
                    "worker": worker,
                    "source_file": source_file,
                    "signature_label": "normal",
                    "event_subtype": "normal_work",
                    "center_seconds": round(float(center), 3),
                    **features,
                })
                added += 1

            print(
                f"{worker:12s} | normal    | {added:4d} signature positions | {source_file}"
            )
            continue

        key = (str(worker), str(source_file))
        if key not in label_lookup:
            raise SystemExit(
                f"ERROR: missing confirmed label for {worker} | {source_file}"
            )

        label_row = label_lookup[key]
        center = float(label_row.suggested_event_time_seconds)
        features = extract_signature(regular, center)

        if features is None:
            raise SystemExit(
                f"ERROR: could not build fall-signature features for "
                f"{worker} | {source_file} | {center:.2f}s"
            )

        subtype = str(label_row.final_event_type)
        signature_label = "near_miss" if subtype == "near_miss" else "fall"

        rows.append({
            "worker": worker,
            "source_file": source_file,
            "signature_label": signature_label,
            "event_subtype": subtype,
            "center_seconds": round(center, 3),
            **features,
        })

        print(
            f"{worker:12s} | {signature_label:9s} | event @ {center:7.2f}s | {source_file}"
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise SystemExit("ERROR: no fall-signature rows created.")

    incidents = result[result["signature_label"].isin(["near_miss", "fall"])]

    if len(incidents) != 8:
        raise SystemExit(
            f"ERROR: expected 8 incident signature rows, created {len(incidents)}."
        )

    counts = incidents["signature_label"].value_counts()
    if counts.get("near_miss", 0) != 4 or counts.get("fall", 0) != 4:
        raise SystemExit("ERROR: expected 4 near-miss and 4 fall signature rows.")

    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(5)
    result.to_csv(OUTPUT_FILE, index=False)

    feature_columns = [
        "pre_impact_min_accel_g",
        "impact_peak_accel_g",
        "impact_peak_gyro_dps",
        "accel_rise_min_to_impact_g",
        "time_min_to_impact_s",
        "pre_impact_low_g_duration_lt_0_8_s",
        "pre_impact_low_g_duration_lt_0_6_s",
        "pre_impact_low_g_duration_lt_0_4_s",
    ]

    summary_rows = []
    for label in ["normal", "near_miss", "fall"]:
        subset = result[result["signature_label"] == label]
        if subset.empty:
            raise SystemExit(f"ERROR: no rows created for '{label}'.")

        for feature in feature_columns:
            series = pd.to_numeric(subset[feature], errors="coerce").dropna()
            if series.empty:
                raise SystemExit(
                    f"ERROR: no valid values for {feature} / {label}."
                )

            summary_rows.append({
                "signature_label": label,
                "feature": feature,
                "count": int(series.count()),
                "median": float(series.median()),
                "q05": float(series.quantile(0.05)),
                "q95": float(series.quantile(0.95)),
                "min": float(series.min()),
                "max": float(series.max()),
            })

    summary = pd.DataFrame(summary_rows)
    summary_numeric = summary.select_dtypes(include=[np.number]).columns
    summary[summary_numeric] = summary[summary_numeric].round(5)
    summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print("Confirmed incident fall-signature features:")
    print(
        incidents[
            [
                "worker",
                "signature_label",
                "event_subtype",
                "pre_impact_min_accel_g",
                "impact_peak_accel_g",
                "accel_rise_min_to_impact_g",
                "time_min_to_impact_s",
                "pre_impact_low_g_duration_lt_0_8_s",
                "pre_impact_low_g_duration_lt_0_6_s",
            ]
        ].to_string(index=False)
    )

    print()
    print("Normal-work reference:")
    normal_summary = summary[summary["signature_label"] == "normal"]
    print(
        normal_summary[
            ["feature", "median", "q05", "q95", "min", "max"]
        ].to_string(index=False)
    )

    print()
    print("Files created:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print()
    print("IMPORTANT:")
    print("This step does NOT assume low-g proves a fall.")
    print("It checks whether a low-g -> impact pattern actually exists in YOUR data.")
    print("Only after reviewing this result will we build the final state-machine rule validation.")


if __name__ == "__main__":
    main()
