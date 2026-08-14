from __future__ import annotations

import pandas as pd


REQUIRED_SIGNAL_KEYS = (
    "acceleration_peak_g",
    "gyroscope_peak_dps",
    "jerk_peak_g_per_second",
    "rotation_integral_deg",
)


def classify_votes(votes):
    """Convert the 4-signal vote count into the dashboard live state."""
    if votes is None or pd.isna(votes):
        return "UNKNOWN"

    votes = int(votes)
    if votes >= 4:
        return "FALL"
    if votes >= 2:
        return "NEAR_MISS"
    return "SAFE"


def _threshold_value(spec):
    if isinstance(spec, dict):
        return spec.get("value")
    return spec


def classify_live_samples(samples, thresholds, window_seconds=2.0):
    """
    Classify each worker from the most recent live-signal window.

    This intentionally mirrors the dashboard's four detector signals:
    acceleration peak, gyroscope peak, jerk peak and trapezoidal gyroscope
    rotation integral. It is a live/indicative state; the permanent event log
    remains the authoritative record of detector events.
    """
    if samples is None or samples.empty:
        return {}

    required_columns = {
        "worker_id",
        "sample_epoch",
        "acceleration_magnitude_g",
        "gyroscope_magnitude_dps",
        "jerk_g_per_second",
    }
    missing = required_columns.difference(samples.columns)
    if missing:
        raise ValueError(
            "Live samples are missing required columns: " + ", ".join(sorted(missing))
        )

    limits = {
        key: _threshold_value(thresholds.get(key))
        for key in REQUIRED_SIGNAL_KEYS
    }

    if any(value is None or pd.isna(value) for value in limits.values()):
        return {}

    newest = float(samples["sample_epoch"].max())
    window = samples.loc[
        samples["sample_epoch"] >= newest - float(window_seconds)
    ].copy()

    states = {}

    for worker_id, group in window.groupby("worker_id"):
        group = group.sort_values("sample_epoch")
        if group.empty:
            continue

        rotation = 0.0
        times = group["sample_epoch"].to_numpy()
        gyros = group["gyroscope_magnitude_dps"].to_numpy()

        for index in range(1, len(times)):
            dt = times[index] - times[index - 1]
            if dt > 0:
                rotation += 0.5 * (gyros[index] + gyros[index - 1]) * dt

        measured = {
            "acceleration_peak_g": float(group["acceleration_magnitude_g"].max()),
            "gyroscope_peak_dps": float(group["gyroscope_magnitude_dps"].max()),
            "jerk_peak_g_per_second": float(group["jerk_g_per_second"].max()),
            "rotation_integral_deg": float(rotation),
        }

        votes = sum(
            int(measured[key] >= float(limits[key]))
            for key in REQUIRED_SIGNAL_KEYS
        )

        states[str(worker_id)] = {
            "votes": votes,
            "status": classify_votes(votes),
            "measured": measured,
        }

    return states
