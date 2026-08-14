from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from live_status import classify_live_samples


# ============================================================
# GROUP 5 - NESSO SAFETY MONITOR
# Local-first | SQLite | live BLE status
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT / "database" / "safety_pipeline.db"
SINGAPORE_TZ = timezone(timedelta(hours=8))
STALE_AFTER_SECONDS = 60
LIVE_CHART_SECONDS = 30
LIVE_STATUS_SECONDS = 2.0


st.set_page_config(
    page_title="NESSO Safety Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    header[data-testid="stHeader"] { background: transparent; }

    .topbar, .section-card, .worker-card {
        background: white;
        border: 1px solid #e5e9f0;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
    }
    .topbar {
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .brand-title {
        font-size: 28px;
        line-height: 1.1;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }
    .brand-subtitle { font-size: 13px; color: #6b7280; }

    .section-card {
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .section-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 3px;
    }
    .section-help { font-size: 12px; color: #6b7280; }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e9f0;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
    }
    div[data-testid="stMetricLabel"] { color: #6b7280; font-size: 12px; }
    div[data-testid="stMetricValue"] {
        color: #111827;
        font-size: 27px;
        font-weight: 800;
    }

    .worker-card {
        border-radius: 14px;
        padding: 16px;
        min-height: 180px;
    }
    .worker-name {
        font-size: 17px;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }
    .worker-id { margin-top: 3px; color: #6b7280; font-size: 12px; }
    .status-pill {
        display: inline-block;
        margin-top: 12px;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .safe { color: #067647; background: #ecfdf3; border: 1px solid #abefc6; }
    .near { color: #b54708; background: #fffaeb; border: 1px solid #fedf89; }
    .fall { color: #b42318; background: #fef3f2; border: 1px solid #fecdca; }
    .offline { color: #475467; background: #f2f4f7; border: 1px solid #d0d5dd; }

    .worker-grid {
        margin-top: 14px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px 18px;
    }
    .mini-label {
        color: #98a2b3;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .mini-value {
        color: #344054;
        font-size: 12px;
        font-weight: 650;
        margin-top: 1px;
    }

    .alert-fall, .alert-near, .all-clear, .info-banner {
        border-radius: 14px;
        padding: 15px 19px;
        margin-bottom: 14px;
        font-size: 13px;
        font-weight: 700;
    }
    .alert-fall { background: #d92d20; color: white; }
    .alert-near { background: #fffaeb; color: #93370d; border: 1px solid #fedf89; }
    .all-clear { background: #ecfdf3; color: #067647; border: 1px solid #abefc6; }
    .info-banner { background: #eff8ff; color: #175cd3; border: 1px solid #b2ddff; }

    .footer-note {
        color: #98a2b3;
        font-size: 11px;
        text-align: center;
        padding-top: 10px;
    }
    button[data-baseweb="tab"] { font-weight: 700; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def parse_datetime(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def sg_time(value):
    dt = parse_datetime(value)
    if dt is None:
        return "Never"
    return dt.astimezone(SINGAPORE_TZ).strftime("%d %b, %H:%M:%S")


def age_text(value):
    dt = parse_datetime(value)
    if dt is None:
        return "No signal yet"
    seconds = max(0, int((utc_now() - dt).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"


def connect_db():
    if not DB_PATH.exists():
        st.error("Database not found. Run `py src\\17_init_database.py` first.")
        st.stop()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def safe_int(row, possible_names, default=0):
    for name in possible_names:
        if name in row.index and not pd.isna(row[name]):
            return int(row[name])
    return default


def safe_float(row, possible_names, default=None):
    for name in possible_names:
        if name in row.index and not pd.isna(row[name]):
            return float(row[name])
    return default


def load_data():
    con = connect_db()
    try:
        workers = pd.read_sql_query(
            """
            SELECT
                w.worker_id, w.worker_name, w.device_id, w.ble_name, w.active,
                ds.connection_status, ds.safety_status, ds.battery_percent,
                ds.queued_events, ds.last_seen_at, ds.last_event_at, ds.updated_at
            FROM workers w
            LEFT JOIN device_status ds
              ON w.worker_id = ds.worker_id AND w.device_id = ds.device_id
            WHERE w.active = 1
            ORDER BY w.worker_id;
            """,
            con,
        )
        events = pd.read_sql_query(
            """
            SELECT
                e.event_id, e.event_uuid, e.worker_id, w.worker_name, e.device_id,
                e.event_time, e.event_type, e.detection_source, e.event_votes,
                e.acceleration_peak_g, e.gyroscope_peak_dps,
                e.jerk_peak_g_per_second, e.rotation_integral_deg,
                e.acknowledged, e.acknowledged_at, e.notes, e.received_at
            FROM events e
            LEFT JOIN workers w ON e.worker_id = w.worker_id
            ORDER BY e.event_time DESC;
            """,
            con,
        )
        detector = pd.read_sql_query(
            """
            SELECT * FROM detector_config
            WHERE is_active = 1
            ORDER BY config_version DESC
            LIMIT 1;
            """,
            con,
        )
        queue = pd.read_sql_query(
            "SELECT * FROM offline_queue ORDER BY queued_at DESC;",
            con,
        )
    finally:
        con.close()
    return workers, events, detector, queue


def load_live_samples(seconds=LIVE_CHART_SECONDS):
    con = connect_db()
    try:
        exists = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='live_samples';"
        ).fetchone()
        if exists is None:
            return pd.DataFrame()
        cutoff = time.time() - seconds
        return pd.read_sql_query(
            """
            SELECT
                ls.worker_id,
                COALESCE(w.worker_name, ls.worker_id) AS worker_name,
                ls.sample_epoch,
                ls.acceleration_magnitude_g,
                ls.gyroscope_magnitude_dps,
                ls.jerk_g_per_second
            FROM live_samples ls
            LEFT JOIN workers w ON w.worker_id = ls.worker_id
            WHERE ls.sample_epoch >= ?
            ORDER BY ls.sample_epoch;
            """,
            con,
            params=(cutoff,),
        )
    finally:
        con.close()


def detector_thresholds(detector_row):
    if detector_row is None:
        return {}
    return {
        "acceleration_peak_g": {
            "label": "Acceleration", "unit": "g",
            "value": safe_float(detector_row, ["acceleration_threshold_g", "accel_threshold_g"]),
            "format": "{:.2f}",
        },
        "gyroscope_peak_dps": {
            "label": "Gyroscope", "unit": "dps",
            "value": safe_float(detector_row, ["gyroscope_threshold_dps", "gyro_threshold_dps"]),
            "format": "{:.0f}",
        },
        "jerk_peak_g_per_second": {
            "label": "Jerk", "unit": "g/s",
            "value": safe_float(detector_row, ["jerk_threshold_g_per_second", "jerk_threshold_gps"]),
            "format": "{:.1f}",
        },
        "rotation_integral_deg": {
            "label": "Rotation", "unit": "deg",
            "value": safe_float(detector_row, ["rotation_threshold_deg", "rotation_threshold_degrees"]),
            "format": "{:.0f}",
        },
    }


def effective_connection(row):
    last_seen = parse_datetime(row["last_seen_at"])
    if last_seen is None:
        return "DISCONNECTED"
    return (
        "CONNECTED"
        if (utc_now() - last_seen).total_seconds() <= STALE_AFTER_SECONDS
        else "DISCONNECTED"
    )


def test_event_mask(events):
    if events.empty:
        return pd.Series(False, index=events.index, dtype=bool)
    type_mask = events["event_type"].fillna("").astype(str).str.upper().eq("TEST_EVENT")
    source_mask = (
        events["detection_source"].fillna("").astype(str).str.upper()
        .isin(["SIMULATION", "OFFLINE_REPLAY"])
    )
    note_mask = events["notes"].fillna("").astype(str).str.contains(
        "simulat|test", case=False, regex=True
    )
    return type_mask | source_mask | note_mask


def incident_classification(votes):
    if votes is None or pd.isna(votes):
        return "—"
    votes = int(votes)
    if votes >= 4:
        return "Fall"
    if votes >= 2:
        return "Near-miss"
    return "Normal"


def worker_card(row, live_status):
    if live_status == "SAFE":
        pill = '<span class="status-pill safe">● SAFE</span>'
    elif live_status == "NEAR_MISS":
        pill = '<span class="status-pill near">● NEAR-MISS</span>'
    elif live_status == "FALL":
        pill = '<span class="status-pill fall">● FALL</span>'
    else:
        pill = '<span class="status-pill offline">● DISCONNECTED</span>'

    battery = "—" if pd.isna(row["battery_percent"]) else f"{float(row['battery_percent']):.0f}%"
    queued = 0 if pd.isna(row["queued_events"]) else int(row["queued_events"])

    html = (
        '<div class="worker-card">'
        f'<div class="worker-name">{row["worker_name"]}</div>'
        f'<div class="worker-id">{row["worker_id"]} · {row["device_id"]}</div>'
        f'{pill}'
        '<div class="worker-grid">'
        '<div><div class="mini-label">Battery</div>'
        f'<div class="mini-value">{battery}</div></div>'
        '<div><div class="mini-label">Queue</div>'
        f'<div class="mini-value">{queued}</div></div>'
        '<div><div class="mini-label">Last seen</div>'
        f'<div class="mini-value">{age_text(row["last_seen_at"])}</div></div>'
        '<div><div class="mini-label">Last event</div>'
        f'<div class="mini-value">{sg_time(row["last_event_at"])}</div></div>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# LOAD NON-LIVE DATA
# ============================================================
workers, events, detector, queue = load_data()
workers = workers.copy()
workers["effective_connection"] = workers.apply(effective_connection, axis=1)

test_mask = test_event_mask(events)
show_test = st.toggle(
    "Show test / simulated data",
    value=False,
    help="Keep this off for the real dashboard view.",
)
visible_events = events.copy() if show_test else events.loc[~test_mask].copy()

connected = int((workers["effective_connection"] == "CONNECTED").sum())
real_event_count = len(events.loc[~test_mask])
pending_queue = 0
if not queue.empty and "sync_status" in queue.columns:
    pending_queue = int(
        queue["sync_status"].fillna("").astype(str).str.upper().eq("PENDING").sum()
    )


# ============================================================
# HEADER / NAVIGATION
# ============================================================
st.markdown(
    """
    <div class="topbar">
        <div class="brand-title">NESSO Safety Monitor</div>
        <div class="brand-subtitle">
            Real-time worker safety monitoring · EG2A17 Group 5 · Local-first prototype
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if connected == 0:
    st.markdown(
        '<div class="info-banner">No fresh NESSO status is being received right now. Start the BLE gateway to begin live monitoring.</div>',
        unsafe_allow_html=True,
    )

overview_tab, events_tab, devices_tab, config_tab = st.tabs(
    ["Overview", "Incident Log", "Devices", "Detector"]
)


# ============================================================
# OVERVIEW
# ============================================================
with overview_tab:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Workers", len(workers))
    k2.metric("Connected", connected)
    k3.metric("Safety Events", real_event_count)
    k4.metric("Pending Queue", pending_queue)
    st.write("")

    @st.fragment(run_every="1s")
    def live_worker_status_panel():
        live_workers, _, live_detector, _ = load_data()
        live_workers = live_workers.copy()
        live_workers["effective_connection"] = live_workers.apply(effective_connection, axis=1)

        samples = load_live_samples()
        thresholds = detector_thresholds(
            None if live_detector.empty else live_detector.iloc[0]
        )
        states = classify_live_samples(
            samples,
            thresholds,
            window_seconds=LIVE_STATUS_SECONDS,
        )

        resolved = []
        for _, row in live_workers.iterrows():
            if row["effective_connection"] == "DISCONNECTED":
                status = "DISCONNECTED"
            else:
                state = states.get(str(row["worker_id"]))
                status = state["status"] if state else "SAFE"
            resolved.append((row, status))

        fall_rows = [(row, status) for row, status in resolved if status == "FALL"]
        near_rows = [(row, status) for row, status in resolved if status == "NEAR_MISS"]

        if fall_rows:
            names = ", ".join(row["worker_name"] for row, _ in fall_rows)
            st.markdown(
                f'<div class="alert-fall">⚠ FALL DETECTED — {names}. Supervisor check required.</div>',
                unsafe_allow_html=True,
            )
        elif near_rows:
            names = ", ".join(row["worker_name"] for row, _ in near_rows)
            st.markdown(
                f'<div class="alert-near">⚠ NEAR-MISS — {names}. Elevated movement detected.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="all-clear">✓ All connected workers are currently below the incident threshold.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Workers</div>
                <div class="section-help">
                    Live state refreshes every second from the same four signal thresholds used below.
                    0–1 votes = SAFE, 2–3 = NEAR-MISS, 4 = FALL.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        for col, (row, status) in zip(cols, resolved):
            with col:
                worker_card(row, status)

    live_worker_status_panel()
    st.write("")

    @st.fragment(run_every="1s")
    def live_signal_panel():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Live signal</div>
                <div class="section-help">
                    Recent sensor magnitude. Dashed lines are the active detector thresholds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        samples = load_live_samples()
        if samples.empty:
            st.info("No live signal yet. Start `py src\\21_live_ble_gateway.py`.")
            return

        thresholds = detector_thresholds(None if detector.empty else detector.iloc[0])
        samples["time"] = pd.to_datetime((samples["sample_epoch"] // 0.2) * 0.2, unit="s")
        binned = (
            samples.groupby(["time", "worker_name"], as_index=False)
            .agg(
                accel=("acceleration_magnitude_g", "max"),
                gyro=("gyroscope_magnitude_dps", "max"),
                jerk=("jerk_g_per_second", "max"),
            )
        )

        def signal_chart(column, title, unit, threshold):
            top = max(float(binned[column].max()), float(threshold or 0))
            top = max(top, 1e-6)
            lines = (
                alt.Chart(binned)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("time:T", title=None),
                    y=alt.Y(f"{column}:Q", title=f"{title} ({unit})", scale=alt.Scale(domain=[0, top * 1.15])),
                    color=alt.Color("worker_name:N", title="Worker"),
                    tooltip=["worker_name:N", alt.Tooltip(f"{column}:Q", format=".2f"), "time:T"],
                )
            )
            if threshold is None:
                return lines.properties(height=220)
            rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
                strokeDash=[6, 4], strokeWidth=2
            ).encode(y="y:Q")
            return (lines + rule).properties(height=220)

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                signal_chart(
                    "accel", "Acceleration", "g",
                    thresholds.get("acceleration_peak_g", {}).get("value"),
                ),
                width="stretch",
            )
        with c2:
            st.altair_chart(
                signal_chart(
                    "gyro", "Gyroscope", "dps",
                    thresholds.get("gyroscope_peak_dps", {}).get("value"),
                ),
                width="stretch",
            )

        states = classify_live_samples(samples, thresholds, window_seconds=LIVE_STATUS_SECONDS)
        rows = []
        for worker_id, state in states.items():
            name_rows = samples.loc[samples["worker_id"].astype(str) == worker_id, "worker_name"]
            worker_name = name_rows.iloc[-1] if not name_rows.empty else worker_id
            measured = state["measured"]
            rows.append(
                {
                    "Worker": worker_name,
                    "Incident": incident_classification(state["votes"]),
                    "Acceleration": f"{measured['acceleration_peak_g']:.2f} g",
                    "Gyroscope": f"{measured['gyroscope_peak_dps']:.0f} dps",
                    "Jerk": f"{measured['jerk_peak_g_per_second']:.1f} g/s",
                    "Rotation": f"{measured['rotation_integral_deg']:.0f} deg",
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    live_signal_panel()
    st.write("")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Recent safety events</div>
            <div class="section-help">Permanent event records stored in SQLite.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if visible_events.empty:
        st.info("No real safety events are stored in the current view.")
    else:
        recent = visible_events.head(8).copy()
        recent["Time"] = recent["event_time"].map(sg_time)
        recent["Worker"] = recent["worker_name"].fillna(recent["worker_id"])
        recent["Incident"] = recent["event_votes"].map(incident_classification)
        st.dataframe(
            recent[
                ["Time", "Worker", "device_id", "Incident", "acceleration_peak_g", "gyroscope_peak_dps", "detection_source"]
            ].rename(
                columns={
                    "device_id": "Device",
                    "acceleration_peak_g": "Peak accel (g)",
                    "gyroscope_peak_dps": "Peak gyro (dps)",
                    "detection_source": "Source",
                }
            ),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# INCIDENT LOG
# ============================================================
with events_tab:
    st.subheader("Incident Log")
    if visible_events.empty:
        st.info("No events match the current view.")
    else:
        log = visible_events.copy()
        log["Time"] = log["event_time"].map(sg_time)
        log["Worker"] = log["worker_name"].fillna(log["worker_id"])
        log["Event"] = log["event_votes"].map(incident_classification)
        log["Acknowledged"] = log["acknowledged"].map({0: "No", 1: "Yes"})
        st.dataframe(
            log[
                [
                    "Time", "Worker", "device_id", "Event",
                    "acceleration_peak_g", "gyroscope_peak_dps",
                    "jerk_peak_g_per_second", "rotation_integral_deg",
                    "detection_source", "Acknowledged", "notes",
                ]
            ].rename(
                columns={
                    "device_id": "Device",
                    "acceleration_peak_g": "Peak accel (g)",
                    "gyroscope_peak_dps": "Peak gyro (dps)",
                    "jerk_peak_g_per_second": "Peak jerk (g/s)",
                    "rotation_integral_deg": "Rotation activity (deg)",
                    "detection_source": "Source",
                    "notes": "Notes",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("Rotation activity is integrated gyroscope magnitude; it is not net body angle.")


# ============================================================
# DEVICES
# ============================================================
with devices_tab:
    st.subheader("Device Status")
    device_view = workers.copy()
    device_view["Connection"] = device_view["effective_connection"]
    device_view["Battery"] = device_view["battery_percent"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):.0f}%"
    )
    device_view["Queued"] = device_view["queued_events"].fillna(0).astype(int)
    device_view["Last seen"] = device_view["last_seen_at"].map(sg_time)
    device_view["Last event"] = device_view["last_event_at"].map(sg_time)
    st.dataframe(
        device_view[
            ["worker_name", "worker_id", "device_id", "Connection", "Battery", "Queued", "Last seen", "Last event"]
        ].rename(
            columns={"worker_name": "Worker", "worker_id": "Worker ID", "device_id": "Device"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        f"A device is shown DISCONNECTED after more than {STALE_AFTER_SECONDS} seconds without a fresh STATUS message."
    )


# ============================================================
# DETECTOR
# ============================================================
with config_tab:
    st.subheader("Validated Stage-1 Detector")
    if detector.empty:
        st.error("No active detector configuration was found.")
    else:
        row = detector.iloc[0]
        required_votes = safe_int(row, ["required_votes_out_of_4", "required_votes"], 2)
        matched = safe_int(row, ["reliable_incidents_matched", "reliable_incidents_detected"], 0)
        tested = safe_int(row, ["reliable_incidents_tested"], 0)
        false_negatives = safe_int(row, ["false_negatives"], 0)

        d1, d2, d3 = st.columns(3)
        d1.metric("Trigger rule", f"{required_votes} signals required")
        d2.metric("Reliable incidents detected", f"{matched}/{tested}" if tested else str(matched))
        d3.metric("False negatives", false_negatives)

        thresholds = detector_thresholds(row)
        threshold_rows = []
        for spec in thresholds.values():
            if spec["value"] is not None:
                threshold_rows.append(
                    {
                        "Signal": spec["label"],
                        "Threshold": "≥ " + spec["format"].format(spec["value"]) + f" {spec['unit']}",
                    }
                )
        st.dataframe(pd.DataFrame(threshold_rows), width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        c1.metric("Startup grace", f"{safe_float(row, ['startup_grace_seconds'], 0):.1f} s")
        c2.metric("Cooldown", f"{safe_float(row, ['cooldown_seconds'], 0):.1f} s")

        st.warning(
            "The internal validation result applies only to the labelled validation recordings. It does not guarantee identical performance on future workers."
        )
        st.caption(
            "Live dashboard labels are vote-based for the demonstration: 0–1 = SAFE, 2–3 = Near-miss, 4 = Fall. Stored detector events remain SAFETY_EVENT records."
        )


st.markdown(
    '<div class="footer-note">Group 5 · NESSO wearable safety monitoring · SQLite local-first prototype</div>',
    unsafe_allow_html=True,
)
