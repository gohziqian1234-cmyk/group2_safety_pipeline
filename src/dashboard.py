from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st


# ============================================================
# GROUP 5 - CLEAN SAFETY MONITOR DASHBOARD
# Local-first | SQLite | Boss-friendly UI
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT / "database" / "safety_pipeline.db"

SINGAPORE_TZ = timezone(timedelta(hours=8))
STALE_AFTER_SECONDS = 60


# -----------------------------
# PAGE
# -----------------------------
st.set_page_config(
    page_title="NESSO Safety Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* top header */
    .topbar {
        background: white;
        border: 1px solid #e5e9f0;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
    }

    .brand-title {
        font-size: 28px;
        line-height: 1.1;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .brand-subtitle {
        font-size: 13px;
        color: #6b7280;
    }

    .section-card {
        background: white;
        border: 1px solid #e5e9f0;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.035);
    }

    .section-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 3px;
    }

    .section-help {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    /* metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e9f0;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
    }

    div[data-testid="stMetricLabel"] {
        color: #6b7280;
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #111827;
        font-size: 27px;
        font-weight: 800;
    }

    /* worker cards */
    .worker-card {
        background: white;
        border: 1px solid #e5e9f0;
        border-radius: 14px;
        padding: 16px;
        min-height: 180px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
    }

    .worker-name {
        font-size: 17px;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }

    .worker-id {
        margin-top: 3px;
        color: #6b7280;
        font-size: 12px;
    }

    .status-pill {
        display: inline-block;
        margin-top: 12px;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }

    .safe {
        color: #067647;
        background: #ecfdf3;
        border: 1px solid #abefc6;
    }

    .event {
        color: #b42318;
        background: #fef3f2;
        border: 1px solid #fecdca;
    }

    .offline {
        color: #475467;
        background: #f2f4f7;
        border: 1px solid #d0d5dd;
    }

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

    /* alerts */
    .alert-red {
        background: #d92d20;
        color: white;
        border-radius: 14px;
        padding: 17px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(217,45,32,.16);
    }

    .alert-red-title {
        font-size: 18px;
        font-weight: 850;
    }

    .alert-red-detail {
        margin-top: 4px;
        font-size: 12px;
        opacity: .94;
    }

    .all-clear {
        background: #ecfdf3;
        color: #067647;
        border: 1px solid #abefc6;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 16px;
        font-size: 13px;
        font-weight: 700;
    }

    .info-banner {
        background: #eff8ff;
        color: #175cd3;
        border: 1px solid #b2ddff;
        border-radius: 12px;
        padding: 11px 14px;
        font-size: 12px;
        margin-bottom: 14px;
    }

    .footer-note {
        color: #98a2b3;
        font-size: 11px;
        text-align: center;
        padding-top: 10px;
    }

    /* tabs */
    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    /* hide streamlit menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# HELPERS
# -----------------------------
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


def table_columns(con, table_name):
    rows = con.execute(f"PRAGMA table_info({table_name});").fetchall()
    return {row["name"] for row in rows}


LIVE_CHART_SECONDS = 30      # how much history the live chart shows


def load_live_samples(seconds=LIVE_CHART_SECONDS):
    """
    Read the recent raw signal the gateway buffers for charting.

    Returns an empty frame if the table does not exist yet, which is the
    normal state before the gateway has run for the first time.
    """
    con = connect_db()
    try:
        exists = con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='live_samples';"
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
                ls.gyroscope_magnitude_dps
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


def load_data():
    con = connect_db()

    try:
        workers = pd.read_sql_query(
            """
            SELECT
                w.worker_id,
                w.worker_name,
                w.device_id,
                w.ble_name,
                w.active,
                ds.connection_status,
                ds.safety_status,
                ds.battery_percent,
                ds.queued_events,
                ds.last_seen_at,
                ds.last_event_at,
                ds.updated_at
            FROM workers w
            LEFT JOIN device_status ds
              ON w.worker_id = ds.worker_id
             AND w.device_id = ds.device_id
            WHERE w.active = 1
            ORDER BY w.worker_id;
            """,
            con,
        )

        events = pd.read_sql_query(
            """
            SELECT
                e.event_id,
                e.event_uuid,
                e.worker_id,
                w.worker_name,
                e.device_id,
                e.event_time,
                e.event_type,
                e.detection_source,
                e.event_votes,
                e.acceleration_peak_g,
                e.gyroscope_peak_dps,
                e.jerk_peak_g_per_second,
                e.rotation_integral_deg,
                e.acknowledged,
                e.acknowledged_at,
                e.notes,
                e.received_at
            FROM events e
            LEFT JOIN workers w
              ON e.worker_id = w.worker_id
            ORDER BY e.event_time DESC;
            """,
            con,
        )

        detector = pd.read_sql_query(
            """
            SELECT *
            FROM detector_config
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

        config_columns = table_columns(con, "detector_config")

    finally:
        con.close()

    return workers, events, detector, queue, config_columns


def enrich_workers(df):
    df = df.copy()

    connections = []
    statuses = []

    for _, row in df.iterrows():
        last_seen = parse_datetime(row["last_seen_at"])

        if last_seen is None:
            effective_connection = "DISCONNECTED"
        else:
            seconds = (utc_now() - last_seen).total_seconds()
            effective_connection = (
                "CONNECTED"
                if seconds <= STALE_AFTER_SECONDS
                else "DISCONNECTED"
            )

        if effective_connection == "DISCONNECTED":
            effective_status = "DISCONNECTED"
        elif str(row["safety_status"]).upper() == "SAFETY_EVENT":
            effective_status = "SAFETY_EVENT"
        else:
            effective_status = "SAFE"

        connections.append(effective_connection)
        statuses.append(effective_status)

    df["effective_connection"] = connections
    df["effective_status"] = statuses
    return df


def test_event_mask(events):
    if events.empty:
        return pd.Series(False, index=events.index, dtype=bool)

    type_mask = (
        events["event_type"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("TEST_EVENT")
    )

    source_mask = (
        events["detection_source"]
        .fillna("")
        .astype(str)
        .str.upper()
        .isin(["SIMULATION", "OFFLINE_REPLAY"])
    )

    note_mask = (
        events["notes"]
        .fillna("")
        .astype(str)
        .str.contains("simulat|test", case=False, regex=True)
    )

    return type_mask | source_mask | note_mask


def worker_card(row):
    status = row["effective_status"]

    if status == "SAFE":
        pill = '<span class="status-pill safe">● SAFE</span>'
    elif status == "SAFETY_EVENT":
        pill = '<span class="status-pill event">● SAFETY EVENT</span>'
    else:
        pill = '<span class="status-pill offline">● DISCONNECTED</span>'

    battery = (
        "—"
        if pd.isna(row["battery_percent"])
        else f"{float(row['battery_percent']):.0f}%"
    )

    queued = (
        0
        if pd.isna(row["queued_events"])
        else int(row["queued_events"])
    )

    html = (
        '<div class="worker-card">'
        f'<div class="worker-name">{row["worker_name"]}</div>'
        f'<div class="worker-id">{row["worker_id"]} · {row["device_id"]}</div>'
        f'{pill}'
        '<div class="worker-grid">'
        '<div>'
        '<div class="mini-label">Battery</div>'
        f'<div class="mini-value">{battery}</div>'
        '</div>'
        '<div>'
        '<div class="mini-label">Queue</div>'
        f'<div class="mini-value">{queued}</div>'
        '</div>'
        '<div>'
        '<div class="mini-label">Last seen</div>'
        f'<div class="mini-value">{age_text(row["last_seen_at"])}</div>'
        '</div>'
        '<div>'
        '<div class="mini-label">Last event</div>'
        f'<div class="mini-value">{sg_time(row["last_event_at"])}</div>'
        '</div>'
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


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


# -----------------------------
# DATA
# -----------------------------
workers, events, detector, queue, config_columns = load_data()
workers = enrich_workers(workers)

test_mask = test_event_mask(events)

# Default: real events only.
show_test = st.toggle(
    "Show test / simulated data",
    value=False,
    help="Code 18 and Code 19 created test records. Keep this off for the real dashboard view.",
)

visible_events = events.copy() if show_test else events.loc[~test_mask].copy()

connected = int((workers["effective_connection"] == "CONNECTED").sum())
active_alerts = int((workers["effective_status"] == "SAFETY_EVENT").sum())
real_event_count = len(events.loc[~test_mask])

pending_queue = 0
if not queue.empty and "sync_status" in queue.columns:
    pending_queue = int(
        queue["sync_status"].fillna("").astype(str).str.upper().eq("PENDING").sum()
    )


# -----------------------------
# HEADER
# -----------------------------
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
        """
        <div class="info-banner">
            No fresh NESSO status is being received right now. This is expected while the real BLE gateway is not running.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# NAV
# -----------------------------
overview_tab, events_tab, devices_tab, config_tab = st.tabs(
    ["Overview", "Incident Log", "Devices", "Detector"]
)


# ============================================================
# OVERVIEW
# ============================================================
with overview_tab:

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Workers", len(workers))
    k2.metric("Connected", connected)
    k3.metric("Safety Events", real_event_count)
    k4.metric("Pending Queue", pending_queue)

    st.write("")

    # Active alert banner
    active_rows = workers.loc[workers["effective_status"] == "SAFETY_EVENT"]

    if not active_rows.empty:
        for _, row in active_rows.iterrows():
            last_event_text = sg_time(row["last_event_at"])
            st.markdown(
                f"""
                <div class="alert-red">
                    <div class="alert-red-title">
                        SAFETY EVENT — {row["worker_name"]}
                    </div>
                    <div class="alert-red-detail">
                        Device {row["device_id"]} · Last event {last_event_text} · Supervisor check required
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="all-clear">
                ✓ No active safety alert is currently reported by a connected device.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Worker section
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Workers</div>
            <div class="section-help">
                Live state is based on the most recent gateway STATUS message.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_cols = st.columns(4)

    for col, (_, row) in zip(card_cols, workers.iterrows()):
        with col:
            worker_card(row)

    st.write("")

    # ------------------------------------------------------------------
    # LIVE SIGNAL CHART
    #
    # In its own fragment so it can refresh on a timer without re-running
    # the whole page - reloading every table once a second would make the
    # dashboard unusable.
    # ------------------------------------------------------------------
    @st.fragment(run_every="1s")
    def live_signal_chart():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Live signal</div>
                <div class="section-help">
                    Acceleration magnitude in g, one line per connected worker,
                    last 30 seconds. Updates once a second while the BLE
                    gateway is running.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        samples = load_live_samples()

        if samples.empty:
            st.info(
                "No live signal yet. Start the BLE gateway "
                "(py src\\21_live_ble_gateway.py) and the chart will fill in."
            )
            return

        # Put both workers on a shared time grid before pivoting.
        #
        # Every sample carries its own microsecond timestamp, so pivoting on
        # the raw value gives one row per sample with the other worker's
        # column empty - and a line chart skips gaps, so nothing is drawn.
        # Binning into fixed 200 ms buckets lines the workers up on the same
        # rows. The bucket takes the maximum rather than the mean, so a short
        # impact peak survives instead of being averaged away.
        samples["time"] = pd.to_datetime(
            (samples["sample_epoch"] // 0.2) * 0.2, unit="s"
        )
        accel = samples.pivot_table(
            index="time",
            columns="worker_name",
            values="acceleration_magnitude_g",
            aggfunc="max",
        )

        st.line_chart(accel, height=260)

        latest = (
            samples.sort_values("sample_epoch")
                   .groupby("worker_name")
                   .last()
                   .reset_index()
        )
        cols = st.columns(max(len(latest), 1))
        for col, (_, row) in zip(cols, latest.iterrows()):
            col.metric(
                row["worker_name"],
                f"{row['acceleration_magnitude_g']:.2f} g",
                f"{row['gyroscope_magnitude_dps']:.0f} dps",
            )

    live_signal_chart()

    st.write("")

    # Recent event section
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Recent safety events</div>
            <div class="section-help">
                Most recent validated Stage-1 safety-event records stored in SQLite.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if visible_events.empty:
        st.info(
            "No real safety events are stored in the current view."
            + (
                f" {int(test_mask.sum())} test record(s) are hidden."
                if int(test_mask.sum()) > 0 and not show_test
                else ""
            )
        )
    else:
        recent = visible_events.head(8).copy()
        recent["Time"] = recent["event_time"].map(sg_time)
        recent["Worker"] = recent["worker_name"].fillna(recent["worker_id"])
        recent["Votes"] = recent["event_votes"].map(
            lambda x: "—" if pd.isna(x) else f"{int(x)}/4"
        )

        st.dataframe(
            recent[
                [
                    "Time",
                    "Worker",
                    "device_id",
                    "Votes",
                    "acceleration_peak_g",
                    "gyroscope_peak_dps",
                    "detection_source",
                ]
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

        # Feature profile of latest event
        latest = visible_events.iloc[0]

        feature_df = pd.DataFrame(
            {
                "Feature": [
                    "Acceleration (g)",
                    "Gyroscope (dps)",
                    "Jerk (g/s)",
                    "Rotation activity (deg)",
                ],
                "Value": [
                    float(latest["acceleration_peak_g"]),
                    float(latest["gyroscope_peak_dps"]),
                    float(latest["jerk_peak_g_per_second"]),
                    float(latest["rotation_integral_deg"]),
                ],
            }
        ).set_index("Feature")

        st.caption("Latest event feature profile")
        st.bar_chart(feature_df)


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
        log["Votes"] = log["event_votes"].map(
            lambda x: "—" if pd.isna(x) else f"{int(x)}/4"
        )
        log["Acknowledged"] = log["acknowledged"].map({0: "No", 1: "Yes"})

        st.dataframe(
            log[
                [
                    "Time",
                    "Worker",
                    "device_id",
                    "event_type",
                    "Votes",
                    "acceleration_peak_g",
                    "gyroscope_peak_dps",
                    "jerk_peak_g_per_second",
                    "rotation_integral_deg",
                    "detection_source",
                    "Acknowledged",
                    "notes",
                ]
            ].rename(
                columns={
                    "device_id": "Device",
                    "event_type": "Event",
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

        st.caption(
            "Rotation activity is integrated gyroscope magnitude. "
            "It is not the worker's net body angle."
        )


# ============================================================
# DEVICE STATUS
# ============================================================
with devices_tab:
    st.subheader("Device Status")

    device_view = workers.copy()

    device_view["Connection"] = device_view["effective_connection"]
    device_view["Safety"] = device_view["effective_status"]
    device_view["Battery"] = device_view["battery_percent"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):.0f}%"
    )
    device_view["Queued"] = device_view["queued_events"].fillna(0).astype(int)
    device_view["Last seen"] = device_view["last_seen_at"].map(sg_time)
    device_view["Last event"] = device_view["last_event_at"].map(sg_time)

    st.dataframe(
        device_view[
            [
                "worker_name",
                "worker_id",
                "device_id",
                "Connection",
                "Safety",
                "Battery",
                "Queued",
                "Last seen",
                "Last event",
            ]
        ].rename(
            columns={
                "worker_name": "Worker",
                "worker_id": "Worker ID",
                "device_id": "Device",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.info(
        f"A device becomes DISCONNECTED on the dashboard when no fresh "
        f"STATUS message has been received for more than {STALE_AFTER_SECONDS} seconds."
    )


# ============================================================
# DETECTOR CONFIG
# ============================================================
with config_tab:
    st.subheader("Validated Stage-1 Detector")

    if detector.empty:
        st.error("No active detector configuration was found.")
    else:
        row = detector.iloc[0]

        required_votes = safe_int(
            row,
            ["required_votes_out_of_4", "required_votes"],
            default=2,
        )

        matched = safe_int(
            row,
            ["reliable_incidents_matched", "reliable_incidents_detected"],
            default=0,
        )

        tested = safe_int(
            row,
            ["reliable_incidents_tested"],
            default=0,
        )

        false_negatives = safe_int(
            row,
            ["false_negatives"],
            default=0,
        )

        d1, d2, d3 = st.columns(3)
        d1.metric("Voting rule", f"{required_votes}/4")
        d2.metric(
            "Reliable incidents detected",
            f"{matched}/{tested}" if tested else str(matched),
        )
        d3.metric("False negatives", false_negatives)

        acc = safe_float(
            row,
            ["acceleration_threshold_g", "accel_threshold_g"],
        )
        gyro = safe_float(
            row,
            ["gyroscope_threshold_dps", "gyro_threshold_dps"],
        )
        jerk = safe_float(
            row,
            ["jerk_threshold_g_per_second", "jerk_threshold_gps"],
        )
        rotation = safe_float(
            row,
            ["rotation_threshold_deg", "rotation_integral_threshold_deg"],
        )
        startup = safe_float(
            row,
            ["startup_grace_seconds"],
            default=0,
        )
        cooldown = safe_float(
            row,
            ["cooldown_seconds"],
            default=0,
        )

        threshold_rows = []

        if acc is not None:
            threshold_rows.append(
                {"Signal": "Acceleration", "Threshold": f"≥ {acc:.5f} g"}
            )
        if gyro is not None:
            threshold_rows.append(
                {"Signal": "Gyroscope", "Threshold": f"≥ {gyro:.5f} dps"}
            )
        if jerk is not None:
            threshold_rows.append(
                {"Signal": "Jerk", "Threshold": f"≥ {jerk:.5f} g/s"}
            )
        if rotation is not None:
            threshold_rows.append(
                {"Signal": "Rotation activity", "Threshold": f"≥ {rotation:.5f} deg"}
            )

        if threshold_rows:
            st.dataframe(
                pd.DataFrame(threshold_rows),
                width="stretch",
                hide_index=True,
            )

        c1, c2 = st.columns(2)
        c1.metric("Startup grace", f"{startup:.1f} s")
        c2.metric("Cooldown", f"{cooldown:.1f} s")

        st.warning(
            "The 0 false-negative result applies only to the 7 reliable labelled "
            "incidents in the current internal validation dataset. It does not "
            "guarantee zero false negatives on future workers."
        )

        st.caption(
            "The current validated output is SAFETY_EVENT. "
            "Fall-vs-near-miss classification was not sufficiently generalised, "
            "so the dashboard does not claim a reliable fall classification."
        )


st.markdown(
    """
    <div class="footer-note">
        Group 5 · NESSO edge safety monitoring · SQLite local-first prototype
    </div>
    """,
    unsafe_allow_html=True,
)