from pathlib import Path
import json
import sqlite3
from datetime import datetime, timezone


# ============================================================
# GROUP 5 - GATEWAY CORE
# Reusable message-processing layer between NESSO/BLE and SQLite.
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT / "database" / "safety_pipeline.db"

ALLOWED_MESSAGE_TYPES = {"STATUS", "EVENT"}
ALLOWED_SAFETY_STATUS = {"SAFE", "SAFETY_EVENT"}
ALLOWED_DETECTION_SOURCES = {"NESSO_EDGE", "BLE_GATEWAY", "OFFLINE_REPLAY", "SIMULATION"}


def utc_now_text():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_database():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DB_PATH}\n"
            "Run src\\17_init_database.py first."
        )

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.row_factory = sqlite3.Row
    return connection


# ============================================================
# LIVE SIGNAL BUFFER
#
# The detector only stores EVENTS. To draw a live chart the dashboard also
# needs the raw signal, so the gateway writes a short rolling window of
# samples here - a few minutes at most, trimmed continuously. This is a
# display buffer, not a recording: the permanent record is still the events
# table.
# ============================================================

# How much history to keep. The dashboard shows less than this; the extra is
# margin so a slow refresh never finds an empty chart.
LIVE_SAMPLE_KEEP_SECONDS = 180


def ensure_live_tables(connection):
    """
    Create the live-sample table if it does not exist.

    Done here rather than only in 17_init_database.py so an existing database
    gains the table automatically, without anyone having to re-initialise and
    risk their stored events.
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS live_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            sample_epoch REAL NOT NULL,
            elapsed_seconds REAL NOT NULL,
            acceleration_magnitude_g REAL NOT NULL,
            gyroscope_magnitude_dps REAL NOT NULL,
            jerk_g_per_second REAL
        );
        """
    )

    # Migrate a table created before jerk was stored. CREATE TABLE IF NOT
    # EXISTS silently leaves an older table alone, so without this an
    # existing database would keep a table missing the column and every
    # insert would fail.
    # Index by position, not by name: this must work whether or not the
    # caller set row_factory, and PRAGMA table_info puts the column name in
    # field 1.
    existing = {
        row[1]
        for row in connection.execute("PRAGMA table_info(live_samples);")
    }
    if "jerk_g_per_second" not in existing:
        connection.execute(
            "ALTER TABLE live_samples ADD COLUMN jerk_g_per_second REAL;"
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_live_samples_time
        ON live_samples(sample_epoch);
        """
    )
    connection.commit()


def insert_live_samples(connection, rows):
    """
    Store a batch of samples.

    Batched on purpose: two boards at 25 Hz is 50 rows a second, and one
    INSERT plus one commit per sample would spend most of the gateway's time
    in SQLite rather than reading Bluetooth.
    """
    if not rows:
        return 0

    connection.executemany(
        """
        INSERT INTO live_samples (
            worker_id, device_id, sample_epoch, elapsed_seconds,
            acceleration_magnitude_g, gyroscope_magnitude_dps,
            jerk_g_per_second
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )
    return len(rows)


def trim_live_samples(connection, keep_seconds=LIVE_SAMPLE_KEEP_SECONDS):
    """Delete samples older than the retention window."""
    cutoff = datetime.now(timezone.utc).timestamp() - keep_seconds
    cursor = connection.execute(
        "DELETE FROM live_samples WHERE sample_epoch < ?;", (cutoff,)
    )
    return cursor.rowcount


def parse_message(raw_message):
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")

    if isinstance(raw_message, str):
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON message: {exc}") from exc
    elif isinstance(raw_message, dict):
        message = dict(raw_message)
    else:
        raise TypeError("Message must be bytes, str, or dict.")

    message_type = str(message.get("message_type", "")).strip().upper()

    if message_type not in ALLOWED_MESSAGE_TYPES:
        raise ValueError(
            f"message_type must be one of {sorted(ALLOWED_MESSAGE_TYPES)}."
        )

    message["message_type"] = message_type
    return message


def require_fields(message, required_fields):
    missing = [
        field for field in required_fields
        if field not in message or message[field] is None
    ]
    if missing:
        raise ValueError(
            "Message is missing required fields:\n"
            + "\n".join(f" - {field}" for field in missing)
        )


def verify_worker_device(connection, worker_id, device_id):
    row = connection.execute(
        """
        SELECT worker_id, worker_name, device_id
        FROM workers
        WHERE worker_id = ? AND device_id = ? AND active = 1;
        """,
        (worker_id, device_id),
    ).fetchone()

    if row is None:
        raise ValueError(
            f"Unknown or mismatched worker/device: {worker_id} / {device_id}"
        )

    return row


def validate_battery(value):
    if value is None:
        return None

    battery = float(value)
    if not 0.0 <= battery <= 100.0:
        raise ValueError("battery_percent must be between 0 and 100.")
    return battery


def validate_event_time(value):
    text = str(value).strip()
    if not text:
        raise ValueError("event_time cannot be empty.")

    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "event_time must be an ISO-8601 timestamp."
        ) from exc

    return text


def process_status(connection, message):
    require_fields(
        message,
        ["worker_id", "device_id", "safety_status", "queued_events"],
    )

    worker_id = str(message["worker_id"]).strip()
    device_id = str(message["device_id"]).strip()
    safety_status = str(message["safety_status"]).strip().upper()

    if safety_status not in ALLOWED_SAFETY_STATUS:
        raise ValueError(
            f"safety_status must be one of {sorted(ALLOWED_SAFETY_STATUS)}."
        )

    queued_events = int(message["queued_events"])
    if queued_events < 0:
        raise ValueError("queued_events cannot be negative.")

    battery = validate_battery(message.get("battery_percent"))
    verify_worker_device(connection, worker_id, device_id)

    now = utc_now_text()

    connection.execute(
        """
        UPDATE device_status
        SET
            connection_status = 'CONNECTED',
            safety_status = ?,
            battery_percent = ?,
            queued_events = ?,
            last_seen_at = ?,
            updated_at = ?
        WHERE worker_id = ? AND device_id = ?;
        """,
        (
            safety_status,
            battery,
            queued_events,
            now,
            now,
            worker_id,
            device_id,
        ),
    )

    return {
        "message_type": "STATUS",
        "worker_id": worker_id,
        "device_id": device_id,
        "safety_status": safety_status,
        "queued_events": queued_events,
        "stored": True,
    }


def process_event(connection, message):
    require_fields(
        message,
        [
            "event_uuid",
            "worker_id",
            "device_id",
            "event_time",
            "event_votes",
            "acceleration_peak_g",
            "gyroscope_peak_dps",
            "jerk_peak_g_per_second",
            "rotation_integral_deg",
        ],
    )

    worker_id = str(message["worker_id"]).strip()
    device_id = str(message["device_id"]).strip()
    event_uuid = str(message["event_uuid"]).strip()
    event_time = validate_event_time(message["event_time"])
    detection_source = str(message.get("detection_source", "NESSO_EDGE")).strip().upper()

    if not event_uuid:
        raise ValueError("event_uuid cannot be empty.")

    if detection_source not in ALLOWED_DETECTION_SOURCES:
        raise ValueError(
            f"detection_source must be one of {sorted(ALLOWED_DETECTION_SOURCES)}."
        )

    votes = int(message["event_votes"])
    if not 0 <= votes <= 4:
        raise ValueError("event_votes must be between 0 and 4.")

    numeric_fields = {
        "acceleration_peak_g": float(message["acceleration_peak_g"]),
        "gyroscope_peak_dps": float(message["gyroscope_peak_dps"]),
        "jerk_peak_g_per_second": float(message["jerk_peak_g_per_second"]),
        "rotation_integral_deg": float(message["rotation_integral_deg"]),
    }

    for field, value in numeric_fields.items():
        if value < 0:
            raise ValueError(f"{field} cannot be negative.")

    verify_worker_device(connection, worker_id, device_id)

    received_at = utc_now_text()

    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO events (
            event_uuid,
            worker_id,
            device_id,
            event_time,
            event_type,
            detection_source,
            event_votes,
            acceleration_peak_g,
            gyroscope_peak_dps,
            jerk_peak_g_per_second,
            rotation_integral_deg,
            acknowledged,
            acknowledged_at,
            notes,
            received_at
        )
        VALUES (?, ?, ?, ?, 'SAFETY_EVENT', ?,
                ?, ?, ?, ?, ?, 0, NULL, ?, ?);
        """,
        (
            event_uuid,
            worker_id,
            device_id,
            event_time,
            detection_source,
            votes,
            numeric_fields["acceleration_peak_g"],
            numeric_fields["gyroscope_peak_dps"],
            numeric_fields["jerk_peak_g_per_second"],
            numeric_fields["rotation_integral_deg"],
            message.get("notes"),
            received_at,
        ),
    )

    inserted = cursor.rowcount == 1

    connection.execute(
        """
        UPDATE device_status
        SET
            connection_status = 'CONNECTED',
            safety_status = 'SAFETY_EVENT',
            last_seen_at = ?,
            last_event_at = ?,
            updated_at = ?
        WHERE worker_id = ? AND device_id = ?;
        """,
        (
            received_at,
            event_time,
            received_at,
            worker_id,
            device_id,
        ),
    )

    return {
        "message_type": "EVENT",
        "worker_id": worker_id,
        "device_id": device_id,
        "event_uuid": event_uuid,
        "detection_source": detection_source,
        "inserted": inserted,
        "duplicate": not inserted,
    }


def process_message(connection, raw_message):
    message = parse_message(raw_message)

    if message["message_type"] == "STATUS":
        return process_status(connection, message)

    if message["message_type"] == "EVENT":
        return process_event(connection, message)

    raise RuntimeError("Unsupported message type reached.")
