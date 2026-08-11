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
