from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone


# ============================================================
# GROUP 5 - TEST EVENT INGESTION INTO SQLITE
# Simulates the exact type of event the BLE gateway will later
# receive from a NESSO device.
#
# NO SENSOR REQUIRED.
# This script tests:
#   1) payload validation
#   2) worker/device matching
#   3) duplicate-event protection
#   4) event insertion
#   5) device-status update
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT / "database" / "safety_pipeline.db"

REQUIRED_EVENT_FIELDS = {
    "event_uuid",
    "worker_id",
    "device_id",
    "event_time",
    "event_type",
    "detection_source",
    "event_votes",
    "acceleration_peak_g",
    "gyroscope_peak_dps",
    "jerk_peak_g_per_second",
    "rotation_integral_deg",
}

ALLOWED_EVENT_TYPES = {"SAFETY_EVENT", "TEST_EVENT"}
ALLOWED_SOURCES = {"NESSO_EDGE", "BLE_GATEWAY", "OFFLINE_REPLAY", "SIMULATION"}


def utc_now_text():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_payload(payload):
    missing = sorted(REQUIRED_EVENT_FIELDS - set(payload))
    if missing:
        raise ValueError(
            "Event payload is missing required fields:\n"
            + "\n".join(f" - {field}" for field in missing)
        )

    if payload["event_type"] not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"Invalid event_type: {payload['event_type']}")

    if payload["detection_source"] not in ALLOWED_SOURCES:
        raise ValueError(
            f"Invalid detection_source: {payload['detection_source']}"
        )

    votes = int(payload["event_votes"])
    if not 0 <= votes <= 4:
        raise ValueError("event_votes must be between 0 and 4.")

    for field in [
        "acceleration_peak_g",
        "gyroscope_peak_dps",
        "jerk_peak_g_per_second",
        "rotation_integral_deg",
    ]:
        value = float(payload[field])
        if value < 0:
            raise ValueError(f"{field} cannot be negative.")


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
            f"Worker/device mismatch or unknown device: "
            f"{worker_id} / {device_id}"
        )

    return row


def ingest_event(connection, payload):
    validate_payload(payload)
    worker = verify_worker_device(
        connection,
        payload["worker_id"],
        payload["device_id"],
    )

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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?);
        """,
        (
            payload["event_uuid"],
            payload["worker_id"],
            payload["device_id"],
            payload["event_time"],
            payload["event_type"],
            payload["detection_source"],
            int(payload["event_votes"]),
            float(payload["acceleration_peak_g"]),
            float(payload["gyroscope_peak_dps"]),
            float(payload["jerk_peak_g_per_second"]),
            float(payload["rotation_integral_deg"]),
            payload.get("notes"),
            received_at,
        ),
    )

    inserted = cursor.rowcount == 1

    # Device status is updated even if the event itself was already received.
    # This lets repeated BLE delivery refresh the "last seen" timestamp without
    # creating duplicate incident rows.
    connection.execute(
        """
        UPDATE device_status
        SET
            connection_status = 'CONNECTED',
            safety_status = 'SAFETY_EVENT',
            last_seen_at = ?,
            last_event_at = ?,
            updated_at = ?
        WHERE device_id = ? AND worker_id = ?;
        """,
        (
            received_at,
            payload["event_time"],
            received_at,
            payload["device_id"],
            payload["worker_id"],
        ),
    )

    return {
        "inserted": inserted,
        "worker_name": worker["worker_name"],
    }


def build_demo_payload():
    # Deterministic UUID:
    # rerunning this test sends the SAME event again.
    # The database should keep only one copy.
    event_uuid = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "group5-safety-pipeline-demo-event-zq-v1",
        )
    )

    return {
        "event_uuid": event_uuid,
        "worker_id": "ZQ",
        "device_id": "ZQ_N1",
        "event_time": utc_now_text(),
        "event_type": "TEST_EVENT",
        "detection_source": "SIMULATION",
        "event_votes": 3,
        "acceleration_peak_g": 4.10,
        "gyroscope_peak_dps": 800.0,
        "jerk_peak_g_per_second": 80.0,
        "rotation_integral_deg": 600.0,
        "notes": (
            "Code 18 simulated gateway-ingestion test. "
            "This is NOT a real worker incident."
        ),
    }


def main():
    print("=" * 86)
    print("GROUP 5 - TEST EVENT INGESTION")
    print("Simulated NESSO event -> SQLite | NO SENSOR REQUIRED")
    print("=" * 86)

    payload = build_demo_payload()

    connection = connect_database()

    try:
        print()
        print("SIMULATED EVENT PAYLOAD:")
        print(f"  event_uuid:       {payload['event_uuid']}")
        print(f"  worker/device:    {payload['worker_id']} / {payload['device_id']}")
        print(f"  event_type:       {payload['event_type']}")
        print(f"  source:           {payload['detection_source']}")
        print(f"  event_votes:      {payload['event_votes']}/4")

        print()
        print("PASS 1 - ingesting event...")
        result_1 = ingest_event(connection, payload)
        connection.commit()
        print(
            "  RESULT:",
            "NEW EVENT INSERTED"
            if result_1["inserted"]
            else "EVENT ALREADY EXISTS - NOT DUPLICATED",
        )

        print()
        print("PASS 2 - sending the SAME event again...")
        result_2 = ingest_event(connection, payload)
        connection.commit()
        print(
            "  RESULT:",
            "NEW EVENT INSERTED"
            if result_2["inserted"]
            else "DUPLICATE SAFELY IGNORED",
        )

        stored = connection.execute(
            """
            SELECT
                event_id,
                event_uuid,
                worker_id,
                device_id,
                event_type,
                event_votes,
                acknowledged,
                received_at
            FROM events
            WHERE event_uuid = ?;
            """,
            (payload["event_uuid"],),
        ).fetchall()

        status = connection.execute(
            """
            SELECT
                device_id,
                worker_id,
                connection_status,
                safety_status,
                last_seen_at,
                last_event_at
            FROM device_status
            WHERE device_id = ?;
            """,
            (payload["device_id"],),
        ).fetchone()

        total_events = connection.execute(
            "SELECT COUNT(*) FROM events;"
        ).fetchone()[0]

        if len(stored) != 1:
            raise RuntimeError(
                f"Duplicate-protection check failed. "
                f"Expected 1 stored copy, found {len(stored)}."
            )

        if status is None:
            raise RuntimeError("Device-status row was not found.")

        if status["connection_status"] != "CONNECTED":
            raise RuntimeError("Device status did not change to CONNECTED.")

        if status["safety_status"] != "SAFETY_EVENT":
            raise RuntimeError("Safety status did not change to SAFETY_EVENT.")

        print()
        print("DATABASE RESULT:")
        print(f"  Stored copies of this event: {len(stored)}")
        print(f"  Total events in database:    {total_events}")
        print(f"  Device connection:           {status['connection_status']}")
        print(f"  Device safety status:        {status['safety_status']}")
        print(f"  Last event:                  {status['last_event_at']}")

        print()
        print("EVENT INGESTION CHECK: PASS")
        print()
        print("WHAT THIS PROVES:")
        print("- The gateway can validate a received event payload.")
        print("- Worker and device IDs are checked before storing.")
        print("- Re-sending the same event does NOT create duplicate rows.")
        print("- A valid event is stored in SQLite.")
        print("- The worker/device status is updated for the dashboard.")
        print("- This test used SIMULATION, not a real sensor event.")

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()