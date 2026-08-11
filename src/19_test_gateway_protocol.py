import json
import uuid
from datetime import datetime, timezone

from gateway_core import connect_database, process_message


# ============================================================
# GROUP 5 - TEST GATEWAY MESSAGE PROTOCOL
# Simulates the messages that BLE will later deliver.
# NO SENSOR REQUIRED.
# ============================================================

TEST_EVENT_UUID = str(
    uuid.uuid5(
        uuid.NAMESPACE_URL,
        "group5-gateway-protocol-test-zq-v1",
    )
)


def utc_now_text():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    print("=" * 88)
    print("GROUP 5 - GATEWAY PROTOCOL TEST")
    print("STATUS + EVENT + duplicate EVENT + STATUS | NO SENSOR REQUIRED")
    print("=" * 88)

    status_safe = {
        "message_type": "STATUS",
        "worker_id": "ZQ",
        "device_id": "ZQ_N1",
        "battery_percent": 82,
        "safety_status": "SAFE",
        "queued_events": 0,
    }

    event = {
        "message_type": "EVENT",
        "event_uuid": TEST_EVENT_UUID,
        "worker_id": "ZQ",
        "device_id": "ZQ_N1",
        "event_time": utc_now_text(),
        "event_votes": 3,
        "acceleration_peak_g": 4.10,
        "gyroscope_peak_dps": 800.0,
        "jerk_peak_g_per_second": 80.0,
        "rotation_integral_deg": 600.0,
        "notes": "Code 19 simulated BLE-protocol event. Not a real incident.",
    }

    connection = connect_database()

    try:
        # Make reruns deterministic for this specific Code 19 test event.
        connection.execute(
            "DELETE FROM events WHERE event_uuid = ?;",
            (TEST_EVENT_UUID,),
        )
        connection.commit()

        print()
        print("1) STATUS MESSAGE: worker is connected and SAFE")
        result = process_message(connection, json.dumps(status_safe))
        connection.commit()
        print(
            f"   {result['worker_id']} / {result['device_id']} -> "
            f"{result['safety_status']}"
        )

        print()
        print("2) EVENT MESSAGE: NESSO reports a SAFETY_EVENT")
        first = process_message(connection, json.dumps(event))
        connection.commit()
        print(
            "   RESULT:",
            "NEW EVENT STORED"
            if first["inserted"]
            else "DUPLICATE IGNORED",
        )

        print()
        print("3) SAME EVENT AGAIN: simulates BLE retry")
        second = process_message(connection, json.dumps(event))
        connection.commit()
        print(
            "   RESULT:",
            "DUPLICATE SAFELY IGNORED"
            if second["duplicate"]
            else "UNEXPECTED SECOND INSERT",
        )

        stored_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_uuid = ?;",
            (TEST_EVENT_UUID,),
        ).fetchone()[0]

        event_row = connection.execute(
            """
            SELECT event_type, detection_source, event_votes
            FROM events
            WHERE event_uuid = ?;
            """,
            (TEST_EVENT_UUID,),
        ).fetchone()

        if stored_count != 1:
            raise RuntimeError(
                f"Expected one stored event, found {stored_count}."
            )

        if event_row["event_type"] != "SAFETY_EVENT":
            raise RuntimeError("Stored event_type is not SAFETY_EVENT.")

        if event_row["detection_source"] != "NESSO_EDGE":
            raise RuntimeError(
                "Stored detection_source is not NESSO_EDGE."
            )

        print()
        print("4) STATUS MESSAGE: after cooldown, worker returns to SAFE")
        final_status = dict(status_safe)
        final_status["battery_percent"] = 81
        process_message(connection, json.dumps(final_status))
        connection.commit()

        status_row = connection.execute(
            """
            SELECT
                connection_status,
                safety_status,
                battery_percent,
                queued_events
            FROM device_status
            WHERE device_id = 'ZQ_N1';
            """
        ).fetchone()

        if status_row["connection_status"] != "CONNECTED":
            raise RuntimeError("Device should be CONNECTED.")

        if status_row["safety_status"] != "SAFE":
            raise RuntimeError("Device should have returned to SAFE.")

        print()
        print("DATABASE CHECK:")
        print(f"   Stored copies of Code 19 event: {stored_count}")
        print(f"   Event type:                     {event_row['event_type']}")
        print(f"   Detection source:               {event_row['detection_source']}")
        print(f"   Event votes:                    {event_row['event_votes']}/4")
        print(f"   Device connection:              {status_row['connection_status']}")
        print(f"   Device safety status:           {status_row['safety_status']}")
        print(f"   Battery:                        {status_row['battery_percent']:.0f}%")
        print(f"   Queued events:                  {status_row['queued_events']}")

        print()
        print("GATEWAY PROTOCOL CHECK: PASS")
        print()
        print("WHAT THIS PROVES:")
        print("- STATUS messages can update live device state.")
        print("- EVENT messages are stored as NESSO_EDGE safety events.")
        print("- BLE retries cannot duplicate the same incident.")
        print("- A later SAFE status can clear the live alert state.")
        print("- The same gateway_core.py can be reused by the real Bleak listener later.")

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()