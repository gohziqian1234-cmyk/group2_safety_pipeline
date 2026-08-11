from pathlib import Path
import sqlite3
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# GROUP 5 SAFETY PIPELINE DATABASE INITIALISATION
# Local-first SQLite storage for workers, device state, events,
# queued offline events, and the validated Stage-1 detector config.
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
PROCESSED = DATA_DIR / "processed"
DATABASE_DIR = PROJECT / "database"
DB_PATH = DATABASE_DIR / "safety_pipeline.db"
CONFIG_CSV = PROCESSED / "final_stage1_detector_config.csv"

WORKERS = [
    {"worker_id": "HJ", "worker_name": "Hong Jean", "device_id": "HJ_N1"},
    {"worker_id": "KT", "worker_name": "Kwanteng", "device_id": "KT_N1"},
    {"worker_id": "PI", "worker_name": "Pierre", "device_id": "PI_N1"},
    {"worker_id": "ZQ", "worker_name": "Ziqian", "device_id": "ZQ_N1"},
]


def utc_now_text():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_database():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def create_schema(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS workers (
            worker_id TEXT PRIMARY KEY,
            worker_name TEXT NOT NULL,
            device_id TEXT NOT NULL UNIQUE,
            ble_name TEXT,
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS device_status (
            device_id TEXT PRIMARY KEY,
            worker_id TEXT NOT NULL,
            connection_status TEXT NOT NULL DEFAULT 'DISCONNECTED'
                CHECK (connection_status IN ('CONNECTED', 'DISCONNECTED')),
            safety_status TEXT NOT NULL DEFAULT 'SAFE'
                CHECK (safety_status IN ('SAFE', 'SAFETY_EVENT', 'DISCONNECTED')),
            battery_percent REAL,
            queued_events INTEGER NOT NULL DEFAULT 0 CHECK (queued_events >= 0),
            last_seen_at TEXT,
            last_event_at TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_uuid TEXT NOT NULL UNIQUE,
            worker_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            event_time TEXT NOT NULL,
            event_type TEXT NOT NULL DEFAULT 'SAFETY_EVENT'
                CHECK (event_type IN ('SAFETY_EVENT', 'TEST_EVENT')),
            detection_source TEXT NOT NULL
                CHECK (detection_source IN ('NESSO_EDGE', 'BLE_GATEWAY', 'OFFLINE_REPLAY', 'SIMULATION')),
            event_votes INTEGER CHECK (event_votes BETWEEN 0 AND 4),
            acceleration_peak_g REAL,
            gyroscope_peak_dps REAL,
            jerk_peak_g_per_second REAL,
            rotation_integral_deg REAL,
            acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
            acknowledged_at TEXT,
            notes TEXT,
            received_at TEXT NOT NULL,
            FOREIGN KEY (worker_id) REFERENCES workers(worker_id),
            FOREIGN KEY (device_id) REFERENCES workers(device_id)
        );

        CREATE TABLE IF NOT EXISTS offline_queue (
            queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_uuid TEXT NOT NULL UNIQUE,
            worker_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            queued_at TEXT NOT NULL,
            sync_status TEXT NOT NULL DEFAULT 'PENDING'
                CHECK (sync_status IN ('PENDING', 'SYNCED', 'FAILED')),
            synced_at TEXT,
            FOREIGN KEY (worker_id) REFERENCES workers(worker_id),
            FOREIGN KEY (device_id) REFERENCES workers(device_id)
        );

        CREATE TABLE IF NOT EXISTS detector_config (
            config_version INTEGER PRIMARY KEY,
            is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            required_votes_out_of_4 INTEGER NOT NULL CHECK (required_votes_out_of_4 BETWEEN 1 AND 4),
            acceleration_threshold_g REAL NOT NULL,
            gyroscope_threshold_dps REAL NOT NULL,
            jerk_threshold_g_per_second REAL NOT NULL,
            rotation_threshold_deg REAL NOT NULL,
            startup_grace_seconds REAL NOT NULL,
            cooldown_seconds REAL NOT NULL,
            reliable_incidents_detected INTEGER,
            reliable_incidents_tested INTEGER,
            false_negatives INTEGER,
            normal_recordings_with_alert INTEGER,
            total_normal_alerts INTEGER,
            source_file TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_events_worker_time
        ON events(worker_id, event_time DESC);

        CREATE INDEX IF NOT EXISTS idx_events_acknowledged
        ON events(acknowledged, event_time DESC);

        CREATE INDEX IF NOT EXISTS idx_queue_status
        ON offline_queue(sync_status, queued_at);
        """
    )


def seed_workers(connection):
    timestamp = utc_now_text()

    for worker in WORKERS:
        connection.execute(
            """
            INSERT INTO workers (
                worker_id, worker_name, device_id, ble_name, active, created_at
            )
            VALUES (?, ?, ?, NULL, 1, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                worker_name = excluded.worker_name,
                device_id = excluded.device_id,
                active = 1;
            """,
            (
                worker["worker_id"],
                worker["worker_name"],
                worker["device_id"],
                timestamp,
            ),
        )

        connection.execute(
            """
            INSERT INTO device_status (
                device_id,
                worker_id,
                connection_status,
                safety_status,
                battery_percent,
                queued_events,
                last_seen_at,
                last_event_at,
                updated_at
            )
            VALUES (?, ?, 'DISCONNECTED', 'DISCONNECTED', NULL, 0, NULL, NULL, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                worker_id = excluded.worker_id,
                updated_at = excluded.updated_at;
            """,
            (
                worker["device_id"],
                worker["worker_id"],
                timestamp,
            ),
        )


def load_detector_config(connection):
    if not CONFIG_CSV.exists():
        raise FileNotFoundError(
            f"Missing final detector config:\n{CONFIG_CSV}\n"
            "Run src\\16_finalize_stage1_detector.py first."
        )

    config = pd.read_csv(CONFIG_CSV)

    if len(config) != 1:
        raise ValueError(
            f"Expected exactly one row in {CONFIG_CSV.name}, found {len(config)}."
        )

    row = config.iloc[0]

    required = [
        "required_votes_out_of_4",
        "threshold_event_peak_acceleration_g",
        "threshold_event_peak_gyroscope_dps",
        "threshold_event_peak_jerk_g_per_second",
        "threshold_event_rotation_integral_deg",
        "startup_grace_seconds",
        "cooldown_seconds",
        "reliable_incidents_matched",
        "reliable_incidents_tested",
        "false_negatives",
        "normal_recordings_with_alert",
        "total_normal_alerts",
    ]

    missing = [column for column in required if column not in config.columns]
    if missing:
        raise ValueError(
            "Final detector config is missing columns:\n"
            + "\n".join(f" - {column}" for column in missing)
        )

    values = {
        "required_votes": int(row["required_votes_out_of_4"]),
        "accel": float(row["threshold_event_peak_acceleration_g"]),
        "gyro": float(row["threshold_event_peak_gyroscope_dps"]),
        "jerk": float(row["threshold_event_peak_jerk_g_per_second"]),
        "rotation": float(row["threshold_event_rotation_integral_deg"]),
        "startup": float(row["startup_grace_seconds"]),
        "cooldown": float(row["cooldown_seconds"]),
        "detected": int(row["reliable_incidents_matched"]),
        "tested": int(row["reliable_incidents_tested"]),
        "fn": int(row["false_negatives"]),
        "normal_recordings": int(row["normal_recordings_with_alert"]),
        "normal_alerts": int(row["total_normal_alerts"]),
    }

    if values["fn"] != 0:
        raise ValueError(
            "Refusing to activate detector config because false_negatives is not zero."
        )

    connection.execute("UPDATE detector_config SET is_active = 0;")

    connection.execute(
        """
        INSERT INTO detector_config (
            config_version,
            is_active,
            required_votes_out_of_4,
            acceleration_threshold_g,
            gyroscope_threshold_dps,
            jerk_threshold_g_per_second,
            rotation_threshold_deg,
            startup_grace_seconds,
            cooldown_seconds,
            reliable_incidents_detected,
            reliable_incidents_tested,
            false_negatives,
            normal_recordings_with_alert,
            total_normal_alerts,
            source_file,
            created_at
        )
        VALUES (1, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(config_version) DO UPDATE SET
            is_active = 1,
            required_votes_out_of_4 = excluded.required_votes_out_of_4,
            acceleration_threshold_g = excluded.acceleration_threshold_g,
            gyroscope_threshold_dps = excluded.gyroscope_threshold_dps,
            jerk_threshold_g_per_second = excluded.jerk_threshold_g_per_second,
            rotation_threshold_deg = excluded.rotation_threshold_deg,
            startup_grace_seconds = excluded.startup_grace_seconds,
            cooldown_seconds = excluded.cooldown_seconds,
            reliable_incidents_detected = excluded.reliable_incidents_detected,
            reliable_incidents_tested = excluded.reliable_incidents_tested,
            false_negatives = excluded.false_negatives,
            normal_recordings_with_alert = excluded.normal_recordings_with_alert,
            total_normal_alerts = excluded.total_normal_alerts,
            source_file = excluded.source_file,
            created_at = excluded.created_at;
        """,
        (
            values["required_votes"],
            values["accel"],
            values["gyro"],
            values["jerk"],
            values["rotation"],
            values["startup"],
            values["cooldown"],
            values["detected"],
            values["tested"],
            values["fn"],
            values["normal_recordings"],
            values["normal_alerts"],
            str(CONFIG_CSV.relative_to(PROJECT)),
            utc_now_text(),
        ),
    )

    return values


def verify_database(connection):
    integrity = connection.execute("PRAGMA integrity_check;").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")

    fk_errors = connection.execute("PRAGMA foreign_key_check;").fetchall()
    if fk_errors:
        raise RuntimeError(f"Foreign-key check failed: {fk_errors}")

    tables = [
        "workers",
        "device_status",
        "events",
        "offline_queue",
        "detector_config",
    ]

    counts = {}
    for table in tables:
        counts[table] = connection.execute(
            f"SELECT COUNT(*) FROM {table};"
        ).fetchone()[0]

    return counts


def main():
    print("=" * 82)
    print("GROUP 5 - INITIALISE LOCAL SAFETY DATABASE")
    print("SQLite | local-first | supports offline queueing")
    print("=" * 82)

    connection = connect_database()

    try:
        create_schema(connection)
        seed_workers(connection)
        config = load_detector_config(connection)
        connection.commit()
        counts = verify_database(connection)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(f"Database: {DB_PATH}")
    print()
    print("Tables ready:")
    for table, count in counts.items():
        print(f"  {table:18s} rows={count}")

    print()
    print("Workers/devices registered:")
    for worker in WORKERS:
        print(
            f"  {worker['worker_id']:2s} | "
            f"{worker['worker_name']:10s} | {worker['device_id']}"
        )

    print()
    print("Validated Stage-1 detector stored as active config v1:")
    print(f"  Required votes: {config['required_votes']}/4")
    print(f"  Acceleration:   >= {config['accel']:.5f} g")
    print(f"  Gyroscope:      >= {config['gyro']:.5f} dps")
    print(f"  Jerk:           >= {config['jerk']:.5f} g/s")
    print(f"  Rotation:       >= {config['rotation']:.5f} deg")
    print(f"  Startup grace:  {config['startup']:.1f} s")
    print(f"  Cooldown:       {config['cooldown']:.1f} s")
    print(f"  Validation:     {config['detected']}/{config['tested']} reliable incidents detected")
    print(f"  False negatives:{config['fn']}")

    print()
    print("DATABASE CHECK: PASS")
    print("No sensor is required for this step.")


if __name__ == "__main__":
    main()