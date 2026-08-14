from __future__ import annotations

import asyncio
import math
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone

from bleak import BleakClient, BleakScanner

from gateway_core import (
    connect_database,
    ensure_live_tables,
    insert_live_samples,
    process_message,
    trim_live_samples,
)


# ============================================================
# EG2A17 GROUP 5 - FINAL LIVE BLE GATEWAY
# NESSO 25 Hz raw IMU -> Python rule detector -> SQLite
# Detection runs HERE in Python. This is NOT NESSO edge detection.
#
# Runs one or more NESSO boards at the same time. Each board gets its own
# BLE session, its own detector instance and its own row in device_status,
# so one worker's incident or dropout never affects another's.
# ============================================================

# ------------------------------------------------------------
# THE BOARDS
#
# ble_name   must match BLE_NAME in that board's Arduino sketch
# worker_id  must match an active row in the SQLite workers table
# device_id  must match that same row
#
# Add a board by adding a line here and flashing its sketch. Nothing else
# in this file, in gateway_core.py or in the dashboard needs changing.
# ------------------------------------------------------------
WORKERS = [
    {"ble_name": "ziqian", "worker_id": "ZQ", "device_id": "ZQ_N1"},
    {"ble_name": "hongjean", "worker_id": "HJ", "device_id": "HJ_N1"},
]

SERVICE_UUID = "bdc766fc-7eee-417f-bbe0-2e71a8a2bf70"
COMBINED_IMU_UUID = "f509416c-3c4b-401e-a768-b25a9e621a91"

SAMPLE_RATE_HZ = 25
DT = 1.0 / SAMPLE_RATE_HZ
SCAN_STEP_SECONDS = 0.5
EVENT_WINDOW_BEFORE_SECONDS = 0.8
EVENT_WINDOW_AFTER_SECONDS = 1.2
STATUS_INTERVAL_SECONDS = 1.0

# How often buffered samples are written to SQLite for the dashboard chart.
# One write a second instead of 25 keeps the gateway responsive to Bluetooth.
LIVE_FLUSH_INTERVAL_SECONDS = 1.0

# How often old chart samples are deleted.
LIVE_TRIM_INTERVAL_SECONDS = 30.0
RECONNECT_DELAY_SECONDS = 2.0
SCAN_TIMEOUT_SECONDS = 12.0

# Approved validated Stage-1 deployment values.
APPROVED_CONFIG = {
    "required_votes_out_of_4": 2,
    "acceleration_threshold_g": 3.28154,
    "gyroscope_threshold_dps": 641.28650,
    "jerk_threshold_g_per_second": 67.40781,
    "rotation_threshold_deg": 524.51258,
    "startup_grace_seconds": 5.0,
    "cooldown_seconds": 8.0,
}


# ============================================================
# SHARED RESOURCES
#
# Two things are shared between the per-worker tasks and both need guarding.
# ============================================================

# Only one BLE scan may run at a time. Windows and BlueZ both dislike
# concurrent discovery, and two workers searching at once produces missed
# devices rather than a clean error.
SCAN_LOCK = asyncio.Lock()

# One SQLite connection is shared by every worker. The tasks all run on the
# same event loop, so writes cannot truly overlap - but this lock makes that
# safety explicit rather than accidental, and keeps it true if an await is
# ever added between an execute and its commit.
DB_LOCK = asyncio.Lock()


class LiveStage1Detector:
    """Online implementation of the validated 25 Hz Stage-1 detector."""

    def __init__(self, connection, config, worker):
        self.connection = connection
        self.config = config
        self.worker = worker
        self.samples = deque()
        self.sample_index = 0
        self.previous_acceleration_magnitude = None
        self.first_sample_utc = None
        self.next_center_seconds = max(
            EVENT_WINDOW_BEFORE_SECONDS,
            float(config["startup_grace_seconds"]),
        )
        self.cooldown_until_center = -math.inf
        self.alert_until_monotonic = 0.0
        self.latest_sample = None

    def alert_active(self):
        return time.monotonic() < self.alert_until_monotonic

    @staticmethod
    def _magnitude(x, y, z):
        return math.sqrt(x * x + y * y + z * z)

    @staticmethod
    def _rotation_integral(window):
        if len(window) < 2:
            return 0.0

        total = 0.0
        for left, right in zip(window, window[1:]):
            dt = right["elapsed_seconds"] - left["elapsed_seconds"]
            if dt > 0:
                total += 0.5 * (
                    left["gyroscope_magnitude_dps"]
                    + right["gyroscope_magnitude_dps"]
                ) * dt
        return total

    def _extract_features(self, center_seconds):
        start = center_seconds - EVENT_WINDOW_BEFORE_SECONDS
        end = center_seconds + EVENT_WINDOW_AFTER_SECONDS
        window = [
            sample
            for sample in self.samples
            if start <= sample["elapsed_seconds"] <= end
        ]

        # Offline validation required at least 20 points in the event window.
        if len(window) < 20:
            return None

        return {
            "acceleration_peak_g": max(
                sample["acceleration_magnitude_g"] for sample in window
            ),
            "gyroscope_peak_dps": max(
                sample["gyroscope_magnitude_dps"] for sample in window
            ),
            "jerk_peak_g_per_second": max(
                sample["jerk_g_per_second"] for sample in window
            ),
            "rotation_integral_deg": self._rotation_integral(window),
        }

    def _vote_count(self, features):
        votes = 0
        votes += int(
            features["acceleration_peak_g"]
            >= self.config["acceleration_threshold_g"]
        )
        votes += int(
            features["gyroscope_peak_dps"]
            >= self.config["gyroscope_threshold_dps"]
        )
        votes += int(
            features["jerk_peak_g_per_second"]
            >= self.config["jerk_threshold_g_per_second"]
        )
        votes += int(
            features["rotation_integral_deg"]
            >= self.config["rotation_threshold_deg"]
        )
        return votes

    def _build_event(self, center_seconds, votes, features):
        if self.first_sample_utc is None:
            event_time = datetime.now(timezone.utc)
        else:
            event_time = self.first_sample_utc + timedelta(seconds=center_seconds)

        return {
            "message_type": "EVENT",
            "event_uuid": str(uuid.uuid4()),
            "worker_id": self.worker["worker_id"],
            "device_id": self.worker["device_id"],
            "event_time": event_time.isoformat(timespec="milliseconds"),
            "detection_source": "BLE_GATEWAY",
            "event_votes": int(votes),
            **features,
            "notes": (
                "Rule-based 25 Hz detector in Python BLE gateway; "
                "validated Stage-1 2-of-4 voting."
            ),
        }

    def ingest(self, values):
        """
        Feed in one BLE sample.

        Returns a list of (event_message, votes, features) for any incident
        confirmed by this sample. Storing is left to the caller, because
        writing to SQLite has to happen under DB_LOCK and this method is
        deliberately synchronous.
        """
        xg, yg, zg, xdeg, ydeg, zdeg = values

        elapsed = self.sample_index * DT
        self.sample_index += 1

        if self.first_sample_utc is None:
            self.first_sample_utc = datetime.now(timezone.utc)

        acceleration_magnitude = self._magnitude(xg, yg, zg)
        gyroscope_magnitude = self._magnitude(xdeg, ydeg, zdeg)

        if self.previous_acceleration_magnitude is None:
            jerk = 0.0
        else:
            # The approved BLE payload has no device timestamp. Because the
            # Arduino source is fixed at 25 Hz, use the validated 0.04 s DT
            # rather than BLE arrival timing, which can be bursty.
            jerk = abs(
                acceleration_magnitude - self.previous_acceleration_magnitude
            ) / DT

        self.previous_acceleration_magnitude = acceleration_magnitude

        self.samples.append(
            {
                "elapsed_seconds": elapsed,
                "acceleration_magnitude_g": acceleration_magnitude,
                "gyroscope_magnitude_dps": gyroscope_magnitude,
                "jerk_g_per_second": jerk,
            }
        )

        # Kept so the session loop can buffer this sample for the dashboard
        # chart without recomputing the magnitudes.
        self.latest_sample = (
            elapsed,
            acceleration_magnitude,
            gyroscope_magnitude,
        )

        # Keep enough history for the 2.0 s feature window plus scan margin.
        keep_after = elapsed - 3.0
        while self.samples and self.samples[0]["elapsed_seconds"] < keep_after:
            self.samples.popleft()

        detections = []

        # The offline detector evaluates a center only after the complete
        # [-0.8 s, +1.2 s] window is available. Reproduce that online.
        while elapsed >= self.next_center_seconds + EVENT_WINDOW_AFTER_SECONDS:
            center = self.next_center_seconds
            self.next_center_seconds += SCAN_STEP_SECONDS

            if center < self.cooldown_until_center:
                continue

            features = self._extract_features(center)
            if features is None:
                continue

            votes = self._vote_count(features)
            if votes < int(self.config["required_votes_out_of_4"]):
                continue

            self.cooldown_until_center = center + float(
                self.config["cooldown_seconds"]
            )
            self.alert_until_monotonic = time.monotonic() + float(
                self.config["cooldown_seconds"]
            )

            detections.append(
                (self._build_event(center, votes, features), votes, features)
            )

        return detections


def parse_ble_payload(raw_data):
    try:
        text = bytes(raw_data).decode("utf-8").strip().strip("\x00")
    except UnicodeDecodeError as exc:
        raise ValueError("BLE payload is not valid UTF-8.") from exc

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 6:
        raise ValueError(
            "Expected BLE payload Xg,Yg,Zg,Xdeg,Ydeg,Zdeg "
            f"but received {len(parts)} fields: {text!r}"
        )

    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"BLE payload contains a non-numeric value: {text!r}") from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"BLE payload contains a non-finite value: {text!r}")

    return values


def load_and_verify_detector_config(connection):
    row = connection.execute(
        """
        SELECT
            required_votes_out_of_4,
            acceleration_threshold_g,
            gyroscope_threshold_dps,
            jerk_threshold_g_per_second,
            rotation_threshold_deg,
            startup_grace_seconds,
            cooldown_seconds
        FROM detector_config
        WHERE is_active = 1
        ORDER BY config_version DESC
        LIMIT 1;
        """
    ).fetchone()

    if row is None:
        raise RuntimeError(
            "No active detector configuration is stored in SQLite. "
            "Run src\\17_init_database.py first."
        )

    config = dict(row)

    for key, approved in APPROVED_CONFIG.items():
        actual = config[key]
        if key == "required_votes_out_of_4":
            matches = int(actual) == int(approved)
        else:
            matches = math.isclose(
                float(actual), float(approved), rel_tol=0.0, abs_tol=1e-6
            )

        if not matches:
            raise RuntimeError(
                "Active SQLite detector configuration does not match the "
                f"approved deployment value for {key}: "
                f"database={actual}, approved={approved}. "
                "No threshold was changed automatically."
            )

    return config


def register_live_ble_name(connection, worker):
    cursor = connection.execute(
        """
        UPDATE workers
        SET ble_name = ?
        WHERE worker_id = ? AND device_id = ? AND active = 1;
        """,
        (worker["ble_name"], worker["worker_id"], worker["device_id"]),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(
            f"Expected active worker/device "
            f"{worker['worker_id']}/{worker['device_id']} in SQLite. "
            "Check the WORKERS list against the workers table, and run "
            "src\\17_init_database.py if the database has not been seeded."
        )
    connection.commit()


async def write_status(connection, worker, safety_status):
    async with DB_LOCK:
        result = process_message(
            connection,
            {
                "message_type": "STATUS",
                "worker_id": worker["worker_id"],
                "device_id": worker["device_id"],
                "safety_status": safety_status,
                "queued_events": 0,
                "battery_percent": None,
            },
        )
        connection.commit()
    return result


async def store_event(connection, event_message):
    async with DB_LOCK:
        result = process_message(connection, event_message)
        connection.commit()
    return result


async def flush_live_samples(connection, rows):
    async with DB_LOCK:
        insert_live_samples(connection, rows)
        connection.commit()


async def trim_live(connection):
    async with DB_LOCK:
        trim_live_samples(connection)
        connection.commit()


async def mark_disconnected(connection, worker):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    async with DB_LOCK:
        connection.execute(
            """
            UPDATE device_status
            SET
                connection_status = 'DISCONNECTED',
                safety_status = 'DISCONNECTED',
                updated_at = ?
            WHERE worker_id = ? AND device_id = ?;
            """,
            (now, worker["worker_id"], worker["device_id"]),
        )
        connection.commit()


async def find_nesso(worker):
    """
    Scan for one board.

    Held under SCAN_LOCK so only one worker scans at a time - concurrent BLE
    discovery is unreliable on both Windows and Linux.
    """
    ble_name = worker["ble_name"]

    def matches(device, advertisement_data):
        names = {
            str(device.name or "").strip().lower(),
            str(advertisement_data.local_name or "").strip().lower(),
        }
        return ble_name.lower() in names

    async with SCAN_LOCK:
        print(f"[{ble_name}] scanning...")
        return await BleakScanner.find_device_by_filter(
            matches,
            timeout=SCAN_TIMEOUT_SECONDS,
        )


async def run_connected_session(device, connection, config, worker):
    ble_name = worker["ble_name"]
    packet_queue = asyncio.Queue(maxsize=250)
    disconnected = asyncio.Event()

    def on_disconnect(_client):
        disconnected.set()

    def on_notification(_sender, data):
        try:
            packet_queue.put_nowait(bytes(data))
        except asyncio.QueueFull:
            # Drop the oldest buffered BLE packet rather than blocking the
            # BLE callback. A visible warning is printed by the consumer.
            try:
                packet_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                packet_queue.put_nowait(bytes(data))
            except asyncio.QueueFull:
                pass

    detector = LiveStage1Detector(connection, config, worker)

    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        await client.start_notify(COMBINED_IMU_UUID, on_notification)
        await write_status(connection, worker, "SAFE")

        print(
            f"[{ble_name}] CONNECTED / SAFE | "
            f"worker={worker['worker_id']} | device={worker['device_id']}"
        )

        last_status = 0.0
        last_drop_warning = 0.0
        last_flush = time.monotonic()
        last_trim = time.monotonic()
        live_buffer = []

        while client.is_connected and not disconnected.is_set():
            try:
                raw_data = await asyncio.wait_for(packet_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if packet_queue.qsize() > 100 and time.monotonic() - last_drop_warning > 5.0:
                print(
                    f"[{ble_name}] WARNING: BLE processing queue is high "
                    f"({packet_queue.qsize()} packets)."
                )
                last_drop_warning = time.monotonic()

            try:
                values = parse_ble_payload(raw_data)
                detections = detector.ingest(values)
            except ValueError as exc:
                print(f"[{ble_name}] IGNORED BAD BLE PAYLOAD: {exc}")
                continue

            # Buffer this sample for the dashboard's live chart.
            if detector.latest_sample is not None:
                elapsed, accel_mag, gyro_mag = detector.latest_sample
                live_buffer.append(
                    (
                        worker["worker_id"],
                        worker["device_id"],
                        time.time(),
                        elapsed,
                        accel_mag,
                        gyro_mag,
                    )
                )

            for event_message, votes, features in detections:
                result = await store_event(connection, event_message)

                print()
                print(f"*** SAFETY_EVENT STORED - {ble_name} "
                      f"({worker['worker_id']}) ***")
                print(f"event_uuid: {result['event_uuid']}")
                print(f"votes:      {votes}/4")
                print(
                    "features:   "
                    f"accel={features['acceleration_peak_g']:.5f} g | "
                    f"gyro={features['gyroscope_peak_dps']:.5f} dps | "
                    f"jerk={features['jerk_peak_g_per_second']:.5f} g/s | "
                    f"rotation={features['rotation_integral_deg']:.5f} deg"
                )
                print("SQLite/dashboard source: BLE_GATEWAY")
                print()

            now = time.monotonic()

            if now - last_flush >= LIVE_FLUSH_INTERVAL_SECONDS and live_buffer:
                await flush_live_samples(connection, live_buffer)
                live_buffer = []
                last_flush = now

            if now - last_trim >= LIVE_TRIM_INTERVAL_SECONDS:
                await trim_live(connection)
                last_trim = now

            if now - last_status >= STATUS_INTERVAL_SECONDS:
                safety_status = "SAFETY_EVENT" if detector.alert_active() else "SAFE"
                await write_status(connection, worker, safety_status)
                last_status = now

        # Do not lose the last part-second of chart data on disconnect.
        if live_buffer:
            await flush_live_samples(connection, live_buffer)

        try:
            await client.stop_notify(COMBINED_IMU_UUID)
        except Exception:
            pass


async def worker_session(connection, config, worker):
    """
    Keep one board connected for as long as the gateway runs.

    Each worker runs one of these concurrently. Anything that goes wrong here
    is contained: a board that will not connect retries on its own without
    stopping the others.
    """
    ble_name = worker["ble_name"]

    while True:
        device = await find_nesso(worker)

        if device is None:
            print(
                f"[{ble_name}] not found. Check Bluetooth is on and the board "
                f"is powered; retrying."
            )
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            continue

        print(f"[{ble_name}] found at {device.address}. Connecting...")

        try:
            await run_connected_session(device, connection, config, worker)
        except Exception as exc:
            print(f"[{ble_name}] session ended: {type(exc).__name__}: {exc}")
        finally:
            await mark_disconnected(connection, worker)

        print(f"[{ble_name}] reconnecting in {RECONNECT_DELAY_SECONDS:.0f} s...")
        await asyncio.sleep(RECONNECT_DELAY_SECONDS)


def select_workers(argv):
    """
    Choose which boards to run.

    With no arguments every board in WORKERS runs. Naming boards on the
    command line runs only those, which is what you want when only one board
    is on the bench - otherwise the gateway spends its time scanning for a
    board that is not switched on.

        py src\\21_live_ble_gateway.py                  both boards
        py src\\21_live_ble_gateway.py ziqian           ziqian only
    """
    if not argv:
        return list(WORKERS)

    wanted = [name.strip().lower() for name in argv]
    by_name = {worker["ble_name"].lower(): worker for worker in WORKERS}

    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        known = ", ".join(sorted(by_name))
        raise SystemExit(
            f"Unknown board name(s): {', '.join(unknown)}.\n"
            f"Configured boards are: {known}"
        )

    return [by_name[name] for name in wanted]


def print_config(config, workers):
    print("Approved active detector config:")
    print(f"  Votes:        {int(config['required_votes_out_of_4'])}/4")
    print(f"  Acceleration: >= {float(config['acceleration_threshold_g']):.5f} g")
    print(f"  Gyroscope:    >= {float(config['gyroscope_threshold_dps']):.5f} dps")
    print(f"  Jerk:         >= {float(config['jerk_threshold_g_per_second']):.5f} g/s")
    print(f"  Rotation:     >= {float(config['rotation_threshold_deg']):.5f} deg")
    print(f"  Startup:      {float(config['startup_grace_seconds']):.1f} s")
    print(f"  Cooldown:     {float(config['cooldown_seconds']):.1f} s")
    print()
    print(f"Boards ({len(workers)}):")
    for worker in workers:
        print(
            f"  {worker['ble_name']:10s} -> "
            f"{worker['worker_id']} / {worker['device_id']}"
        )
    print()
    print(f"Service UUID: {SERVICE_UUID}")
    print(f"IMU UUID:     {COMBINED_IMU_UUID}")
    print("Payload:      Xg,Yg,Zg,Xdeg,Ydeg,Zdeg")
    print("Sampling:     25 Hz")
    print("Detection:    Python BLE gateway (BLE_GATEWAY), not NESSO edge")
    print("Press Ctrl+C to stop.")
    print("=" * 76)


async def main():
    workers = select_workers(sys.argv[1:])
    connection = connect_database()

    try:
        ensure_live_tables(connection)
        config = load_and_verify_detector_config(connection)

        # Fail before any BLE work if a board is not registered in SQLite -
        # a typo in WORKERS is much easier to understand here than as a
        # foreign-key error at the moment of the first detection.
        for worker in workers:
            register_live_ble_name(connection, worker)

        print_config(config, workers)

        # One task per board. return_exceptions keeps a crash in one session
        # from silently cancelling the other worker's task.
        results = await asyncio.gather(
            *(worker_session(connection, config, worker) for worker in workers),
            return_exceptions=True,
        )
        for worker, result in zip(workers, results):
            if isinstance(result, Exception):
                print(f"[{worker['ble_name']}] stopped: {result!r}")

    finally:
        for worker in workers:
            try:
                await mark_disconnected(connection, worker)
            except Exception:
                pass
        connection.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLive BLE gateway stopped by user.")
