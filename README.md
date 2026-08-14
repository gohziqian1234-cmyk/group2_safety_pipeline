# EG2A17 Group 5 NESSO Wearable Safety Project

Automated wearable safety pipeline using a NESSO N1 IMU, Bluetooth Low Energy, a Python rule-based live gateway, SQLite, and a Streamlit dashboard.

## Final lesson architecture

```text
NESSO 25 Hz Arduino
    -> BLE raw IMU data
    -> src/21_live_ble_gateway.py
    -> rule-based SAFETY_EVENT detection
    -> SQLite
    -> src/dashboard.py (Streamlit)
```

**Important:** the lesson-ready version performs the validated rule-based detector in the **Python BLE gateway**. It must not be described as full NESSO edge detection.

## Final hardware / BLE settings

All four boards run at the same time. Every setting below is identical on both -
only the BLE name differs, because that is what tells the gateway which
worker a board belongs to.

| BLE name | Worker | Device | Arduino sketch |
|----------|--------|--------|----------------|
| `ziqian` | `ZQ` (Ziqian) | `ZQ_N1` | `NESSO_ziqian_25Hz.ino` |
| `hongjean` | `HJ` (Hong Jean) | `HJ_N1` | `NESSO_hongjean_25Hz.ino` |
| `kwanteng` | `KT` (Kwanteng) | `KT_N1` | `NESSO_kwanteng_25Hz.ino` |
| `pierre` | `PI` (Pierre) | `PI_N1` | `NESSO_pierre_25Hz.ino` |

- Sampling rate: **25 Hz**
- Service UUID: `bdc766fc-7eee-417f-bbe0-2e71a8a2bf70`
- Combined accelerometer + gyroscope UUID: `f509416c-3c4b-401e-a768-b25a9e621a91`
- BLE payload: `Xg,Yg,Zg,Xdeg,Ydeg,Zdeg`

**Every board needs its own BLE name.** Two boards advertising the same name
would be matched to the same worker, and their two data streams would be
interleaved into one detector - which produces meaningless features rather
than an obvious error.

### Adding another board

Add a line to `WORKERS` at the top of `src/21_live_ble_gateway.py`:

```python
WORKERS = [
    {"ble_name": "ziqian",   "worker_id": "ZQ", "device_id": "ZQ_N1"},
    {"ble_name": "hongjean", "worker_id": "HJ", "device_id": "HJ_N1"},
    {"ble_name": "kwanteng", "worker_id": "KT", "device_id": "KT_N1"},
    {"ble_name": "pierre",   "worker_id": "PI", "device_id": "PI_N1"},
]
```

then copy a sketch and change its `BLE_NAME`. `worker_id` and `device_id`
must match an active row in the SQLite `workers` table - `17_init_database.py`
already seeds ZQ, HJ, KT and PI. Nothing else needs changing: `gateway_core.py`
and the Streamlit dashboard are already per-worker.

## Validated detector configuration

The live gateway reads the active detector configuration from SQLite and refuses to run if it differs from these approved deployment values:

- Required votes: **2 out of 4**
- Acceleration: **>= 3.28154 g**
- Gyroscope: **>= 641.28650 dps**
- Jerk: **>= 67.40781 g/s**
- Rotation activity: **>= 524.51258 deg**
- Startup grace: **5 seconds**
- Cooldown: **8 seconds**

The validation and derivation evidence remains under `src/01...16`, `data/processed/`, and `reports/`.

## First-time Python setup

From the repository root:

```bat
py -m pip install -r requirements_live.txt
py src\17_init_database.py
```

`17_init_database.py` loads the validated final detector configuration into `database/safety_pipeline.db`.

## Exact lesson run sequence

1. **Upload the sketches.** One per board - `NESSO_ziqian_25Hz.ino`,
   `NESSO_hongjean_25Hz.ino`, `NESSO_kwanteng_25Hz.ino`,
   `NESSO_pierre_25Hz.ino`. They differ only in `BLE_NAME`.
2. **Turn on Bluetooth** on the Windows laptop and power the boards.
3. From the repository root, run the real BLE gateway:

   ```bat
   py src\21_live_ble_gateway.py
   ```

   That connects to all four boards. To run a subset - useful when the
   others are not on the bench, since otherwise the gateway spends its time
   scanning for boards that are switched off:

   ```bat
   py src\21_live_ble_gateway.py ziqian hongjean
   ```

4. In a second terminal, run the Streamlit dashboard:

   ```bat
   py -m streamlit run src\dashboard.py
   ```

   Alternatively, after dependencies are installed, `run_live_system.bat` opens both programs.

5. Confirm the gateway prints `CONNECTED / SAFE` for every board you
   powered, and that the dashboard shows them connected. Every gateway line
   is prefixed with the board name so they are easy to tell apart.
6. Perform only a **safe, controlled movement** appropriate for the lesson demonstration.
7. When the validated 2-of-4 rule is met, confirm a **SAFETY_EVENT** is written to SQLite and appears on the Streamlit dashboard.
8. The event source should be recorded as **`BLE_GATEWAY`**, because detection runs in Python for this lesson-ready version.

## Main project files

### Offline analysis and detector evidence

`src/01_check_files.py` through `src/16_finalize_stage1_detector.py` contain the cleaning, analysis, feature engineering, rule validation, cross-validation, replay, rearm tuning, and final detector derivation evidence.

### Database and live system

- `src/17_init_database.py` - SQLite schema, workers, and validated detector config
- `src/gateway_core.py` - reusable status/event persistence layer
- `src/21_live_ble_gateway.py` - real BLE listener + Python Stage-1 detector
- `src/dashboard.py` - Streamlit safety dashboard
- `NESSO_ziqian_25Hz.ino` - NESSO 25 Hz raw-IMU BLE firmware (Ziqian)
- `NESSO_hongjean_25Hz.ino`, `NESSO_kwanteng_25Hz.ino`,
  `NESSO_pierre_25Hz.ino` - the same firmware with a different `BLE_NAME`
- `requirements_live.txt` - live Python dependencies
- `run_live_system.bat` - Windows lesson launcher
- `database/safety_pipeline.db` - local project database

### Data and evidence

- `data/raw/` - original sensor recordings
- `data/processed/` - canonical cleaned data, feature/rule validation outputs, final detector results
- `reports/` - plots and report/PPT evidence

## Live-system notes

The Arduino sends raw accelerometer and gyroscope values only. The Python gateway reconstructs the approved 25 Hz detector features, applies the fixed 2-of-4 vote rule, uses the 5-second startup grace and 8-second cooldown, stores real events as `BLE_GATEWAY`, and updates the device status used by the Streamlit dashboard.

### How the boards stay independent

Each board gets its own BLE session, its own `LiveStage1Detector` instance and
its own row in `device_status`. One worker's incident, cooldown or dropout has
no effect on the other. Two details make that safe:

- **Scanning is serialised.** Only one BLE scan runs at a time, because
  concurrent discovery is unreliable on both Windows and Linux - it produces
  missed devices rather than a clean error.
- **Database writes are serialised.** The tasks share one SQLite connection.
  They all run on the same event loop so writes cannot truly overlap, but the
  lock makes that guarantee explicit rather than accidental.

### Known detector behaviour on the recorded data

Recorded in `data/processed/final_stage1_detector_config.csv` and
`fixed_event_rule_final.csv`, and worth stating plainly before a
demonstration:

- 7 of 7 reliable incidents detected, 0 false negatives.
- The ziqian fall recording is **not** detected. It is classified as an
  uncertain incident rather than a reliable one, so it does not count against
  the detection rate - but it does mean this detector has a known miss.
- 1 of 4 normal-work recordings raises an alert; the per-window false
  positive rate is 0.46%.
