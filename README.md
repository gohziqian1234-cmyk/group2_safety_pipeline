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

- BLE name: `ziqian`
- Sampling rate: **25 Hz**
- Service UUID: `bdc766fc-7eee-417f-bbe0-2e71a8a2bf70`
- Combined accelerometer + gyroscope UUID: `f509416c-3c4b-401e-a768-b25a9e621a91`
- BLE payload: `Xg,Yg,Zg,Xdeg,Ydeg,Zdeg`
- Arduino sketch: `NESSO_ziqian_25Hz.ino`

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

1. **Upload `NESSO_ziqian_25Hz.ino`** to the NESSO N1.
2. **Turn on Bluetooth** on the Windows laptop and power the NESSO.
3. From the repository root, run the real BLE gateway:

   ```bat
   py src\21_live_ble_gateway.py
   ```

4. In a second terminal, run the Streamlit dashboard:

   ```bat
   py -m streamlit run src\dashboard.py
   ```

   Alternatively, after dependencies are installed, `run_live_system.bat` opens both programs.

5. Confirm the gateway/dashboard shows **CONNECTED / SAFE** for Ziqian.
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
- `NESSO_ziqian_25Hz.ino` - final NESSO 25 Hz raw-IMU BLE firmware
- `requirements_live.txt` - live Python dependencies
- `run_live_system.bat` - Windows lesson launcher
- `database/safety_pipeline.db` - local project database

### Data and evidence

- `data/raw/` - original sensor recordings
- `data/processed/` - canonical cleaned data, feature/rule validation outputs, final detector results
- `reports/` - plots and report/PPT evidence

## Live-system notes

The Arduino sends raw accelerometer and gyroscope values only. The Python gateway reconstructs the approved 25 Hz detector features, applies the fixed 2-of-4 vote rule, uses the 5-second startup grace and 8-second cooldown, stores real events as `BLE_GATEWAY`, and updates the device status used by the Streamlit dashboard.
