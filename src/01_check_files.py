from pathlib import Path
import re

import pandas as pd


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
RAW_DATA_FOLDER = PROJECT_FOLDER / "data" / "raw"

SAMPLE_RATE_HZ = 25


# These are the final standard column names used by the group.
REQUIRED_SENSOR_COLUMNS = [
    "Xg",
    "Yg",
    "Zg",
    "Xdeg",
    "Ydeg",
    "Zdeg",
]


# ============================================================
# COLUMN NAME ALIASES
#
# Different teammates may use different column names.
# Python converts them into one standard format automatically.
# ============================================================

COLUMN_ALIASES = {
    # Time columns
    "timestamp": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",

    "elapsedseconds": "elapsed_seconds",
    "elapsedsecond": "elapsed_seconds",
    "elapsedtime": "elapsed_seconds",
    "elapsed": "elapsed_seconds",

    # Accelerometer X
    "xg": "Xg",
    "accx": "Xg",
    "accelx": "Xg",
    "accelerometerx": "Xg",
    "accelerationx": "Xg",
    "ax": "Xg",

    # Accelerometer Y
    "yg": "Yg",
    "accy": "Yg",
    "accely": "Yg",
    "accelerometery": "Yg",
    "accelerationy": "Yg",
    "ay": "Yg",

    # Accelerometer Z
    "zg": "Zg",
    "accz": "Zg",
    "accelz": "Zg",
    "accelerometerz": "Zg",
    "accelerationz": "Zg",
    "az": "Zg",

    # Gyroscope X
    "xdeg": "Xdeg",
    "xdegs": "Xdeg",
    "xdps": "Xdeg",
    "gyrox": "Xdeg",
    "gyroscopex": "Xdeg",
    "angularvelocityx": "Xdeg",

    # Gyroscope Y
    "ydeg": "Ydeg",
    "ydegs": "Ydeg",
    "ydps": "Ydeg",
    "gyroy": "Ydeg",
    "gyroscopey": "Ydeg",
    "angularvelocityy": "Ydeg",

    # Gyroscope Z
    "zdeg": "Zdeg",
    "zdegs": "Zdeg",
    "zdps": "Zdeg",
    "gyroz": "Zdeg",
    "gyroscopez": "Zdeg",
    "angularvelocityz": "Zdeg",
}


def normalize_text(value: str) -> str:
    """
    Convert text into a simple comparison format.

    Example:
    'Near Miss' becomes 'nearmiss'
    'gyro_x' becomes 'gyrox'
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower()
    )


def normalize_columns(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert different teammate column names into
    the same standard column names.
    """

    rename_dictionary = {}

    for original_column in dataframe.columns:
        normalized_column = normalize_text(
            original_column
        )

        if normalized_column in COLUMN_ALIASES:
            rename_dictionary[original_column] = (
                COLUMN_ALIASES[normalized_column]
            )

    output = dataframe.rename(
        columns=rename_dictionary
    ).copy()

    output.columns = [
        str(column).strip()
        for column in output.columns
    ]

    return output


def get_worker_name(csv_file: Path) -> str:
    """
    Get the worker automatically from the folder name.

    Example:
    data/raw/ziqian/file.csv
    becomes worker = ziqian
    """

    parent_folder = normalize_text(
        csv_file.parent.name
    )

    # Files stored inside worker subfolders
    if parent_folder not in {
        "raw",
        "data",
        ""
    }:
        return (
            csv_file.parent.name
            .strip()
            .lower()
            .replace(" ", "_")
        )

    # Fallback when a file is directly inside data/raw
    file_name = normalize_text(csv_file.stem)

    if "ziqian" in file_name:
        return "ziqian"

    if (
        "kwanteng" in file_name
        or file_name.startswith("kt")
    ):
        return "kwanteng"

    if (
        "hongjean" in file_name
        or "hj" in file_name
    ):
        return "hong_jean"

    if (
        "pierre" in file_name
        or "pieere" in file_name
    ):
        return "pierre"

    return "unknown_worker"


def get_action(csv_file: Path) -> str:
    """
    Identify the activity automatically.

    The group has exactly three activity types:
    normal work, near miss and fall-like.
    """

    file_name = normalize_text(csv_file.stem)

    # Check near miss first.
    if (
        "nearmiss" in file_name
        or "nearmissed" in file_name
    ):
        return "near_miss"

    if (
        "fall" in file_name
        or "die" in file_name
    ):
        return "fall_like"

    if "normal" in file_name:
        return "normal_work"

    # Your group collected only three activity types.
    # A file without fall/near-miss keywords is treated
    # as the normal-work recording.
    return "normal_work"


def check_csv(csv_file: Path) -> dict:
    """
    Read and validate one CSV.
    """

    dataframe = pd.read_csv(csv_file)

    # Standardise different column naming styles.
    dataframe = normalize_columns(dataframe)

    worker = get_worker_name(csv_file)
    action = get_action(csv_file)

    problems = []
    warnings = []

    if dataframe.empty:
        problems.append(
            "CSV contains no sensor rows."
        )

    missing_sensor_columns = [
        column
        for column in REQUIRED_SENSOR_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_sensor_columns:
        problems.append(
            "Missing sensor columns: "
            f"{missing_sensor_columns}"
        )

    # Time fields are useful, but they can be reconstructed.
    if "elapsed_seconds" not in dataframe.columns:
        warnings.append(
            "elapsed_seconds is missing. "
            "It can be reconstructed later using 25 Hz."
        )

    if "timestamp" not in dataframe.columns:
        warnings.append(
            "timestamp is missing. "
            "Relative elapsed time will be used."
        )

    if worker == "unknown_worker":
        warnings.append(
            "Worker could not be identified automatically."
        )

    status = (
        "VALID"
        if len(problems) == 0
        else "INVALID"
    )

    return {
        "file_name": csv_file.name,
        "worker": worker,
        "action": action,
        "rows": len(dataframe),
        "columns": dataframe.columns.tolist(),
        "status": status,
        "problems": problems,
        "warnings": warnings,
    }


def main() -> None:
    print("=" * 70)
    print("GROUP 2 AUTOMATIC SENSOR DATA CHECK")
    print("=" * 70)

    # Automatically discover all CSV files.
    csv_files = sorted(
        RAW_DATA_FOLDER.rglob("*.csv")
    )

    print(
        f"Files discovered: {len(csv_files)}"
    )
    print()

    valid_count = 0
    invalid_count = 0

    for csv_file in csv_files:
        print("-" * 70)

        try:
            result = check_csv(csv_file)

            print(
                f"File: {result['file_name']}"
            )
            print(
                f"Worker: {result['worker']}"
            )
            print(
                f"Action: {result['action']}"
            )
            print(
                f"Rows: {result['rows']}"
            )
            print(
                f"Status: {result['status']}"
            )

            for warning in result["warnings"]:
                print(
                    f"Warning: {warning}"
                )

            for problem in result["problems"]:
                print(
                    f"Problem: {problem}"
                )

            # Display actual columns for invalid files.
            if result["status"] == "INVALID":
                print(
                    "Columns found: "
                    f"{result['columns']}"
                )

            if result["status"] == "VALID":
                valid_count += 1
            else:
                invalid_count += 1

        except Exception as error:
            invalid_count += 1

            print(f"File: {csv_file.name}")
            print("Status: INVALID")
            print(
                f"Problem: "
                f"{type(error).__name__}: {error}"
            )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(
        f"Total files: {len(csv_files)}"
    )
    print(
        f"Valid files: {valid_count}"
    )
    print(
        f"Invalid files: {invalid_count}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()