from pathlib import Path

import pandas as pd


# Find the project folder automatically.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Raw files from every teammate are stored here.
RAW_DATA_FOLDER = PROJECT_FOLDER / "data" / "raw"


REQUIRED_COLUMNS = [
    "timestamp",
    "elapsed_seconds",
    "Xg",
    "Yg",
    "Zg",
    "Xdeg",
    "Ydeg",
    "Zdeg",
]


def get_worker_name(csv_file: Path) -> str:
    """
    Use the worker folder name automatically.

    Example:
    data/raw/ziqian/file.csv
    worker = ziqian
    """

    return (
        csv_file.parent.name
        .strip()
        .lower()
        .replace(" ", "_")
    )


def get_action(csv_file: Path) -> str:
    """
    Detect the action automatically from the filename.
    No file renaming is required.
    """

    file_name = csv_file.stem.lower()

    if "near_miss" in file_name:
        return "near_miss"

    if "normal_work" in file_name:
        return "normal_work"

    if "fall" in file_name or "die" in file_name:
        return "fall_like"

    return "unknown"


def check_csv(csv_file: Path) -> dict:
    """
    Check one sensor CSV file.
    """

    dataframe = pd.read_csv(csv_file)

    worker = get_worker_name(csv_file)
    action = get_action(csv_file)

    problems = []

    if dataframe.empty:
        problems.append("CSV contains no sensor rows.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        problems.append(
            f"Missing columns: {missing_columns}"
        )

    if action == "unknown":
        problems.append(
            "The action could not be identified."
        )

    return {
        "file_name": csv_file.name,
        "worker": worker,
        "action": action,
        "rows": len(dataframe),
        "status": (
            "VALID"
            if len(problems) == 0
            else "INVALID"
        ),
        "problems": problems,
    }


def main() -> None:
    print("=" * 70)
    print("GROUP 2 AUTOMATIC SENSOR DATA CHECK")
    print("=" * 70)

    # Find every CSV inside every teammate folder.
    csv_files = sorted(
        RAW_DATA_FOLDER.rglob("*.csv")
    )

    print(f"Files discovered: {len(csv_files)}")
    print()

    valid_count = 0
    invalid_count = 0

    for csv_file in csv_files:
        print("-" * 70)

        try:
            result = check_csv(csv_file)

            print(f"File: {result['file_name']}")
            print(f"Worker: {result['worker']}")
            print(f"Action: {result['action']}")
            print(f"Rows: {result['rows']}")
            print(f"Status: {result['status']}")

            for problem in result["problems"]:
                print(f"Problem: {problem}")

            if result["status"] == "VALID":
                valid_count += 1
            else:
                invalid_count += 1

        except Exception as error:
            invalid_count += 1

            print(f"File: {csv_file.name}")
            print("Status: INVALID")
            print(
                f"Problem: {type(error).__name__}: {error}"
            )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(csv_files)}")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {invalid_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()