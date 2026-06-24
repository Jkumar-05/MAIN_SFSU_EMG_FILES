#!/usr/bin/env python3
"""
Plot EMG data saved from the STM32 serial recorder.

Examples:
    python3 visualization.py ../EMG_Data_6-17-1026.csv
    python3 visualization.py ../EMG_Data_6-17-1026.csv --columns value_1 value_2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

LOCAL_MATPLOTLIB_CACHE = Path(__file__).resolve().parent / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MATPLOTLIB_CACHE))
LOCAL_MATPLOTLIB_CACHE.mkdir(exist_ok=True)

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is not installed. Install it with: python3 -m pip install matplotlib"
    ) from exc


PREFERRED_TIME_COLUMNS = ("elapsed_s", "time_s", "time", "timestamp", "sample")
IGNORED_NUMERIC_COLUMNS = {"wall_time", "raw_line"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load EMG values from a CSV file and graph them."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        default=None,
        help="Path to the EMG CSV file. Default: ../EMG_Data_6-17-1026.csv",
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=None,
        help="Path to the EMG CSV file. Default: ../EMG_Data_6-17-1026.csv",
    )
    parser.add_argument(
        "-c",
        "--columns",
        nargs="+",
        help="EMG data columns to plot. Default: auto-detect numeric value columns.",
    )
    parser.add_argument(
        "--title",
        default="STM32 EMG Data",
        help="Plot title. Default: STM32 EMG Data",
    )
    return parser.parse_args()


def to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file does not have a header row.")

        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    return reader.fieldnames, rows


def find_time_column(fieldnames: list[str], rows: list[dict[str, str]]) -> str | None:
    for column in PREFERRED_TIME_COLUMNS:
        if column in fieldnames and any(to_float(row.get(column)) is not None for row in rows):
            return column

    return None


def find_data_columns(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    time_column: str | None,
) -> list[str]:
    data_columns = []

    for column in fieldnames:
        if column == time_column or column in IGNORED_NUMERIC_COLUMNS:
            continue

        numeric_count = sum(to_float(row.get(column)) is not None for row in rows)
        if numeric_count > 0:
            data_columns.append(column)

    return data_columns


def series_from_rows(
    rows: list[dict[str, str]], column: str
) -> tuple[list[int], list[float]]:
    sample_numbers = []
    values = []

    for sample_number, row in enumerate(rows, start=1):
        value = to_float(row.get(column))
        if value is None:
            continue

        sample_numbers.append(sample_number)
        values.append(value)

    return sample_numbers, values


def plot_emg(
    csv_path: Path,
    columns: list[str] | None,
    title: str,
) -> None:
    fieldnames, rows = load_rows(csv_path)
    time_column = find_time_column(fieldnames, rows)
    data_columns = columns or find_data_columns(fieldnames, rows, time_column)

    missing_columns = [column for column in data_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"Column(s) not found in CSV: {', '.join(missing_columns)}")

    if not data_columns:
        raise ValueError("No numeric EMG columns found to plot.")

    fig, ax = plt.subplots(figsize=(10, 6))

    for column in data_columns:
        if time_column is None:
            x_values, y_values = series_from_rows(rows, column)
        else:
            x_values = []
            y_values = []
            for row in rows:
                x_value = to_float(row.get(time_column))
                y_value = to_float(row.get(column))
                if x_value is None or y_value is None:
                    continue
                x_values.append(x_value)
                y_values.append(y_value)

        if y_values:
            ax.plot(x_values, y_values, marker="o", markersize=3, linewidth=1.2, label=column)

    ax.set_title(title)
    ax.set_xlabel(time_column if time_column is not None else "sample number")
    ax.set_ylabel("EMG reading")
    ax.grid(True, alpha=0.3)

    if len(data_columns) > 1:
        ax.legend()

    fig.tight_layout()
    plt.show()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.input_file or args.csv_file or "../EMG_Data_6-17-1026.csv")

    try:
        plot_emg(
            csv_path=csv_path,
            columns=args.columns,
            title=args.title,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
