#!/usr/bin/env python3
"""
Live plot EMG samples using the same serial input/parsing code as EMG_Data.py.

This script only plots. It does not write a CSV file.

Examples:
    python3 Core/Src/live_plot_emg_data.py --list-ports
    python3 Core/Src/live_plot_emg_data.py --port /dev/cu.usbmodem1103 --baud 200000
    python3 Core/Src/live_plot_emg_data.py --duration 30 --window-seconds 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
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

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "pyserial is not installed. Install it with: python3 -m pip install pyserial"
    ) from exc

try:
    from EMG_Data import choose_default_port, numbers_from_line, print_ports
except ImportError as exc:
    raise SystemExit(
        "Could not import EMG_Data.py. Run this script from the project root or Core/Src."
    ) from exc


ADC_MIN = 0
ADC_MAX = 4095
DEFAULT_BAUD = 200000
DEFAULT_MAX_VALUES = 1
DEFAULT_PLOT_RATE_HZ = 30.0
DEFAULT_WINDOW_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot live EMG serial data using EMG_Data.py parsing."
    )
    parser.add_argument(
        "-p",
        "--port",
        help="Serial port, for example /dev/cu.usbmodem1103 or /dev/cu.usbserial-0001.",
    )
    parser.add_argument(
        "-b",
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate used by the STM32 firmware. Default: {DEFAULT_BAUD}.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Seconds to plot before stopping. Omit to run until Ctrl+C.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.05,
        help="Serial read timeout in seconds. Default: 0.05.",
    )
    parser.add_argument(
        "--max-values",
        type=int,
        default=DEFAULT_MAX_VALUES,
        help=f"Numeric values to plot from each serial line. Default: {DEFAULT_MAX_VALUES}.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Seconds of recent data visible on the plot. Default: {DEFAULT_WINDOW_SECONDS:g}.",
    )
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=DEFAULT_PLOT_RATE_HZ,
        help=f"Maximum graph refresh rate in Hz. Default: {DEFAULT_PLOT_RATE_HZ:g}.",
    )
    parser.add_argument(
        "--autoscale-y",
        action="store_true",
        help="Autoscale the y-axis instead of using the 12-bit ADC range 0-4095.",
    )
    parser.add_argument(
        "--title",
        default="Live EMG Data",
        help="Window title and plot title. Default: Live EMG Data.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        help="Path to save parsed timestamps and values as CSV.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Show available serial ports and exit.",
    )
    return parser.parse_args()


def create_plot(
    series_count: int,
    title: str,
    window_seconds: float,
    autoscale_y: bool,
) -> tuple[plt.Figure, plt.Axes, list[plt.Line2D]]:
    fig, ax = plt.subplots(figsize=(11, 6))
    lines: list[plt.Line2D] = []

    for index in range(series_count):
        (line,) = ax.plot([], [], linewidth=1.4, label=f"value_{index + 1}")
        lines.append(line)

    ax.set_title(title)
    ax.set_xlabel("elapsed time (seconds)")
    ax.set_ylabel("EMG value")
    ax.set_xlim(0, max(window_seconds, 1.0))
    if not autoscale_y:
        ax.set_ylim(ADC_MIN, ADC_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax, lines


def trim_old_samples(
    times: deque[float],
    values_by_series: list[deque[float]],
    latest_time: float,
    window_seconds: float,
) -> None:
    cutoff = latest_time - window_seconds
    while times and times[0] < cutoff:
        times.popleft()
        for series in values_by_series:
            series.popleft()


def update_plot(
    ax: plt.Axes,
    lines: list[plt.Line2D],
    times: deque[float],
    values_by_series: list[deque[float]],
    window_seconds: float,
    autoscale_y: bool,
) -> None:
    if not times:
        return

    for line, values in zip(lines, values_by_series):
        line.set_data(list(times), list(values))

    latest_time = times[-1]
    ax.set_xlim(max(0.0, latest_time - window_seconds), max(window_seconds, latest_time))

    if autoscale_y:
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)


def run_live_plot(
    port: str,
    baud: int,
    duration: float | None,
    timeout: float,
    max_values: int,
    window_seconds: float,
    plot_rate: float,
    autoscale_y: bool,
    title: str,
    save_csv: Path | None = None,
) -> None:
    if max_values < 1:
        raise ValueError("--max-values must be at least 1.")
    if window_seconds <= 0:
        raise ValueError("--window-seconds must be greater than 0.")

    sample_count = 0
    plotted_count = 0
    start_time = time.monotonic()
    plot_interval = 1.0 / plot_rate if plot_rate > 0 else 0.0
    last_plot_update = 0.0

    times: deque[float] = deque()
    values_by_series: list[deque[float]] = [deque() for _ in range(max_values)]
    fig, ax, lines = create_plot(max_values, title, window_seconds, autoscale_y)
    fig.canvas.manager.set_window_title(title)

    plt.ion()
    plt.show(block=False)

    csv_file = None
    csv_writer = None
    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = save_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        header = ["wall_time", "elapsed_s", "raw_line"] + [
            f"value_{index + 1}" for index in range(max_values)
        ]
        csv_writer.writerow(header)

    try:
        with serial.Serial(port=port, baudrate=baud, timeout=timeout) as ser:
            ser.reset_input_buffer()
            print(f"Plotting EMG data from {port} at {baud} baud")
            if save_csv is not None:
                print(f"Saving parsed samples to CSV: {save_csv}")
            print("Press Ctrl+C to stop.")

            while plt.fignum_exists(fig.number):
                elapsed = time.monotonic() - start_time
                if duration is not None and elapsed >= duration:
                    break

                raw_bytes = ser.readline()
                if not raw_bytes:
                    plt.pause(0.001)
                    continue

                line = raw_bytes.decode("utf-8", errors="replace").strip()
                values = numbers_from_line(line)
                sample_count += 1

                if not values:
                    print(f"{sample_count:>6}  {elapsed:>9.3f}s  skipped: {line}")
                    if csv_writer is not None:
                        empty_cells = ["" for _ in range(max_values)]
                        row = [
                            datetime.now().isoformat(timespec="milliseconds"),
                            f"{elapsed:.6f}",
                            line,
                        ] + empty_cells
                        csv_writer.writerow(row)
                        csv_file.flush()
                    continue

                times.append(elapsed)
                plotted_values = values[:max_values]
                for index in range(max_values):
                    value = plotted_values[index] if index < len(plotted_values) else float("nan")
                    values_by_series[index].append(value)

                trim_old_samples(times, values_by_series, elapsed, window_seconds)
                plotted_count += 1

                values_text = ", ".join(f"{value:g}" for value in plotted_values)
                print(f"{plotted_count:>6}  {elapsed:>9.3f}s  {values_text}")

                if csv_writer is not None:
                    value_cells = [
                        f"{plotted_values[index]:g}" if index < len(plotted_values) else ""
                        for index in range(max_values)
                    ]
                    row = [
                        datetime.now().isoformat(timespec="milliseconds"),
                        f"{elapsed:.6f}",
                        line,
                    ] + value_cells
                    csv_writer.writerow(row)
                    csv_file.flush()

                if elapsed - last_plot_update >= plot_interval:
                    update_plot(
                        ax=ax,
                        lines=lines,
                        times=times,
                        values_by_series=values_by_series,
                        window_seconds=window_seconds,
                        autoscale_y=autoscale_y,
                    )
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                    last_plot_update = elapsed
    finally:
        if csv_file is not None:
            csv_file.close()

    print(f"Read {sample_count} serial lines and plotted {plotted_count} samples.")


def main() -> int:
    args = parse_args()

    if args.list_ports:
        print_ports()
        return 0

    port = args.port or choose_default_port()

    try:
        run_live_plot(
            port=port,
            baud=args.baud,
            duration=args.duration,
            timeout=args.timeout,
            max_values=args.max_values,
            window_seconds=args.window_seconds,
            plot_rate=args.plot_rate,
            autoscale_y=args.autoscale_y,
            title=args.title,
            save_csv=args.save_csv,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (ValueError, serial.SerialException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
