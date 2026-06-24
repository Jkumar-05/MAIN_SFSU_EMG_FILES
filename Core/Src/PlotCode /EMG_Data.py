#!/usr/bin/env python3
"""
Record EMG samples streamed from an STM32 serial port.

Examples:
    python3 Src/EMG_Data.py --list-ports
    python3 Src/EMG_Data.py --port /dev/cu.usbmodem1103 --baud 200000
    python3 Src/EMG_Data.py --duration 30 --output Data/emg_trial.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:
    raise SystemExit(
        "pyserial is not installed. Install it with: python3 -m pip install pyserial"
    ) from exc


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Store EMG data streamed from an STM32 over USB serial."
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
        default=200000,
        help="Baud rate used by the STM32 firmware. Default: 200000.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="CSV output path. Default: Data/emg_YYYYmmdd_HHMMSS.csv.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Seconds to record. Omit to record until Ctrl+C.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Serial read timeout in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--max-values",
        type=int,
        default=8,
        help="Maximum numeric values to split into CSV columns. Default: 8.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Show available serial ports and exit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a live plot while recording.",
    )
    parser.add_argument(
        "--plot-window",
        type=float,
        default=10.0,
        help="Seconds of data to keep visible in the live plot. Default: 10.",
    )
    parser.add_argument(
        "--plot-value",
        type=int,
        default=1,
        help="Numeric value column to plot, using 1 for value_1. Default: 1.",
    )
    return parser.parse_args()


def available_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def print_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("Available serial ports:")
    for port in ports:
        description = port.description or "no description"
        print(f"  {port.device}  ({description})")


def choose_default_port() -> str:
    ports = available_ports()
    preferred = [
        port
        for port in ports
        if "usbmodem" in port.lower()
        or "usbserial" in port.lower()
        or "wchusbserial" in port.lower()
    ]

    if preferred:
        return preferred[0]

    if len(ports) == 1:
        return ports[0]

    print_ports()
    raise SystemExit("Choose a port with --port.")


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("Data") / f"emg_{timestamp}.csv"


def numbers_from_line(line: str) -> list[float]:
    return [float(match.group(0)) for match in NUMBER_PATTERN.finditer(line)]


class LiveEmgPlot:
    def __init__(self, window_s: float) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit(
                "matplotlib is not installed. Install it with: python3 -m pip install matplotlib"
            ) from exc

        self.plt = plt
        self.window_s = window_s
        self.times: deque[float] = deque()
        self.values: deque[float] = deque()
        self.last_draw = 0.0

        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(12, 5))
        (self.line,) = self.axis.plot([], [], color="b", linewidth=1.0)
        self.axis.set_title("Live Raw EMG Signal")
        self.axis.set_xlabel("Time (s)")
        self.axis.set_ylabel("Amplitude (ADC Value)")
        self.axis.grid(True, linestyle="--", alpha=0.7)
        self.figure.tight_layout()
        self.figure.show()

    def add_sample(self, elapsed: float, value: float) -> None:
        self.times.append(elapsed)
        self.values.append(value)

        while self.times and elapsed - self.times[0] > self.window_s:
            self.times.popleft()
            self.values.popleft()

        now = time.monotonic()
        if now - self.last_draw < 0.05:
            return

        self.last_draw = now
        self.line.set_data(list(self.times), list(self.values))

        x_min = max(0.0, elapsed - self.window_s)
        x_max = max(self.window_s, elapsed)
        self.axis.set_xlim(x_min, x_max)

        if self.values:
            y_min = min(self.values)
            y_max = max(self.values)
            if y_min == y_max:
                y_min -= 1
                y_max += 1
            margin = (y_max - y_min) * 0.1
            self.axis.set_ylim(y_min - margin, y_max + margin)

        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self) -> None:
        self.plt.ioff()
        self.plt.show()


def record_emg(
    port: str,
    baud: int,
    output_path: Path,
    duration: float | None,
    timeout: float,
    max_values: int,
    plot: bool,
    plot_window: float,
    plot_value: int,
) -> None:
    if plot_value < 1:
        raise SystemExit("--plot-value must be 1 or greater.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    sample_count = 0
    live_plot = LiveEmgPlot(plot_window) if plot else None

    try:
        with serial.Serial(
            port=port, baudrate=baud, timeout=timeout
        ) as ser, output_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            value_columns = [f"value_{index}" for index in range(1, max_values + 1)]
            writer.writerow(["wall_time", "elapsed_s", "raw_line", *value_columns])

            print(f"Recording from {port} at {baud} baud")
            print(f"Writing to {output_path}")
            if live_plot is not None:
                print(f"Live plotting value_{plot_value}")
            print("Press Ctrl+C to stop.")

            while True:
                if duration is not None and time.monotonic() - start_time >= duration:
                    break

                raw_bytes = ser.readline()
                if not raw_bytes:
                    continue

                elapsed = time.monotonic() - start_time
                wall_time = datetime.now().isoformat(timespec="milliseconds")
                line = raw_bytes.decode("utf-8", errors="replace").strip()
                values = numbers_from_line(line)
                value_cells = values[:max_values]
                value_cells.extend([""] * (max_values - len(value_cells)))

                writer.writerow([wall_time, f"{elapsed:.6f}", line, *value_cells])
                csv_file.flush()
                sample_count += 1

                if live_plot is not None and len(values) >= plot_value:
                    live_plot.add_sample(elapsed, values[plot_value - 1])

                values_text = ", ".join(f"{value:g}" for value in values)
                print(f"{sample_count:>6}  {elapsed:>9.3f}s  {values_text or line}")
    finally:
        if live_plot is not None:
            live_plot.close()

    print(f"Saved {sample_count} samples to {output_path}")


def main() -> int:
    args = parse_args()

    if args.list_ports:
        print_ports()
        return 0

    port = args.port or choose_default_port()
    output_path = Path(args.output) if args.output else default_output_path()

    try:
        record_emg(
            port=port,
            baud=args.baud,
            output_path=output_path,
            duration=args.duration,
            timeout=args.timeout,
            max_values=args.max_values,
            plot=args.plot,
            plot_window=args.plot_window,
            plot_value=args.plot_value,
        )
    except KeyboardInterrupt:
        print("\nStopped recording.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
