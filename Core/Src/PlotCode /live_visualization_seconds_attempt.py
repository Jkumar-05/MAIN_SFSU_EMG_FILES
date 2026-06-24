#!/usr/bin/env python3
"""
Read EMG samples streamed from an STM32 serial port and show a live plot.

The serial reading/parsing behavior intentionally matches EMG_Data.py. The only
added behavior is plotting the numeric EMG values as they arrive.

Examples:
    python3 Core/Src/live_visualization_seconds_attempt.py --list-ports
    python3 Core/Src/live_visualization_seconds_attempt.py --port /dev/cu.usbmodem1103 --baud 200000
    python3 Core/Src/live_visualization_seconds_attempt.py --duration 30
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import deque
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
    from serial.tools import list_ports
except ImportError as exc:
    raise SystemExit(
        "pyserial is not installed. Install it with: python3 -m pip install pyserial"
    ) from exc


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read EMG data streamed from an STM32 over USB serial and plot it live."
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
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Seconds to run. Omit to run until Ctrl+C.",
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
        default=1,
        help="Maximum numeric values to plot from each serial line. Default: 1.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Number of recent samples to keep on the plot. Default: 200.",
    )
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=30.0,
        help="Maximum graph refresh rate in Hz. Default: 30.",
    )
    parser.add_argument(
        "--title",
        default="STM32 EMG Live Visualization",
        help="Window title and plot title. Default: STM32 EMG Live Visualization.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Show available serial ports and exit.",
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


def numbers_from_line(line: str) -> list[float]:
    return [float(match.group(0)) for match in NUMBER_PATTERN.finditer(line)]


def create_plot(
    num_series: int, title: str
) -> tuple[plt.Figure, plt.Axes, list[plt.Line2D]]:
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []

    for index in range(num_series):
        (line_obj,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"value_{index + 1}",
        )
        lines.append(line_obj)

    ax.set_title(title)
    ax.set_xlabel("elapsed seconds")
    ax.set_ylabel("EMG value")
    ax.set_ylim(0, 4500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax, lines


def update_plot(
    ax: plt.Axes,
    lines: list[plt.Line2D],
    time_history: deque[float],
    value_history: list[deque[float]],
) -> None:
    for line_obj, series in zip(lines, value_history):
        line_obj.set_data(time_history, list(series))

    ax.relim()
    ax.autoscale_view(scalex=True, scaley=False)


def run_live_visualization(
    port: str,
    baud: int,
    duration: float | None,
    timeout: float,
    max_values: int,
    window_size: int,
    plot_rate: float,
    title: str,
) -> None:
    start_time = time.monotonic()
    sample_count = 0
    plot_interval = 1.0 / plot_rate if plot_rate > 0 else 0.0
    last_plot_update = 0.0

    time_history: deque[float] = deque(maxlen=window_size)
    value_history: list[deque[float]] = [
        deque(maxlen=window_size) for _ in range(max_values)
    ]

    fig, ax, lines = create_plot(max_values, title)
    fig.canvas.manager.set_window_title(title)
    plt.ion()
    plt.show(block=False)

    with serial.Serial(port=port, baudrate=baud, timeout=timeout) as ser:
        print(f"Reading from {port} at {baud} baud")
        print("Press Ctrl+C to stop.")

        while True:
            if duration is not None and time.monotonic() - start_time >= duration:
                break

            raw_bytes = ser.readline()
            if not raw_bytes:
                plt.pause(0.01)
                continue

            elapsed = time.monotonic() - start_time
            line = raw_bytes.decode("utf-8", errors="replace").strip()
            values = numbers_from_line(line)
            sample_count += 1

            values_text = ", ".join(f"{value:g}" for value in values)
            print(f"{sample_count:>6}  {elapsed:>9.3f}s  {values_text or line}")

            if values:
                time_history.append(elapsed)
                plotted_values = values[:max_values]

                for index in range(max_values):
                    if index < len(plotted_values):
                        value_history[index].append(plotted_values[index])
                    else:
                        value_history[index].append(float("nan"))

                if elapsed - last_plot_update >= plot_interval:
                    update_plot(ax, lines, time_history, value_history)
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                    last_plot_update = elapsed

    print(f"Read {sample_count} samples.")

    if plt.fignum_exists(fig.number):
        print("Streaming ended. Close the plot window to exit.")
        plt.ioff()
        plt.show(block=True)


def main() -> int:
    args = parse_args()

    if args.list_ports:
        print_ports()
        return 0

    port = args.port or choose_default_port()

    try:
        run_live_visualization(
            port=port,
            baud=args.baud,
            duration=args.duration,
            timeout=args.timeout,
            max_values=args.max_values,
            window_size=args.window_size,
            plot_rate=args.plot_rate,
            title=args.title,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
