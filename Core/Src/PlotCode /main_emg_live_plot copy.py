#!/usr/bin/env python3
"""
Multithreaded real-time EMG signal plotter for STM32L476RG.

This version separates serial reading from graph drawing:
    - Thread 1: continuously reads EMG samples from the STM32
    - Main thread: updates the matplotlib graph

Usage:
    python3 emg_plot_threaded.py --list-ports
    python3 emg_plot_threaded.py --port /dev/cu.usbmodem141403
    python3 emg_plot_threaded.py --port /dev/cu.usbmodem141403 --baud 200000 --window 5
    python3 emg_plot_threaded.py --port /dev/cu.usbmodem141403 --save emg_session.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# ── dependency checks ────────────────────────────────────────────────────────

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    raise SystemExit("pyserial is not installed.\n  pip install pyserial")

try:
    import matplotlib
    matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg is unavailable
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
except ImportError:
    raise SystemExit("matplotlib is not installed.\n  pip install matplotlib")

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
ENVELOPE_WINDOW = 50        # samples used for moving-average envelope
UPDATE_INTERVAL_MS = 40     # animation frame interval (~25 fps)
MAX_POINTS = 100_000        # max samples kept in memory


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multithreaded real-time EMG plotter for STM32L476RG.")
    p.add_argument("-p", "--port",
                   help="Serial port (e.g. /dev/cu.usbmodem141403 or COM3).")
    p.add_argument("-b", "--baud", type=int, default=200000,
                   help="Baud rate. Default: 200000.")
    p.add_argument("-w", "--window", type=float, default=10.0,
                   help="Seconds of history to display. Default: 10.")
    p.add_argument("-c", "--column", type=int, default=1,
                   help="1-based index of the numeric value to plot. Default: 1.")
    p.add_argument("--timeout", type=float, default=0.1,
                   help="Serial read timeout in seconds. Default: 0.1.")
    p.add_argument("--save", default=None,
                   help="Optional CSV path to log all samples.")
    p.add_argument("--csv", nargs="?", const="session.csv", default=None,
                   help="Write samples to CSV. Optional path; when omitted defaults to 'session.csv'.")
    p.add_argument("--list-ports", action="store_true",
                   help="Print available serial ports and exit.")
    p.add_argument("--no-envelope", action="store_true",
                   help="Disable the RMS envelope overlay.")
    return p.parse_args()


# ── port helpers ─────────────────────────────────────────────────────────────

def print_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}  ({port.description or 'no description'})")


def auto_select_port() -> str:
    ports = [p.device for p in list_ports.comports()]
    preferred = [p for p in ports if any(k in p.lower()
                 for k in ("usbmodem", "usbserial", "wchusbserial", "ttyacm", "ttyusb"))]
    if preferred:
        return preferred[0]
    if len(ports) == 1:
        return ports[0]
    print_ports()
    raise SystemExit("Multiple ports found — specify one with --port.")


# ── number extraction ─────────────────────────────────────────────────────────

def extract_numbers(line: str) -> list[float]:
    return [float(m.group()) for m in NUMBER_PATTERN.finditer(line)]


# ── live threaded plotter ─────────────────────────────────────────────────────

class ThreadedEMGPlotter:
    def __init__(
        self,
        port: str,
        baud: int,
        window_s: float,
        column: int,
        timeout: float,
        save_path: str | None,
        show_envelope: bool,
    ) -> None:
        self.port = port
        self.baud = baud
        self.window_s = window_s
        self.col_idx = column - 1          # convert to 0-based
        self.timeout = timeout
        self.show_envelope = show_envelope

        # Shared ring buffers. The serial thread writes; the graph thread reads.
        self.times: deque[float] = deque(maxlen=MAX_POINTS)
        self.values: deque[float] = deque(maxlen=MAX_POINTS)
        self.data_lock = threading.Lock()

        self.start_time = time.monotonic()
        self.sample_count = 0
        self.bad_line_count = 0
        self.ser: serial.Serial | None = None

        # Used to stop the serial thread cleanly when the plot window closes.
        self.stop_event = threading.Event()
        self.reader_thread: threading.Thread | None = None

        # Optional CSV logging. This happens in the serial thread so every sample is logged.
        self.csv_file = None
        self.csv_writer = None
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["wall_time", "elapsed_s", "value"])
            print(f"Logging to {path}")

        self._build_figure()

    # ── figure construction ───────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 7), facecolor="#0f0f0f")
        self.fig.canvas.manager.set_window_title("EMG Real-Time Monitor — Threaded")

        gs = gridspec.GridSpec(
            2, 1, height_ratios=[4, 1], hspace=0.35,
            left=0.07, right=0.97, top=0.91, bottom=0.09,
        )

        self.ax_sig = self.fig.add_subplot(gs[0])
        self.ax_sig.set_facecolor("#111111")
        self.ax_sig.tick_params(colors="#aaaaaa")
        for spine in self.ax_sig.spines.values():
            spine.set_edgecolor("#333333")
        self.ax_sig.set_title("Live Raw EMG Signal — Threaded Serial Reader", color="#eeeeee",
                              fontsize=13, fontweight="bold", pad=10)
        self.ax_sig.set_xlabel("Time (s)", color="#aaaaaa")
        self.ax_sig.set_ylabel("Amplitude (ADC)", color="#aaaaaa")
        self.ax_sig.grid(True, linestyle="--", linewidth=0.5,
                         color="#2a2a2a", alpha=0.8)

        self.line_raw, = self.ax_sig.plot(
            [], [], color="#00aaff", linewidth=0.8, label="Raw EMG", zorder=2
        )
        self.line_env, = self.ax_sig.plot(
            [], [], color="#ff6600", linewidth=1.5, linestyle="--",
            label="RMS Envelope", zorder=3, visible=self.show_envelope
        )
        self.ax_sig.legend(loc="upper right", facecolor="#1a1a1a",
                           edgecolor="#444444", labelcolor="#cccccc",
                           fontsize=9)

        self.status_text = self.ax_sig.text(
            0.01, 0.97, "", transform=self.ax_sig.transAxes,
            color="#88ff88", fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor="#333333", alpha=0.8),
        )

        self.ax_stats = self.fig.add_subplot(gs[1])
        self.ax_stats.set_facecolor("#111111")
        self.ax_stats.set_axis_off()
        self.stats_text = self.ax_stats.text(
            0.5, 0.5, "Waiting for EMG samples...", transform=self.ax_stats.transAxes,
            color="#cccccc", fontsize=10, ha="center", va="center",
            family="monospace",
        )

        plt.ion()

    # ── serial open / reader thread / close ───────────────────────────────────

    def _open_serial(self) -> None:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
            )
            # Clear old bytes that may have been sitting in the USB buffer.
            self.ser.reset_input_buffer()
            print(f"Opened {self.port} at {self.baud} baud.")
        except serial.SerialException as exc:
            raise SystemExit(f"Cannot open {self.port}: {exc}")

    def _reader_loop(self) -> None:
        """Continuously read serial data without blocking matplotlib."""
        if self.ser is None:
            return

        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline()
            except serial.SerialException as exc:
                print(f"\nSerial read error: {exc}")
                self.stop_event.set()
                break

            if not raw:
                continue

            elapsed = time.monotonic() - self.start_time
            line = raw.decode("utf-8", errors="replace").strip()
            nums = extract_numbers(line)

            if not nums or self.col_idx >= len(nums):
                self.bad_line_count += 1
                continue

            value = nums[self.col_idx]

            # Lock only around the shared buffer update.
            with self.data_lock:
                self.times.append(elapsed)
                self.values.append(value)
                self.sample_count += 1

            if self.csv_writer:
                wall = datetime.now().isoformat(timespec="milliseconds")
                self.csv_writer.writerow([wall, f"{elapsed:.6f}", value])
                # Flush less often than every line for better performance.
                if self.sample_count % 100 == 0:
                    self.csv_file.flush()

    def _start_reader_thread(self) -> None:
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            name="EMGSerialReader",
            daemon=True,
        )
        self.reader_thread.start()
        print("Started serial reader thread.")

    def _close_serial(self) -> None:
        self.stop_event.set()

        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)

        if self.ser and self.ser.is_open:
            self.ser.close()

        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()

    # ── animation callback ────────────────────────────────────────────────────

    def _snapshot_buffers(self) -> tuple[list[float], list[float], int, int]:
        """Copy shared data quickly, then release the lock before plotting."""
        with self.data_lock:
            return (
                list(self.times),
                list(self.values),
                self.sample_count,
                self.bad_line_count,
            )

    def _update(self, _frame: int) -> list:
        t, v, sample_count, bad_line_count = self._snapshot_buffers()

        if not t:
            self.status_text.set_text("Waiting for serial data...")
            return [self.line_raw, self.line_env]

        now = t[-1]

        # Clip to visible window.
        x_min = max(0.0, now - self.window_s)
        start_idx = 0
        for i, ti in enumerate(t):
            if ti >= x_min:
                start_idx = i
                break

        t_vis = t[start_idx:]
        v_vis = v[start_idx:]

        self.line_raw.set_data(t_vis, v_vis)

        # RMS envelope.
        if self.show_envelope and len(v_vis) >= ENVELOPE_WINDOW:
            arr = np.array(v_vis, dtype=float)
            mean = np.mean(arr)
            arr_c = arr - mean
            kernel = np.ones(ENVELOPE_WINDOW) / ENVELOPE_WINDOW
            rms = np.sqrt(np.convolve(arr_c ** 2, kernel, mode="same")) + mean
            self.line_env.set_data(t_vis, rms)
        else:
            self.line_env.set_data([], [])

        self.ax_sig.set_xlim(x_min, max(x_min + self.window_s, now))
        self.ax_sig.set_ylim(0, 4095)

        rate = sample_count / now if now > 0 else 0
        self.status_text.set_text(
            f"Samples: {sample_count:,}   Elapsed: {now:.1f}s   "
            f"~{rate:.0f} Hz   Bad lines: {bad_line_count:,}"
        )

        if v_vis:
            arr = np.array(v_vis)
            self.stats_text.set_text(
                f"Window stats —  "
                f"Min: {arr.min():.1f}   "
                f"Max: {arr.max():.1f}   "
                f"Mean: {arr.mean():.1f}   "
                f"Std: {arr.std():.1f}   "
                f"Peak-to-peak: {arr.max() - arr.min():.1f}"
            )

        return [self.line_raw, self.line_env]

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._open_serial()
        self._start_reader_thread()

        print(f"Plotting column {self.col_idx + 1} | window {self.window_s}s | "
              f"Press Ctrl+C or close the window to stop.")

        try:
            self._anim = FuncAnimation(
                self.fig,
                self._update,
                interval=UPDATE_INTERVAL_MS,
                blit=False,
                cache_frame_data=False,
            )
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self._close_serial()
            print(f"Total samples recorded: {self.sample_count:,}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.list_ports:
        print_ports()
        return 0

    port = args.port or auto_select_port()

    # Prefer explicit --csv if provided, otherwise fall back to --save
    save_path = args.csv if args.csv is not None else args.save

    plotter = ThreadedEMGPlotter(
        port=port,
        baud=args.baud,
        window_s=args.window,
        column=args.column,
        timeout=args.timeout,
        save_path=save_path,
        show_envelope=not args.no_envelope,
    )
    plotter.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
