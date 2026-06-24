import serial
import time
import csv
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

PORT = "/dev/cu.usbmodem12203"
BAUD = 115200

WINDOW_SECONDS = 10
PLOT_UPDATE_HZ = 30

Y_MIN = 0
Y_MAX = 4095

DEFAULT_FILENAME = "emg_recording.csv"

ser = serial.Serial(PORT, BAUD, timeout=0.02)
time.sleep(2)
ser.reset_input_buffer()

times = deque()
values = deque()
recorded_rows = []

start = time.time()
last_plot = 0
latest_value = None

running = True
paused = False
filename = DEFAULT_FILENAME

plt.ion()
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.28)

line, = ax.plot([], [], linewidth=1)

ax.set_title("Live EMG / STM32 Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("ADC Value")
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlim(0, WINDOW_SECONDS)
ax.grid(True)

status_text = ax.text(
    0.02,
    0.95,
    "ADC: ---",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=16
)

file_text = ax.text(
    0.02,
    0.89,
    f"File: {filename}",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=11
)

state_text = ax.text(
    0.02,
    0.84,
    "Recording",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=11
)

# GUI controls
pause_ax = plt.axes([0.10, 0.08, 0.18, 0.075])
stop_ax = plt.axes([0.72, 0.08, 0.18, 0.075])
filename_ax = plt.axes([0.33, 0.09, 0.34, 0.05])

pause_button = Button(pause_ax, "Pause")
stop_button = Button(stop_ax, "Stop / Save")
filename_box = TextBox(filename_ax, "Filename: ", initial=DEFAULT_FILENAME)


def clean_filename(name):
    name = name.strip()

    if name == "":
        name = DEFAULT_FILENAME

    if not name.endswith(".csv"):
        name += ".csv"

    return name


def set_filename(text):
    global filename
    filename = clean_filename(text)
    file_text.set_text(f"File: {filename}")
    fig.canvas.draw_idle()


def toggle_pause(event=None):
    global paused
    paused = not paused

    if paused:
        pause_button.label.set_text("Resume")
        state_text.set_text("Paused")
    else:
        pause_button.label.set_text("Pause")
        state_text.set_text("Recording")

    fig.canvas.draw_idle()


def stop_and_save(event=None):
    global running, filename

    filename = clean_filename(filename_box.text)
    running = False

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "emg_value"])
        writer.writerows(recorded_rows)

    print(f"\nSaved {len(recorded_rows)} samples to {filename}")

    try:
        ser.close()
    except Exception:
        pass

    plt.close(fig)


def on_key(event):
    if event.key == " ":
        toggle_pause()
    elif event.key == "q":
        stop_and_save()


pause_button.on_clicked(toggle_pause)
stop_button.on_clicked(stop_and_save)
filename_box.on_submit(set_filename)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show(block=False)

print("Live EMG recording GUI running.")
print("Space = pause/resume")
print("q = stop/save")
print("Use filename box before pressing Stop / Save.")

try:
    while running:
        if not paused:
            raw = ser.readline().decode(errors="ignore").strip()

            if raw != "":
                try:
                    latest_value = int(raw)
                except ValueError:
                    pass

            # Drain serial backlog and keep newest value
            while ser.in_waiting > 0:
                raw = ser.readline().decode(errors="ignore").strip()

                if raw == "":
                    continue

                try:
                    latest_value = int(raw)
                except ValueError:
                    continue

            if latest_value is not None:
                t = time.time() - start

                times.append(t)
                values.append(latest_value)
                recorded_rows.append([t, latest_value])

                while times and times[0] < t - WINDOW_SECONDS:
                    times.popleft()
                    values.popleft()

        now = time.time()

        if latest_value is not None and now - last_plot >= 1 / PLOT_UPDATE_HZ:
            current_t = time.time() - start

            line.set_data(list(times), list(values))
            ax.set_xlim(max(0, current_t - WINDOW_SECONDS), current_t + 0.1)
            ax.set_ylim(Y_MIN, Y_MAX)

            status_text.set_text(f"ADC: {latest_value}")

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            last_plot = now

        plt.pause(0.001)

except KeyboardInterrupt:
    stop_and_save()

finally:
    if ser.is_open:
        ser.close()