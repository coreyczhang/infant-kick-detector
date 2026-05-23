"""
Selective Kick Detector for Infant CP Device
=============================================
Reads dual-ankle accelerometer data (left + right leg), detects kick events,
classifies them as selective (one leg) or non-selective (both legs within 200ms),
logs each event to CSV, and plots live accelerometer traces.

Hardware: Two Seeed XIAO nRF52840 Sense boards, each with built-in LSM6DS3TRC IMU.
Each XIAO streams "X: val  Y: val  Z: val m/s²" over USB serial at 100Hz.
Values are in m/s² — converted to g internally (divide by 9.81).

When no hardware is connected, auto-falls back to simulated data.

Usage:
    python kick_detector.py                          # simulate
    python kick_detector.py --sim                    # force simulation
    python kick_detector.py --left /dev/tty.usbmodem101 --right /dev/tty.usbmodem102
    python kick_detector.py --left /dev/tty.usbmodem101 --sim-right  # one real, one sim
    python kick_detector.py --threshold 1.2          # custom kick threshold in g
"""

import argparse
import csv
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ   = 100          # accelerometer sample rate
WINDOW_MS        = 200          # simultaneity window in milliseconds
WINDOW_SAMPLES   = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)  # 20 samples
PLOT_WINDOW_SEC  = 5            # seconds of history shown in live plot
PLOT_SAMPLES     = SAMPLE_RATE_HZ * PLOT_WINDOW_SEC
REFRACTORY_MS    = 300          # minimum ms between kicks on the same leg
REFRACTORY_SAMP  = int(SAMPLE_RATE_HZ * REFRACTORY_MS / 1000)
CSV_DIR          = Path("kick_logs")


# ---------------------------------------------------------------------------
# Simulated Data Source
# ---------------------------------------------------------------------------

class SimulatedSensor:
    """
    Generates realistic infant kicking patterns for one leg.

    Model:
      - Resting gravity component ~1g on z-axis
      - Background noise ~0.05g std on each axis
      - Kicks: Poisson-distributed arrivals at `kick_rate` Hz, each producing
        a sharp Gaussian acceleration burst (~1.5–3g peak) lasting ~150ms.
    """

    def __init__(self, kick_rate=0.8, seed=None):
        self.rng = np.random.default_rng(seed)
        self.kick_rate = kick_rate          # average kicks per second
        self._t = 0.0                       # current time in seconds
        self._dt = 1.0 / SAMPLE_RATE_HZ
        self._kick_queue: list[tuple[float, float]] = []  # (start_time, peak_g)

    def _schedule_kick(self):
        """Randomly schedule the next kick using exponential inter-arrival times."""
        interval = self.rng.exponential(1.0 / self.kick_rate)
        peak = self.rng.uniform(1.5, 3.0)
        self._kick_queue.append((self._t + interval, peak))

    def read(self) -> tuple[float, float, float]:
        """Return one (x, y, z) acceleration sample in g-forces."""
        if not self._kick_queue:
            self._schedule_kick()

        # background noise + resting gravity on z
        x = self.rng.normal(0.0, 0.05)
        y = self.rng.normal(0.0, 0.05)
        z = self.rng.normal(1.0, 0.05)

        # add any active kick bursts
        kicks_to_remove = []
        for i, (t_start, peak) in enumerate(self._kick_queue):
            if self._t >= t_start:
                # Gaussian burst lasting ~150ms (sigma = 0.05s)
                age = self._t - t_start
                burst = peak * np.exp(-0.5 * (age / 0.05) ** 2)
                z += burst
                # retire kick after 5-sigma
                if age > 5 * 0.05:
                    kicks_to_remove.append(i)
                    self._schedule_kick()

        for i in reversed(kicks_to_remove):
            self._kick_queue.pop(i)

        self._t += self._dt
        return float(x), float(y), float(z)


class SynchronizedSimulation:
    """
    Drives two SimulatedSensors, injecting occasional synchronized (bilateral)
    kicks to create a realistic mix of selective and non-selective events.
    """

    def __init__(self, selective_ratio=0.65):
        # Left kicks slightly more often to reflect asymmetric CP presentation
        self._left  = SimulatedSensor(kick_rate=0.7, seed=42)
        self._right = SimulatedSensor(kick_rate=0.6, seed=99)
        self._selective_ratio = selective_ratio
        self._rng = np.random.default_rng(7)
        self._bilateral_offset = 0  # samples remaining until bilateral kick ends

    def read(self) -> tuple[tuple, tuple]:
        """Return ((lx,ly,lz), (rx,ry,rz)) for one time step."""
        # Occasionally force bilateral synchrony to produce non-selective events
        if self._bilateral_offset > 0:
            self._bilateral_offset -= 1
        elif self._rng.random() < (1 - self._selective_ratio) / SAMPLE_RATE_HZ:
            # Inject a bilateral kick: temporarily align both kick queues
            shared_peak = self._rng.uniform(1.5, 2.5)
            self._left._kick_queue.insert(0,  (self._left._t,  shared_peak))
            self._right._kick_queue.insert(0, (self._right._t, shared_peak))
            self._bilateral_offset = REFRACTORY_SAMP

        return self._left.read(), self._right.read()


# ---------------------------------------------------------------------------
# Serial Data Source (reads from XIAO nRF52840 Sense over USB serial)
# ---------------------------------------------------------------------------

MS2_TO_G = 1.0 / 9.81  # convert m/s² to g-forces

class XIAOSerialSensor:
    """
    Reads one XIAO nRF52840 Sense over USB serial.

    Expected line format (from code.py on the XIAO):
        X: 3.93  Y: 6.47  Z: 6.71 m/s²

    Values are in m/s² and are converted to g internally.
    """

    def __init__(self, port: str, label: str = ""):
        import serial  # type: ignore
        self._ser = serial.Serial(port, baudrate=115200, timeout=1.0)
        self._label = label
        print(f"Opened {label} sensor on: {port}")

    def read(self) -> tuple[float, float, float]:
        """Return (x, y, z) in g-forces."""
        while True:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line.startswith("X:"):
                continue
            try:
                # parse "X: 3.93  Y: 6.47  Z: 6.71 m/s²"
                parts = line.replace("m/s²", "").split()
                x = float(parts[1]) * MS2_TO_G
                y = float(parts[3]) * MS2_TO_G
                z = float(parts[5]) * MS2_TO_G
                return x, y, z
            except (IndexError, ValueError):
                continue

    def close(self):
        self._ser.close()


class DualXIAOSource:
    """Wraps two XIAOSerialSensor instances into a dual-leg source."""

    def __init__(self, left_port: str, right_port: str):
        self._left  = XIAOSerialSensor(left_port,  "LEFT")
        self._right = XIAOSerialSensor(right_port, "RIGHT")

    def read(self) -> tuple[tuple, tuple]:
        return self._left.read(), self._right.read()

    def close(self):
        self._left.close()
        self._right.close()


class SingleXIAOSource:
    """
    Uses one real XIAO for one leg and simulation for the other.
    Useful for testing with only one sensor connected.
    """

    def __init__(self, real_port: str, real_leg: str = "left"):
        self._real   = XIAOSerialSensor(real_port, real_leg.upper())
        self._sim    = SimulatedSensor(kick_rate=0.6, seed=99)
        self._real_leg = real_leg

    def read(self) -> tuple[tuple, tuple]:
        real_xyz = self._real.read()
        sim_xyz  = self._sim.read()
        if self._real_leg == "left":
            return real_xyz, sim_xyz
        return sim_xyz, real_xyz


# ---------------------------------------------------------------------------
# BLE Data Source (wireless, uses bleak)
# ---------------------------------------------------------------------------

class BLESource:
    """
    Receives IMU data from one or two XIAO nRF52840 Sense boards over BLE.
    Uses Nordic UART Service (NUS) to receive x,y,z in m/s².
    If only XIAO-LEFT is found, simulates the right leg.
    """

    UART_RX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

    def __init__(self):
        import asyncio
        from bleak import BleakScanner, BleakClient  # type: ignore
        self._asyncio = asyncio
        self._BleakScanner = BleakScanner
        self._BleakClient  = BleakClient
        self._left_xyz  = (0.0, 0.0, 1.0)
        self._right_xyz = (0.0, 0.0, 1.0)
        self._sim = SimulatedSensor(kick_rate=0.6, seed=99)
        self._has_right = False
        self._loop = asyncio.new_event_loop()
        self._left_addr  = None
        self._right_addr = None
        self._left_client  = None
        self._right_client = None
        self._loop.run_until_complete(self._connect())

    def _parse(self, data: bytearray):
        try:
            x, y, z = data.decode().strip().split(",")
            return float(x) * MS2_TO_G, float(y) * MS2_TO_G, float(z) * MS2_TO_G
        except Exception:
            return None

    def _left_handler(self, sender, data):
        parsed = self._parse(data)
        if parsed:
            self._left_xyz = parsed

    def _right_handler(self, sender, data):
        parsed = self._parse(data)
        if parsed:
            self._right_xyz = parsed

    async def _connect(self):
        print("Scanning for XIAO sensors (10s)...")
        devices = await self._BleakScanner.discover(timeout=10.0)
        left_dev = right_dev = None
        for d in devices:
            if d.name == "XIAO-LEFT":
                left_dev = d
                print(f"Found XIAO-LEFT: {d.address}")
            elif d.name == "XIAO-RIGHT":
                right_dev = d
                print(f"Found XIAO-RIGHT: {d.address}")

        if not left_dev:
            raise RuntimeError("XIAO-LEFT not found — make sure it's powered on and advertising")

        self._left_client = self._BleakClient(left_dev.address)
        await self._left_client.connect()
        await self._left_client.start_notify(self.UART_RX_UUID, self._left_handler)
        print("Connected to XIAO-LEFT ✅")

        if right_dev:
            self._right_client = self._BleakClient(right_dev.address)
            await self._right_client.connect()
            await self._right_client.start_notify(self.UART_RX_UUID, self._right_handler)
            self._has_right = True
            print("Connected to XIAO-RIGHT ✅")
        else:
            print("XIAO-RIGHT not found — simulating right leg")

    def read(self) -> tuple[tuple, tuple]:
        # pump BLE events
        self._loop.run_until_complete(self._asyncio.sleep(0.01))
        right = self._right_xyz if self._has_right else self._sim.read()
        return self._left_xyz, right

    def close(self):
        async def _disconnect():
            if self._left_client:
                await self._left_client.disconnect()
            if self._right_client:
                await self._right_client.disconnect()
        self._loop.run_until_complete(_disconnect())


# ---------------------------------------------------------------------------
# Kick Detector
# ---------------------------------------------------------------------------

class KickDetector:
    """
    Detects kick events on a single leg using a magnitude threshold.

    A kick is triggered when the acceleration magnitude exceeds `threshold`
    and the leg has not kicked within the refractory period.
    """

    def __init__(self, label: str, threshold: float):
        self.label = label
        self.threshold = threshold
        self._since_last_kick = REFRACTORY_SAMP  # start ready
        self.last_kick_time: float | None = None
        self.last_magnitude: float = 0.0

    def update(self, x: float, y: float, z: float) -> bool:
        """Feed one sample; returns True if a new kick is detected."""
        mag = float(np.sqrt(x**2 + y**2 + z**2))
        self.last_magnitude = mag
        self._since_last_kick += 1

        kicked = mag > self.threshold and self._since_last_kick >= REFRACTORY_SAMP
        if kicked:
            self._since_last_kick = 0
            self.last_kick_time = time.time()

        return kicked


# ---------------------------------------------------------------------------
# Selectivity Classifier
# ---------------------------------------------------------------------------

class SelectivityClassifier:
    """
    Classifies each kick as selective or non-selective using a 200ms window.

    Maintains a deque of recent kick timestamps per leg; a kick is non-selective
    if the other leg kicked within WINDOW_MS milliseconds.
    """

    def __init__(self):
        self._left_times:  deque[float] = deque()
        self._right_times: deque[float] = deque()

    def _prune(self, q: deque, now: float):
        cutoff = now - WINDOW_MS / 1000.0
        while q and q[0] < cutoff:
            q.popleft()

    def record(self, leg: str, t: float) -> bool:
        """
        Record a kick on `leg` at time `t`.
        Returns True if the kick is selective (the other leg did NOT kick
        within the simultaneity window).
        """
        now = t
        self._prune(self._left_times,  now)
        self._prune(self._right_times, now)

        if leg == "left":
            other = self._right_times
            self._left_times.append(t)
        else:
            other = self._left_times
            self._right_times.append(t)

        return len(other) == 0  # selective if other leg has no recent kicks


# ---------------------------------------------------------------------------
# Session Logger
# ---------------------------------------------------------------------------

class SessionLogger:
    """Writes each kick event to a CSV file with timestamp, leg, selective, magnitude."""

    def __init__(self):
        CSV_DIR.mkdir(exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = CSV_DIR / f"session_{session_id}.csv"
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "leg", "selective", "magnitude_g"])
        print(f"Logging to: {self._path}")

    def log(self, leg: str, selective: bool, magnitude: float):
        ts = datetime.now().isoformat(timespec="milliseconds")
        self._writer.writerow([ts, leg, selective, f"{magnitude:.4f}"])
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Live Plotter
# ---------------------------------------------------------------------------

class LivePlotter:
    """
    Scrolling real-time plot showing acceleration magnitude for both legs
    with kick markers overlaid.
    """

    def __init__(self, threshold: float):
        self._threshold = threshold
        self._left_mag:  deque[float] = deque([0.0] * PLOT_SAMPLES, maxlen=PLOT_SAMPLES)
        self._right_mag: deque[float] = deque([0.0] * PLOT_SAMPLES, maxlen=PLOT_SAMPLES)
        self._left_kicks:  list[int] = []   # sample indices of left kicks
        self._right_kicks: list[int] = []
        self._sample_count = 0
        self._lock = threading.Lock()

        self._fig, (self._ax_l, self._ax_r) = plt.subplots(
            2, 1, figsize=(10, 5), sharex=True
        )
        self._fig.suptitle(
            "Infant Selective Kick Monitor", fontsize=13, fontweight="bold"
        )
        self._setup_axes()

    def _setup_axes(self):
        for ax, label, color in [
            (self._ax_l, "Left Leg",  "#2196F3"),
            (self._ax_r, "Right Leg", "#FF5722"),
        ]:
            ax.set_ylabel("|accel| (g)")
            ax.set_title(label, color=color, fontsize=10)
            ax.axhline(self._threshold, color="gray", linestyle="--",
                       linewidth=0.8, label=f"Threshold ({self._threshold}g)")
            ax.set_ylim(0, 4)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        self._ax_r.set_xlabel("Time (samples)")

    def push(self, l_mag: float, r_mag: float,
             l_kicked: bool, r_kicked: bool):
        with self._lock:
            self._left_mag.append(l_mag)
            self._right_mag.append(r_mag)
            if l_kicked:
                self._left_kicks.append(self._sample_count % PLOT_SAMPLES)
            if r_kicked:
                self._right_kicks.append(self._sample_count % PLOT_SAMPLES)
            self._sample_count += 1

    def _animate(self, _frame):
        with self._lock:
            t = np.arange(PLOT_SAMPLES)
            for ax, mag, kicks, color in [
                (self._ax_l, list(self._left_mag),  self._left_kicks,  "#2196F3"),
                (self._ax_r, list(self._right_mag), self._right_kicks, "#FF5722"),
            ]:
                ax.clear()
                ax.plot(t, mag, color=color, linewidth=0.8)
                ax.axhline(self._threshold, color="gray", linestyle="--",
                           linewidth=0.8, label=f"Threshold ({self._threshold}g)")
                for k in kicks:
                    ax.axvline(k, color="green", alpha=0.5, linewidth=1.0)
                ax.set_ylim(0, 4)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
        self._ax_l.set_title("Left Leg",  color="#2196F3", fontsize=10)
        self._ax_r.set_title("Right Leg", color="#FF5722", fontsize=10)
        self._ax_r.set_xlabel("Time (samples, last 5s)")
        self._ax_l.set_ylabel("|accel| (g)")
        self._ax_r.set_ylabel("|accel| (g)")
        # Trim old kick markers outside the visible window
        self._left_kicks  = [k for k in self._left_kicks  if k >= 0][-20:]
        self._right_kicks = [k for k in self._right_kicks if k >= 0][-20:]

    def start(self):
        self._ani = animation.FuncAnimation(
            self._fig, self._animate, interval=100, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def flush(self):
        plt.pause(0.001)


# ---------------------------------------------------------------------------
# Main acquisition loop
# ---------------------------------------------------------------------------

def run(source, threshold: float, no_plot: bool):
    classifier = SelectivityClassifier()
    logger     = SessionLogger()
    left_det   = KickDetector("left",  threshold)
    right_det  = KickDetector("right", threshold)
    plotter    = None if no_plot else LivePlotter(threshold)

    if plotter:
        plotter.start()

    # Session-level counters for the summary line
    counts = {"left": 0, "right": 0, "both": 0,
              "selective": 0, "non_selective": 0}

    print(f"\n{'='*60}")
    print(f"  Selective Kick Monitor  |  threshold={threshold}g")
    print(f"  Simultaneity window: {WINDOW_MS}ms  |  100Hz sampling")
    print(f"{'='*60}")
    print(f"  {'TIME':12}  {'LEG':6}  {'SELECTIVE':10}  {'MAG (g)':8}")
    print(f"  {'-'*46}")

    try:
        while True:
            left_xyz, right_xyz = source.read()

            l_kicked = left_det.update(*left_xyz)
            r_kicked = right_det.update(*right_xyz)

            l_mag = left_det.last_magnitude
            r_mag = right_det.last_magnitude

            now = time.time()

            # Process detected kicks
            events: list[tuple[str, bool, float]] = []

            if l_kicked and r_kicked:
                # Both fired this sample — treat as a bilateral event
                sel_l = classifier.record("left",  now)
                sel_r = classifier.record("right", now)
                selective = sel_l and sel_r  # both must be uncontested
                events.append(("both", selective, max(l_mag, r_mag)))
                counts["both"] += 1
            else:
                if l_kicked:
                    selective = classifier.record("left", now)
                    events.append(("left", selective, l_mag))
                    counts["left"] += 1
                if r_kicked:
                    selective = classifier.record("right", now)
                    events.append(("right", selective, r_mag))
                    counts["right"] += 1

            for leg, selective, mag in events:
                if selective:
                    counts["selective"] += 1
                else:
                    counts["non_selective"] += 1

                tag  = "SELECTIVE    ✓" if selective else "non-selective  "
                ts   = datetime.now().strftime("%H:%M:%S.%f")[:12]
                leg_label = f"{leg:6}"
                print(f"  {ts}  {leg_label}  {tag}  {mag:.3f}g")
                logger.log(leg, selective, mag)

            if plotter:
                plotter.push(l_mag, r_mag, l_kicked, r_kicked)
                plotter.flush()
            else:
                # Sleep to match 100Hz when not plotting
                time.sleep(1.0 / SAMPLE_RATE_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
        total = counts["left"] + counts["right"] + counts["both"]
        print(f"\n{'='*60}")
        print(f"  Session summary")
        print(f"  Total kick events : {total}")
        print(f"  Left              : {counts['left']}")
        print(f"  Right             : {counts['right']}")
        print(f"  Bilateral         : {counts['both']}")
        print(f"  Selective         : {counts['selective']}")
        print(f"  Non-selective     : {counts['non_selective']}")
        if total:
            pct = 100 * counts["selective"] / total
            print(f"  Selectivity rate  : {pct:.1f}%")
        print(f"{'='*60}\n")
        if plotter:
            plt.close("all")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Infant selective kick detector")
    parser.add_argument("--left",      type=str, default=None,
                        help="Serial port for LEFT ankle XIAO (e.g. /dev/tty.usbmodem101)")
    parser.add_argument("--right",     type=str, default=None,
                        help="Serial port for RIGHT ankle XIAO")
    parser.add_argument("--sim-right", action="store_true",
                        help="Simulate right leg (use with --left)")
    parser.add_argument("--sim-left",  action="store_true",
                        help="Simulate left leg (use with --right)")
    parser.add_argument("--ble",       action="store_true",
                        help="Use Bluetooth BLE (XIAO-LEFT and XIAO-RIGHT)")
    parser.add_argument("--sim",       action="store_true",
                        help="Force full simulation (no hardware)")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="Kick detection threshold in g (default: 1.5)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Disable live matplotlib plot")
    args = parser.parse_args()

    if args.ble:
        print("Bluetooth BLE mode")
        source = BLESource()
    elif args.left and args.right:
        print(f"Dual XIAO mode: left={args.left}  right={args.right}")
        source = DualXIAOSource(args.left, args.right)
    elif args.left and args.sim_right:
        print(f"Single XIAO mode: left={args.left}, right=simulated")
        source = SingleXIAOSource(args.left, real_leg="left")
    elif args.right and args.sim_left:
        print(f"Single XIAO mode: right={args.right}, left=simulated")
        source = SingleXIAOSource(args.right, real_leg="right")
    elif args.left:
        print(f"Single XIAO mode: left={args.left}, right=simulated")
        source = SingleXIAOSource(args.left, real_leg="left")
    else:
        print("No hardware specified — using simulated infant kicking data.\n")
        source = SynchronizedSimulation()

    run(source, threshold=args.threshold, no_plot=args.no_plot)


if __name__ == "__main__":
    main()
