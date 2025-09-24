import struct
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

LABELS = {
    "imu": ['acc.x', 'acc.y', 'acc.z', 'gyro.x', 'gyro.y', 'gyro.z', 'mag.x', 'mag.y', 'mag.z'],
    "barometer": ['barometer.temperature', 'barometer.pressure'],
    "ppg": ['ppg.red', 'ppg.ir', 'ppg.green', 'ppg.ambient'],
    "bone_acc": ['bone_acc.x', 'bone_acc.y', 'bone_acc.z']
}

# --- Hjälpklass ---
class SensorAccessor:
    def __init__(self, df: pd.DataFrame, labels: list):
        self._df = df
        self._data = {}
        groups = defaultdict(list)

        for label in labels:
            parts = label.split('.')
            if len(parts) == 2:
                group, field = parts
                if label in df:
                    groups[group].append(label)
            else:
                if label in df:
                    self._data[label] = df[label]

        for group, columns in groups.items():
            short_names = [label.split('.')[1] for label in columns]
            subdf = df[columns].copy()
            subdf.columns = short_names
            self._data[group] = subdf

        self._full_df = pd.concat(self._data.values(), axis=1) if self._data else df

    def __getitem__(self, key):
        return self._data.get(key, None)

    def to_dataframe(self):
        return self._full_df

# --- Huvudklass för .oe ---
class SensorDataset:
    SENSOR_SID = {"imu": 0, "barometer": 1, "microphone": 2, "ppg": 4, "bone_acc": 7}

    def __init__(self, filename):
        self.filename = filename
        self.data = defaultdict(list)
        self.df = pd.DataFrame()
        self.parse()
        self._build_accessors()

    def parse(self):
        FILE_HEADER_FORMAT = '<HQ'
        FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FORMAT)
        with open(self.filename, 'rb') as f:
            header = f.read(FILE_HEADER_SIZE)
            if len(header) < FILE_HEADER_SIZE:
                raise ValueError("Not a valid .oe file")
            version, timestamp = struct.unpack(FILE_HEADER_FORMAT, header)

            while True:
                hdr = f.read(10)
                if len(hdr) < 10:
                    break
                sid, size, t_us = struct.unpack('<BBQ', hdr)
                if size > 192 or sid > 7:
                    break
                payload = f.read(size)
                if len(payload) < size:
                    break
                timestamp_s = t_us / 1e6

                if sid == 0:  # IMU
                    values = struct.unpack('<9f', payload)
                    self.data[sid].append((timestamp_s, values))

    def _build_accessors(self):
        if 0 in self.data and self.data[0]:
            times, values = zip(*self.data[0])
            df = pd.DataFrame(values, columns=LABELS["imu"])
            df["timestamp"] = times
            df.set_index("timestamp", inplace=True)
            self.df = df

    def get_dataframe(self):
        return self.df

# --- Player Load funktion ---
def compute_player_load(df: pd.DataFrame):
    dx = df["acc.x"].diff()
    dy = df["acc.y"].diff()
    dz = df["acc.z"].diff()
    pl_series = np.sqrt(dx**2 + dy**2 + dz**2).fillna(0)
    pl_total = pl_series.sum()
    return pl_series, pl_total

# --- Huvudkörning ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot Player Load from .oe file")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to .oe file")
    args = parser.parse_args()

    ds = SensorDataset(args.in_path)
    df = ds.get_dataframe()

    if df.empty:
        print("❌ No IMU data found in file")
        exit()

    pl_series, pl_total = compute_player_load(df)
    pl_cumulative = pl_series.cumsum()

    duration = df.index[-1] - df.index[0]

    print(f"File: {os.path.basename(args.in_path)}")
    print(f"Total Player Load : {pl_total:.2f}")
    print(f"Duration (s)      : {duration:.2f}")
    print(f"PL per second     : {pl_total/duration:.2f}")
    print(f"PL per minute     : {pl_total/(duration/60):.2f}")

    # --- Rita grafer ---
    plt.figure(figsize=(12,6))

    plt.subplot(2,1,1)
    plt.plot(df.index, pl_series, color="blue")
    plt.title("Player Load per tidssteg")
    plt.xlabel("Tid (s)")
    plt.ylabel("PL")

    plt.subplot(2,1,2)
    plt.plot(df.index, pl_cumulative, color="green")
    plt.title("Ackumulerad Player Load")
    plt.xlabel("Tid (s)")
    plt.ylabel("Total PL")

    plt.tight_layout()
    plt.show()
