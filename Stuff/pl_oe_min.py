#Kör med python pl_oe_min.py --infile filnamn.oe

import struct, os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_oe_acc(path: str) -> pd.DataFrame:
    """
    Läser acc.x/y/z från OpenEarable .oe (IMU, SID=0, 9 floats/pkt).
    Returnerar DataFrame med index = timestamp (s) och kolumner acc.x/y/z.
    """
    FILE_HDR_FMT = "<HQ"   # uint16 version, uint64 epoch(us)
    PKT_HDR_FMT  = "<BBQ"  # sid(uint8), size(uint8), t_us(uint64)
    ACC_FMT      = "<9f"   # 9 float32: acc(x,y,z), gyro(x,y,z), mag(x,y,z)

    times, acc = [], []

    with open(path, "rb") as f:
        hdr = f.read(struct.calcsize(FILE_HDR_FMT))
        if len(hdr) != struct.calcsize(FILE_HDR_FMT):
            raise ValueError("Ogiltigt .oe-huvud")
        version, epoch = struct.unpack(FILE_HDR_FMT, hdr)

        while True:
            h = f.read(struct.calcsize(PKT_HDR_FMT))
            if len(h) < struct.calcsize(PKT_HDR_FMT):
                break
            sid, size, t_us = struct.unpack(PKT_HDR_FMT, h)
            if size > 192 or sid > 7:
                break
            payload = f.read(size)
            if len(payload) < size:
                break

            if sid == 0 and size == struct.calcsize(ACC_FMT):
                v = struct.unpack(ACC_FMT, payload)
                ax, ay, az = float(v[0]), float(v[1]), float(v[2])
                times.append(t_us / 1e6)  # sekunder
                acc.append((ax, ay, az))

    if not times:
        return pd.DataFrame(columns=["acc.x", "acc.y", "acc.z"])

    df = pd.DataFrame(acc, columns=["acc.x", "acc.y", "acc.z"])
    df["timestamp"] = times
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df

def compute_player_load(df: pd.DataFrame):
    dx, dy, dz = df["acc.x"].diff(), df["acc.y"].diff(), df["acc.z"].diff()
    pl = np.sqrt(dx**2 + dy**2 + dz**2).fillna(0.0)
    return pl, float(pl.sum())

def main():
    ap = argparse.ArgumentParser(description="Compute & plot Player Load from .oe")
    ap.add_argument("--infile", required=True, help="Path to .oe file")
    args = ap.parse_args()

    df = read_oe_acc(args.infile)
    if df.empty:
        print("❌ Hittade ingen IMU-acc i filen.")
        return

    pl_series, pl_total = compute_player_load(df)
    duration = float(df.index[-1] - df.index[0]) if len(df) > 1 else 0.0

    print(f"Fil: {os.path.basename(args.infile)}")
    print(f"Total Player Load : {pl_total:.3f}")
    if duration > 0:
        print(f"Duration (s)      : {duration:.2f}")
        print(f"PL per second     : {pl_total/duration:.3f}")
        print(f"PL per minute     : {pl_total/(duration/60):.3f}")

    # Plot
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(df.index, pl_series)
    plt.title("Player Load per tidssteg")
    plt.xlabel("Tid (s)"); plt.ylabel("PL"); plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(df.index, pl_series.cumsum())
    plt.title("Ackumulerad Player Load")
    plt.xlabel("Tid (s)"); plt.ylabel("Total PL"); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

