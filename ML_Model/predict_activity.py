import argparse, json, os
import numpy as np
import pandas as pd

# Arguments

p = argparse.ArgumentParser()
p.add_argument("--csv",   required=True, type=str, help="New CSV to run inference on")
p.add_argument("--stats", required=True, type=str, help="norm_stats.json from training")
p.add_argument("--model", type=str, default=None, help="Keras model, e.g. activity_model.keras")
p.add_argument("--tflite", type=str, default=None, help="TFLite model, e.g. activity_model.tflite")
p.add_argument("--sr",     type=float, default=50.0, help="Sample rate Hz (used for windows)")
p.add_argument("--win",    type=float, default=2.0,  help="Window length in seconds")
p.add_argument("--stride", type=float, default=1.0,  help="Step in seconds between windows")
args = p.parse_args()

if (args.model is None) and (args.tflite is None):
    raise SystemExit("Specify either --model (Keras) or --tflite (TFLite).")

# Load normalization stats + metadata

with open(args.stats, "r", encoding="utf-8") as f:
    meta = json.load(f)
mean = np.array(meta["mean"], dtype=np.float32).reshape(1,1,-1)
std  = np.array(meta["std"],  dtype=np.float32).reshape(1,1,-1)
features = meta["features"]
labels = meta["labels"]
trained_window = int(meta.get("window", round(args.sr * args.win)))

# Ensure runtime window matches training window
WINDOW = trained_window
STRIDE = int(round(args.sr * args.stride))

print("Features (expected):", features)
print("Labels (ordinal):", labels)
print("WINDOW:", WINDOW, "STRIDE:", STRIDE)

# Read new CSV

if not os.path.exists(args.csv):
    raise FileNotFoundError(f"Missing: {args.csv}")
df = pd.read_csv(args.csv)

# Drop irrelevant / create exact feature matrix in correct order
missing = [c for c in features if c not in df.columns]
if missing:
    raise SystemExit(f"Missing feature columns in {args.csv}: {missing}")

X_all = df[features].astype(np.float32).values

# (Optional) Time column for output – tries to find a timestamp column
time_col = None
for cand in ["time", "Time", "timestamp", "Timestamp", "time_s"]:
    if cand in df.columns:
        time_col = cand
        break
time_vals = df[time_col].values if time_col else None

# Build windows (over the file)

Xs = []
starts = []  # row index for window start
times_start = []
n = len(X_all)
for start in range(0, n - WINDOW + 1, STRIDE):
    end = start + WINDOW
    Xs.append(X_all[start:end])
    starts.append(start)
    if time_vals is not None:
        times_start.append(time_vals[start])

if not Xs:
    raise SystemExit("No windows created. Check win/stride/sample-rate.")

X = np.stack(Xs, axis=0)  # (n_win, WINDOW, n_feat)

# Normalize with training statistics

Xn = (X - mean) / std

# Load model and predict

probs = None
pred_ids = None

if args.model:
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    probs = model.predict(Xn, verbose=0)
    pred_ids = probs.argmax(axis=1)

elif args.tflite:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    probs = np.zeros((len(Xn), len(labels)), dtype=np.float32)
    for i, x in enumerate(Xn):
        x_in = x[np.newaxis, ...].astype(np.float32)  # (1, W, C)
        interpreter.set_tensor(inp["index"], x_in)
        interpreter.invoke()
        probs[i] = interpreter.get_tensor(out["index"])[0]
    pred_ids = probs.argmax(axis=1)

pred_labels = [labels[i] for i in pred_ids]

# Save predictions.csv

out = pd.DataFrame({
    "window_idx": np.arange(len(pred_labels)),
    "start_row": starts,
    "pred_label": pred_labels
})

# add class probabilities
for j, lab in enumerate(labels):
    out[f"p_{lab}"] = probs[:, j]

# add start time if we found a time column
if time_vals is not None:
    out["start_time"] = times_start

out.to_csv("predictions.csv", index=False)
print("Saved predictions.csv (first rows):")
print(out.head())

# Quick summary + majority class

from collections import Counter
cnt = Counter(pred_labels)
print("\nPredictions per class:", dict(cnt))
majority_label = cnt.most_common(1)[0][0]
print("Majority class for the entire file:", majority_label)
print("✅ Done.")
