# train_har_windows.py (fixed)
import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

p = argparse.ArgumentParser()
p.add_argument("--jump",   type=str, default="edgeml_jump.csv")
p.add_argument("--run",    type=str, default="edgeml_run.csv")
p.add_argument("--still",  type=str, default="edgeml_still.csv")
p.add_argument("--sr",     type=float, default=50.0, help="Sample rate Hz")
p.add_argument("--win",    type=float, default=2.0,  help="Window length (sec)")
p.add_argument("--stride", type=float, default=1.0,  help="Stride (sec)")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--bs",     type=int, default=64)
p.add_argument("--no-plot", action="store_true")
args = p.parse_args()

# --- helpers ---
def load_and_tag(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saknas: {path}")
    df = pd.read_csv(path)

    # Exkludera uppenbara icke-features
    exclude = {
        "label", "Label", "activity", "Activity", "ActivityLabel",
        "label_activity_sample", "time", "Time", "timestamp", "Timestamp"
    }
    # Endast numeriska features (ej tidskolumner)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in exclude and not str(c).lower().startswith("time")]

    if len(feat_cols) == 0:
        raise ValueError(f"Inga numeriska feature-kolumner i {path}")

    # Släng rader som har NaN i features (för säkerhets skull)
    df = df.dropna(subset=feat_cols).copy()
    df["label"] = label
    return df, feat_cols

df_jump,  feat_cols_j = load_and_tag(args.jump,  "jump")
df_run,   feat_cols_r = load_and_tag(args.run,   "run")
df_still, feat_cols_s = load_and_tag(args.still, "still")

# Gemensamma features i samma ordning som jump-filen
feat_cols = [c for c in feat_cols_j if c in feat_cols_r and c in feat_cols_s]
if len(feat_cols) == 0:
    raise ValueError("Hittade inga gemensamma feature-kolumner i alla tre filer.")

print("Feature-kolumner:", feat_cols)

WINDOW = int(round(args.sr * args.win))
STRIDE = int(round(args.sr * args.stride))

def windows_from_df(df, label):
    Xs, ys = [], []          # <-- FIX
    data = df[feat_cols].values.astype(np.float32)
    n = len(data)
    for start in range(0, n - WINDOW + 1, STRIDE):
        end = start + WINDOW
        Xs.append(data[start:end])
        ys.append(label)
    return np.array(Xs), np.array(ys)

Xj, yj = windows_from_df(df_jump,  "jump")
Xr, yr = windows_from_df(df_run,   "run")
Xs, ys = windows_from_df(df_still, "still")

X = np.concatenate([Xj, Xr, Xs], axis=0)
y = np.concatenate([yj, yr, ys], axis=0)

print("Antal fönster:", {lbl:int((y==lbl).sum()) for lbl in np.unique(y)})
print("X shape:", X.shape)

# --- split & normalisering (fit på train) ---
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

mean = X_train.mean(axis=(0,1), keepdims=True)
std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std

# --- class weights ---
classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
print("Class weights:", {le.inverse_transform([k])[0]: v for k,v in class_weight.items()})

# --- model ---
inputs = layers.Input(shape=(WINDOW, len(feat_cols)))
x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(le.classes_), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

cbs = [
    callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5, monitor="val_accuracy"),
    callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy")
]

hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.bs,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=1
)

# --- utvärdera ---
probs = model.predict(X_val, verbose=0)
preds = probs.argmax(axis=1)

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification report:")
print(classification_report(y_val, preds, target_names=le.classes_))
cm = confusion_matrix(y_val, preds)
print("Confusion matrix:\n", cm)

# --- spara ---
model.save("activity_model.keras")
with open("norm_stats.json", "w") as f:
    json.dump({"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist(),
               "features": feat_cols, "labels": le.classes_.tolist(),
               "window": WINDOW}, f, indent=2)

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    open("activity_model.tflite", "wb").write(converter.convert())
    print("Sparade TFLite: activity_model.tflite")
except Exception as e:
    print("Kunde inte exportera TFLite:", e)

print("✅ Klar.")
