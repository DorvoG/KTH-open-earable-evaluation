import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


# Read files
df_jump = pd.read_csv("edgeml_jump.csv")
df_run = pd.read_csv("edgeml_run.csv")
df_still = pd.read_csv("edgeml_still.csv")

df_jump["label"] = "jump"
df_run["label"] = "run"
df_still["label"] = "still"

df_all = pd.concat([df_jump, df_run, df_still], ignore_index=True)

print("Data shape:", df_all.shape)
print("Labels:", df_all["label"].value_counts())

# Create window (2 sec = 200 sampel at 50 Hz)

SAMPLE_RATE = 50
WINDOW_SEC = 2
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC

feature_cols = [c for c in df_all.columns if c not in ["label", "time"]]

def create_windows(df, window_size):
    X, y = [], []
    data = df[feature_cols].values
    labels = df["label"].values
    num_windows = len(df) // window_size
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        if end <= len(df):
            X.append(data[start:end])
            y.append(labels[start:end][0])
    return np.array(X), np.array(y)

X, y = create_windows(df_all, WINDOW_SIZE)

print("Windows:", X.shape, y.shape)

# Label encoding

le = LabelEncoder()
y_enc = le.fit_transform(y)

# 4. Train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

# --------------------------
# Build 1D-CNN-model
# --------------------------
model = models.Sequential([
    layers.Conv1D(32, 5, activation="relu", input_shape=(WINDOW_SIZE, len(feature_cols))),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

# Train

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# Evaluate test data
test_probs = model.predict(X_test)
y_pred = np.argmax(test_probs, axis=1)

# Text-report (precision/recall/F1)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Matrix (numbers)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (råa tal):\n", cm)

# Plot and save picture
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
plt.title("Confusion Matrix (val/test)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# 7. Save model 

model.save("activity_model.h5")

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("activity_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Träning klar. Modell sparad som activity_model.h5 och activity_model.tflite")
