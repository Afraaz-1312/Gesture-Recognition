import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json

DATA_DIR = "gesture_data"
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 126  # 2 hands * 63 features each

# --- Load data ---
labels = sorted(os.listdir(DATA_DIR))  # e.g. ['Bye', 'Hello', 'Help', 'None']
label_to_index = {label: idx for idx, label in enumerate(labels)}
print("Label mapping:", label_to_index)

X = []
y = []

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            sequence = np.load(os.path.join(folder, filename))
            if sequence.shape == (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                X.append(sequence)
                y.append(label_to_index[label])
            else:
                print(f"Skipping malformed file: {folder}/{filename}, shape={sequence.shape}")

X = np.array(X)
y = np.array(y)

print(f"Final dataset shape: X={X.shape}, y={y.shape}")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Class weights (compensate for imbalance) ---
class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class weights:", class_weights)

# --- Model ---
num_classes = len(labels)

model = keras.Sequential([
    keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=8,
    class_weight=class_weights,
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nFinal test accuracy: {test_acc:.2%}")

# --- Save model + label mapping ---
model.save("gesture_model.keras")
with open("label_mapping.json", "w") as f:
    json.dump(label_to_index, f)

print("Model saved to gesture_model.keras")
print("Label mapping saved to label_mapping.json")

from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test).argmax(axis=1)
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print("Labels order:", labels)
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))