import numpy as np
from src.config import (
    TRAIN_DIR, IMG_SIZE, MAX_IMAGES_PER_CLASS, SEED,
    CONV_FILTERS, CONV2_FILTERS, DENSE_UNITS, LEARNING_RATE,
    BATCH_SIZE, EPOCHS, VALID_SIZE, EARLY_STOPPING, PATIENCE,
    METRIC_THRESHOLD, MODEL_DIR, MODEL_NAME
)
from src.loader import load_images_from_folder
from src.split import train_valid_split
from src.cnn_model import build_cnn_model
from src.metrics import compute_classification_metrics, find_best_threshold
import tensorflow as tf
import os

# 1. Load Data
print("Loading images...")
X, y = load_images_from_folder(
    TRAIN_DIR,
    img_size=IMG_SIZE,
    max_images_per_class=MAX_IMAGES_PER_CLASS,
    shuffle=True,
    seed=SEED
)
print(f"Loaded {len(X)} images.")

# 2. Split Data
print("Splitting into train and validation...")
x_train, x_valid, y_train, y_valid = train_valid_split(
    X, y, valid_size=VALID_SIZE, seed=SEED, shuffle=True
)
print(f"Train shape: {x_train.shape}, Valid shape: {x_valid.shape}")

# 3. Build Model
print("Building model...")
model = build_cnn_model(
    img_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    conv_filters=CONV_FILTERS,
    conv2_filters=CONV2_FILTERS,
    dense_units=DENSE_UNITS,
    learning_rate=LEARNING_RATE
)

# 4. Train Model
EarlyStopping = tf.keras.callbacks.EarlyStopping

callbacks = []
if EARLY_STOPPING:
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    ))

print("Training model...")
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_valid, y_valid),
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# 5. Evaluate Model
print("Evaluating on validation set...")
y_pred_probs = model.predict(x_valid).flatten()

# Find best threshold for F1
best_thresh, best_f1 = find_best_threshold(y_valid, y_pred_probs, metric='f1')
print(f"Best threshold by F1: {best_thresh:.3f} (F1={best_f1:.3f})")

# Metrics at default threshold
metrics_default = compute_classification_metrics(y_valid, y_pred_probs, threshold=METRIC_THRESHOLD)
# Metrics at best threshold
metrics_best = compute_classification_metrics(y_valid, y_pred_probs, threshold=best_thresh)

print("\nMetrics @ Default Threshold (0.5):")
for k, v in metrics_default.items():
    print(f"{k.capitalize()}: {v:.4f}" if isinstance(v, float) else f"{k.capitalize()}: {v}")

print("\nMetrics @ Best Threshold:")
for k, v in metrics_best.items():
    print(f"{k.capitalize()}: {v:.4f}" if isinstance(v, float) else f"{k.capitalize()}: {v}")

# 6. Save Model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.h5")
model.save(model_path)
print(f"\nModel saved to: {model_path}")

# 7. Optionally: Save threshold and metrics
import json
with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}_threshold.json"), "w") as f:
    json.dump({'best_threshold': float(best_thresh)}, f)
print(f"Best threshold saved.")

# (Optional) Run parameter search
# from src.eval_configs import grid_search
# results = grid_search(X, y, model_param_grid, train_param_grid, n_seeds=2)
