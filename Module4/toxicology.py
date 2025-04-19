import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import warnings

# Suppress the specific deprecation warnings
warnings.filterwarnings('ignore', message='please use MorganGenerator')

# Set seeds for reproducibility
np.random.seed(456)
tf.random.set_seed(456)

# Step 1: Load the Tox21 Dataset
# If you want to use the recommended MorganGenerator directly (optional):
# featurizer = dc.feat.MorganFingerprint(radius=2, size=1024)
# loader = dc.data.CSVLoader(tasks=["NR-AR"], smiles_field="smiles", featurizer=featurizer)
# But for now, we'll stick with the standard loading and just suppress warnings:
_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Step 2: Remove extra datasets (focus on first task only)
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

# Hyperparameters - increased hidden layer size
d = 1024  # Input feature dimension
n_hidden = 100  # Increased from 50 to 100 neurons
learning_rate = 0.001
n_epochs = 10
batch_size = 100
dropout_rate = 0.3  # Slightly increased dropout to prevent overfitting

# Step 4-7: Define the model with TensorFlow 2 (Sequential API)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d,)),
    tf.keras.layers.Dense(n_hidden, activation='relu'),  # Single larger hidden layer with 100 neurons
    tf.keras.layers.Dropout(dropout_rate),  # Increased dropout rate
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Define class weights to address imbalance
# Using a more moderate weight for better precision-recall balance
class_weights = {0: 1.0, 1: 5.0}  # Weight toxic examples 5x instead of 10x

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.AUC(), 
             tf.keras.metrics.Precision(), 
             tf.keras.metrics.Recall()]
)

# Setup TensorBoard
log_dir = '/tmp/fcnet-tox21-tf2'
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1, 
    write_graph=True
)

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    mode='min',          # Stop when it starts increasing
    patience=2,          # Wait 2 epochs before stopping
    restore_best_weights=True  # Restore weights from best epoch
)

# Step 8: Implement mini-batch training with TensorFlow 2's fit method
history = model.fit(
    train_X, train_y,
    epochs=n_epochs,
    batch_size=batch_size,
    validation_data=(valid_X, valid_y),
    callbacks=[tensorboard_callback, early_stopping],  # Add callbacks
    class_weight=class_weights,  # Apply class weights
    verbose=1
)

# Make predictions
valid_y_pred = model.predict(valid_X)
valid_y_pred_binary = (valid_y_pred > 0.5).astype(int).flatten()

# Calculate evaluation metrics with default threshold (0.5)
accuracy = accuracy_score(valid_y, valid_y_pred_binary)
auc = roc_auc_score(valid_y, valid_y_pred)
precision = precision_score(valid_y, valid_y_pred_binary)
recall = recall_score(valid_y, valid_y_pred_binary)
f1 = 2 * (precision * recall) / (precision + recall + 1e-7)  # avoid div by zero

print("\nModel Performance on Validation Set (threshold=0.5):")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Find optimal threshold for F1 score
print("\nFinding optimal threshold for balanced precision-recall...")
thresholds = np.arange(0.1, 0.9, 0.05)
best_thresh = 0.5
best_f1 = f1

for thresh in thresholds:
    pred_binary = (valid_y_pred > thresh).astype(int).flatten()
    try:
        prec = precision_score(valid_y, pred_binary)
        rec = recall_score(valid_y, pred_binary)
        f1_score = 2 * (prec * rec) / (prec + rec + 1e-7)
        
        print(f"Threshold: {thresh:.2f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_score:.4f}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_thresh = thresh
    except:
        print(f"Error with threshold {thresh:.2f}")

print(f"\nOptimal threshold: {best_thresh:.2f}")
print(f"Best F1 Score: {best_f1:.4f}")

# Recalculate metrics with optimal threshold
valid_y_pred_binary = (valid_y_pred > best_thresh).astype(int).flatten()
accuracy = accuracy_score(valid_y, valid_y_pred_binary)
precision = precision_score(valid_y, valid_y_pred_binary)
recall = recall_score(valid_y, valid_y_pred_binary)

print("\nModel Performance with Optimal Threshold:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {best_f1:.4f}")

# Step 9: Display instructions for TensorBoard
print("\nTo view TensorBoard, run the following command in your terminal:")
print(f"tensorboard --logdir={log_dir}")

# Plot the loss curve (alternative to TensorBoard)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('tox21_loss_curve.png')
plt.show()