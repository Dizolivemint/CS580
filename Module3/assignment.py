import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(101)
tf.random.set_seed(101)

# Generating random linear data
# There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

# 1) Plot the training data
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.title('Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

# Make sure both x and y are float32
x = x.astype(np.float32)
y = y.astype(np.float32)

# Normalize the data
x_mean = np.mean(x)
x_std = np.std(x)
x_normalized = (x - x_mean) / x_std

# Define the model (simplest possible approach)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1], dtype=tf.float32)
])

# Initialize with reasonable weights (closer to what we expect)
initial_weights = [np.array([[1.0]], dtype=np.float32),  # Weight close to 1
                   np.array([0.0], dtype=np.float32)]    # Bias close to 0
model.set_weights(initial_weights)

# Compile with a smaller learning rate for stability
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mse'
)

# Add a callback to stop early if the model diverges
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=10,
    min_delta=0.0001,
    restore_best_weights=True
)

# Train with careful monitoring
history = model.fit(
    x_normalized, 
    y,
    epochs=1000,
    verbose=1,
    callbacks=[early_stopping]
)

# Print final weights
weights = model.get_weights()
final_W = weights[0][0][0]
final_b = weights[1][0]

print(f"\nTraining completed!")
print(f"Final Weight (normalized scale): {final_W:.4f}")
print(f"Final Bias: {final_b:.4f}")

# Calculate the actual slope in the original scale
actual_slope = final_W / x_std
actual_intercept = final_b - (final_W * x_mean / x_std)

print(f"Final Weight (original scale): {actual_slope:.4f}")
print(f"Final Intercept (original scale): {actual_intercept:.4f}")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()

# Plot the fitted line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Training Data')

# Creating values to plot the fitted line in the original scale
x_plot = np.linspace(0, 50, 100)
y_pred = actual_slope * x_plot + actual_intercept

plt.plot(x_plot, y_pred, 'r-', linewidth=2, label='Fitted Line')
plt.title('Linear Regression Result')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# Calculate R-squared to evaluate model fit
y_mean = np.mean(y)
ss_total = np.sum((y - y_mean) ** 2)
y_pred_train = model.predict(x_normalized, verbose=0)
ss_residual = np.sum((y - y_pred_train.flatten()) ** 2)
r_squared = 1 - (ss_residual / ss_total)

print(f"\nModel evaluation:")
print(f"R-squared: {r_squared:.4f}")

# Predictions on sample points
print("\nPredictions on sample points:")
sample_points = [0, 10, 20, 30, 40, 50]
for point in sample_points:
    point_normalized = (point - x_mean) / x_std
    prediction = model.predict(np.array([point_normalized]), verbose=0)[0][0]
    print(f"X = {point}, Predicted Y = {prediction:.2f}")