# src/config.py

# Dataset and paths
TRAIN_DIR = 'train/'       # Or absolute path if needed
IMG_SIZE = (94, 125)
MAX_IMAGES_PER_CLASS = 2048

# Randomness
SEED = 42

# Model hyperparameters
CONV_FILTERS = 24
CONV2_FILTERS = 48         # Use None if single conv
DENSE_UNITS = 128
LEARNING_RATE = 1e-4

# Training settings
BATCH_SIZE = 32
EPOCHS = 10  # Changed from 3 to 10 as per assignment
VALID_SIZE = 0.1
EARLY_STOPPING = True
PATIENCE = 2

# Metrics
METRIC_THRESHOLD = 0.5     # Used for metrics, can be tuned

# Saving
MODEL_DIR = 'models/'
MODEL_NAME = 'cnn_cats_dogs'