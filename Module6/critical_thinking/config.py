# src/config.py

# Dataset and paths
TRAIN_DIR = 'train/'       # Or absolute path if needed
IMG_SIZE = (94, 125)
MAX_IMAGES_PER_CLASS = 8192

# Randomness
SEED = 42

# Model hyperparameters
CONV_FILTERS = 24
CONV2_FILTERS = 48
DENSE_UNITS = 128
LEARNING_RATE = 1e-4

# Training settings
BATCH_SIZE = 128
EPOCHS = 10
VALID_SIZE = 0.1
EARLY_STOPPING = False
PATIENCE = 2

# Metrics
METRIC_THRESHOLD = 0.5     # Used for metrics, can be tuned

# Saving
MODEL_DIR = 'models/'
MODEL_NAME = 'cnn_cats_dogs'