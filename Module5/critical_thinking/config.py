# src/config.py

# Model parameters
LATENT_DIM = 100
IMAGE_SHAPE = (32, 32, 3)
LEARNING_RATE = 0.0002
BETA1 = 0.5  # For Adam optimizer

# Training parameters
BATCH_SIZE = 32
EPOCHS = 15000
SAMPLE_INTERVAL = 2500

# Data parameters
CIFAR10_CLASS_ID = 8  # Ships (0: airplane, 1: car, 2: bird, etc.)

# Output settings
OUTPUT_DIR = "output"
IMAGES_DIR = "output/images"

# Random seed for reproducibility 
RANDOM_SEED = 42