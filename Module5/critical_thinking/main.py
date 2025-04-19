# main.py
import argparse
import os
import tensorflow as tf
import numpy as np

from src.models.gan import CIFAR10GAN
from src.utils.data_loader import load_cifar10_data
from src.train import train_gan

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a GAN on CIFAR10 dataset')
    parser.add_argument('--epochs', type=int, default=15000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sample_interval', type=int, default=2500, help='Interval to save generated images')
    parser.add_argument('--class_id', type=int, default=8, help='CIFAR10 class ID to train on')
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train = load_cifar10_data(class_id=args.class_id)
    
    # Initialize GAN
    gan = CIFAR10GAN()
    
    # Train the GAN
    g_losses, d_losses = train_gan(
        gan, 
        X_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        sample_interval=args.sample_interval, 
        output_dir=output_dir
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()