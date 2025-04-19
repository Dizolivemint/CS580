# src/train.py
import time
import numpy as np
import os
from utils.visualization import save_loss_plot, save_generated_images

def train_gan(gan, X_train, epochs=15000, batch_size=32, sample_interval=2500, output_dir='output'):
    """
    Train the GAN model
    
    Args:
        gan: The GAN model to train
        X_train: Training data
        epochs: Number of training epochs
        batch_size: Size of the training batch
        sample_interval: Interval to save generated images
        output_dir: Directory to save outputs
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # For tracking losses
    g_losses = []
    d_losses = []
    
    # For tracking training time
    start_time = time.time()
    
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, gan.latent_dim))
        gen_imgs = gan.generator.predict(noise)
        
        # Adversarial ground truths with label smoothing
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid += 0.05 * np.random.random(valid.shape)
        fake += 0.05 * np.random.random(fake.shape)
        
        # Train the discriminator
        d_loss_real = gan.discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = gan.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Train the generator to fool the discriminator
        noise = np.random.normal(0, 1, (batch_size, gan.latent_dim))
        g_loss = gan.combined.train_on_batch(noise, valid)
        
        # Store losses
        g_losses.append(g_loss)
        d_losses.append(d_loss[0])
        
        # Print progress
        elapsed_time = time.time() - start_time
        if epoch % 100 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}] time: {elapsed_time:.2f}s")
        
        # Save generated image samples at specified intervals
        if epoch % sample_interval == 0:
            save_generated_images(gan.generator, gan.latent_dim, epoch, f"{output_dir}/images")
            
    # Save loss history
    save_loss_plot(g_losses, d_losses, output_dir)
    
    return g_losses, d_losses