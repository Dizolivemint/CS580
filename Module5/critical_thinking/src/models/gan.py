# src/models/gan.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import numpy as np

from models.generator import build_generator
from models.discriminator import build_discriminator

class CIFAR10GAN:
    def __init__(self, img_shape=(32, 32, 3), latent_dim=100, learning_rate=0.0002):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.optimizer = Adam(learning_rate, beta_1=0.5)
        
        # Build and compile the discriminator
        self.discriminator = build_discriminator(img_shape)
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = build_generator(latent_dim, img_shape[2])
        
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # The generator takes noise as input and generates images
        z = Input(shape=(latent_dim,))
        img = self.generator(z)
        
        # The discriminator determines validity of the generated images
        valid = self.discriminator(img)
        
        # The combined model
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        # For storing training history
        self.g_losses = []
        self.d_losses = []
    
    def train_step(self, real_imgs, batch_size):
        """
        Perform one training step
        """
        # Generate random noise
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Generate a batch of fake images
        gen_imgs = self.generator.predict(noise)
        
        # Train the discriminator
        # Add some noise to labels for improved training
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid += 0.05 * np.random.random(valid.shape)
        fake += 0.05 * np.random.random(fake.shape)
        
        d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        g_loss = self.combined.train_on_batch(noise, valid)
        
        return d_loss, g_loss