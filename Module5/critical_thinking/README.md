# Enhanced GAN for CIFAR10 - Portfolio Project

This project implements an enhanced Generative Adversarial Network (GAN) for generating images based on the CIFAR10 dataset. The implementation extends the basic requirements of the assignment with modern techniques for improved stability and quality.

## Project Overview

The goal is to train a GAN to generate realistic images similar to those in the CIFAR10 dataset. By default, the model is trained on class 8 (ships), but you can easily change this to any class you prefer.

## Key Features and Enhancements

This implementation includes several modern enhancements beyond the basic assignment requirements:

1. **Residual Connections**: Added to the generator for better gradient flow and higher quality generation
2. **Spectral Normalization**: Applied to the discriminator for improved stability during training
3. **Training Monitoring**: Tracks and visualizes loss metrics over time
4. **Proper Initialization**: Uses appropriate weight initialization for GAN training
5. **Label Smoothing**: Adds noise to the labels to reduce mode collapse
6. **Improved Architecture**: Both the generator and discriminator have enhanced architectures

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- Python 3.6+

## How to Use

1. Install the required dependencies:
   ```
   pip install tensorflow numpy matplotlib
   ```

2. Run the training:
   ```python
   # Initialize the GAN
   cifar_gan = CIFAR10GAN()
   
   # Train with specified parameters
   cifar_gan.train(epochs=15000, batch_size=32, sample_interval=2500)
   ```

3. The script will create a `gan_output` directory with:
   - Generated images at specified intervals during training
   - A plot of loss history

## Analysis Instructions

For the portfolio requirements, you need to:

1. **First Epoch Analysis**: 
   - Examine the images from epoch 0 (saved as `gan_output/cifar10_gan_epoch_0.png`)
   - Take a screenshot for your report
   - Note the randomness, lack of structure, and noise

2. **Last Epoch Analysis**: 
   - Examine the images from the final epoch (saved as `gan_output/cifar10_gan_epoch_15000.png`)
   - Take a screenshot for your report
   - Look for improvements in coherence, realism, and structure

3. **Performance Analysis**:
   - Examine the loss history plot (`gan_output/loss_history.png`)
   - Discuss convergence, stability, and quality evolution
   - Comment on whether the GAN successfully learned the data distribution
   - Discuss any mode collapse or convergence issues

## Customization Options

You can customize several aspects of the GAN:

1. **Target Class**: Change the class selection in `load_cifar10_data()` to focus on different CIFAR10 classes:
   ```python
   # Class 0: airplane
   # Class 1: automobile
   # Class 2: bird
   # Class 3: cat
   # Class 4: deer
   # Class 5: dog
   # Class 6: frog
   # Class 7: horse
   # Class 8: ship
   # Class 9: truck
   X = X[y.flatten() == 8]  # Change 8 to any class number from 0-9
   ```

2. **Architecture**: Modify the network architectures in `build_generator()` and `build_discriminator()`

3. **Training Parameters**: Adjust learning rate, batch size, and epochs in the `train()` method

## Expected Training Time

With a modern GPU, training for 15,000 epochs may take several hours. For testing purposes, you can reduce the number of epochs (e.g., to 1,000) to get faster results.

## Final Report Guidelines

Your final report should include:

1. Screenshots of generated images from epoch 0 and the final epoch
2. Analysis of the GAN's performance over time
3. Discussion of the improvements seen in the generated images
4. Explanation of any challenges encountered during training
5. Comments on the effectiveness of the enhancements added to the basic implementation