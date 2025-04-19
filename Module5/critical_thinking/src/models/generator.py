# src/models/generator.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.initializers import RandomNormal

def build_generator(latent_dim=100, channels=3):
    """
    Build an improved generator with residual connections
    
    Args:
        latent_dim (int): Dimension of the latent space
        channels (int): Number of channels in output image (3 for RGB)
        
    Returns:
        Model: Keras generator model
    """
    init = RandomNormal(stddev=0.02)
    model = Sequential()
    
    # Foundation for 8x8 feature maps
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim, 
                    kernel_initializer=init))
    model.add(Reshape((8, 8, 128)))
    model.add(BatchNormalization(momentum=0.8))
    
    # Upsample to 16x16
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=init))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    # Residual block
    residual = model.output
    x = Conv2D(128, kernel_size=3, padding="same", kernel_initializer=init)(residual)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
    x = Conv2D(128, kernel_size=3, padding="same", kernel_initializer=init)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.add([x, residual])
    
    # Upsample to 32x32
    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=3, padding="same", kernel_initializer=init)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
    
    # Final conv layer
    x = Conv2D(channels, kernel_size=3, padding="same", kernel_initializer=init)(x)
    img = Activation("tanh")(x)
    
    # Create the model
    input_layer = model.input
    model = Model(input_layer, img)
    
    return model