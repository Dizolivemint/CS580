import tensorflow as tf

def build_cnn_model(img_shape, conv_filters=24, conv2_filters=None, dense_units=128, learning_rate=1e-4):
    layers, models = tf.keras.layers, tf.keras.models
    inputs = tf.keras.Input(shape=img_shape)
    x = layers.Conv2D(conv_filters, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    if conv2_filters is not None:
        x = layers.Conv2D(conv2_filters, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
