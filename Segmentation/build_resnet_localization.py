import tensorflow as tf

# Define residual block
def residual_block(x, filters):
    shortcut = x  # Save input for skip connection
    
    # First convolution layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Second convolution layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add shortcut to output of convolutions
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

# Build ResNet model for localization
def build_resnet_localization(input_shape, output_channels):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution layer
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Add residual blocks
    for _ in range(3):  # Repeat block
        x = residual_block(x, 64)

    # Upsampling layers for localization
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)

    # Output layer for mask prediction
    outputs = tf.keras.layers.Conv2D(output_channels, kernel_size=1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model

# Example usage for localization
input_shape = (248, 496, 1)  # Updated input shape for MRI images (height=248, width=496, grayscale)
output_channels = 1  # Number of output channels (binary mask)

resnet_localization_model = build_resnet_localization(input_shape, output_channels)
resnet_localization_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the summary
resnet_localization_model.summary()

