import tensorflow as tf

# Define residual block
def residual_block(x, filters):
    """
    Creates a residual block with two convolutional layers and a skip connection.

    Parameters:
        x (tf.Tensor): Input tensor to the residual block.
        filters (int): Number of filters for both convolutional layers.

    Returns:
        tf.Tensor: Output tensor after applying the residual block.
    """
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
    """
    Builds a ResNet-based model for image localization tasks with upsampling layers for mask prediction.

    Parameters:
        input_shape (tuple of int): Shape of the input image
        output_channels (int): Number of channels in the output, typically 1 for binary masks.

    Returns:
        tf.keras.Model: A Keras Model for image localization with input shape input_shape 
                        and output shape (input_shape[0], input_shape[1], output_channels).
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution layer
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)  # Downsample once
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)  # Downsample again

    # Add residual blocks
    for _ in range(3):  # Repeat block
        x = residual_block(x, 64)

    # Upsampling layers to restore dimensions
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # From (62, 124) to (124, 248)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # From (124, 248) to (248, 496)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)

    # Final output layer for mask prediction
    outputs = tf.keras.layers.Conv2D(output_channels, kernel_size=1, activation='sigmoid')(x)  # Output shape (248, 496, 1)

    model = tf.keras.models.Model(inputs, outputs)
    return model

input_shape = (248, 496, 1)  # Input shape (height, width, channels)
output_channels = 1  # Number of output channels (binary mask)

resnet_localization_model = build_resnet_localization(input_shape, output_channels)
resnet_localization_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the summary
resnet_localization_model.summary()

