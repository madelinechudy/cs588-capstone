# RESNET FRAMEWORK 
import tensorflow as tf

# Define residual block
def residual_block(x, filters):
    """
    Builds a residual block with two convolutional layers and a skip connection.

    Parameters:
        x (tf.Tensor): Input tensor to the residual block.
        filters (int): The number of filters for the convolutional layers.

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

# Build ResNet model
def build_resnet(input_shape, num_classes):
    """
    Builds a ResNet model.

    Parameters:
        input_shape (tuple of int): The shape of the input images, e.g., `(224, 224, 3)` for RGB images.
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Model: A Keras Model representing the ResNet architecture.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution layer
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Add residual blocks
    for _ in range(3):  # Repeat block
        x = residual_block(x, 64)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Global Average Pooling
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Output layer

    model = tf.keras.models.Model(inputs, outputs)
    return model

# Example usage
input_shape = (248, 496, 1)  # Updated input shape (grayscale image size)
num_classes = 4  # Number of classes for classification

resnet_model = build_resnet(input_shape, num_classes)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the summary
resnet_model.summary()

 

