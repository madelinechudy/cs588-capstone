# RESNET FRAMEWORK 
import tensorflow as tf
from tensorflow.keras import layers, models

#Define residual block
def residual_block(x, filters):
    shortcut = x  #Save input for skip connection
    
    #First convolution layer
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    #Second convolution layer
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    #Add shortcut to output of convolutions
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

#Build ResNet model
def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    #Initial convolution layer
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    #Add residual blocks
    for _ in range(3):  #Repeat block
        x = residual_block(x, 64)

    x = layers.GlobalAveragePooling2D()(x)  #Global Average Pooling
    outputs = layers.Dense(num_classes, activation='softmax')(x)  #Output layer

    model = models.Model(inputs, outputs)
    return model

# Example usage
input_shape = (224, 224, 3)  # Example input shape (image size)
num_classes = 10  # Number of classes for classification

resnet_model = build_resnet(input_shape, num_classes)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Print the summary
resnet_model.summary()
 

