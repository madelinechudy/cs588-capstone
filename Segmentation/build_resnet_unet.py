#RESNET/UNET -- possibly we could do it this way, one file both integrated

import tensorflow as tf
from tensorflow.keras import layers, models

#Define residual block
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

#Build ResNet encoder
def build_resnet_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    #Initial Convolution
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    #Residual blocks
    skip_connections = []
    for filters in [64, 128, 256]:
        x = residual_block(x, filters)
        skip_connections.append(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    return inputs, skip_connections, x

#Build U-Net with ResNet encoder
def build_resnet_unet(input_shape, num_classes):
    inputs, skip_connections, x = build_resnet_encoder(input_shape)

    #Decoder path
    for i in range(len(skip_connections)-1, -1, -1):
        x = layers.Conv2DTranspose(64 * (2 ** i), kernel_size=2, strides=2, padding='same')(x)
        x = layers.concatenate([x, skip_connections[i]])
        x = residual_block(x, 64 * (2 ** i))

    #Output layer
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

#Example usage
input_shape = (256, 256, 3)  #Example input shape
num_classes = 10  #Number of classes for segmentation

resnet_unet_model = build_resnet_unet(input_shape, num_classes)
resnet_unet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Print model summary
resnet_unet_model.summary()
