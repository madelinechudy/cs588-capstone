import tensorflow as tf
import numpy as np 

def build_unet(input_shape):
    input = tf.keras.layers.Input(input_shape)

    #Encoder Path
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(input)
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPool2D((2,2))(c1)

    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(input)
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPool2D((2,2))(c1)

    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(input)
    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPool2D((2,2))(c1)

    #Bottleneck
    c1 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(p1)
    c1 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(c1)

    #Decoder Path
    u1 = tf.keras.layers.UpSampling2D((2,2))(c1)
    concatenate layer
    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c1)
    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c1)

    u1 = tf.keras.layers.UpSampling2D((2,2))(c1)
    concatenate layer
    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c1)
    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c1)

    u1 = tf.keras.layers.UpSampling2D((2,2))(c1)
    concatenate layer
    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c1)
    c1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c1)

    outputs = Conv2D(c1)

    model = tf.keras.Model(input, outputs)

    model.compile(optimizer, loss, metrics)
    model.save
    return model


# pooling size
# convulation filter and kernel use and size
# padding 
# interpolative resize
# concatenate
# ask about pretraining and why specific numbers were chosen