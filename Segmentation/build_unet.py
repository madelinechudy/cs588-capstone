import os
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split

def build_unet(input_shape):
    """
    Function that builds the U-Net Convolutional Neural Network. 

    Params:
        input_shape (tuple): Input size of the image shapes (height, width, channel).

    Return:
        Model compilation.
    """
    # Input Layer
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder Path
    # First Convolutional Block
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPool2D((2,2))(c1)

    # Second Convolutional Block
    c2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPool2D((2,2))(c2)

    # Third Convolutional Block
    c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPool2D((2,2))(c3)

    # Bottleneck
    c4 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(c4)

    # Decoder Path
    # First Upsampling Block
    u1 = tf.keras.layers.UpSampling2D((2,2))(c4)
    u1 = tf.keras.layers.concatenate([u1, c3])
    c5 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(u1)
    c5 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c5)

    # Second Upsampling Block
    u2 = tf.keras.layers.UpSampling2D((2,2))(c5)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(u2)
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c6)

    # Third Upsampling Block
    u3 = tf.keras.layers.UpSampling2D((2,2))(c6)
    u3 = tf.keras.layers.concatenate([u3, c1])
    c7 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(u3)
    c7 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c7)

    # Sigmoid activation for binary segmentation 
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def pretrain_data():
    """
    Function to use random noise to pre-train the model. 

    Allows faster convergence during real training, increasing performance.
    """
    train_images = np.random.rand(100, 224, 224, 3)
    train_masks = np.random.randint(0, 2, (100, 224, 224, 1))

    return train_images, train_masks
    

if __name__ == "__main__":

    train_images, train_masks = pretrain_data()

    input_shape = (224, 224, 3)
    model = build_unet(input_shape)

    model.fit(train_images, train_masks, epochs=10, batch_size=8)

    if not os.path.exists('../cs588-capstone/Segmentation/Models'):
        os.makedirs('../cs588-capstone/Segmentation/Models')
    model.save('../cs588-capstone/Segmentation/Models/pretrained_unet_model.keras')



