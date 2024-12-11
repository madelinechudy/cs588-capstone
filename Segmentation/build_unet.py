import os
import plaidml.keras
plaidml.keras.install_backend()
import keras
import numpy as np
from sklearn.model_selection import train_test_split


def dice_coefficient(y_true, y_pred):
    """
    Dice Coefficient for segmentation evaluation.

    Parameters:
        y_true (tensor): Ground truth masks.
        y_pred (tensor): Predicted masks.

    Returns:
        float: Dice coefficient score.
    """
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + 1e-7)


def build_unet(input_shape):
    """
    Function that builds the U-Net Convolutional Neural Network.

    Parameters:
        input_shape (tuple): Input size of the image shapes (height, width, channel).

    Returns:
        keras.Model: Compiled U-Net model.
    """
    # Input Layer
    inputs = keras.layers.Input(input_shape)

    # Encoder Path
    # First Convolutional Block
    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    # Second Convolutional Block
    c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    # Third Convolutional Block
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder Path
    # First Upsampling Block
    u1 = keras.layers.UpSampling2D((2, 2))(c4)
    u1 = keras.layers.concatenate([u1, c3])
    c5 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c5 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    # Second Upsampling Block
    u2 = keras.layers.UpSampling2D((2, 2))(c5)
    u2 = keras.layers.concatenate([u2, c2])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    # Third Upsampling Block
    u3 = keras.layers.UpSampling2D((2, 2))(c6)
    u3 = keras.layers.concatenate([u3, c1])
    c7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    c7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    # Sigmoid activation for binary segmentation
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = keras.Model(inputs, outputs)

    # Compile the model with Dice coefficient and binary cross-entropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coefficient])

    return model


def pretrain_data():
    """
    Function to use random noise to pre-train the model.

    Returns:
        tuple: Randomly generated training images and masks.
    """
    train_images = np.random.rand(100, 224, 224, 3).astype(np.float32)
    train_masks = np.random.randint(0, 2, (100, 224, 224, 1)).astype(np.float32)

    return train_images, train_masks


if __name__ == "__main__":
    # Generate pretraining data
    train_images, train_masks = pretrain_data()

    # Build and train the model
    input_shape = (224, 224, 3)
    model = build_unet(input_shape)

    # Train the model on random data
    model.fit(train_images, train_masks, epochs=10, batch_size=8)

    # Save the pretrained model
    model_save_path = '../cs588-capstone/Segmentation/Models/pretrained_unet_model.keras'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)



