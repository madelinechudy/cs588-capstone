import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from build_resnet import build_resnet

data_dir = '../cs588-capstone/Data/Training'
batch_size = 64
input_shape = (248, 496, 1)  
epochs = 10

# Multi Classification
label_map = {'no_dementia': 0, 'very_mild_dementia': 1, 'mild_dementia': 2, 'moderate_dementia': 3}

def data_generator(image_files, batch_size, input_shape):
    """
    Yields batches of images and corresponding labels dynamically for training or validation.

    Args:
        image_files (list of str): List of file paths to the images.
        batch_size (int): The number of samples per batch.
        input_shape (tuple of int): The desired shape of the images as (height, width).

    Yields:
        tuple:
            batch_images (numpy.ndarray): A batch of preprocessed images with shape 
              (batch_size, input_shape[0], input_shape[1], 1). Images are normalized to the range [0, 1].
            batch_labels (numpy.ndarray): Corresponding labels for the batch of images.
    """
    while True:
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)
        for start_idx in range(0, len(image_files), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                img = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                label = labels[idx]
                if img is not None:
                    img = cv2.resize(img, (input_shape[1], input_shape[0]))
                    img = np.expand_dims(img / 255.0, axis=-1)  # Normalize and add channel dim
                    batch_images.append(img)
                    batch_labels.append(label)

            yield np.array(batch_images), np.array(batch_labels)

# Load file paths and split data into training and validation sets
image_files = []
labels = []
for label, label_value in label_map.items():
    label_dir = os.path.join(data_dir, label)
    for file_name in os.listdir(label_dir):
        image_files.append(os.path.join(label_dir, file_name))
        labels.append(label_value)

train_images, val_images, train_labels, val_labels = train_test_split(
    image_files, labels, test_size=0.2, random_state=42
)

# Data generators
train_gen = data_generator(train_images, batch_size, input_shape)
val_gen = data_generator(val_images, batch_size, input_shape)

# Build and compile the model
resnet_model = build_resnet(input_shape, num_classes=len(label_map))
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with specified steps per epoch and validation steps
history = resnet_model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=len(train_images) // batch_size,
    validation_data=val_gen,
    validation_steps=len(val_images) // batch_size
)

# Define the directory path
base_path = 'Models/ResNet/Classification'
if not os.path.exists(base_path):
    os.makedirs(base_path)

# Save model and history
model_path = os.path.join(base_path, 'resnet_model.keras')
resnet_model.save(model_path)

history_path = os.path.join(base_path, 'resnet_history.npy')
np.save(history_path, history.history)

# Plot training history
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(base_path, 'resnet_accuracy.png'))
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(base_path, 'resnet_loss.png'))
plt.close()

