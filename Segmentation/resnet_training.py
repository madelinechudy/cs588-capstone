import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from build_resnet import build_resnet

data_dir = '../cs588-capstone/Data/Training'  
batch_size = 64
input_shape = (224, 224, 3) 

# Multi Classification 
label_map = {'no_dementia': 0, 'very_mild_dementia': 1, 'mild_dementia': 2, 'moderate_dementia': 3}

def data_generator(data_dir, batch_size, input_shape):
    """
    Yields batches of images and labels from directories dynamically for training.
    
    Parameters:
        data_dir (string): Directory to the dataset images.
        batch_size (int): Size of batches.
        input_shape (array): Shape of the images by pixel count and channels. 

    Yields:
        tuple, containing:
            batch_images (np.array): The normalized image data as a numpy array, given from a batch.
            batch_labels (np.array): A numpy array of labels corresponding to each image, given from a batch.

    """
    image_files = []
    labels = []

    # Collect file paths and their respective labels
    for label, label_value in label_map.items():
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            image_files.append(file_path)
            labels.append(label_value)
    
    # Shuffle data indices
    indices = np.arange(len(image_files))
    np.random.shuffle(indices)

    # Preprocessing
    while True:
        for start_idx in range(0, len(image_files), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img = cv2.imread(image_files[idx])
                if img is not None:
                    # Resize and normalize image
                    img = cv2.resize(img, (input_shape[1], input_shape[0]))
                    img = img / 255.0  # Normalize to [0, 1] range (not truly binary --> look to making binary)
                    batch_images.append(img)
                    batch_labels.append(labels[idx])
            
            yield np.array(batch_images), np.array(batch_labels)

# Create dataset off tf.data API to streamline batches
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(data_dir, batch_size, input_shape),
    output_signature=(
        tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# Shuffle the dataset created from tf.data.Dataset
train_dataset = train_dataset.shuffle(buffer_size=1024).prefetch(tf.data.experimental.AUTOTUNE)

# Build and compile the model
resnet_model = build_resnet(input_shape, num_classes=4)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = resnet_model.fit(train_dataset, epochs=10, steps_per_epoch=1086)

# Check if the directory exists and create it if not
if not os.path.exists('../cs588-capstone/Segmentation/Models'):
    os.makedirs('../cs588-capstone/Segmentation/Models')

# Save model
resnet_model.save('../cs588-capstone/Segmentation/Models/resnet_model.keras')

# Preservation of training history
history_path = '../cs588-capstone/Segmentation/Models/ResNet/resnet_history.npy'

if not os.path.exists(os.path.dirname(history_path)):
    os.makedirs(os.path.dirname(history_path))
np.save(history_path, history.history)

# Visualization Training History
if not os.path.exists('Models/ResNet'):
    os.makedirs('Models/ResNet')

# Plot and save accuracy images
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Models/ResNet/resnet_accuracy.png')
plt.close()

# Draw and save loss images
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Models/ResNet/resnet_loss.png')
plt.close() 