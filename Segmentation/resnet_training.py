import numpy as np
import tensorflow as tf
from build_resnet import build_resnet
import os
import matplotlib.pyplot as plt

# Load preprocessed images and labels
X_train = np.load('../cs588-capstone/Data/Processed/train_images.npy') # Shape should be (num_samples, height, width, channels)
y_train = np.load('../cs588-capstone/Data/Processed/train_labels.npy') # Shape should be (num_samples,)

# Convert the labels to a TensorFlow tensor
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

# Create a TensorFlow dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Print out the shape of the dataset
for images, labels in train_dataset.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

input_shape = (224, 224, 3)  
resnet_model = build_resnet(input_shape, num_classes=4)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Use the dataset for training
history = resnet_model.fit(train_dataset, epochs=10, batch_size=32)

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
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Models/ResNet/resnet_accuracy.png')
plt.close()

# Draw and save loss images
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Models/ResNet/resnet_loss.png')
plt.close() 