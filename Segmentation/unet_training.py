import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from build_unet import build_unet
import matplotlib.pyplot as plt

# Load preprocessed data
train_images = np.load('../cs588-capstone/Data/Processed/train_images.npy')
train_masks = np.load('../cs588-capstone/Data/Processed/train_masks.npy')  

# Divide into training set and validation set
X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# Load pretrained model from build_unet
input_shape = (224, 224, 3)
unet_model = tf.keras.models.load_model('../cs588-capstone/Segmentation/Models/pretrained_unet_model.keras')

# Train the model and save the history
history = unet_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
unet_model.save('../cs588-capstone/Segmentation/Models/unet_model.keras')

# Preservation of training history
history_path = '../cs588-capstone/Segmentation/Models/UNet/unet_history.npy'

if not os.path.exists(os.path.dirname(history_path)):
    os.makedirs(os.path.dirname(history_path))
np.save(history_path, history.history)

# Visualization Training History
if not os.path.exists('Models/UNet'):
    os.makedirs('Models/UNet')

# Plot and save accuracy images
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Models/UNet/unet_accuracy.png')
plt.close()

# Plot and save loss images
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Models/UNet/unet_loss.png')
plt.close()