import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from build_unet import build_unet
from skimage.morphology import remove_small_objects

data_dir = '/content/cs588-capstone/Data/Training'
mask_dir = '/content/cs588-capstone/Data/Processed/Masks'
output_dir = '/content/cs588-capstone/Segmentation/Models/UNet'
batch_size = 32
input_shape = (248, 496, 1) 
epochs = 10

label_map = {'no_dementia': 0, 'very_mild_dementia': 1, 'mild_dementia': 2, 'moderate_dementia': 3}

# Combined Dice and Binary Cross-Entropy Loss
def combined_loss(y_true, y_pred):
    """
    Computes a combined loss for binary segmentation tasks, consisting of:
      Binary Cross-Entropy loss.
      Dice loss (a measure of overlap between predicted and true masks).

    Parameters:
        y_true (tf.Tensor): Ground truth binary masks. Shape: (batch_size, height, width, 1).
        y_pred (tf.Tensor): Predicted masks, with values in the range [0, 1].
                            Shape: (batch_size, height, width, 1).

    Returns:
        tf.Tensor: The combined loss value for the batch. This is the sum of BCE and Dice loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = 1 - numerator / (denominator + tf.keras.backend.epsilon())
    return bce + dice

# Data generator with augmentation and postprocessing
def data_generator(image_files, mask_files, batch_size, input_shape, augment=False):
    """
    A generator that yields batches of images and corresponding masks for training or validation.

    Parameters:
        image_files (list of str): List of file paths to the input images.
        mask_files (list of str): List of file paths to the corresponding masks.
        batch_size (int): The number of samples per batch.
        input_shape (tuple of int): The desired shape of the images and masks as (height, width).
        augment (bool, optional): If True, applies random data augmentation such as flipping. 
                                  Defaults to False.

    Yields:
        tuple:
            batch_images (numpy.ndarray): Batch of preprocessed images with shape 
              (batch_size, height, width, 1), normalized to the range [0, 1].
            batch_masks (numpy.ndarray): Batch of preprocessed binary masks with shape 
              (batch_size, height, width, 1).
    """
    while True:
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)
        for start_idx in range(0, len(image_files), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_images = []
            batch_masks = []

            for idx in batch_indices:
                img = cv2.imread(image_files[idx])
                mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
                if img is not None and mask is not None:
                    img = cv2.resize(img, (input_shape[1], input_shape[0]))
                    img = img / 255.0

                    mask = cv2.resize(mask, (input_shape[1], input_shape[0]))
                    mask = np.where(mask > 127, 1, 0)
                    mask = remove_small_objects(mask > 0.5, min_size=50).astype(np.uint8)

                    if augment:
                        if np.random.rand() > 0.5:
                            img = np.fliplr(img)
                            mask = np.fliplr(mask)
                        if np.random.rand() > 0.5:
                            img = np.flipud(img)
                            mask = np.flipud(mask)

                    batch_images.append(img)
                    batch_masks.append(mask)

            yield np.array(batch_images), np.expand_dims(np.array(batch_masks), -1)

# Load file paths and split data into training and validation sets
image_files = []
mask_files = []
for label, label_value in label_map.items():
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for file_name in os.listdir(label_dir):
            image_files.append(os.path.join(label_dir, file_name))
            mask_files.append(os.path.join(mask_dir, f"mask_{label_value}_{file_name}"))

train_images, val_images, train_masks, val_masks = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)

# Data generators
train_gen = data_generator(train_images, train_masks, batch_size, input_shape, augment=True)
val_gen = data_generator(val_images, val_masks, batch_size, input_shape, augment=False)

# Calculate class weights
class_counts = {label: len([f for f in os.listdir(os.path.join(data_dir, label))]) for label in label_map.keys()}
total_samples = sum(class_counts.values())
class_weights = {label_map[k]: total_samples / (len(label_map) * v) for k, v in class_counts.items()}

# Build and compile model
unet_model = build_unet(input_shape)
unet_model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_dir, 'unet_best_model.keras'), save_best_only=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = unet_model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=len(train_images) // batch_size,
    validation_data=val_gen,
    validation_steps=len(val_images) // batch_size,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, lr_scheduler]
)

# Save the final model and training history
unet_model.save(os.path.join(output_dir, 'unet_model.keras'))
np.save(os.path.join(output_dir, 'unet_history.npy'), history.history)

# Plot accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'unet_accuracy.png'))
plt.close()

# Plot loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'unet_loss.png'))
plt.close()


