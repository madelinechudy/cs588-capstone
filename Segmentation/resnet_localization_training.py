import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from build_resnet import build_resnet_localization  # Import the updated ResNet localization function
from skimage.morphology import remove_small_objects

# Directory paths and parameters
data_dir = '../cs588-capstone/Data/Training'
mask_dir = '../cs588-capstone/Data/Processed/Masks'
output_dir = '../cs588-capstone/Segmentation/Models/ResNet'
batch_size = 32
input_shape = (496, 248, 1)  # Adjusted for actual image dimensions and grayscale
epochs = 10

label_map = {'no_dementia': 0, 'very_mild_dementia': 1, 'mild_dementia': 2, 'moderate_dementia': 3}

# Combined Dice and Binary Cross-Entropy Loss
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = 1 - numerator / (denominator + tf.keras.backend.epsilon())
    return bce + dice

# Data generator with augmentation and postprocessing
def data_generator(image_files, mask_files, batch_size, input_shape, augment=False):
    while True:
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)
        for start_idx in range(0, len(image_files), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_images = []
            batch_masks = []

            for idx in batch_indices:
                img = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)  # Load grayscale image
                mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
                if img is not None and mask is not None:
                    img = cv2.resize(img, (input_shape[1], input_shape[0]))  # Resize to (248, 496)
                    img = img / 255.0

                    mask = cv2.resize(mask, (input_shape[1], input_shape[0]))  # Resize to (248, 496)
                    mask = np.where(mask > 127, 1, 0)
                    mask = remove_small_objects(mask > 0.5, min_size=50).astype(np.uint8)

                    if augment:
                        if np.random.rand() > 0.5:
                            img = np.fliplr(img)
                            mask = np.fliplr(mask)
                        if np.random.rand() > 0.5:
                            img = np.flipud(img)
                            mask = np.flipud(mask)

                    batch_images.append(np.expand_dims(img, axis=-1))  # Add channel dimension
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

# Build and compile ResNet-based model
resnet_model = build_resnet_localization(input_shape, output_channels=1)
resnet_model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_dir, 'resnet_best_model.keras'), save_best_only=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = resnet_model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=len(train_images) // batch_size,
    validation_data=val_gen,
    validation_steps=len(val_images) // batch_size,
    callbacks=[checkpoint_cb, lr_scheduler]
)

# Save the final model and training history
resnet_model.save(os.path.join(output_dir, 'resnet_model.keras'))
np.save(os.path.join(output_dir, 'resnet_history.npy'), history.history)

# Plot training history
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'resnet_accuracy.png'))
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'resnet_loss.png'))
plt.close()
