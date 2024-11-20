import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from build_unet import build_unet  
from skimage.morphology import remove_small_objects

data_dir = '../cs588-capstone/Data/Training'
output_dir = '../cs588-capstone/Segmentation/Models/UNet'
batch_size = 32
input_shape = (224, 224, 3)  
epochs = 10

label_map = {'no_dementia': 0, 'very_mild_dementia': 1, 'mild_dementia': 2, 'moderate_dementia': 3}

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

def data_generator(data_dir, mask_dir, batch_size, input_shape, label_map):
    image_files = []
    mask_files = []

    for label, label_value in label_map.items():
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                image_files.append(file_path)
                
                # Construct mask filename
                mask_filename = f"mask_{label_value}_{file_name}"
                mask_files.append(os.path.join(mask_dir, mask_filename))

    # Shuffle data
    indices = np.arange(len(image_files))
    np.random.shuffle(indices)

    while True:
        for start_idx in range(0, len(image_files), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_images = []
            batch_masks = []

            for idx in batch_indices:
                img = cv2.imread(image_files[idx])  # Load RGB image
                mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
                if img is not None and mask is not None:
                    # Resize and normalize image
                    img = cv2.resize(img, (input_shape[1], input_shape[0]))
                    img = img / 255.0  # Normalize to [0, 1]

                    # Resize and binarize mask
                    mask = cv2.resize(mask, (input_shape[1], input_shape[0]))
                    mask = np.where(mask > 127, 1, 0)  # Binary mask (0 or 1)

                    batch_images.append(img)
                    batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)

def postprocess_mask(mask):
    mask = remove_small_objects(mask > 0.5, min_size=50)  
    return mask

def visualize_predictions(model, dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, (images, masks) in enumerate(dataset.take(5)):
        preds = model.predict(images)
        for i in range(len(images)):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(images[i, :, :, :])
            plt.title("Input MRI (RGB)")
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i, :, :], cmap='gray')
            plt.title("Ground Truth")
            plt.subplot(1, 3, 3)
            plt.imshow(preds[i, :, :, 0] > 0.5, cmap='gray')
            plt.title("Prediction")
            plt.savefig(os.path.join(output_dir, f"prediction_{idx}_{i}.png"))
            plt.close()

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(data_dir, '../cs588-capstone/Data/Processed/Masks', batch_size, input_shape, label_map),
    output_signature=(
        tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
        tf.TensorSpec(shape=(None, input_shape[0], input_shape[1]), dtype=tf.int32)
    )
)

train_dataset = train_dataset.shuffle(buffer_size=1024).prefetch(tf.data.experimental.AUTOTUNE)
unet_model = build_unet(input_shape)
unet_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# work on model checkpoint callback
"""checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_dir, 'unet_best_model.keras'), save_best_only=True
)"""

# Train the model (!! incorporate checkpoint callback)
'''history = unet_model.fit(train_dataset, epochs=epochs, steps_per_epoch=len(os.listdir(data_dir)) // batch_size, callbacks=[checkpoint_cb])'''

# Save the final model
unet_model.save(os.path.join(output_dir, 'unet_model.keras'))

# Save training history
history_path = os.path.join(output_dir, 'unet_history.npy')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
np.save(history_path, history.history)

# Plot and save accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'unet_accuracy.png'))
plt.close()

# Plot and save loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'unet_loss.png'))
plt.close()

# Visualize predictions
'''visualize_predictions(unet_model, train_dataset, '../cs588-capstone/Segmentation/Predictions')'''