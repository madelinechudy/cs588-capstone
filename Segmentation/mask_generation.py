import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Define paths and parameters
image_dir = '../cs588-capstone/Data/Training'
output_dir = '../cs588-capstone/Data/Processed/Masks'
batch_size = 64
target_size = (248, 496)  # Correct dimensions: (height, width)

label_map = {'no_dementia': 0, 'very_mild_dementia': 1, 'mild_dementia': 2, 'moderate_dementia': 3}

def preprocess_image(file_path, label, original_filename):
    """
    Loads and preprocesses an image for training or evaluation.

    Args:
        file_path (str): The path to the image file.
        label (int or str): The label associated with the image.
        original_filename (str): The original filename of the image.

    Returns:
        tuple:
            image (tf.Tensor): A preprocessed image tensor with normalized pixel values
              in the range [0, 1] and resized to the specified target dimensions.
            label (int or str): The input label, unchanged.
            original_filename (str): The input original filename, unchanged.
    """

    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (target_size[0], target_size[1]))  # Resize to 496x248
    image = image / 255.0  # Normalize to [0, 1] range
    return image, label, original_filename

def load_data(image_dir):
    """
    Creates lists of image file paths, labels, and original filenames for a given directory structure.

    Args:
        image_dir (str): The path to the directory containing labeled subdirectories of images.
            Each subdirectory should correspond to a class label, with its name as the label key.

    Returns:
        tuple:
            image_files (list of str): List of full paths to image files.
            labels (list of int): List of integer labels corresponding to each image, 
              as mapped by the label_map dictionary.
            original_filenames (list of str): List of original filenames of the images 
    """
    image_files = []
    labels = []
    original_filenames = []
    
    for label, label_value in label_map.items():
        label_dir = os.path.join(image_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            image_files.append(img_path)
            labels.append(label_value)
            original_filenames.append(img_file)  # Keep track of the original filename

    return image_files, labels, original_filenames

# Get the file paths, labels, and filenames
image_files, labels, original_filenames = load_data(image_dir)

# Convert lists to TensorFlow dataset
file_paths_ds = tf.data.Dataset.from_tensor_slices((image_files, labels, original_filenames))
dataset = file_paths_ds.map(lambda x, y, z: preprocess_image(x, y, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle, batch, and prefetch
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def generate_and_save_masks(dataset, output_dir=output_dir):
    """
    Generates binary masks from an input dataset, processes them, and saves them to an output directory.

    Args:
        dataset (tf.data.Dataset): A dataset containing batches of images, labels, and filenames.
            Each batch is expected to be a tuple of:
                batch_images (tf.Tensor): A batch of images in the range [0, 1].
                batch_labels (tf.Tensor): Corresponding labels for the images.
                batch_filenames (tf.Tensor): Original filenames of the images.
        output_dir (str, optional): The directory where the generated masks will be saved. 
            Defaults to output_dir.

    Returns:
        None: This function does not return any value. Masks are saved to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for batch_images, batch_labels, batch_filenames in dataset:
        batch_images_np = batch_images.numpy() * 255  # Convert back to 0-255 range for OpenCV
        batch_images_gray = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in batch_images_np]

        # Apply adaptive thresholding for better detail retention
        batch_masks = [cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2) for img in batch_images_gray]
        
        # Enhance details using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        batch_masks = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for mask in batch_masks]

        # Resize masks to match target size (496x248)
        batch_masks = [cv2.resize(mask, (target_size[1], target_size[0])) for mask in batch_masks]  # Width, Height order

        # Save each mask in the batch
        for i in range(batch_images.shape[0]):
            mask = batch_masks[i] * 255  # Convert to [0, 255] range 
            label = batch_labels[i].numpy()
            original_filename = batch_filenames[i].numpy().decode('utf-8')  # Convert filename to string

            # Naming convention that titles mask, label_map label, and original filename name for correspondence
            file_name = f"mask_{label}_{original_filename}"
            cv2.imwrite(os.path.join(output_dir, file_name), mask)

generate_and_save_masks(dataset)

