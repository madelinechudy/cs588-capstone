import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def load_images(image_dir):
    """
    Loads and preprocesses images from a given directory.

    Params:
        image_dir (str): Path to the directory containing images.

    Return:
        tuple: A tuple containing an array of preprocessed images and a list of image filenames.
    """
    # Empty arrays: images will hold the np.array data of each image. image_files will hold the image names.
    images = []
    image_files = []

    # Searching through each folder in the given directory.
    for label_dir in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_dir)
        if os.path.isdir(label_path): # If the path is a folder.
            # Taking each image within each folder in the given directory.
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)  # Load image using OpenCV
                if image is not None:
                    image = cv2.resize(image, (224, 224))  # Resize image to 224x224
                    images.append(image)
                    image_files.append(img_file)
    images = np.array(images)  # Convert list of images to NumPy array
    return images, image_files


def save_masks(masks, mask_dir, image_files):
    """
    Saves generated masks to a specified directory with corresponding filenames.

    Params:
        masks (np.array): Array of generated segmentation masks.
        mask_dir (str): Path to the directory where masks will be saved.
        image_files (list): List of image filenames to name the corresponding mask files.
    """
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)  # Create directory if it doesn't exist
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(mask_dir, f"{image_files[idx]}.png")
        cv2.imwrite(mask_path, mask * 255)  # Save mask as a binary image (0 or 255)


def visualize_mask(mask):
    """
    Display a single mask using matplotlib.

    Params:
        mask (np.array): A single mask to be visualized.
    """
    plt.imshow(mask, cmap='gray')  # Display the mask in grayscale
    plt.show()


if __name__ == "__main__":
    # Paths for the images and masks
    image_dir = '../cs588-capstone/Data/Training'
    mask_dir = 'C:/Users/matt/Desktop/train_masks'

    # Load pre-trained segmentation model
    model = tf.keras.models.load_model('../cs588-capstone/Segmentation/Models/pretrained_unet_model.keras')  
    images, image_files = load_images(image_dir)

    # Preprocess images for prediction
    images = images.astype('float32') / 255.0  # Normalize images to [0, 1]
    
    # Generate segmentation masks using the pre-trained model
    masks = model.predict(images)
    
    # Binarize the masks (convert to 0 or 1)
    masks = (masks > 0.5).astype(np.uint8)

    # Visualize some of the generated masks for inspection
    for i in range(5):
        visualize_mask(masks[i, :, :, 0])

    # Rename image files for mask saving
    image_files = [f"img_{idx}" for idx in range(len(images))]

    # Save the generated masks
    save_masks(masks, mask_dir, image_files)