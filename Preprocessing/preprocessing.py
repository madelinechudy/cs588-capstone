import os
import numpy as np
import cv2


def load_data(data_dir):
    '''
    Loads and labels image data from a directory for a dementia classification.

    Input:
        data_dir (str): The path to the root directory containing subdirectories of image files. 
                        Each subdirectory is named according to the dementia label it represents.
    
    Returns:
        tuple, containing:
            images: A numpy array of the loaded and resized image data.
            labels: A numpy array of integer labels corresponding to each image based on the condition from label_map.
    '''
    labels = []
    images = []

    # Define label mapping
    label_map = {'no_dementia': 0, 'very_mild_dementia': 1, 'mild_dementia': 2, 'moderate_dementia': 3}

    # Iterate over each label
    for label, value in label_map.items():
        path = os.path.join(data_dir, label)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (224, 224))  # Resize image to fit model input
                images.append(image)
                labels.append(value)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def preprocess_data(images, labels):
    '''
    Normalizes image data and prepares labels.

    Parameters:
        images (np.array): Array of image data, btained from the output of the `load_data()` function.
        labels (np.array): Array of image integer labels, obtained from the output of the `load_data()` function.

    Returns:
        tuple, containing:
            images: The normalized image data as a numpy array.
            labels: A numpy array of labels corresponding to each image.
    '''
    # Normalize pixel values by dividing by 255. 
    images = images.astype('float32') / 255.0
    labels = np.array(labels)

    return images, labels


if __name__ == "__main__":
    train_data_dir = os.path.abspath('../cs588-capstone/Data/Training')
    test_data_dir = os.path.abspath('../cs588-capstone/Data/Testing')

    train_images, train_labels = load_data(train_data_dir)
    test_images, test_labels = load_data(test_data_dir)

    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)

    # Saving of processed data
    if not os.path.exists('../cs588-capstone/Data/Processed'):
        os.makedirs('../cs588-capstone/Data/Processed')

    np.save('../cs588-capstone/Data/Processed/train_images.npy', train_images)
    np.save('../cs588-capstone/Data/Processed/train_labels.npy', train_labels)
    np.save('../cs588-capstone/Data/Processed/test_images.npy', test_images)
    np.save('../cs588-capstone/Data/Processed/test_labels.npy', test_labels)