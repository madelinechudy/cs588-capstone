# predict.py
import tensorflow as tf
import numpy as np
import os
import cv2

# Define classes
classes = ['no_dementia', 'very_mild_dementia', 'mild_dementia', 'moderate_dementia']

# Load pretrained model
def load_model(model_path):
    """
    Parameters:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: Loaded Keras model.
    """
    return tf.keras.models.load_model(model_path)

# Perform prediction
def predict(model, image_tensor):
    """
    Perform prediction using model on a single image tensor.

    Parameters:
        model (tf.keras.Model): Pretrained model.
        image_tensor (np.ndarray): Input image tensor with shape (1, height, width, channels).

    Returns:
        str: Predicted class label.
    """
    output = model(image_tensor)  # Get model output
    print(f"Raw model output: {output.numpy()}")  # Print the raw output for debugging

    # Check if the output is softmax probabilities for multi-class classification
    if output.shape[-1] > 1:
        predicted_class = np.argmax(output.numpy(), axis=-1)
    else:  # For binary classification (if needed)
        predicted_class = (output.numpy() > 0.5).astype(int)

    return classes[predicted_class.item()]

# Preprocess image to match training pipeline
def preprocess_image(image_path):
    """
    Preprocess the input image for prediction.

    Parameters:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: Preprocessed image tensor ready for prediction.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    img = cv2.resize(img, (496, 248))  # Resize to model input size
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Main function to run a single prediction
def main(image_path, resnet_model_path):
    """
    Main function to load data, perform predictions, and display results.

    Parameters:
        image_path (str): Path to the input image file.
        resnet_model_path (str): Path to the pretrained ResNet model.
    """
    # Load model
    resnet_model = load_model(resnet_model_path)

    # Preprocess the image
    img_tensor = preprocess_image(image_path)
    print(f"Image shape before prediction: {img_tensor.shape}")

    # Perform prediction
    prediction = predict(resnet_model, img_tensor)
    print(f"Predicted Class: {prediction}")

    return prediction

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict dementia stages using ResNet model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--resnet_model_path', type=str, required=True, help='Path to the pretrained ResNet model.')
    args = parser.parse_args()

    # Ensure input files exist
    if not os.path.exists(args.image_path):
        print(f"Error: The file '{args.image_path}' does not exist.")
        exit(1)
    if not os.path.exists(args.resnet_model_path):
        print(f"Error: The file '{args.resnet_model_path}' does not exist.")
        exit(1)

    # Run prediction
    main(args.image_path, args.resnet_model_path)