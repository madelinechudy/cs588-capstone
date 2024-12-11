import tensorflow as tf
import numpy as np
import os
import cv2

#Define classes
classes = ['no_dementia', 'very_mild_dementia', 'mild_dementia', 'moderate_dementia']

#Load pretrained model
def load_model(model_path):
    """
    Parameters:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: Loaded Keras model.
    """
    return tf.keras.models.load_model(model_path)

#Perform prediction
def predict(model, image_tensor):
    """
    Perform prediction using model on a single image tensor.

    Parameters:
        model (tf.keras.Model): Pretrained model.
        image_tensor (np.ndarray): Input image tensor with shape (1, height, width, channels).

    Returns:
        str: Predicted class label.
    """
    output = model(image_tensor)  #Get model output
    print(f"Raw model output: {output.numpy()}")  #Print the raw output for debugging

    #Check if the output is softmax probabilities for multi-class classification
    if output.shape[-1] > 1:
        predicted_class = np.argmax(output.numpy(), axis=-1)
    else:  #For binary classification (if needed)
        predicted_class = (output.numpy() > 0.5).astype(int)

    return classes[predicted_class.item()]

#Generator function to load images in batches (no masks needed for test set)
def load_images_in_batches(image_dir, batch_size):
    """
    Load images from directory in batches to save memory. This function
    will recursively walk through subdirectories to find images.

    Parameters:
        image_dir (str): Path to the directory containing test images.
        batch_size (int): The batch size to load at once.

    Yields:
        np.ndarray: A batch of images.
    """
    images = []

    #Walk through all subdirectories in the image directory
    for root, _, files in os.walk(image_dir):
        for img_file in files:
            img_path = os.path.join(root, img_file)
            if os.path.isfile(img_path):
                #Read and resize image, convert to grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #Read as grayscale
                if img is not None:
                    img = cv2.resize(img, (496, 248))  #Resize to model input size (width, height)
                    images.append(img)

            #Once the batch size is met, yield the batch
            if len(images) == batch_size:
                yield np.array(images)[..., np.newaxis]  #Add channel dimension (for grayscale)
                images = []  #Reset list after yielding

    #Yield any remaining images that didn't fill a complete batch
    if len(images) > 0:
        yield np.array(images)[..., np.newaxis]  #Add channel dimension (for grayscale)

#Main function to run predictions (modified for test images only)
def main(test_images_dir, resnet_model_path, batch_size):
    """
    Main function to load data, perform predictions, and display results.

    Parameters:
        test_images_dir (str): Path to the test images directory.
        resnet_model_path (str): Path to the pretrained ResNet model.
        batch_size (int): The batch size for processing images.
    """
    #Load model
    resnet_model = load_model(resnet_model_path)

    #Load test images in batches
    for batch_idx, batch_images in enumerate(load_images_in_batches(test_images_dir, batch_size)):
        print(f"Processing batch {batch_idx + 1} with {batch_images.shape[0]} images.")  #Debug print
        
        #Normalize test images to [0, 1] just like during training
        batch_images = batch_images.astype('float32') / 255.0

        #Perform predictions for each batch of images
        for i, img in enumerate(batch_images):
            img_tensor = np.expand_dims(img, axis=0)  #Add batch dimension (shape: (1, 248, 496, 1))
            
            #Check shape and values of the input tensor for debugging
            print(f"Image shape before prediction: {img_tensor.shape}")
            
            prediction = predict(resnet_model, img_tensor)
            print(f"Batch {batch_idx + 1}, Image {i + 1}: Predicted Class: {prediction}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict dementia stages using ResNet model.')
    parser.add_argument('--test_images_dir', type=str, required=True, help='Path to the directory containing test images.')
    parser.add_argument('--resnet_model_path', type=str, required=True, help='Path to the pretrained ResNet model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images.')
    args = parser.parse_args()

    #Ensure input directories/files exist
    if not os.path.exists(args.test_images_dir):
        print(f"Error: The directory '{args.test_images_dir}' does not exist.")
        exit(1)
    if not os.path.exists(args.resnet_model_path):
        print(f"Error: The file '{args.resnet_model_path}' does not exist.")
        exit(1)

    #Run prediction
    main(args.test_images_dir, args.resnet_model_path, args.batch_size)
