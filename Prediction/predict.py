#Prediction file

import tensorflow as tf
import numpy as np
import os
import torch
import cv2

#Define classes
classes = ['no_dementia', 'very_mild_dementia', 'mild_dementia', 'moderate_dementia']

#Load pretrained models
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)  #Load TensorFlow Keras model
    return model

#Perform prediction
def predict(model, image_tensor):
    #Perform the prediction
    output = model(image_tensor)  #Shape of output could be (1, 224, 224, 3) or (1, num_classes)

    #If it's a multi-class classification problem (e.g., segmentation), get the class with the highest probability
    if len(output.shape) == 4 and output.shape[-1] > 1:  #multi-class output (e.g., segmentation)
        predicted_class = np.argmax(output, axis=-1)  #(1, 224, 224), find the max class for each pixel
    else:  #single-class output, classification
        predicted_class = np.argmax(output, axis=-1)  # (1, num_classes)

    #For segmentation, predicted_class will have shape (1, height, width)
    if predicted_class.ndim == 3:  #For segmentation, predicted_class might be (1, height, width)
        predicted_class = predicted_class[0]  #Remove batch dimension

    #For segmentation, there might be multiple pixel-wise predictions.
    #Convert to human-readable class labels
    classes = ['no_dementia', 'very_mild_dementia', 'mild_dementia', 'moderate_dementia']

    #If it's segmentation, return the predicted class map
    if len(predicted_class.shape) == 2:
        return predicted_class  #Return the whole class map

    #Otherwise, it's a single class classification
    return classes[predicted_class.item()]  #Convert to human-readable class label

#Load images and corresponding masks
def load_images_and_masks(image_dir, masks_dir):
    #Check if image_dir is a directory or file
    if image_dir.endswith('.npy'):
        images = np.load(image_dir)  #Load images directly from the .npy file
    else:
        #Directory loading logic as before
        images = []
        for img_file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_file)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (224, 224))  #Resize image if necessary
                    images.append(img)
        images = np.array(images)

    #Similarly for masks
    if not masks_dir.endswith('.npy'):
        masks = []
        for mask_file in os.listdir(masks_dir):
            mask_path = os.path.join(masks_dir, mask_file)
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (224, 224))  #Resize mask if necessary
                masks.append(mask)
        masks = np.array(masks)
    else:
        masks = np.load(masks_dir)  #Load masks directly if in .npy format

    return images, masks

#Main function to run predictions
def main(test_images_dir, masks_dir, unet_model_path, resnet_model_path):
    #Load models
    unet_model = load_model(unet_model_path)
    resnet_model = load_model(resnet_model_path)

    #Load masked images (this returns a tuple: test_images, masks)
    test_images, masks = load_images_and_masks(test_images_dir, masks_dir)

    #Normalize test images to [0, 1]
    test_images = test_images.astype('float32') / 255.0  

    #Convert to PyTorch tensor (NCHW format: Batch x Channels x Height x Width)
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)

    #Convert PyTorch tensor to NumPy array with correct shape (NHWC format: Batch x Height x Width x Channels)
    test_images_np = test_images_tensor.permute(0, 2, 3, 1).numpy()

    #Perform predictions using the TensorFlow/Keras model
    #Add batch dimension for each image before passing to the model
    unet_predictions = [predict(unet_model, np.expand_dims(img, axis=0)) for img in test_images_np]
    resnet_predictions = [predict(resnet_model, np.expand_dims(img, axis=0)) for img in test_images_np]

    #Output predictions
    for i, (unet_pred, resnet_pred) in enumerate(zip(unet_predictions, resnet_predictions)):
        print(f"Image {i}: U-Net Prediction: {unet_pred}, ResNet Prediction: {resnet_pred}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict dementia stages using U-Net and ResNet models.')
    parser.add_argument('--test_images_dir', type=str, required=True, help='Path to the .npy file containing preprocessed test images.')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to masks directory.')
    parser.add_argument('--unet_model_path', type=str, required=True, help='Path to the pretrained U-Net model.')
    parser.add_argument('--resnet_model_path', type=str, required=True, help='Path to the pretrained ResNet model.')
    args = parser.parse_args()

    #Ensure input .npy file exists
    if not os.path.exists(args.test_images_dir):
        print(f"Error: The directory '{args.test_images_dir}' does not exist.")
        exit(1)
    if not os.path.exists(args.masks_dir):
        print(f"Error: The directory '{args.masks_dir}' does not exist.")
        exit(1)

    #Run prediction
    main(args.test_images_dir, args.masks_dir, args.unet_model_path, args.resnet_model_path)
