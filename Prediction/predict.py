import torch
import numpy as np
import os

#Define classes
classes = ['no_dementia', 'very_mild_dementia', 'mild_dementia', 'moderate_dementia']

#Load pretrained models
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

#Perform prediction
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

#Main function to run predictions
def main(test_images_path, unet_model_path, resnet_model_path):
    #Load models
    unet_model = load_model(unet_model_path)
    resnet_model = load_model(resnet_model_path)

    #Load preprocessed images
    test_images = np.load(test_images_path)
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32)

    #Ensure model input shape matches
    if len(test_images_tensor.shape) == 3:  #If only one channel
        test_images_tensor = test_images_tensor.unsqueeze(1)  #Add channel dimension if needed

    #Perform predictions for each image
    unet_predictions = []
    resnet_predictions = []

    for image_tensor in test_images_tensor:
        #Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        #Get predictions from both models
        unet_prediction = predict(unet_model, image_tensor)
        resnet_prediction = predict(resnet_model, image_tensor)

        unet_predictions.append(unet_prediction)
        resnet_predictions.append(resnet_prediction)

    #Output predictions
    for i, (unet_pred, resnet_pred) in enumerate(zip(unet_predictions, resnet_predictions)):
        print(f'Image {i}: U-Net Prediction: {unet_pred}, ResNet Prediction: {resnet_pred}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict dementia stages using U-Net and ResNet models.')
    parser.add_argument('../cs588-capstone/Data/Processed/test_images.npy', type=str)
    parser.add_argument('../cs588-capstone/Segmentation/Models/pretrained_unet_model.keras', type=str)
    parser.add_argument('../cs588-capstone/Segmentation/Models/pretrained_resnet_model.keras', type=str)

    args = parser.parse_args()

    #Ensure input .npy file exists
    if not os.path.exists(args.test_images_path):
        print(f"Error: The file '{args.test_images_path}' does not exist.")
        exit(1)

    #Run prediction
    main(args.test_images_path, args.unet_model_path, args.resnet_model_path)
