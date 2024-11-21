import os
import argparse
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

def load_models(unet_model_path, resnet_model_path):
    print(f"Loading U-Net model from: {unet_model_path}")
    print(f"Loading ResNet model from: {resnet_model_path}")
    unet_model = tf.keras.models.load_model(unet_model_path)
    resnet_model = tf.keras.models.load_model(resnet_model_path)
    return unet_model, resnet_model

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        #Read the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))  #Resize to match model input
        image = np.array(image) / 255.0  #Normalize the image
        image = np.expand_dims(image, axis=0)  #Add batch dimension

        #Get predictions from both models
        unet_prediction = unet_model.predict(image)
        resnet_prediction = resnet_model.predict(image)

        #Assuming your models output classification labels, modify accordingly
        unet_predicted_class = np.argmax(unet_prediction, axis=-1)
        resnet_predicted_class = np.argmax(resnet_prediction, axis=-1)

        return jsonify({
            'unet_prediction': int(unet_predicted_class[0]),
            'resnet_prediction': int(resnet_predicted_class[0])
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")  #Log errors to console
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Flask app for MRI prediction.")
    parser.add_argument('--unet_model_path', type=str, required=True, 
                        help='Path to the U-Net model file.')
    parser.add_argument('--resnet_model_path', type=str, required=True, 
                        help='Path to the ResNet model file.')

    args = parser.parse_args()
    unet_model, resnet_model = load_models(args.unet_model_path, args.resnet_model_path)
    app.run(debug=True)
