from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = None

@app.before_request
def load_model():
    global model
    model_path = r"C:\Users\maddi\Documents\cs588-capstone\Segmentation\Models\ResNet\Classification\resnet_model.keras"
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs for details.'}), 500

    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        # Save and preprocess the image
        image_path = 'uploaded_image.png'
        file.save(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError('Invalid image format or corrupted image.')

        img = cv2.resize(img, (496, 248))  # Resize to match the model input
        img = img.astype('float32') / 255.0  # Normalize to [0, 1]
        img_tensor = np.expand_dims(img, axis=-1)  # Add channel dimension
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(img_tensor)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        class_names = ['no_dementia', 'very_mild_dementia', 'mild_dementia', 'moderate_dementia']
        result = class_names[predicted_class]

        # Return prediction result as JSON
        return jsonify({'prediction': result})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
