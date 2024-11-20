from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# Load the model (change path as necessary)
model = tf.keras.models.load_model('path_to_your_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))  # Resize to match model input
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Get model prediction
        prediction = model.predict(image)
        # Assuming your model outputs a classification label, modify accordingly
        predicted_class = np.argmax(prediction, axis=-1)

        return jsonify({'prediction': str(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
