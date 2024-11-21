import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; 

function App() {
  const [image, setImage] = useState(null);
  const [predictions, setPredictions] = useState(null); //Store both predictions
  const [errorMessage, setErrorMessage] = useState(''); //To display any errors

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    setImage(file);
    setPredictions(null); //Clear previous predictions
    setErrorMessage(''); //Clear previous errors
  };

  const handleSubmit = async () => {
    if (!image) {
      setErrorMessage('Please select an image before submitting.');
      return;
    }

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      //Handle predictions from both models
      setPredictions({
        unet: response.data.unet_prediction,
        resnet: response.data.resnet_prediction,
      });
    } catch (error) {
      console.error('Error uploading image:', error);
      setErrorMessage('There was an error processing your request. Please try again.');
    }
  };

  return (
    <div className="App">
      <h1>MRI Image Prediction</h1>

      <input type="file" onChange={handleImageChange} />
      <button onClick={handleSubmit}>Submit</button>

      {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}

      {predictions && (
        <div>
          <h2>Predictions</h2>
          <p><strong>U-Net Prediction:</strong> {predictions.unet}</p>
          <p><strong>ResNet Prediction:</strong> {predictions.resnet}</p>
        </div>
      )}
    </div>
  );
}

export default App;
