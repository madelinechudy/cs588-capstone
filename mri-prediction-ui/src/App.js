import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    setImage(file);
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('file', image);

    try {
      // Replace 'http://localhost:5000/predict' with the actual backend URL
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className="App">
      <h1>MRI Image Prediction</h1>
      <input type="file" onChange={handleImageChange} />
      <button onClick={handleSubmit}>Submit</button>

      {prediction && (
        <div>
          <h2>Prediction: {prediction}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
