import React, { useState } from "react";
import axios from "axios";
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setPrediction("");
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please upload an image file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      if (response.data.error) {
        setError(response.data.error);
      } else {
        setPrediction(response.data.prediction);
      }
    } catch (err) {
      // Check for network errors
      if (err.response) {
        setError(`Server Error: ${err.response.data.error || "Unknown error occurred."}`);
      } else if (err.request) {
        setError("Network Error: Unable to reach the server. Please check your connection.");
      } else {
        setError(`Error: ${err.message}`);
      }
    }
  };

  return (
    <div className="App">
      <h1>Alzheimer's Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button type="submit">Submit</button>
      </form>
      {error && <div style={{ color: "red" }}>Error: {error}</div>}
      {prediction && <div style={{ color: "green" }}>Prediction: {prediction}</div>}
    </div>
  );
}

export default App;
