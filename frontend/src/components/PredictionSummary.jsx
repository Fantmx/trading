// src/components/PredictionSummary.jsx
import React, { useEffect, useState } from "react";

const PredictionSummary = ({ asset }) => {
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const res = await fetch(`/api/trade/prediction?asset=${encodeURIComponent(asset)}`);
;
        const json = await res.json();
        setPrediction(json);
      } catch (err) {
        console.error("Failed to fetch prediction:", err);
      }
    };

    fetchPrediction();
  }, [asset]);

  if (!prediction) {
    return (
      <div className="bg-gray-700 rounded p-4 mb-6">
        <p>Loading prediction...</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-700 rounded p-4 mb-6">
      <h2 className="text-xl font-semibold mb-2">Prediction for {asset}</h2>
      <p>
        <strong>Action:</strong> {prediction.action} <br />
        <strong>Confidence:</strong> {prediction.confidence}%
      </p>
    </div>
  );
};

export default PredictionSummary;
