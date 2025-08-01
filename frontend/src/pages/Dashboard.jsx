import React, { useEffect, useState } from "react";
import PriceChart from "../components/PriceChart";

function Dashboard() {
  const [signal, setSignal] = useState(null);
  const [dataPoints, setDataPoints] = useState([
    { time: "09:00", price: 29100 },
    { time: "09:30", price: 29130 },
    { time: "10:00", price: 29080 },
    { time: "10:30", price: 29160 },
    { time: "11:00", price: 29210 },
  ]);

  useEffect(() => {
    fetch("http://localhost:8000/api/trade/signal")
      .then(res => res.json())
      .then(setSignal);
  }, []);

  return (
    <div style={{ padding: "2rem" }}>
      <h1><strong>AI Trading Dashboard</strong></h1>
      {signal ? (
        <p>{signal.symbol}: {signal.action} with {Math.round(signal.confidence * 100)}% confidence</p>
      ) : (
        <p>Loading signal...</p>
      )}

      <PriceChart dataPoints={dataPoints} />
    </div>
  );
}

export default Dashboard;
