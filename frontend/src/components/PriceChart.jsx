import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
} from "chart.js";

// Register the required chart components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const PriceChart = ({ dataPoints }) => {
  const data = {
    labels: dataPoints.map((point) => point.time),
    datasets: [
      {
        label: "BTC/USD",
        data: dataPoints.map((point) => point.price),
        fill: false,
        borderColor: "rgb(75, 192, 192)",
        tension: 0.2,
      },
    ],
  };

  return (
    <div style={{ maxWidth: "800px", marginTop: "2rem" }}>
      <Line data={data} />
    </div>
  );
};

export default PriceChart;
