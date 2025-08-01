// src/components/ChartComponent.jsx
import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const ChartComponent = ({ asset }) => {
  const [data, setData] = useState([]);
  const [timeInterval, setTimeInterval] = useState("daily");
  const [days, setDays] = useState(30);

  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      try {
        const res = await fetch(
          `/api/trade/price-history?asset=${encodeURIComponent(asset)}&days=${days}&interval=${timeInterval}`
        );
        if (!res.ok) throw new Error(`Error ${res.status}: ${res.statusText}`);
        const json = await res.json();
        if (isMounted) setData(json);
      } catch (err) {
        console.error("Failed to fetch price data:", err);
        setData([]);
      }
    };

    fetchData(); // Initial load
    const id = setInterval(fetchData, 30000); // Refresh every 30s

    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, [asset, days, timeInterval]);

  return (
    <div className="bg-gray-800 rounded p-4 mb-6">
      <h2 className="text-xl font-semibold mb-2">Price Chart for {asset}</h2>

      <div className="flex gap-4 mb-4">
        <select
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="bg-gray-700 text-white p-1 rounded"
        >
          <option value={1}>1 Day</option>
          <option value={7}>7 Days</option>
          <option value={30}>30 Days</option>
          <option value={90}>90 Days</option>
        </select>

        <select
          value={timeInterval}
          onChange={(e) => setTimeInterval(e.target.value)}
          className="bg-gray-700 text-white p-1 rounded"
        >
          <option value="hourly">Hourly</option>
          <option value="daily">Daily</option>
        </select>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" tick={{ fill: "#aaa" }} />
          <YAxis tick={{ fill: "#aaa" }} domain={["auto", "auto"]} />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#00bcd4"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ChartComponent;
