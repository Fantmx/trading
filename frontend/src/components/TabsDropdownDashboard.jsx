// TabsDropdownDashboard.js
import React, { useState, useEffect } from "react";
import ChartComponent from "./ChartComponent"; // You should implement this separately
import PredictionSummary from "./PredictionSummary"; // You should implement this separately

const assetCategories = {
  Crypto: ["BTC/USD", "ETH/USD"],
  Stocks: ["AAPL", "GOOGL", "TSLA"],
  Bonds: ["BND", "TLT"],
  Currency: ["EUR/USD", "JPY/USD", "GBP/USD"]
};

const TabsDropdownDashboard = () => {
  const [activeTab, setActiveTab] = useState("Crypto");
  const [selectedAsset, setSelectedAsset] = useState(assetCategories["Crypto"][0]);

  useEffect(() => {
    setSelectedAsset(assetCategories[activeTab][0]);
  }, [activeTab]);

  return (
    <div className="p-6 text-white bg-gray-900 min-h-screen">
      <h1 className="text-4xl font-bold mb-4">AI Trading Dashboard</h1>

      {/* Tabs */}
      <div className="flex space-x-4 mb-4">
        {Object.keys(assetCategories).map((category) => (
          <button
            key={category}
            onClick={() => setActiveTab(category)}
            className={`px-4 py-2 rounded font-semibold transition duration-200 ease-in-out ${
              activeTab === category
                ? "bg-cyan-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {category}
          </button>
        ))}
      </div>

      {/* Dropdown */}
      <div className="mb-6">
        <label className="mr-2 font-semibold">Select Asset:</label>
        <select
          value={selectedAsset}
          onChange={(e) => setSelectedAsset(e.target.value)}
          className="p-2 bg-gray-800 text-white rounded border border-gray-700"
        >
          {assetCategories[activeTab].map((asset) => (
            <option key={asset} value={asset}>
              {asset}
            </option>
          ))}
        </select>
      </div>

      {/* Prediction Summary */}
      <PredictionSummary asset={selectedAsset} />

      {/* Chart */}
      <ChartComponent asset={selectedAsset} />
    </div>
  );
};

export default TabsDropdownDashboard;
