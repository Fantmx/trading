// src/App.jsx
import React from "react";
import Dashboard from "./pages/Dashboard";
import TabsDropdownDashboard from "./components/TabsDropdownDashboard";

function App() {
  return (
    <div className="App">
      <TabsDropdownDashboard />
      <Dashboard />
    </div>
  );
}

export default App;
