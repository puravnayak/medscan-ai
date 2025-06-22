import React from "react";
import Chatbot from "./components/ChatBot";

function App() {
  return (
    <div className="App">
      <h1 style={{ textAlign: "center", marginTop: "20px", fontSize: "1.8em" }}>
        💬 MedScan.AI
      </h1>

      <Chatbot />
    </div>
  );
}

export default App;
