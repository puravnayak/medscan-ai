import React, { useState, useEffect, useRef } from "react";
import "./ChatBot.css";


// const BASE_URL = "http://127.0.0.1:8000"; // Local testing
const BASE_URL = "https://medscan-ai.onrender.com"; // Deployed backend


const getOrCreateSessionId = () => {
  const existing = localStorage.getItem("medscan_session_id");
  if (existing) return existing;

  const newId = `user-${Date.now()}`;
  localStorage.setItem("medscan_session_id", newId);
  return newId;
};

const DEFAULT_BOT_MESSAGE = {
  from: "bot",
  text: "Hi! Describe your symptoms to get started.",
};

const Chatbot = () => {
  const sessionId = useRef(getOrCreateSessionId());
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState(() => {
    const stored = localStorage.getItem("medscan_messages");
    return stored ? JSON.parse(stored) : [DEFAULT_BOT_MESSAGE];
  });
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem("medscan_messages", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { from: "user", text: input }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId.current,
          symptom_text: input,
        }),
      });

      const data = await res.json();
      const predictions = data.predictions
        .map(
          (item, idx) =>
            `${idx + 1}. ${item.disease} — ${Math.round(item.probability * 100)}%`
        )
        .join("\n");

      let botReply = `Possible conditions:\n${predictions}`;
      if (data.disclaimer) botReply += `\n\n⚠️ ${data.disclaimer}`;

      setMessages([...newMessages, { from: "bot", text: botReply }]);
    } catch (err) {
      setMessages([
        ...newMessages,
        {
          from: "bot",
          text: "❌ Error fetching prediction. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const resetSession = async () => {
    await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId.current,
        symptom_text: "",
      }),
    });

    localStorage.removeItem("medscan_messages");
    setMessages([DEFAULT_BOT_MESSAGE]);
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="chatbot-container">
        <h2 style={{ textAlign: "center", padding: "16px", margin: 0, background: "#1976d2", color: "white" }}>
            🩺 MedScan.AI - Symptom Checker
        </h2>
      <div className="chat-window">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.from}`}>
            {msg.text.split("\n").map((line, idx) => (
              <p key={idx}>{line}</p>
            ))}
          </div>
        ))}
        {loading && <div className="message bot">⏳ Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <input
          type="text"
          placeholder="Describe your symptoms..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button onClick={sendMessage}>Send</button>
        <button
          onClick={resetSession}
          style={{
            marginLeft: "10px",
            backgroundColor: "#e74c3c",
            color: "white",
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
