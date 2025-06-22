import React from 'react';

function MessageList({ messages }) {
  return (
    <div style={{ maxHeight: 400, overflowY: "auto", marginBottom: 10 }}>
      {messages.map((msg, idx) => (
        <div
          key={idx}
          style={{
            backgroundColor: msg.sender === "user" ? "#e0f7fa" : "#f1f8e9",
            padding: 10,
            borderRadius: 8,
            marginBottom: 5
          }}
        >
          <strong>{msg.sender === "user" ? "You" : "MedScan.AI"}</strong>:<br />
          <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{msg.text}</pre>
        </div>
      ))}
    </div>
  );
}

export default MessageList;
