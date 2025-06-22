import React, { useState } from 'react';

function MessageInput({ onSend }) {
  const [input, setInput] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", gap: 10 }}>
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        placeholder="Describe your symptoms..."
        style={{ flex: 1, padding: 10, fontSize: 16 }}
      />
      <button type="submit" style={{ padding: "0 20px", fontSize: 16 }}>
        Send
      </button>
    </form>
  );
}

export default MessageInput;
