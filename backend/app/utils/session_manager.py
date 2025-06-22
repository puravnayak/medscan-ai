from collections import defaultdict

_session_store = defaultdict(str)

def append_symptoms(session_id: str, new_text: str) -> str:
    _session_store[session_id] += " " + new_text.strip()
    return _session_store[session_id]

def reset_session(session_id: str):
    _session_store[session_id] = ""

def get_full_symptom_text(session_id: str) -> str:
    return _session_store[session_id].strip()
