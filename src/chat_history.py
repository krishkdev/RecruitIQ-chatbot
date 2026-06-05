import os
import json
from datetime import datetime


class ChatHistory:
    def __init__(self, log_dir="chat_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{session_id}.json")
        self.messages = []

    def add(self, role, content, mode):
        entry = {
            "role": role,
            "content": content,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        }
        self.messages.append(entry)
        with open(self.filepath, "w") as f:
            json.dump(self.messages, f, indent=2)

    def load_session(self, filepath):
        with open(filepath, "r") as f:
            self.messages = json.load(f)

    @staticmethod
    def list_sessions(log_dir="chat_logs"):
        if not os.path.exists(log_dir):
            return []
        files = [
            os.path.join(log_dir, fname)
            for fname in os.listdir(log_dir)
            if fname.endswith(".json")
        ]
        return sorted(files, reverse=True)
