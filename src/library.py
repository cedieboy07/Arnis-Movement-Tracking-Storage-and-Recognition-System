import os
import torch
from collections import deque


class MoveLibrary:
    def __init__(self, db_dir="moves_db", history_len=6):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.moves = {}          # name -> {"vector": Tensor, "raw": np.ndarray}
        self.history = deque(maxlen=history_len)
        self.load_all()

    def save_move(self, name, normalized_vec, raw_landmarks):
        safe_name = name.strip().replace(" ", "_")
        path = os.path.join(self.db_dir, f"{safe_name}.pt")
        payload = {
            "name": name,
            "vector": torch.tensor(normalized_vec, dtype=torch.float32),
            "raw": raw_landmarks,
        }
        torch.save(payload, path)
        self.moves[name] = payload
        print(f"[SAVED] Stance '{name}' stored in {path}")

    def load_all(self):
        self.moves.clear()
        for fname in sorted(os.listdir(self.db_dir)):
            if fname.endswith(".pt"):
                payload = torch.load(os.path.join(self.db_dir, fname), weights_only=False)
                self.moves[payload["name"]] = payload
        print(f"[LOADED] {len(self.moves)} saved stances: {list(self.moves.keys())}")

    def log_detected_move(self, name, score):
        """Appends move to history only if it differs from the most recent entry."""
        if not self.history or self.history[-1][0] != name:
            self.history.append((name, score))

    def get_move_names(self):
        return list(self.moves.keys())