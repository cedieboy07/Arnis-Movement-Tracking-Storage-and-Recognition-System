from collections import deque
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np
import torch


class MoveLibrary:
    def __init__(self, storage_dir="moves_db", history_size=5, log_cooldown_sec=1.5):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.compilations = {}       # {comp_name: {move_name: payload}}
        self.active_compilation = None
        self.history = deque(maxlen=history_size)
        self.log_cooldown_sec = log_cooldown_sec
        self._last_logged_move = None
        self._last_logged_time = 0.0
        self.load_all()

    @property
    def moves(self):
        """Returns the moves dict for the currently selected compilation (or all moves if 'ALL')."""
        if self.active_compilation == "ALL":
            merged = {}
            for comp_moves in self.compilations.values():
                merged.update(comp_moves)
            return merged
        if self.active_compilation and self.active_compilation in self.compilations:
            return self.compilations[self.active_compilation]
        return {}

    @staticmethod
    def _sanitize_filename(name):
        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip()
        return safe.replace(" ", "_") or "Untitled"

    def _move_path(self, name, compilation=None):
        comp = self._sanitize_filename(compilation or self.active_compilation or "Custom_Stances")
        folder = self.storage_dir / comp
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{self._sanitize_filename(name)}.pt"

    def save_move(self, name, feature_vector, raw_landmarks=None, raw_sequence=None, compilation="Custom_Stances"):
        """
        Saves a recorded or imported stance inside moves_db/<compilation>/<name>.pt.
        """
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Move name cannot be empty.")

        vec = np.asarray(feature_vector, dtype=np.float32)
        if vec.shape != (115,):
            raise ValueError(f"Expected 115-D feature vector, got shape {vec.shape}.")

        comp_name = self._sanitize_filename(compilation)
        payload = {
            "name": clean_name,
            "compilation": comp_name,
            "vector": torch.from_numpy(vec),
            "raw": np.asarray(raw_landmarks, dtype=np.float32) if raw_landmarks is not None else None,
            "raw_sequence": np.asarray(raw_sequence, dtype=np.float32) if raw_sequence is not None else None,
            "created_at": datetime.now().isoformat(),
        }
        path = self._move_path(clean_name, compilation=comp_name)
        torch.save(payload, path)

        self.compilations.setdefault(comp_name, {})[clean_name] = payload
        if self.active_compilation is None:
            self.active_compilation = comp_name
        print(f"[SAVED] Stance '{clean_name}' stored in {path}")

    def load_all(self):
        """Loads all compilations from subfolders inside moves_db/."""
        self.compilations.clear()

        # Migrate any loose root-level .pt files into moves_db/Custom_Stances/
        root_pts = list(self.storage_dir.glob("*.pt"))
        if root_pts:
            default_dir = self.storage_dir / "Custom_Stances"
            default_dir.mkdir(parents=True, exist_ok=True)
            for pt_file in root_pts:
                shutil.move(str(pt_file), str(default_dir / pt_file.name))

        for folder in sorted(p for p in self.storage_dir.iterdir() if p.is_dir()):
            comp_name = folder.name
            comp_moves = {}
            for file_path in sorted(folder.glob("*.pt")):
                try:
                    payload = torch.load(file_path, map_location="cpu", weights_only=False)
                    name = payload.get("name", file_path.stem)
                    if payload["vector"].shape == (115,):
                        payload["compilation"] = comp_name
                        comp_moves[name] = payload
                except Exception as exc:
                    print(f"[WARN] Skipped corrupted stance file {file_path.name}: {exc}")
            if comp_moves:
                self.compilations[comp_name] = comp_moves

        comp_names = list(self.compilations.keys())
        if comp_names and self.active_compilation not in comp_names and self.active_compilation != "ALL":
            self.active_compilation = comp_names[0]
        elif not comp_names:
            self.active_compilation = None

        total = sum(len(m) for m in self.compilations.values())
        print(f"[LIBRARY] Loaded {total} stance(s) across {len(comp_names)} compilation(s): {comp_names}")

    def get_compilation_names(self):
        return list(self.compilations.keys())

    def set_active_compilation(self, comp_name):
        if comp_name == "ALL" or comp_name in self.compilations:
            self.active_compilation = comp_name

    def get_total_move_count(self):
        return sum(len(m) for m in self.compilations.values())

    def delete_move(self, name):
        """Deletes a saved stance from its compilation folder and memory."""
        target_comp = None
        for comp_name, comp_moves in self.compilations.items():
            if name in comp_moves:
                target_comp = comp_name
                break
        if target_comp is None:
            return False

        path = self._move_path(name, compilation=target_comp)
        if path.exists():
            path.unlink()
        self.compilations[target_comp].pop(name, None)
        if not self.compilations[target_comp]:
            self.compilations.pop(target_comp, None)
            if self.active_compilation == target_comp:
                rem = self.get_compilation_names()
                self.active_compilation = rem[0] if rem else None
        return True

    def get_move_names(self):
        return list(self.moves.keys())

    def log_detected_move(self, move_name, score):
        """Appends a newly detected move to the recent history feed without spamming every frame."""
        if not move_name:
            return
        now = datetime.now().timestamp()
        if move_name == self._last_logged_move and (now - self._last_logged_time) < self.log_cooldown_sec:
            return
        self._last_logged_move = move_name
        self._last_logged_time = now
        self.history.append({
            "name": move_name,
            "score": round(score, 1),
            "time": datetime.now().strftime("%H:%M:%S"),
        })