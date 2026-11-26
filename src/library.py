from collections import Counter, deque
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import torch


class MoveLibrary:
    """Manages saved Arnis stances/sequences (.pt) and recognized session history."""

    FEATURE_DIM = 115
    _NOISE_LABELS = frozenset({"unknown", "none", "null"})
    _SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

    def __init__(self, db_dir="moves_db", history_len=6):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.moves = {}          # name -> {"name": str, "vector": Tensor, "raw": np.ndarray | None}
        self._move_files = {}    # name -> Path
        self.history = deque(maxlen=history_len)
        self.session_log = []
        self.load_all()

    @classmethod
    def _validate_display_name(cls, name):
        """Validates a user-facing move name and rejects empty or path-traversal strings."""
        if not isinstance(name, str):
            raise ValueError("Move name must be a string.")
        cleaned = name.strip()
        if not cleaned or cleaned in {".", ".."}:
            raise ValueError("Move name cannot be empty or '.' / '..'.")
        if any(sep in cleaned for sep in ("/", "\\", "\x00")) or ".." in cleaned:
            raise ValueError(f"Move name contains unsafe path characters: {name!r}")
        return cleaned

    @classmethod
    def _sanitize_name(cls, name):
        """Converts a validated move name into a filesystem-safe stem."""
        cleaned = cls._validate_display_name(name)
        safe = cls._SAFE_NAME_RE.sub("_", cleaned.replace(" ", "_"))
        safe = re.sub(r"_+", "_", safe).strip(".")
        if not safe or not safe.strip("._") or safe in {".", ".."}:
            raise ValueError(f"Move name produces an invalid filename: {name!r}")
        return safe

    def _move_path(self, name):
        """Builds a resolved .pt path strictly inside self.db_dir."""
        safe_name = self._sanitize_name(name)
        db_root = self.db_dir.resolve()
        path = (db_root / f"{safe_name}.pt").resolve()
        if path.parent != db_root:
            raise ValueError(f"Resolved path escapes moves_db directory: {path}")
        return path

    def _resolve_existing_path(self, name):
        """Finds the .pt file for a move by tracked path or sanitized filename."""
        tracked = self._move_files.get(name)
        if tracked is not None and tracked.exists():
            return tracked
        candidate = self._move_path(name)
        if candidate.exists():
            return candidate
        return None

    @classmethod
    def _validate_vector(cls, vec):
        """Validates and converts a stance (115,) or sequence (T, 115) into float32 CPU Tensor."""
        if vec is None:
            raise ValueError("Feature vector cannot be None.")
        try:
            tensor = torch.as_tensor(vec, dtype=torch.float32).detach().cpu().clone()
        except Exception as exc:
            raise ValueError(f"Could not convert vector to float32 Tensor: {exc}") from exc

        if tensor.ndim == 1:
            if tensor.shape != (cls.FEATURE_DIM,):
                raise ValueError(
                    f"Expected 1D stance vector of shape ({cls.FEATURE_DIM},), got {tuple(tensor.shape)}."
                )
        elif tensor.ndim == 2:
            if tensor.shape[0] < 1 or tensor.shape[1] != cls.FEATURE_DIM:
                raise ValueError(
                    f"Expected 2D sequence tensor of shape (T, {cls.FEATURE_DIM}) with T >= 1, "
                    f"got {tuple(tensor.shape)}."
                )
        else:
            raise ValueError(
                f"Expected 1D ({cls.FEATURE_DIM},) or 2D (T, {cls.FEATURE_DIM}) tensor, "
                f"got {tensor.ndim}D shape {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _prepare_raw_landmarks(raw_landmarks, is_sequence):
        """Normalizes raw landmark payloads while keeping (33, 3) HUD compatibility."""
        if raw_landmarks is None:
            return None, None
        arr = (
            raw_landmarks.detach().cpu().numpy().astype(np.float32)
            if torch.is_tensor(raw_landmarks)
            else np.asarray(raw_landmarks, dtype=np.float32)
        )
        if is_sequence and arr.ndim == 3 and len(arr) > 0:
            # Keep a single representative (33, 3) frame in 'raw' so hud.draw_ghost stays compatible.
            return arr[len(arr) // 2].copy(), arr
        return arr, None

    def save_move(self, name, normalized_vec, raw_landmarks=None):
        """Saves a static (115,) stance or dynamic (T, 115) sequence as a .pt file."""
        clean_name = self._validate_display_name(name)
        vector = self._validate_vector(normalized_vec)
        path = self._move_path(clean_name)

        raw_frame, raw_seq = self._prepare_raw_landmarks(
            raw_landmarks, is_sequence=(vector.ndim == 2)
        )
        payload = {
            "name": clean_name,
            "vector": vector,
            "raw": raw_frame,
        }
        if raw_seq is not None:
            payload["raw_sequence"] = raw_seq

        torch.save(payload, path)
        self.moves[clean_name] = payload
        self._move_files[clean_name] = path
        print(f"[SAVED] Stance '{clean_name}' ({tuple(vector.shape)}) stored in {path}")

    def load_all(self):
        """Loads and validates all .pt stance/sequence files from self.db_dir."""
        self.moves.clear()
        self._move_files.clear()
        for path in sorted(self.db_dir.glob("*.pt")):
            try:
                payload = torch.load(path, weights_only=False)
                if not isinstance(payload, dict) or "name" not in payload or "vector" not in payload:
                    raise ValueError("Missing required keys ('name', 'vector').")
                clean_name = self._validate_display_name(payload["name"])
                vector = self._validate_vector(payload["vector"])
            except Exception as exc:
                print(f"[WARN] Skipping invalid move file '{path.name}': {exc}")
                continue

            entry = dict(payload)
            entry["name"] = clean_name
            entry["vector"] = vector
            entry.setdefault("raw", None)
            self.moves[clean_name] = entry
            self._move_files[clean_name] = path
        print(f"[LOADED] {len(self.moves)} saved stances: {list(self.moves.keys())}")

    def delete_move(self, name):
        """Deletes a saved stance/sequence .pt file and removes it from memory."""
        clean_name = self._validate_display_name(name)
        path = self._resolve_existing_path(clean_name)
        in_memory = clean_name in self.moves

        if not in_memory and path is None:
            return False

        if path is not None and path.exists():
            path.unlink()
        self.moves.pop(clean_name, None)
        self._move_files.pop(clean_name, None)
        print(f"[DELETED] Stance '{clean_name}' removed.")
        return True

    def rename_move(self, old_name, new_name):
        """Renames a saved stance/sequence .pt file without overwriting existing targets."""
        old_clean = self._validate_display_name(old_name)
        new_clean = self._validate_display_name(new_name)

        if old_clean == new_clean or old_clean not in self.moves or new_clean in self.moves:
            return False

        old_path = self._resolve_existing_path(old_clean)
        new_path = self._move_path(new_clean)
        if old_path is None or (new_path.exists() and new_path != old_path):
            return False

        entry = dict(self.moves[old_clean])
        entry["name"] = new_clean
        torch.save(entry, new_path)
        if old_path != new_path and old_path.exists():
            old_path.unlink()

        self.moves.pop(old_clean, None)
        self._move_files.pop(old_clean, None)
        self.moves[new_clean] = entry
        self._move_files[new_clean] = new_path
        print(f"[RENAMED] '{old_clean}' -> '{new_clean}' ({new_path})")
        return True

    def log_detected_move(self, name, score):
        """Appends move to rolling history & session log, ignoring noise and consecutive duplicates."""
        if not isinstance(name, str):
            return
        clean_name = name.strip()
        if not clean_name or clean_name.lower() in self._NOISE_LABELS:
            return

        score_val = float(score)
        if not self.history or self.history[-1][0] != clean_name:
            self.history.append((clean_name, score_val))
            self.session_log.append({
                "index": len(self.session_log) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "move": clean_name,
                "score": score_val,
            })

    def export_session_log(self, output_path=None, fmt="json"):
        """Exports the recorded session combo log as JSON or CSV and returns the output Path."""
        if output_path is not None:
            out_path = Path(output_path)
            suffix = out_path.suffix.lower().lstrip(".")
            fmt_norm = suffix if suffix in {"json", "csv"} else fmt.lower().lstrip(".")
        else:
            fmt_norm = fmt.lower().lstrip(".")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = self.db_dir / f"session_combo_{stamp}.{fmt_norm}"

        if fmt_norm not in {"json", "csv"}:
            raise ValueError(f"Unsupported export format: {fmt_norm!r}. Expected 'json' or 'csv'.")

        events = list(self.session_log)
        if not events and self.history:
            events = [
                {"index": idx, "timestamp": "", "move": m_name, "score": float(m_score)}
                for idx, (m_name, m_score) in enumerate(self.history, start=1)
            ]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt_norm == "json":
            combo_sequence = [e["move"] for e in events]
            payload = {
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "summary": {
                    "total_moves": len(events),
                    "unique_moves": len(set(combo_sequence)),
                    "combo_sequence": combo_sequence,
                    "move_counts": dict(Counter(combo_sequence)),
                },
                "events": events,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "timestamp", "move", "score"])
                writer.writeheader()
                writer.writerows(events)

        print(f"[EXPORTED] Session combo log ({len(events)} moves) -> {out_path}")
        return out_path

    def get_move_names(self):
        return list(self.moves.keys())
