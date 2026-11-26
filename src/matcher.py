import numpy as np
import torch
import torch.nn.functional as F


class PoseMatcher:
    def __init__(self, threshold=0.80):
        """
        Compares a live 115-D normalized feature vector against saved stances
        using PyTorch L2-normalized cosine similarity.
        """
        self.threshold = threshold

    def match(self, live_vector, saved_moves):
        """
        Data Contract:
          Input: live_vector (np.ndarray shape (115,)), saved_moves (dict from MoveLibrary)
          Output: (best_move_name or None, score_percent float 0.0-100.0)
        """
        if live_vector is None or not saved_moves:
            return None, 0.0

        live_tensor = torch.as_tensor(live_vector, dtype=torch.float32).flatten()
        if live_tensor.shape[0] != 115 or torch.norm(live_tensor) < 1e-6:
            return None, 0.0

        names = list(saved_moves.keys())
        bank_tensors = torch.stack([saved_moves[name]["vector"].float() for name in names])

        similarities = F.cosine_similarity(live_tensor.unsqueeze(0), bank_tensors, dim=1)
        best_idx = int(torch.argmax(similarities).item())
        best_sim = float(similarities[best_idx].item())

        score_percent = max(0.0, min(100.0, best_sim * 100.0))
        if best_sim >= self.threshold:
            return names[best_idx], score_percent

        return None, score_percent