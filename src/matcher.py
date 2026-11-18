import torch
import torch.nn.functional as F


class PoseMatcher:
    def __init__(self, threshold=0.88):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def match(self, live_vec, saved_moves):
        """
        Compares live normalized vector (99,) against saved moves dict.
        Returns (best_name, best_score_percent) or (None, 0.0).
        """
        if live_vec is None or not saved_moves:
            return None, 0.0

        live_tensor = torch.tensor(live_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        best_name = None
        best_score = -1.0

        with torch.no_grad():
            for name, data in saved_moves.items():
                target_tensor = data["vector"].to(self.device).unsqueeze(0)
                score = F.cosine_similarity(live_tensor, target_tensor).item()
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score >= self.threshold:
            return best_name, round(best_score * 100, 1)
        return None, round(best_score * 100, 1)