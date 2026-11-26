import torch
import torch.nn.functional as F


class PoseMatcher:
    def __init__(self, threshold=0.88):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def match(self, live_vec, saved_moves):
        """
        Compare a live 115-value feature vector against saved stance vectors.
        Returns the best matching name and a score from 0.0 to 100.0.
        """
        if live_vec is None or not saved_moves:
            return None, 0.0

        live_tensor = torch.as_tensor(
            live_vec, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if live_tensor.numel() != 115:
            raise ValueError(
                f"Live feature vector must contain 115 values; "
                f"got {live_tensor.numel()}."
            )

        names = []
        vectors = []
        for name, data in saved_moves.items():
            raw_vec = torch.as_tensor(data["vector"], dtype=torch.float32, device=self.device)
            if raw_vec.ndim == 2 and raw_vec.shape[1] == 115:
                continue  # Skip dynamic sequences during static stance matching
            vector = raw_vec.reshape(-1)
            if vector.numel() != 115:
                raise ValueError(
                    f"Saved move {name!r} must contain 115 feature values; "
                    f"got {vector.numel()}."
                )
            names.append(name)
            vectors.append(vector)

        if not vectors:
            return None, 0.0

        target_tensor = torch.stack(vectors)
        with torch.no_grad():
            scores = F.cosine_similarity(
                live_tensor.unsqueeze(0), target_tensor, dim=1
            )

        if not torch.isfinite(scores).all():
            raise ValueError("Feature vectors must contain only finite values.")

        best_index = scores.argmax().item()
        best_score = scores[best_index].item()
        score_percent = round(max(0.0, min(best_score, 1.0)) * 100, 1)

        if best_score >= self.threshold:
            return names[best_index], score_percent
        return None, score_percent
