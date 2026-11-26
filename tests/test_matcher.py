import unittest

import torch

from src.matcher import PoseMatcher


class PoseMatcherTests(unittest.TestCase):
    def setUp(self):
        self.matcher = PoseMatcher()
        self.live_vector = torch.zeros(115)
        self.live_vector[0] = 1.0

    def test_match_returns_best_move_and_similarity_percent(self):
        exact_match = self.live_vector.clone()
        lower_match = torch.zeros(115)
        lower_match[0] = 0.8
        lower_match[1] = 0.6

        name, score = self.matcher.match(
            self.live_vector,
            {
                "lower": {"vector": lower_match},
                "exact": {"vector": exact_match},
            },
        )

        self.assertEqual((name, score), ("exact", 100.0))

    def test_below_threshold_returns_similarity_and_clamps_negative_scores(self):
        below_threshold = torch.zeros(115)
        below_threshold[0] = 0.8
        below_threshold[1] = 0.6
        name, score = self.matcher.match(
            self.live_vector, {"below": {"vector": below_threshold}}
        )
        self.assertIsNone(name)
        self.assertEqual(score, 80.0)

        opposite = -self.live_vector
        name, score = self.matcher.match(
            self.live_vector, {"opposite": {"vector": opposite}}
        )
        self.assertIsNone(name)
        self.assertEqual(score, 0.0)

    def test_missing_live_vector_or_saved_moves_returns_no_match(self):
        self.assertEqual(self.matcher.match(None, {"move": {}}), (None, 0.0))
        self.assertEqual(self.matcher.match(self.live_vector, {}), (None, 0.0))

    def test_rejects_vectors_with_wrong_feature_count(self):
        with self.assertRaisesRegex(ValueError, "115"):
            self.matcher.match(torch.zeros(99), {"move": {"vector": torch.zeros(115)}})

        with self.assertRaisesRegex(ValueError, "115"):
            self.matcher.match(self.live_vector, {"move": {"vector": torch.zeros(99)}})

    def test_rejects_non_finite_feature_values(self):
        invalid_vector = self.live_vector.clone()
        invalid_vector[0] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            self.matcher.match(invalid_vector, {"move": {"vector": self.live_vector}})


if __name__ == "__main__":
    unittest.main()
