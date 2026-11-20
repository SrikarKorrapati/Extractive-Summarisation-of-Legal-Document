import unittest

import numpy as np

from legal_lexrank.data import ROLES
from legal_lexrank.ranking import DEFAULT_ROLE_WEIGHTS, rank_sentences, select_indices


class LexRankTests(unittest.TestCase):
    def test_defaults_promote_rulings_when_sentences_are_equally_similar(self):
        self.assertEqual(tuple(DEFAULT_ROLE_WEIGHTS), ROLES)
        self.assertEqual(list(DEFAULT_ROLE_WEIGHTS.values()), [0.1, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2])
        self.assertAlmostEqual(sum(DEFAULT_ROLE_WEIGHTS.values()), 1.0)
        ranking = rank_sentences(np.ones((7, 7)), ROLES)
        self.assertEqual(list(ranking), [5, 6, 3, 4, 0, 1, 2])
        self.assertEqual(select_indices(ranking, 0.4), [3, 5, 6])

    def test_weighted_edges_can_rank_below_legacy_similarity_threshold(self):
        cosine = np.ones((4, 4))
        labels = ["A", "B", "B", "C"]
        ranking = rank_sentences(cosine, labels, {"A": 0.01, "B": 0.02, "C": 0.04})
        self.assertEqual(list(ranking), [3, 1, 2, 0])
        self.assertEqual(select_indices(ranking, 0.5), [1, 3])
        np.testing.assert_array_equal(cosine, np.ones((4, 4)))

    def test_diagonal_ignored_and_dangling_nodes_safe(self):
        cosine = [[10, 0.8, -1, 0], [0.8, 1, 0, 0.8], [-1, 0, 100, 0], [0, 0.8, 0, 1]]
        labels = ["a"] * 4
        self.assertEqual(list(rank_sentences(cosine, labels, None)), [1, 0, 3, 2])
        self.assertEqual(list(rank_sentences(cosine, labels, {"a": 0})), [0, 1, 2, 3])

    def test_budget_rounds_up_and_handles_tiny_inputs(self):
        self.assertEqual(select_indices([2, 1, 0], 0.4), [1, 2])
        self.assertEqual(select_indices([0], 0.3), [0])
        self.assertEqual(select_indices([], 0.5), [])
        self.assertEqual(list(rank_sentences(np.empty((0, 0)), [], None)), [])


if __name__ == "__main__":
    unittest.main()
