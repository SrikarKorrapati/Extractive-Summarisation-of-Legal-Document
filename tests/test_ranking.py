import unittest

import numpy as np

from legal_lexrank.ranking import fit_weights, rank_sentences, select_indices


class RoleWeightTests(unittest.TestCase):
    def test_estimators_distinguish_typical_and_extreme_cases(self):
        # Four case profiles along a line. Equal document weighting ignores the
        # large size of the first document, and the medians favor the majority.
        weights = fit_weights([["A"] * 40, ["B"], ["B"], ["B"]], ["A", "B", "C"])
        np.testing.assert_allclose(list(weights["mean"].values()), [0.25, 0.75, 0])
        np.testing.assert_allclose(list(weights["chebyshev"].values()), [0.5, 0.5, 0], atol=1e-8)
        np.testing.assert_allclose(list(weights["geometric_median"].values()), [0, 1, 0], atol=1e-8)

    def test_geometric_median_can_leave_coincident_arithmetic_mean(self):
        # A shares [0, .5, .6, .6, .8] average .5 but their median is .6.
        documents = [["A"] * count + ["B"] * (10 - count) for count in [0, 5, 6, 6, 8]]
        weights = fit_weights(documents, ["A", "B"])
        self.assertAlmostEqual(weights["geometric_median"]["A"], 0.6, places=8)

    def test_geometric_median_at_120_degree_vertex(self):
        # All three simplex profiles are distinct and the angle at the first is 120 degrees.
        docs = [list("ABCC"), list("AACC"), list("ABBC")]
        weights = fit_weights(docs, ["A", "B", "C"])
        np.testing.assert_allclose(list(weights["geometric_median"].values()), [0.25, 0.25, 0.5])

    def test_geometric_median_near_120_degree_vertex(self):
        docs = [list("ABCC"), list("AACC"), list("A" * 2501 + "B" * 4999 + "C" * 2500)]
        median = fit_weights(docs, ["A", "B", "C"])["geometric_median"]
        center = np.array(list(median.values()))
        profiles = np.array([[.25, .25, .5], [.5, 0, .5], [.2501, .4999, .25]])
        offsets = center - profiles
        distances = np.linalg.norm(offsets, axis=1)
        self.assertGreater(distances.min(), 0)
        np.testing.assert_allclose((offsets / distances[:, None]).sum(axis=0), 0, atol=1e-6)

    def test_identical_profiles_and_unused_role(self):
        weights = fit_weights([["A", "A", "B"], ["B", "A", "A"]], ["A", "B", "C"])
        for estimate in weights.values():
            np.testing.assert_allclose(list(estimate.values()), [2 / 3, 1 / 3, 0])

    def test_invalid_training_data(self):
        for cases in [[], [[]], [["other"]]]:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                fit_weights(cases, ["A"])


class LexRankTests(unittest.TestCase):
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
