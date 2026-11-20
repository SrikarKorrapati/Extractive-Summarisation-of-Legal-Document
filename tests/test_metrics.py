import unittest

from rouge_score.rouge_scorer import RougeScorer

from legal_lexrank.metrics import SELECTION_METRIC, score_summaries


class RougeTests(unittest.TestCase):
    def test_exact_agreement_with_rouge_score(self):
        scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        examples = [
            ("The COURTS ruled on § 12(b).", "A court rules on section 12-B."),
            ("Appeal appeal allowed.", "Appeal allowed, allowed, appeal."),
            ("A one.\nB two.", "B two.\nA one."),
            ("Café court’s order", "cafe COURT'S orders"),
            ("appeal", "appeals"),
            ("", "a decision"),
            ("an order", ""),
            ("...", "§"),
            (("appeal grant stay " * 60) + "dismiss", "stay grant appeal " * 45),
        ]
        for summary, reference in examples:
            with self.subTest(summary=summary, reference=reference):
                expected = {}
                for metric, score in scorer.score(reference, summary).items():
                    expected[f"{metric}_precision"] = score.precision
                    expected[f"{metric}_recall"] = score.recall
                    expected[f"{metric}_f1"] = score.fmeasure
                actual = score_summaries(summary, [reference])
                self.assertEqual(actual, expected)
                self.assertTrue(all(isinstance(value, float) for value in actual.values()))

    def test_average_each_measure_over_distinct_references(self):
        scores = score_summaries("courts ruled", ["court rules", "court", "court"])
        self.assertEqual(scores["rouge1_precision"], 0.75)
        self.assertEqual(scores["rouge1_recall"], 1.0)
        self.assertAlmostEqual(scores["rouge1_f1"], 5 / 6)
        self.assertEqual(scores["rouge2_precision"], 0.5)
        self.assertEqual(scores["rouge2_recall"], 0.5)
        self.assertEqual(scores["rouge2_f1"], 0.5)
        self.assertEqual(SELECTION_METRIC, "rougeL_f1")
        self.assertAlmostEqual(scores[SELECTION_METRIC], 5 / 6)

    def test_repeated_calls_return_independent_scores(self):
        scores = score_summaries("court appeal", ["court appeal"])
        scores[SELECTION_METRIC] = -1
        self.assertEqual(score_summaries("court appeal", ["court appeal"])[SELECTION_METRIC], 1.0)

    def test_missing_references_and_accidental_string(self):
        with self.assertRaisesRegex(ValueError, "At least one reference"):
            score_summaries("court", [])
        with self.assertRaises(TypeError):
            score_summaries("court", "court")


if __name__ == "__main__":
    unittest.main()
