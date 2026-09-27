import unittest

from autoresearch_bench import candidate_metrics


class CandidateMetricsTests(unittest.TestCase):
    def test_precision_and_recall_count_false_candidates_and_multiple_truths(self):
        metrics = candidate_metrics(
            {
                "q1": {"hit", "false"},
                "q2": {"hit2"},
                "q3": {"false2"},
            },
            {
                "q1": {"hit", "miss"},
                "q2": {"hit2"},
                "q3": set(),
            },
        )

        self.assertEqual(metrics["candidate_pairs"], 4)
        self.assertEqual(metrics["true_pairs"], 3)
        self.assertEqual(metrics["retrieved_true_pairs"], 2)
        self.assertEqual(metrics["candidate_precision"], 0.5)
        self.assertAlmostEqual(metrics["candidate_recall"], 2 / 3)

    def test_empty_candidate_set_has_zero_precision_and_recall(self):
        metrics = candidate_metrics({"q1": set()}, {"q1": {"true"}})

        self.assertEqual(metrics["candidate_precision"], 0.0)
        self.assertEqual(metrics["candidate_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
