import unittest

from model_probe_metrics import rank_lexical_rows, rank_probe_rows, ranking_metrics


class ModelProbeMetricsTests(unittest.TestCase):
    def test_embedding_ranking_is_deterministic_and_reports_pair_and_row_recall(self):
        rows = [
            {"bucket": "0", "s1_id": "q1", "target_id": "a", "label": "1", "lexical_rank": "3"},
            {"bucket": "0", "s1_id": "q1", "target_id": "b", "label": "0", "lexical_rank": "1"},
            {"bucket": "0", "s1_id": "q1", "target_id": "c", "label": "0", "lexical_rank": "2"},
            {"bucket": "1", "s1_id": "q2", "target_id": "d", "label": "1", "lexical_rank": "1"},
        ]
        query_vectors = {"q1": [1.0, 0.0], "q2": [0.0, 1.0]}
        target_vectors = {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.0, 1.0],
            "d": [0.0, 1.0],
        }

        ranked = rank_probe_rows(rows, query_vectors, target_vectors)
        self.assertEqual([row["target_id"] for row in ranked[("0", "q1")]], ["a", "b", "c"])
        metrics = ranking_metrics(ranked)
        self.assertEqual(metrics["positive_pairs"], 2)
        self.assertEqual(metrics["negative_pairs"], 2)
        self.assertEqual(metrics["recall_at_k"]["1"], 1.0)
        self.assertEqual(metrics["positive_row_coverage_at_k"]["1"], 1.0)
        lexical_metrics = ranking_metrics(rank_lexical_rows(rows))
        self.assertEqual(lexical_metrics["recall_at_k"]["1"], 0.5)
        self.assertAlmostEqual(lexical_metrics["mean_reciprocal_rank"], 2 / 3)
        self.assertEqual(metrics["mean_reciprocal_rank"], 1.0)

    def test_metrics_include_positives_only_when_they_are_in_the_probe_pool(self):
        rows = [
            {"bucket": "0", "s1_id": "q1", "target_id": "a", "label": "1", "lexical_rank": "2"},
            {"bucket": "0", "s1_id": "q1", "target_id": "b", "label": "0", "lexical_rank": "1"},
        ]
        ranked = rank_probe_rows(rows, {"q1": [1.0]}, {"a": [1.0], "b": [0.5]})
        metrics = ranking_metrics(ranked)
        self.assertEqual(metrics["positive_pairs"], 1)
        self.assertEqual(metrics["recall_at_k"]["1"], 1.0)


if __name__ == "__main__":
    unittest.main()
