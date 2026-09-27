import unittest
from unittest.mock import patch

import retrieval
from retrieval import ProductionRetrievalIndex, _limit_candidates, parse_entity_record, retrieve_candidates_for_record


class CandidateEvidenceLimitTests(unittest.TestCase):
    def setUp(self):
        self.index = ProductionRetrievalIndex()
        self.index.index_record("z-strong", "Alpha Beta Company", "12 Northwind Avenue Industrial Park", "US")
        self.index.index_record("a-weak", "Alpha Beta", "", "US")
        self.index.index_record("b-weak", "Alpha Beta", "", "US")
        self.query = parse_entity_record("Alpha Beta Company", "12 Northwind Avenue Industrial Park", "US")

    def test_four_character_rare_name_token_rescues_candidate(self):
        index = ProductionRetrievalIndex()
        index.index_record("rare", "Koru Holdings", "", "US")
        query = parse_entity_record("Koru Analytics", "", "US")

        self.assertEqual({"rare"}, retrieve_candidates_for_record(query, index))

    def test_cap_preserves_high_signal_then_caps_weak_candidates(self):
        with patch.object(retrieval, "MAX_LOW_EVIDENCE_CANDIDATES_PER_QUERY", 1, create=True):
            candidates = retrieve_candidates_for_record(self.query, self.index)

        self.assertEqual({"a-weak", "z-strong"}, candidates)

    def test_high_weight_candidates_survive_low_evidence_cap(self):
        candidates = {"strong-a", "strong-b", "weak-a", "weak-b"}
        evidence = [
            ({"strong-a"}, 4),
            ({"strong-b"}, 5),
            ({"weak-a"}, 1),
            ({"weak-a"}, 1),
            ({"weak-a"}, 1),
            ({"weak-a"}, 1),
            ({"weak-b"}, 1),
        ]

        self.assertEqual(
            {"strong-a", "strong-b", "weak-a"},
            _limit_candidates(candidates, evidence, 1),
        )

    def test_zero_cap_preserves_uncapped_candidates(self):
        with patch.object(retrieval, "MAX_LOW_EVIDENCE_CANDIDATES_PER_QUERY", 0, create=True):
            candidates = retrieve_candidates_for_record(self.query, self.index)

        self.assertEqual({"a-weak", "b-weak", "z-strong"}, candidates)


if __name__ == "__main__":
    unittest.main()
