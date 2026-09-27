import sqlite3
import tempfile
import unittest
from pathlib import Path

from baseline import build_index, get_candidates
from evaluation import macro_f05
from preprocessing import normalize_basic, normalize_country
from study_candidates import (
    _build_fts,
    analyze_missed_pairs,
    exact_candidates,
    fts_candidates,
    model_probe_pairs,
    rank_candidates,
)


class StudyCandidateTests(unittest.TestCase):
    def build_index(self, directory, targets):
        root = Path(directory)
        s2 = root / "source2.tsv"
        s3 = root / "source3.tsv"
        header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        s2.write_text(header + "".join("\t".join(row) + "\n" for row in targets), encoding="utf-8")
        s3.write_text(header, encoding="utf-8")
        conn = build_index(s2, s3, root / "targets.sqlite")
        _build_fts(conn, "target_name_fts", "name_norm, address_norm, tokenize='unicode61'")
        _build_fts(conn, "target_name_trigram", "name_norm, tokenize='trigram'")
        return conn

    @staticmethod
    def query(country="US", name="Common Name", address="1 Main"):
        return {
            "country": normalize_country(country),
            "name_norm": normalize_basic(name),
            "address_norm": normalize_basic(address),
        }

    def test_uncapped_exact_recovers_true_target_from_broad_key(self):
        with tempfile.TemporaryDirectory() as directory:
            targets = [(f"id-{i:02}", "Common Name", "1 Main" if i == 50 else f"{i} Road", "US") for i in range(51)]
            conn = self.build_index(directory, targets)
            try:
                query = self.query()
                baseline, _ = get_candidates(conn, "US", "Common Name")
                recovered = exact_candidates(conn, query)
                self.assertNotIn("id-50", baseline)
                self.assertIn("id-50", recovered)
            finally:
                conn.close()

    def test_name_token_fts_recovers_typo_without_exact_key(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = self.build_index(directory, [("alias", "Acme Holdings", "West Avenue", "US")])
            try:
                query = self.query(name="Acme Holdngs", address="Elsewhere")
                self.assertNotIn("alias", exact_candidates(conn, query))
                self.assertIn("alias", fts_candidates(conn, query, fields=("name_norm",)))
            finally:
                conn.close()

    def test_trigram_recovers_typo_without_whole_word_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = self.build_index(directory, [("typo", "Acme Holdings", "West Avenue", "US")])
            try:
                query = self.query(name="Acm Holdng", address="Elsewhere")
                self.assertNotIn("typo", exact_candidates(conn, query))
                self.assertNotIn("typo", fts_candidates(conn, query, fields=("name_norm",)))
                self.assertIn("typo", fts_candidates(conn, query, fields=("name_trigram",)))
            finally:
                conn.close()

    def test_same_name_in_different_country_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = self.build_index(directory, [("foreign", "Common Name", "1 Main", "India")])
            try:
                self.assertNotIn("foreign", exact_candidates(conn, self.query(country="US")))
                self.assertNotIn("foreign", fts_candidates(conn, self.query(country="US")))
            finally:
                conn.close()

    def test_blank_name_target_is_omitted_not_counted_as_name_key_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = self.build_index(directory, [("blank", "", "1 Main", "US")])
            try:
                result = analyze_missed_pairs(conn, [{"s1_id": "s1", "target_id": "blank", **self.query()}], {"blank": {"name_norm": "", "address_norm": "1 main", "country": "us"}})
                self.assertEqual(result["causes"]["blank_target_country_or_name"], 1)
                self.assertEqual(result["causes"]["no_shared_name_key"], 0)
                self.assertEqual(sum(result["causes"].values()), result["missed_true_pairs"])
            finally:
                conn.close()

    def test_singleton_contributes_to_counts_and_macro_f05(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = self.build_index(directory, [("hit", "Common Name", "1 Main", "US")])
            try:
                counts = analyze_missed_pairs(conn, [{"s1_id": "singleton", "target_id": None, **self.query()}])
                self.assertEqual(counts["singleton_rows"], 1)
                self.assertEqual(macro_f05({"singleton": set()}, {"singleton": set()}), 1.0)
            finally:
                conn.close()

    def test_rank_cutoff_uses_lexicographic_id_tie_break(self):
        ranked = rank_candidates(
            {"z-id": {"name_norm": "same", "address_norm": "same"}, "a-id": {"name_norm": "same", "address_norm": "same"}},
            self.query(name="same", address="same"),
            k=1,
        )
        self.assertEqual([row["entity_id"] for row in ranked], ["a-id"])

    def test_model_probe_does_not_count_truth_outside_raw_candidate_pool(self):
        rows = model_probe_pairs(
            s1_id="s1",
            truth_ids={"outside"},
            baseline_ids=set(),
            raw_candidate_ids={"inside"},
            target_rows={"inside": {"country": "us"}},
        )
        self.assertEqual(rows, [])

    def test_cause_counts_sum_to_missed_true_pair_count(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = self.build_index(directory, [("present", "Other Name", "1 Main", "US")])
            try:
                result = analyze_missed_pairs(conn, [
                    {"s1_id": "s1", "target_id": "absent", **self.query()},
                    {"s1_id": "s2", "target_id": "present", **self.query()},
                ])
                self.assertEqual(sum(result["causes"].values()), result["missed_true_pairs"])
                self.assertEqual(result["missed_true_pairs"], 2)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
