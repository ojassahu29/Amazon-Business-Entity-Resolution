import contextlib
import io
import sqlite3

import csv
import tempfile
import unittest
from pathlib import Path

from baseline import (
    _report_holdout,
    active_keys_for_records,
    build_index,
    build_production_index,
    get_production_candidates,
    select_threshold,
    write_predictions,
)
from evaluation import f_beta_per_s1
from retrieval import ProductionRetrievalIndex


class BaselineTests(unittest.TestCase):
    def test_holdout_reports_micro_pair_precision_and_recall(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE targets (entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, address_norm TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO targets VALUES (?, ?, ?)",
            [("t1", "us", ""), ("t2", "us", ""), ("t3", "us", "")],
        )
        truths = {"s1a": {"t1", "t2"}, "s1b": set(), "s1c": {"t3"}}
        candidates = {
            "s1a": {"t1": "", "t2": "", "false": ""},
            "s1b": {"false-s": ""},
            "s1c": {"t3": ""},
        }
        scores = {
            "s1a": {"t1": 0.9, "t2": 0.2, "false": 0.8},
            "s1b": {"false-s": 0.9},
            "s1c": {"t3": 0.9},
        }
        rows = {s1_id: ("name", "address", "US") for s1_id in truths}
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                _report_holdout(conn, scores, candidates, truths, 0.5, rows)
        finally:
            conn.close()

        self.assertIn("holdout_pair_precision=0.500000", output.getvalue())
        self.assertIn("holdout_pair_recall=0.666667", output.getvalue())

    def test_test_outputs_preserve_empty_rows_and_filter_by_address(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = root / "test_source1.tsv"
            source2 = root / "test_source2.tsv"
            source3 = root / "test_source3.tsv"
            output = root / "output"
            source1.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "fr-1\tCafe & Sons\t10 Main Street\tFrance\n"
                "fr-2\tNo Such Company\t\tFrance\n"
                "match-1\tShared Name\t10 Main Street\tUS\n",
                encoding="utf-8",
            )
            source2.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "target-fr\tCafe Sons\t10 Main Street\tFrance\n"
                "true-s2\tShared Name\t10 Main Street\tUS\n"
                "false-s2\tShared Name\t99 Remote Road\tUS\n",
                encoding="utf-8",
            )
            source3.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "true-s3\tShared Name\t10 Main Street\tUS\n",
                encoding="utf-8",
            )
            active_keys = active_keys_for_records([
                ("fr-1", "Cafe & Sons", "10 Main Street", "France"),
                ("fr-2", "No Such Company", "", "France"),
                ("match-1", "Shared Name", "10 Main Street", "US"),
            ])
            conn, index = build_production_index(source2, source3, root / "index.sqlite", active_keys)
            try:
                self.assertIsInstance(index, ProductionRetrievalIndex)
                self.assertEqual(write_predictions(index, conn, source1, 0.7, output), 3)
            finally:
                conn.close()

            with (output / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as f:
                candidates = list(csv.reader(f, delimiter="\t"))
            with (output / "matching_results.tsv").open(encoding="utf-8", newline="") as f:
                matches = list(csv.reader(f, delimiter="\t"))
            self.assertEqual(candidates[0], ["source1_entity_id", "candidate_entity_ids"])
            self.assertEqual(matches[0], ["source1_entity_id", "matched_entity_ids"])
            self.assertEqual(len(candidates), 4)
            self.assertEqual(len(matches), 4)
            self.assertEqual(candidates[1], ["fr-1", "target-fr"])
            self.assertEqual(matches[1], ["fr-1", "target-fr"])
            self.assertEqual(candidates[2], ["fr-2", ""])
            self.assertEqual(matches[2], ["fr-2", ""])
            self.assertEqual(candidates[3], ["match-1", "false-s2,true-s2,true-s3"])
            self.assertEqual(matches[3], ["match-1", "true-s2,true-s3"])

    def test_production_candidates_are_scoped_to_query_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source2 = root / "source2.tsv"
            source3 = root / "source3.tsv"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            source2.write_text(
                header
                + "target\tShared Name\t10 Quartz Harbor Road\tUS\n"
                + "secondary\tOther Merchant\t10 Quartz Harbor Road\tUS\n"
                + "unrelated\tUnqueried Business\t7 Other Avenue\tUS\n",
                encoding="utf-8",
            )
            source3.write_text(header, encoding="utf-8")
            active_keys = active_keys_for_records([
                ("query", "Shared Name", "10 Quartz Harbor Road", "US"),
            ])
            conn, index = build_production_index(source2, source3, root / "index.sqlite", active_keys)
            try:
                candidates = get_production_candidates(index, conn, "Shared Name", "10 Quartz Harbor Road", "US")
                self.assertEqual(set(candidates), {"target", "secondary"})
                self.assertNotIn(("us", "unqueried"), index.idx_name_norm)
            finally:
                conn.close()

    def test_input_read_errors_include_the_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.tsv"
            valid = root / "valid.tsv"
            malformed.write_text("wrong_header\nrow\n", encoding="utf-8")
            valid.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, str(malformed)):
                build_index(malformed, valid, root / "index.sqlite")
    def test_threshold_selection_rejects_singleton_only_sample(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            select_threshold({"s1": {}}, {"s1": set()})

    def test_duplicate_target_ids_fail_with_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source2 = root / "source2.tsv"
            source3 = root / "source3.tsv"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            source2.write_text(header + "duplicate\tName\t1 Main\tUS\n", encoding="utf-8")
            source3.write_text(header + "duplicate\tName\t1 Main\tUS\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, str(source3)):
                build_index(source2, source3, root / "index.sqlite")

    def test_empty_singleton_and_threshold_tie_contract(self):
        self.assertEqual(f_beta_per_s1(set(), set()), 1)
        self.assertEqual(f_beta_per_s1({"false"}, set()), 0)
        threshold, score = select_threshold(
            {"s1": {"target": 0.9}},
            {"s1": {"target"}},
        )
        self.assertEqual((threshold, score), (0.85, 1.0))


if __name__ == "__main__":
    unittest.main()
