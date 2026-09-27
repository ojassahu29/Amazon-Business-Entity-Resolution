import io
import json
import os
import sqlite3
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path
from baseline import build_index
from study_candidates import _build_fts, _open_resume_index, _publish_outputs, run_study


class StudyProgressTests(unittest.TestCase):
    def test_reused_full_index_publishes_durable_method_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset"
            (data / "train").mkdir(parents=True)
            (data / "test").mkdir()
            ids = {}
            for bucket in (0, 1):
                bucket_ids = []
                index = 0
                while len(bucket_ids) < 2:
                    s1_id = f"s1-{bucket}-{index}"
                    if zlib.crc32(s1_id.encode("utf-8")) % 200 == bucket:
                        bucket_ids.append(s1_id)
                    index += 1
                ids[bucket] = bucket_ids

            source_header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            (data / "train" / "train_source1.tsv").write_text(
                source_header + "".join(
                    f"{ids[bucket][row]}\tAcme {bucket}\t{row} Main Street\tUS\n"
                    for bucket in (0, 1) for row in (0, 1)
                ),
                encoding="utf-8",
            )
            (data / "train" / "train_source2.tsv").write_text(
                source_header + "".join(f"target-{bucket}\tAcme {bucket}\t0 Main Street\tUS\n" for bucket in (0, 1)),
                encoding="utf-8",
            )
            (data / "train" / "train_source3.tsv").write_text(source_header, encoding="utf-8")
            (data / "train" / "train_ground_truth.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\n" + "".join(
                    f"{ids[bucket][0]}\ttarget-{bucket}\n{ids[bucket][1]}\t\n" for bucket in (0, 1)
                ),
                encoding="utf-8",
            )
            for filename in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
                (data / "test" / filename).write_text(source_header, encoding="utf-8")

            output = root / "output"
            workdir = root / "tmp-study-resume"
            workdir.mkdir(parents=True)
            conn = build_index(
                data / "train" / "train_source2.tsv",
                data / "train" / "train_source3.tsv",
                workdir / "index.sqlite",
            )
            _build_fts(conn, "target_name_fts", "name_norm, address_norm, tokenize='unicode61'")
            conn.execute("CREATE VIRTUAL TABLE target_name_trigram USING fts5(name_norm, tokenize='trigram', content='targets', content_rowid='rowid')")
            conn.commit()
            conn.close()
            with self.assertRaisesRegex(RuntimeError, "FTS content integrity"):
                _open_resume_index(workdir)
            conn = sqlite3.connect(workdir / "index.sqlite")
            conn.execute("DROP TABLE target_name_trigram")
            _build_fts(conn, "target_name_trigram", "name_norm, tokenize='trigram'")
            conn.close()
            with redirect_stdout(io.StringIO()) as first_log:
                run_study(data, output, expected_control_recall=None, resume_index=workdir)
            first_report = json.loads((output / "shortlist-study.json").read_text(encoding="utf-8"))
            probe = (output / "shortlist-probe.tsv").read_bytes()
            progress_path = output / "shortlist-study-progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(len(progress["completed_methods"]), 10)
            self.assertEqual(progress["last_checkpoint"]["stage"], "complete")
            progress["status"] = "running"
            progress_path.write_text(json.dumps(progress), encoding="utf-8")

            with redirect_stdout(io.StringIO()) as resumed_log:
                run_study(data, output, expected_control_recall=None, resume_index=workdir)
            resumed_report = json.loads((output / "shortlist-study.json").read_text(encoding="utf-8"))
            for bucket, methods in first_report["methods"].items():
                for method, metrics in methods.items():
                    if method == "baseline_miss_causes":
                        continue
                    for key in ("true_pair_candidate_recall", "recall_at_k", "candidate_counts"):
                        self.assertEqual(metrics[key], resumed_report["methods"][bucket][method][key])
            self.assertEqual(probe, (output / "shortlist-probe.tsv").read_bytes())
            self.assertNotIn("method=control completed_s1=0", resumed_log.getvalue())
            self.assertIn("method=tokens_trigram completed_s1=0", resumed_log.getvalue())
            for bucket, methods in first_report["methods"].items():
                self.assertEqual(methods["baseline_miss_causes"], methods["control"]["miss_causes"])
            self.assertIn("method=control completed_s1=0", first_log.getvalue())
            self.assertIn("method_complete", first_log.getvalue())
            self.assertTrue(progress_path.exists())
    def test_failed_second_output_replace_restores_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "shortlist-study.json"
            probe = root / "shortlist-probe.tsv"
            report.write_text("old report", encoding="utf-8")
            probe.write_text("old probe", encoding="utf-8")
            report_tmp = root / "report.new"
            probe_tmp = root / "probe.new"
            report_tmp.write_text("new report", encoding="utf-8")
            probe_tmp.write_text("new probe", encoding="utf-8")
            original_replace = os.replace
            failed = False

            def fail_once(source, destination):
                nonlocal failed
                if Path(destination) == report and not failed:
                    failed = True
                    raise OSError("simulated report replacement failure")
                return original_replace(source, destination)

            with patch("study_candidates.os.replace", side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError, "previous outputs restored"):
                    _publish_outputs(report_tmp, probe_tmp, root)
            self.assertEqual(report.read_text(encoding="utf-8"), "old report")
            self.assertEqual(probe.read_text(encoding="utf-8"), "old probe")
if __name__ == "__main__":
    unittest.main()
