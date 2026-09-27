from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import tempfile
import zlib
from collections.abc import Iterable
from pathlib import Path

from rapidfuzz import fuzz

from data_loader import DatasetPaths, iter_ground_truth, iter_source, parse_match_ids
from evaluation import f_beta_per_s1, macro_f05
from preprocessing import normalize_basic, normalize_country

try:
    from .retrieval import ProductionRetrievalIndex, parse_entity_record, retrieve_candidates_for_record
except ImportError:
    from retrieval import ProductionRetrievalIndex, parse_entity_record, retrieve_candidates_for_record

CHUNK_SIZE = 50_000
THRESHOLDS = (0.0, 0.25, 0.5, 0.7, 0.85)


def _source_chunks(path: Path):
    try:
        yield from iter_source(path, chunksize=CHUNK_SIZE)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _ground_truth_chunks(path: Path):
    try:
        yield from iter_ground_truth(path, chunksize=CHUNK_SIZE)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def active_keys_for_records(
    records: Iterable[tuple[str, str, str, str]],
) -> dict[str, set[tuple[str, str]]]:
    keys: dict[str, set[tuple[str, str]]] = {
        "idx_name_norm": set(),
        "idx_name_sorted": set(),
        "idx_name_compact": set(),
        "idx_compact_prefix5": set(),
        "idx_name_tokens": set(),
        "idx_name_stopwords": set(),
        "idx_addr_tokens": set(),
        "idx_addr_numbers": set(),
    }
    for _, name, address, country in records:
        parsed = parse_entity_record(name, address, country)
        c = parsed["country"]
        keys["idx_name_norm"].add((c, parsed["name_norm"]))
        keys["idx_name_sorted"].add((c, parsed["name_sorted"]))
        keys["idx_name_compact"].add((c, parsed["name_compact"]))
        if parsed["prefix5"]:
            keys["idx_compact_prefix5"].add((c, parsed["prefix5"]))
        keys["idx_name_tokens"].update((c, token) for token in parsed["info_name"])
        keys["idx_name_stopwords"].update((c, token) for token in parsed["stop_name"])
        keys["idx_addr_tokens"].update((c, token) for token in parsed["info_addr"])
        keys["idx_addr_numbers"].update((c, number) for number in parsed["all_nums"])
    return keys


def build_production_index(
    s2_path: Path,
    s3_path: Path,
    db_path: Path,
    active_keys: dict[str, set[tuple[str, str]]],
) -> tuple[sqlite3.Connection, ProductionRetrievalIndex]:
    index = ProductionRetrievalIndex(active_keys)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE targets ("
            "entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, address_norm TEXT NOT NULL)"
        )
        insert = "INSERT INTO targets VALUES (?, ?, ?)"
        for path in (s2_path, s3_path):
            for frame in _source_chunks(path):
                rows = []
                for row in frame.itertuples(index=False, name=None):
                    entity_id, name, address, country = map(str, row)
                    index.index_record(entity_id, name, address, country)
                    rows.append((entity_id, normalize_basic(country), normalize_basic(address)))
                try:
                    conn.executemany(insert, rows)
                except sqlite3.Error as exc:
                    raise RuntimeError(f"Failed to index {path}: {exc}") from exc
        conn.commit()
        return conn, index
    except Exception:
        conn.close()
        raise

def build_index(s2_path: Path, s3_path: Path, db_path: Path) -> sqlite3.Connection:
    """Build the full SQLite table used by candidate-method analysis."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE targets ("
            "entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, name_norm TEXT NOT NULL, "
            "name_sorted TEXT NOT NULL, name_compact TEXT NOT NULL, address_norm TEXT NOT NULL)"
        )
        insert = "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?)"
        for path in (s2_path, s3_path):
            for frame in _source_chunks(path):
                rows = []
                for row in frame.itertuples(index=False, name=None):
                    entity_id, name, address, country = map(str, row)
                    name_norm = normalize_basic(name)
                    country_norm = normalize_country(country)
                    if not name_norm or not country_norm:
                        continue
                    rows.append((
                        entity_id,
                        country_norm,
                        name_norm,
                        " ".join(sorted(name_norm.split())),
                        name_norm.replace(" ", ""),
                        normalize_basic(address),
                    ))
                try:
                    conn.executemany(insert, rows)
                except sqlite3.Error as exc:
                    raise RuntimeError(f"Failed to index {path}: {exc}") from exc
        conn.executescript(
            "CREATE INDEX targets_country_name ON targets(country, name_norm);"
            "CREATE INDEX targets_country_sorted ON targets(country, name_sorted);"
            "CREATE INDEX targets_country_compact ON targets(country, name_compact);"
        )
        conn.commit()
        return conn
    except Exception:
        conn.close()
        raise

def get_candidates(
    conn: sqlite3.Connection,
    country: str,
    name: str,
) -> tuple[dict[str, str], int]:
    country_norm = normalize_country(country)
    name_norm = normalize_basic(name)
    if not country_norm or not name_norm:
        return {}, 0
    keys = (
        ("name_norm", name_norm),
        ("name_sorted", " ".join(sorted(name_norm.split()))),
        ("name_compact", name_norm.replace(" ", "")),
    )
    candidates: dict[str, str] = {}
    discarded = 0
    for column, key in keys:
        if not key:
            continue
        rows = conn.execute(
            f"SELECT entity_id, address_norm FROM targets "
            f"WHERE country = ? AND {column} = ? LIMIT 51",
            (country_norm, key),
        ).fetchall()
        if len(rows) == 51:
            discarded += 1
            continue
        candidates.update(rows)
    return candidates, discarded
def get_production_candidates(
    index: ProductionRetrievalIndex,
    conn: sqlite3.Connection,
    name: str,
    address: str,
    country: str,
) -> dict[str, str]:
    parsed = parse_entity_record(name, address, country)
    candidate_ids = retrieve_candidates_for_record(parsed, index)
    candidates: dict[str, str] = {}
    ordered_ids = list(candidate_ids)
    for start in range(0, len(ordered_ids), 500):
        batch = ordered_ids[start:start + 500]
        placeholders = ",".join("?" for _ in batch)
        candidates.update(conn.execute(
            f"SELECT entity_id, address_norm FROM targets WHERE entity_id IN ({placeholders})",
            batch,
        ).fetchall())
    if len(candidates) != len(candidate_ids):
        raise RuntimeError("Retrieval index returned candidate IDs missing from the target table")
    return candidates


def pair_score(s1_address_norm: str, target_address_norm: str) -> float:
    if not s1_address_norm or not target_address_norm:
        return 0.0
    return fuzz.ratio(s1_address_norm, target_address_norm) / 100


def select_threshold(
    scores: dict[str, dict[str, float]], truths: dict[str, set[str]],
) -> tuple[float, float]:
    if not scores or not truths:
        raise ValueError("Threshold selection requires a nonempty labeled sample")
    if not any(truths.values()):
        raise ValueError("Threshold selection sample has no positive links")
    best_threshold: float | None = None
    best_score = -1.0
    for threshold in THRESHOLDS:
        predictions = {
            s1_id: {target_id for target_id, score in values.items() if score >= threshold}
            for s1_id, values in scores.items()
        }
        score = macro_f05(predictions, truths)
        if score > best_score or (score == best_score and (best_threshold is None or threshold > best_threshold)):
            best_threshold, best_score = threshold, score
    assert best_threshold is not None
    return best_threshold, best_score


def _sample_metrics(
    conn: sqlite3.Connection,
    index: ProductionRetrievalIndex,
    s1_rows: dict[str, tuple[str, str, str]],
    truths: dict[str, set[str]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    scores: dict[str, dict[str, float]] = {}
    candidates_by_s1: dict[str, dict[str, str]] = {}
    for s1_id, (name, address, country) in s1_rows.items():
        candidates = get_production_candidates(index, conn, name, address, country)
        address_norm = normalize_basic(address)
        candidates_by_s1[s1_id] = candidates
        scores[s1_id] = {
            target_id: pair_score(address_norm, target_address)
            for target_id, target_address in candidates.items()
        }
    if set(s1_rows) != set(truths):
        raise ValueError("Selected training S1 rows and ground-truth IDs do not match")
    return scores, candidates_by_s1


def _report_holdout(
    conn: sqlite3.Connection,
    scores: dict[str, dict[str, float]],
    candidates: dict[str, dict[str, str]],
    truths: dict[str, set[str]],
    threshold: float,
    s1_rows: dict[str, tuple[str, str, str]],
) -> None:
    if not truths:
        raise ValueError("Hold-out bucket is empty")
    positive = {s1_id: truth for s1_id, truth in truths.items() if truth}
    true_links = sum(map(len, positive.values()))
    if not true_links:
        raise ValueError("Hold-out bucket has no positive links; candidate recall is undefined")
    found_pairs = sum(len(candidates[s1_id].keys() & truth) for s1_id, truth in positive.items())
    found_rows = sum(bool(candidates[s1_id].keys() & truth) for s1_id, truth in positive.items())
    counts = [len(candidates[s1_id]) for s1_id in truths]
    predictions = {
        s1_id: {target_id for target_id, score in scores[s1_id].items() if score >= threshold}
        for s1_id in truths
    }
    singleton_scores = [f_beta_per_s1(predictions[s1_id], set()) for s1_id, truth in truths.items() if not truth]
    positive_scores = [f_beta_per_s1(predictions[s1_id], truth) for s1_id, truth in positive.items()]
    predicted_links = sum(len(predicted) for predicted in predictions.values())
    matched_links = sum(
        len(predictions[s1_id] & truth)
        for s1_id, truth in truths.items()
    )
    print(f"holdout_pair_precision={matched_links / predicted_links if predicted_links else 0.0:.6f}")
    print(f"holdout_pair_recall={matched_links / true_links:.6f}")

    print(f"holdout_macro_f05={macro_f05(predictions, truths):.6f}")
    print(f"holdout_singleton_score={sum(singleton_scores) / len(singleton_scores):.6f}" if singleton_scores else "holdout_singleton_score=n/a")
    print(f"holdout_positive_row_score={sum(positive_scores) / len(positive_scores):.6f}")
    print(f"candidate_pair_recall={found_pairs / true_links:.6f}")
    print(f"positive_rows_with_candidate_recall={found_rows / len(positive):.6f}")
    print(f"mean_candidate_count={sum(counts) / len(counts):.6f}")
    print(f"rows_without_candidates={sum(count == 0 for count in counts)}")
    print(f"holdout_evaluated_s1_rows={len(truths)}")
    country = {s1_id: normalize_basic(row[2]) for s1_id, row in s1_rows.items()}
    ids = {target_id for truth in positive.values() for target_id in truth}
    target_countries = {}
    if ids:
        target_ids = list(ids)
        for start in range(0, len(target_ids), 500):
            batch = target_ids[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            target_countries.update(conn.execute(
                f"SELECT entity_id, country FROM targets WHERE entity_id IN ({placeholders})", batch
            ).fetchall())
    country_mismatch = sum(
        country[s1_id] != target_countries[target_id]
        for s1_id, truth in positive.items()
        for target_id in truth
        if target_id in target_countries and target_id not in candidates[s1_id]
    )
    print(f"missed_true_pairs_with_country_mismatch={country_mismatch}")


def _load_selected_training(
    paths: DatasetPaths,
) -> tuple[dict[int, dict[str, set[str]]], dict[int, dict[str, tuple[str, str, str]]]]:
    truths = {0: {}, 1: {}}
    for frame in _ground_truth_chunks(paths.train_ground_truth):
        for s1_id, value in frame.itertuples(index=False, name=None):
            bucket = zlib.crc32(str(s1_id).encode("utf-8")) % 200
            if bucket in truths:
                truths[bucket][str(s1_id)] = set(parse_match_ids(value))
    selected = set(truths[0]) | set(truths[1])
    rows: dict[int, dict[str, tuple[str, str, str]]] = {0: {}, 1: {}}
    for frame in _source_chunks(paths.train_source1):
        for s1_id, name, address, country in frame.itertuples(index=False, name=None):
            bucket = zlib.crc32(str(s1_id).encode("utf-8")) % 200
            if str(s1_id) in selected and bucket in rows:
                rows[bucket][str(s1_id)] = (str(name), str(address), str(country))
    missing = selected.difference(rows[0]).difference(rows[1])
    if missing:
        raise ValueError(f"Selected ground-truth S1 IDs missing from train_source1: {next(iter(missing))}")
    return truths, rows


def write_predictions(
    index: ProductionRetrievalIndex,
    conn: sqlite3.Connection,
    s1_path: Path,
    threshold: float,
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_tmp = output_dir / "candidate_pairs.tsv.tmp"
    matching_tmp = output_dir / "matching_results.tsv.tmp"
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    count = 0
    try:
        with candidate_tmp.open("w", encoding="utf-8", newline="") as candidate_file, matching_tmp.open("w", encoding="utf-8", newline="") as matching_file:
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
            candidate_writer.writerow(("source1_entity_id", "candidate_entity_ids"))
            matching_writer.writerow(("source1_entity_id", "matched_entity_ids"))
            for frame in _source_chunks(s1_path):
                for s1_id, name, address, country in frame.itertuples(index=False, name=None):
                    s1_id = str(s1_id)
                    address = str(address)
                    candidate_map = get_production_candidates(index, conn, str(name), address, str(country))
                    candidate_ids = sorted(candidate_map)
                    address_norm = normalize_basic(address)
                    match_ids = sorted(
                        target_id for target_id, target_address in candidate_map.items()
                        if pair_score(address_norm, target_address) >= threshold
                    )
                    candidate_writer.writerow((s1_id, ",".join(candidate_ids)))
                    matching_writer.writerow((s1_id, ",".join(match_ids)))
                    count += 1
        candidate_tmp.replace(candidate_path)
        matching_tmp.replace(matching_path)
        return count
    except Exception:
        candidate_tmp.unlink(missing_ok=True)
        matching_tmp.unlink(missing_ok=True)
        raise


def run(data_dir: Path, output_dir: Path) -> None:
    paths = DatasetPaths(data_dir)
    paths.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    truths, rows = _load_selected_training(paths)
    training_records = (
        (s1_id, *row)
        for bucket in (0, 1)
        for s1_id, row in rows[bucket].items()
    )
    training_keys = active_keys_for_records(training_records)
    with tempfile.TemporaryDirectory(prefix="tmp-index-", dir=output_dir) as temp_dir:
        conn, index = build_production_index(
            paths.train_source2, paths.train_source3, Path(temp_dir) / "train.sqlite", training_keys
        )
        try:
            scores, candidates = {}, {}
            for bucket in (0, 1):
                scores[bucket], candidates[bucket] = _sample_metrics(
                    conn, index, rows[bucket], truths[bucket]
                )
            threshold, train_score = select_threshold(scores[0], truths[0])
            print(f"selected_threshold={threshold:.2f}")
            print(f"threshold_selection_s1_rows={len(truths[0])}")
            print(f"threshold_selection_macro_f05={train_score:.6f}")
            _report_holdout(conn, scores[1], candidates[1], truths[1], threshold, rows[1])
        finally:
            conn.close()

    test_records = (
        (str(s1_id), str(name), str(address), str(country))
        for frame in _source_chunks(paths.test_source1)
        for s1_id, name, address, country in frame.itertuples(index=False, name=None)
    )
    test_keys = active_keys_for_records(test_records)
    with tempfile.TemporaryDirectory(prefix="tmp-index-", dir=output_dir) as temp_dir:
        conn, index = build_production_index(
            paths.test_source2, paths.test_source3, Path(temp_dir) / "test.sqlite", test_keys
        )
        try:
            count = write_predictions(index, conn, paths.test_source1, threshold, output_dir)
        finally:
            conn.close()
    print(f"test_s1_rows_written={count}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen production retrieval baseline")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default="output", type=Path)
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir)
    except Exception as exc:
        print(f"baseline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
