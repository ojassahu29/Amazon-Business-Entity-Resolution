from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from baseline import active_keys_for_records
from data_loader import DatasetPaths, iter_ground_truth, iter_source, parse_match_ids
from retrieval import ProductionRetrievalIndex, retrieve_candidates_batch

SAMPLE_SIZE = 5_000
INDEX_CACHE_VERSION = 1
# ponytail: bump this if target indexing/parsing changes; hash its code only if experiment scope expands beyond selector rules.
TARGET_CHUNK_SIZE = 50_000
QUERY_BATCH_SIZE = 250


def candidate_metrics(
    candidates_by_s1: dict[str, set[str]],
    truths_by_s1: dict[str, set[str]],
) -> dict[str, int | float]:
    candidate_pairs = sum(len(candidates) for candidates in candidates_by_s1.values())
    true_pairs = sum(len(truths) for truths in truths_by_s1.values())
    retrieved_true_pairs = sum(
        len(candidates_by_s1.get(s1_id, set()) & truths)
        for s1_id, truths in truths_by_s1.items()
    )
    return {
        "candidate_pairs": candidate_pairs,
        "true_pairs": true_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
        "candidate_precision": retrieved_true_pairs / candidate_pairs if candidate_pairs else 0.0,
        "candidate_recall": retrieved_true_pairs / true_pairs if true_pairs else 0.0,
    }


def _load_sample(paths: DatasetPaths, seed: int, sample_size: int) -> tuple[dict[str, set[str]], dict[str, dict[str, str]]]:
    all_truths: dict[str, set[str]] = {}
    for frame in iter_ground_truth(paths.train_ground_truth):
        for s1_id, matched_ids in frame.itertuples(index=False, name=None):
            all_truths[str(s1_id)] = set(parse_match_ids(matched_ids))

    sample_ids = random.Random(seed).sample(sorted(all_truths), min(sample_size, len(all_truths)))
    truths = {s1_id: all_truths[s1_id] for s1_id in sample_ids}
    remaining = set(sample_ids)
    records: dict[str, dict[str, str]] = {}
    for frame in iter_source(paths.train_source1):
        for s1_id, name, address, country in frame.itertuples(index=False, name=None):
            s1_id = str(s1_id)
            if s1_id in remaining:
                records[s1_id] = {
                    "business_name": str(name),
                    "business_address": str(address),
                    "country": str(country),
                }
                remaining.remove(s1_id)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Sampled ground-truth IDs missing from Source 1: {min(remaining)}")
    return truths, records


def _cache_identity(paths: DatasetPaths, seed: int, sample_size: int) -> dict[str, Any]:
    input_files = (
        paths.train_ground_truth,
        paths.train_source1,
        paths.train_source2,
        paths.train_source3,
    )
    return {
        "version": INDEX_CACHE_VERSION,
        "seed": seed,
        "sample_size": sample_size,
        "inputs": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in input_files
        ],
    }


def _cache_path(identity: dict[str, Any]) -> Path:
    root = Path(os.environ.get(
        "AUTORESEARCH_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "business-entity-resolution-autoresearch"),
    ))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return root / f"selector-index-{key}.pkl"


def _index_cache(index: ProductionRetrievalIndex) -> dict[str, Any]:
    return {
        "idx_name_norm": index.idx_name_norm,
        "idx_name_sorted": index.idx_name_sorted,
        "idx_name_compact": index.idx_name_compact,
        "idx_compact_prefix5": index.idx_compact_prefix5,
        "idx_name_tokens": index.idx_name_tokens,
        "idx_name_stopwords": index.idx_name_stopwords,
        "idx_addr_tokens": index.idx_addr_tokens,
        "idx_addr_numbers": index.idx_addr_numbers,
        "n_indexed": index.n_indexed,
    }


def _build_index(paths: DatasetPaths, records: dict[str, dict[str, str]]) -> ProductionRetrievalIndex:
    active_keys = active_keys_for_records(
        (s1_id, row["business_name"], row["business_address"], row["country"])
        for s1_id, row in records.items()
    )
    index = ProductionRetrievalIndex(active_keys)
    total = 0
    for source, path in (("S2", paths.train_source2), ("S3", paths.train_source3)):
        source_total = 0
        print(f"INDEX_START source={source} path={path}", flush=True)
        for frame in iter_source(path, chunksize=TARGET_CHUNK_SIZE):
            for row in frame.itertuples(index=False, name=None):
                index.index_record(*(str(value) for value in row))
            source_total += len(frame)
            total += len(frame)
            if source_total % 500_000 == 0:
                print(f"INDEX_PROGRESS source={source} rows={source_total} total={total}", flush=True)
        print(f"INDEX_DONE source={source} rows={source_total}", flush=True)
    print(f"INDEX_READY targets={index.n_indexed} active_keys={sum(map(len, active_keys.values()))}", flush=True)
    return index


def _load_or_build_index(
    paths: DatasetPaths,
    records: dict[str, dict[str, str]],
    identity: dict[str, Any],
) -> ProductionRetrievalIndex:
    cache_file = _cache_path(identity)
    if cache_file.is_file():
        print(f"INDEX_CACHE_HIT path={cache_file}", flush=True)
        with cache_file.open("rb") as file:
            cached = pickle.load(file)
        if cached.get("identity") != identity:
            raise ValueError(f"Index cache identity mismatch: {cache_file}")
        return ProductionRetrievalIndex.from_cache_dict(cached["index"])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"INDEX_CACHE_MISS path={cache_file}", flush=True)
    index = _build_index(paths, records)
    fd, temporary_name = tempfile.mkstemp(prefix=cache_file.name, suffix=".tmp", dir=cache_file.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump({"identity": identity, "index": _index_cache(index)}, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, cache_file)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return index


def run_benchmark(data_dir: Path, seed: int, sample_size: int) -> dict[str, int | float]:
    paths = DatasetPaths(data_dir)
    for path in (paths.train_ground_truth, paths.train_source1, paths.train_source2, paths.train_source3):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    print(f"WORKLOAD seed={seed} s1_sample={sample_size} targets=train_source2+train_source3", flush=True)
    truths, records = _load_sample(paths, seed, sample_size)
    print(f"SAMPLE_READY queries={len(records)} true_pairs={sum(map(len, truths.values()))}", flush=True)
    identity = _cache_identity(paths, seed, sample_size)
    index = _load_or_build_index(paths, records, identity)

    candidates_by_s1: dict[str, set[str]] = {}
    ordered_records = list(records.items())
    for start in range(0, len(ordered_records), QUERY_BATCH_SIZE):
        batch = dict(ordered_records[start:start + QUERY_BATCH_SIZE])
        candidates_by_s1.update(retrieve_candidates_batch(batch, index))
        completed = min(start + len(batch), len(ordered_records))
        print(f"QUERY_PROGRESS completed={completed} total={len(ordered_records)}", flush=True)

    metrics = candidate_metrics(candidates_by_s1, truths)
    metrics["mean_candidates"] = metrics["candidate_pairs"] / len(records) if records else 0.0
    metrics["elapsed_seconds"] = time.perf_counter() - started
    print(f"METRIC candidate_precision={metrics['candidate_precision']:.9f}", flush=True)
    print(f"METRIC candidate_recall={metrics['candidate_recall']:.9f}", flush=True)
    print(f"METRIC candidate_pairs={metrics['candidate_pairs']}", flush=True)
    print(f"METRIC retrieved_true_pairs={metrics['retrieved_true_pairs']}", flush=True)
    print(f"METRIC mean_candidates={metrics['mean_candidates']:.6f}", flush=True)
    print(f"METRIC elapsed_seconds={metrics['elapsed_seconds']:.3f}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure production selector precision and recall on a fixed labeled sample")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.seed, args.sample_size)


if __name__ == "__main__":
    main()
