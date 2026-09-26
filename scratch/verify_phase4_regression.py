"""
Phase 4 Exact Behavioral Regression Test.
Verifies that code/business_entity_resolution/src/retrieval.py reproduces the exact
candidate pairs stored in output/golden_reference_seed42.pkl.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"))

from retrieval import (
    ProductionRetrievalIndex,
    parse_entity_record,
    retrieve_candidates_batch,
    retrieve_candidates_for_record,
)


def compute_distribution_stats(counts: list[int]) -> dict[str, float]:
    arr = np.array(counts, dtype=float)
    return {
        "total": int(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
        "min": int(np.min(arr)),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("PHASE 4: EXACT BEHAVIORAL REGRESSION VERIFICATION")
    print("=" * 80, flush=True)

    # 1. Load Golden Reference
    ref_path = Path("output/golden_reference_seed42.pkl")
    if not ref_path.exists():
        raise FileNotFoundError(f"Golden reference not found at {ref_path}")

    print(f"Loading Phase 2 Golden Reference from {ref_path}...", flush=True)
    with open(ref_path, "rb") as f:
        golden_candidates: dict[str, list[str]] = pickle.load(f)

    # 2. Load Ground Truth
    print("Loading Ground Truth...", flush=True)
    gt: dict[str, list[str]] = {}
    for chunk in pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
        for row in chunk.itertuples(index=False):
            m = row.matched_entity_ids.strip()
            gt[row.source1_entity_id] = [x.strip() for x in m.split(",") if x.strip()] if m else []

    val_s1_ids = random.Random(42).sample(sorted(gt.keys()), 5000)
    val_s1_set = set(val_s1_ids)
    val_gt = {s: gt[s] for s in val_s1_ids}
    total_val_positives = sum(len(v) for v in val_gt.values())

    # 3. Load S1 records
    print("Loading S1 records...", flush=True)
    val_s1_records: dict[str, dict[str, str]] = {}
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
        mask = chunk["entity_id"].isin(val_s1_set)
        for eid, name, addr, country in zip(
            chunk.loc[mask, "entity_id"],
            chunk.loc[mask, "business_name"],
            chunk.loc[mask, "business_address"],
            chunk.loc[mask, "country"],
        ):
            val_s1_records[eid] = {
                "business_name": name,
                "business_address": addr,
                "country": country,
            }

    # 4. Load ProductionRetrievalIndex
    cache_path = Path("output/canonical_combo3_index_cache.pkl")
    print(f"Loading ProductionRetrievalIndex from {cache_path}...", flush=True)
    index = ProductionRetrievalIndex.load_cache(cache_path)

    # 5. Execute production retrieval via src/retrieval.py entry point
    print("Executing production batch retrieval via retrieve_candidates_batch...", flush=True)
    t_retrieval_start = time.time()
    
    # We pass the records in sorted s1 order to maintain determinism
    ordered_s1_records = [(s1_id, val_s1_records[s1_id]["business_name"], val_s1_records[s1_id]["business_address"], val_s1_records[s1_id]["country"]) for s1_id in val_s1_ids]
    new_candidates_dict: dict[str, set[str]] = retrieve_candidates_batch(ordered_s1_records, index)
    print(f"Production retrieval completed in {time.time() - t_retrieval_start:.1f}s", flush=True)

    # 6. Detailed Pairwise and Set Equality Comparison
    print("\nComparing candidate sets against Golden Reference...", flush=True)
    
    hasher = hashlib.sha256()
    new_candidate_counts: list[int] = []
    total_retrieved_matches = 0

    old_total_pairs = sum(len(cands) for cands in golden_candidates.values())
    new_total_pairs = sum(len(cands) for cands in new_candidates_dict.values())

    missing_in_new = 0
    extra_in_new = 0
    divergent_queries = 0
    first_divergence_info = None

    for s1_id in val_s1_ids:
        old_set = set(golden_candidates[s1_id])
        new_set = new_candidates_dict[s1_id]

        # Check for set equality per query
        diff_old_minus_new = old_set - new_set
        diff_new_minus_old = new_set - old_set

        if diff_old_minus_new or diff_new_minus_old:
            divergent_queries += 1
            missing_in_new += len(diff_old_minus_new)
            extra_in_new += len(diff_new_minus_old)
            if first_divergence_info is None:
                first_divergence_info = {
                    "s1_id": s1_id,
                    "old_count": len(old_set),
                    "new_count": len(new_set),
                    "missing_sample": list(diff_old_minus_new)[:5],
                    "extra_sample": list(diff_new_minus_old)[:5],
                }

        # True matches
        gt_set = set(val_gt[s1_id])
        m_set = new_set & gt_set
        total_retrieved_matches += len(m_set)

        # Canonical hashing in sorted order
        sorted_new_cands = sorted(new_set)
        new_candidate_counts.append(len(sorted_new_cands))
        for cand_id in sorted_new_cands:
            hasher.update(f"{s1_id}\t{cand_id}\n".encode("utf-8"))

    new_sha256 = hasher.hexdigest()
    expected_sha256 = "873c791862d91ae0c91f26047c4787af50de93e32d06db878e0d5802956f2c5c"
    sha256_match = (new_sha256 == expected_sha256)
    candidate_set_equality = (divergent_queries == 0 and missing_in_new == 0 and extra_in_new == 0)

    stats = compute_distribution_stats(new_candidate_counts)
    recall = total_retrieved_matches / total_val_positives * 100.0

    print("=" * 80)
    print("PHASE 4 REGRESSION TEST RESULTS")
    print("=" * 80)
    print(f"Old Candidate-Pair Count: {old_total_pairs:,}")
    print(f"New Candidate-Pair Count: {new_total_pairs:,}")
    print(f"Old - New Count:         {missing_in_new:,}")
    print(f"New - Old Count:         {extra_in_new:,}")
    print(f"Divergent Queries:       {divergent_queries:,} / {len(val_s1_ids):,}")
    print(f"Exact Candidate-Set EQ:  {candidate_set_equality}")
    print(f"New SHA-256 Digest:      {new_sha256}")
    print(f"Expected SHA-256 Digest: {expected_sha256}")
    print(f"SHA-256 Exact Match:     {sha256_match}")
    print(f"Retrieved True Matches:  {total_retrieved_matches:,} (Expected: 16,370)")
    print(f"Retrieval Recall:        {recall:.4f}% (Expected: 94.5478%)")
    print(f"Candidate Mean / S1:     {stats['mean']:.2f} (Expected: 1711.03)")
    print(f"Candidate Median:        {stats['median']:.1f} (Expected: 233.5)")
    print(f"Candidate P90:           {stats['p90']:.1f} (Expected: 3792.4)")
    print(f"Candidate P95:           {stats['p95']:.1f} (Expected: 8339.2)")
    print(f"Candidate P99:           {stats['p99']:.1f} (Expected: 26163.2)")
    print(f"Candidate Max:           {stats['max']:,} (Expected: 59,456)")
    print(f"Candidate Min:           {stats['min']:,} (Expected: 1)")
    print("=" * 80, flush=True)

    if not candidate_set_equality or not sha256_match:
        print("\n[CRITICAL ERROR] Candidate sets differ from Golden Reference!", flush=True)
        if first_divergence_info:
            print("First Divergence Details:", json.dumps(first_divergence_info, indent=2), flush=True)
        sys.exit(1)

    print("\n>>> PRIMARY INVARIANT SATISFIED: set(old) == set(new) (100% BIT-FOR-BIT IDENTICAL) <<<\n", flush=True)
    print(f"Total verification time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
