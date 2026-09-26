"""
Generate the Pre-Refactor Golden Reference for Frozen C3 + Secondary A + C2-A.
Uses the exact validated implementation from evaluate_retrieval_pruning.py.
Validation sample: 5,000 S1 queries, seed=42, full S2+S3 universe via canonical cache.
Outputs:
- output/golden_reference_seed42.pkl (ignored by git via output/*.pkl)
- output/golden_reference_seed42_manifest.json (metadata, hash digests, distribution metrics)
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import pickle
import random
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"))

from blocking import ADDR_STOPWORDS, NAME_STOPWORDS
from preprocessing import (
    compact,
    normalize_basic,
    sorted_tokens,
    tokenize,
)


def extract_address_numbers(addr: str) -> set[str]:
    raw_nums = re.findall(r"\b\d+[/\w-]*\b", addr)
    nums = set()
    for n in raw_nums:
        clean_n = n.strip(" ,.-/#")
        if len(clean_n) >= 2 and clean_n not in {"2019", "2020", "2021", "2022", "2023", "2024", "2025"}:
            nums.add(clean_n.lower())
    return nums


def get_token_overlap_candidates(token_sets: list[set[str]], min_overlap: int) -> set[str]:
    if len(token_sets) < min_overlap:
        return set()
    token_sets = sorted(token_sets, key=len)
    if min_overlap == 2 and len(token_sets) <= 6:
        res = set()
        for a, b in combinations(token_sets, 2):
            res |= (a & b)
        return res
    elif min_overlap == 3 and len(token_sets) <= 4:
        res = set()
        for a, b, c in combinations(token_sets, 3):
            res |= (a & b & c)
        return res

    hits: dict[str, int] = defaultdict(int)
    for s in token_sets:
        for eid in s:
            hits[eid] += 1
    return {eid for eid, cnt in hits.items() if cnt >= min_overlap}


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
    cache_path = Path("output/canonical_combo3_index_cache.pkl")
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at {cache_path}")

    print(f"Loading canonical index cache from {cache_path}...", flush=True)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    idx_name_norm = cache["idx_name_norm"]
    idx_name_sorted = cache["idx_name_sorted"]
    idx_name_compact = cache["idx_name_compact"]
    idx_compact_prefix5 = cache["idx_compact_prefix5"]
    idx_name_tokens = cache["idx_name_tokens"]
    idx_name_stopwords = cache["idx_name_stopwords"]
    idx_addr_tokens = cache["idx_addr_tokens"]
    idx_addr_numbers = cache["idx_addr_numbers"]

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
    print(f"Validation sample: {len(val_s1_ids):,} S1 queries, {total_val_positives:,} true matches.", flush=True)

    print("Loading Source 1 records...", flush=True)
    val_s1_records: dict[str, dict[str, str]] = {}
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
        mask = chunk["entity_id"].isin(val_s1_set)
        for eid, name, addr, country in zip(
            chunk.loc[mask, "entity_id"],
            chunk.loc[mask, "business_name"],
            chunk.loc[mask, "business_address"],
            chunk.loc[mask, "country"],
        ):
            val_s1_records[eid] = {"name": name, "addr": addr, "country": normalize_basic(country)}

    # Parse S1 records
    print("Parsing S1 records...", flush=True)
    s1_parsed: dict[str, dict[str, Any]] = {}
    for s1_id in val_s1_ids:
        rec = val_s1_records[s1_id]
        c = rec["country"]
        name = rec["name"]
        addr = rec["addr"]
        nn = normalize_basic(name)
        ns = sorted_tokens(name)
        nc = compact(name)
        p5 = nc[:5] if len(nc) >= 5 else ""
        toks_n = tokenize(name)
        toks_a = tokenize(addr)
        info_n = [t for t in toks_n if len(t) >= 3 and t not in NAME_STOPWORDS]
        stop_n = [t for t in toks_n if t in NAME_STOPWORDS]
        info_a = [t for t in toks_a if len(t) >= 3 and t not in ADDR_STOPWORDS]
        nums = extract_address_numbers(addr)
        bldg = {n for n in nums if not re.match(r"^\d{5,6}$", n)}

        s1_parsed[s1_id] = {
            "country": c,
            "raw_name": name,
            "raw_addr": addr,
            "name_norm": nn,
            "name_sorted": ns,
            "name_compact": nc,
            "prefix5": p5,
            "info_name": info_n,
            "stop_name": stop_n,
            "info_addr": info_a,
            "all_nums": nums,
            "bldg_nums": bldg,
        }

    # Generate Candidate Pairs for Frozen Retrieval: C3 + Secondary A (DF <= 2000) + C2-A (DF <= 500)
    print("\nExecuting frozen retrieval (C3 + Secondary A + C2-A)...", flush=True)
    golden_candidates: dict[str, list[str]] = {}
    total_retrieved_matches = 0
    candidate_counts: list[int] = []

    # Stream-compute a deterministic SHA-256 over all canonical candidate pairs
    # Canonical candidate pair: f"{s1_id}\t{cand_id}\n" sorted globally
    hasher = hashlib.sha256()

    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]

        # 1. Combo 3 primary
        c1 = idx_name_norm.get((c, p["name_norm"]), set())
        c2 = idx_name_sorted.get((c, p["name_sorted"]), set())
        c3 = idx_name_compact.get((c, p["name_compact"]), set())
        name_sets = [idx_name_tokens.get((c, t), set()) for t in p["info_name"] if (c, t) in idx_name_tokens]
        c4 = get_token_overlap_candidates(name_sets, min_overlap=2)
        addr_sets = [idx_addr_tokens.get((c, t), set()) for t in p["info_addr"] if (c, t) in idx_addr_tokens]
        c5 = get_token_overlap_candidates(addr_sets, min_overlap=3)
        c_set = c1 | c2 | c3 | c4 | c5

        rare_info = [idx_name_tokens[(c, t)] for t in p["info_name"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
        if rare_info and p["stop_name"]:
            s_union = set()
            for st in p["stop_name"]:
                s_union |= idx_name_stopwords.get((c, st), set())
            if s_union:
                for n_set in rare_info:
                    c_set |= (n_set & s_union)

        valid_nums = [n for n in p["all_nums"] if len(n) >= 3 and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 500]
        valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
        if valid_nums and valid_addrs:
            a_union = set().union(*valid_addrs)
            for num in valid_nums:
                c_set |= (idx_addr_numbers[(c, num)] & a_union)

        rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
        if len(rare_addrs) >= 2:
            c_set |= get_token_overlap_candidates(rare_addrs, min_overlap=2)

        for t in p["info_name"]:
            if len(t) >= 5 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 50:
                c_set |= idx_name_tokens[(c, t)]

        p5 = p["prefix5"]
        if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= 50:
            c_set |= idx_compact_prefix5[(c, p5)]

        # 2. Secondary A: >=2 shared address tokens, DF <= 2000
        sec_a_rare = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
        if len(sec_a_rare) >= 2:
            c_set |= get_token_overlap_candidates(sec_a_rare, min_overlap=2)

        # 3. C2-A: same country + bldg num + >=1 address token DF <= 500
        b_nums = [n for n in p["bldg_nums"] if (c, n) in idx_addr_numbers]
        a_toks = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
        if b_nums and a_toks:
            a_u = set().union(*a_toks)
            for n in b_nums:
                c_set |= (idx_addr_numbers[(c, n)] & a_u)

        # Evaluate matches
        gt_set = set(val_gt[s1_id])
        m_set = c_set & gt_set
        total_retrieved_matches += len(m_set)

        # Sort candidate list deterministically for canonical representation
        sorted_cands = sorted(c_set)
        golden_candidates[s1_id] = sorted_cands
        candidate_counts.append(len(sorted_cands))

        for cand_id in sorted_cands:
            hasher.update(f"{s1_id}\t{cand_id}\n".encode("utf-8"))

    stats = compute_distribution_stats(candidate_counts)
    sha256_digest = hasher.hexdigest()
    recall = total_retrieved_matches / total_val_positives * 100.0

    print("=" * 80)
    print("PRE-REFACTOR GOLDEN REFERENCE METRICS")
    print("=" * 80)
    print(f"S1 Queries:              {len(val_s1_ids):,}")
    print(f"Total Ground Truth:      {total_val_positives:,}")
    print(f"Retrieved True Matches:  {total_retrieved_matches:,}")
    print(f"Retrieval Recall:        {recall:.4f}% ({total_retrieved_matches}/{total_val_positives})")
    print(f"Total Candidate Pairs:   {stats['total']:,}")
    print(f"Mean candidates/S1:      {stats['mean']:.2f}")
    print(f"Median candidates/S1:    {stats['median']:.1f}")
    print(f"P90 candidates/S1:       {stats['p90']:.1f}")
    print(f"P95 candidates/S1:       {stats['p95']:.1f}")
    print(f"P99 candidates/S1:       {stats['p99']:.1f}")
    print(f"Max candidates/S1:       {stats['max']:,}")
    print(f"Min candidates/S1:       {stats['min']:,}")
    print(f"Candidate Pair SHA-256:  {sha256_digest}")
    print("=" * 80, flush=True)

    # Verification against expected values
    expected_matches = 16370
    expected_candidates = 8555167
    expected_recall = 94.55

    assert total_retrieved_matches == expected_matches, f"Matches mismatch! Got {total_retrieved_matches}, expected {expected_matches}"
    assert stats["total"] == expected_candidates, f"Candidates mismatch! Got {stats['total']}, expected {expected_candidates}"
    assert round(recall, 2) == expected_recall, f"Recall mismatch! Got {recall:.2f}%, expected {expected_recall}%"
    print("\n>>> ALL VALIDATION ASSERTIONS PASSED! EXACT MATCH WITH VALIDATED SEED=42 BASELINE <<<\n", flush=True)

    # Save reference locally (ignored by git via output/*.pkl)
    ref_pkl_path = Path("output/golden_reference_seed42.pkl")
    print(f"Saving exact candidate dictionary to {ref_pkl_path}...", flush=True)
    with open(ref_pkl_path, "wb") as f:
        pickle.dump(golden_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "architecture": "Combo 3 + Secondary A (DF <= 2000) + C2-A (DF <= 500)",
        "implementation_source": "code/business_entity_resolution/src/evaluate_retrieval_pruning.py",
        "validation_sample": {
            "num_s1": len(val_s1_ids),
            "seed": 42,
            "total_gt_matches": total_val_positives,
        },
        "metrics": {
            "retrieved_matches": total_retrieved_matches,
            "recall_percent": round(recall, 4),
            "total_candidate_pairs": stats["total"],
            "mean_candidates": stats["mean"],
            "median_candidates": stats["median"],
            "p90": stats["p90"],
            "p95": stats["p95"],
            "p99": stats["p99"],
            "max": stats["max"],
            "min": stats["min"],
        },
        "sha256_digest": sha256_digest,
        "reference_file": str(ref_pkl_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = Path("output/golden_reference_seed42_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}.", flush=True)
    print(f"Total time elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
