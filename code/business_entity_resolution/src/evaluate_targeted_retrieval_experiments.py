"""
Controlled Retrieval Experiments on Top of Frozen Combo 3 Baseline.

Experiments:
- Baseline: Frozen Combo 3 (Retrieval Baseline B)
- Experiment A: Combo 3 + Secondary Blocker (2 shared address tokens, DF <= 2,000)
- Experiment B: Combo 3 + Secondary Blocker (single informative name token, len >= 4, DF <= 100)
- Experiment C: Combo 3 + Secondary Blocker A + Secondary Blocker B

Evaluated on the exact same 5,000-S1 validation split (seed=42) against full S2 + S3 (10.3M records).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import (
    ADDR_STOPWORDS,
    NAME_STOPWORDS,
)
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


def get_token_overlap_candidates(
    token_sets: list[set[str]],
    min_overlap: int,
) -> set[str]:
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


def load_ground_truth(path: Path) -> dict[str, list[str]]:
    gt = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        for row in chunk.itertuples(index=False):
            matched = row.matched_entity_ids.strip()
            gt[row.source1_entity_id] = (
                [m.strip() for m in matched.split(",") if m.strip()]
                if matched
                else []
            )
    return gt


def run_experiments(
    data_dir: Path,
    output_file: Path,
    sample_size: int = 5_000,
    seed: int = 42,
) -> None:
    print("=" * 80, flush=True)
    print("CONTROLLED RETRIEVAL EXPERIMENTS ON TOP OF FROZEN COMBO 3 BASELINE", flush=True)
    print(f"Validation sample size: {sample_size:,} S1 entities (seed={seed})", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"

    # Step 1: Load Ground Truth and create validation split
    print("\n[Step 1] Loading ground truth & creating validation split...", flush=True)
    gt = load_ground_truth(gt_path)
    all_s1_ids = sorted(gt.keys())
    rng = random.Random(seed)
    val_s1_ids = rng.sample(all_s1_ids, min(sample_size, len(all_s1_ids)))
    val_s1_set = set(val_s1_ids)

    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}
    val_s2_matches = sum(sum(1 for m in matches if m.startswith("S2-")) for matches in val_gt.values())
    val_s3_matches = sum(sum(1 for m in matches if m.startswith("S3-")) for matches in val_gt.values())
    total_val_matches = val_s2_matches + val_s3_matches

    print(f"  Validation S1 entities: {len(val_s1_ids):,}", flush=True)
    print(f"  Total true matches: {total_val_matches:,} (S2: {val_s2_matches:,}, S3: {val_s3_matches:,})", flush=True)

    # Step 2: Load S1 records & prepare query structures
    print("\n[Step 2] Loading validation S1 records & preparing query structures...", flush=True)
    val_s1_records: dict[str, dict] = {}
    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        mask = chunk["entity_id"].isin(val_s1_set)
        for eid, name, addr, country in zip(
            chunk.loc[mask, "entity_id"],
            chunk.loc[mask, "business_name"],
            chunk.loc[mask, "business_address"],
            chunk.loc[mask, "country"],
        ):
            val_s1_records[eid] = {
                "entity_id": eid,
                "business_name": name,
                "business_address": addr,
                "country": country,
            }

    s1_parsed: dict[str, dict] = {}
    for s1_id, rec in val_s1_records.items():
        country = normalize_basic(rec["country"])
        name = rec["business_name"]
        addr = rec["business_address"]

        nn = normalize_basic(name)
        ns = sorted_tokens(name)
        nc = compact(name)
        p5 = nc[:5] if len(nc) >= 5 else ""

        toks_name = tokenize(name)
        toks_addr = tokenize(addr)
        nums_addr = extract_address_numbers(addr)

        info_name = [t for t in toks_name if len(t) >= 3 and t not in NAME_STOPWORDS]
        stop_name = [t for t in toks_name if t in NAME_STOPWORDS]
        info_addr = [t for t in toks_addr if len(t) >= 3 and t not in ADDR_STOPWORDS]

        s1_parsed[s1_id] = {
            "country": country,
            "name_norm": nn,
            "name_sorted": ns,
            "name_compact": nc,
            "prefix5": p5,
            "info_name_tokens": info_name,
            "stop_name_tokens": stop_name,
            "info_addr_tokens": info_addr,
            "addr_numbers": nums_addr,
        }

    # Step 3: Load cached indices
    cache_path = output_file.parent / "canonical_combo3_index_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at {cache_path}. Run analyze_missed_matches.py first.")

    print(f"\n[Step 3] Loading precomputed Combo 3 indices from {cache_path}...", flush=True)
    t_cache_start = time.time()
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    idx_name_norm = cache_data["idx_name_norm"]
    idx_name_sorted = cache_data["idx_name_sorted"]
    idx_name_compact = cache_data["idx_name_compact"]
    idx_compact_prefix5 = cache_data["idx_compact_prefix5"]
    idx_name_tokens = cache_data["idx_name_tokens"]
    idx_name_stopwords = cache_data["idx_name_stopwords"]
    idx_addr_tokens = cache_data["idx_addr_tokens"]
    idx_addr_numbers = cache_data["idx_addr_numbers"]
    print(f"  Loaded cached indices in {time.time()-t_cache_start:.1f}s!", flush=True)

    # Step 4: Generate Baseline Combo 3 Candidates
    print("\n[Step 4] Generating baseline Combo 3 candidates...", flush=True)
    t_combo_start = time.time()
    combo3_cands: dict[str, set[str]] = {}

    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]
        c1 = idx_name_norm.get((c, p["name_norm"]), set())
        c2 = idx_name_sorted.get((c, p["name_sorted"]), set())
        c3 = idx_name_compact.get((c, p["name_compact"]), set())

        name_sets = [idx_name_tokens.get((c, t), set()) for t in p["info_name_tokens"] if (c, t) in idx_name_tokens]
        c4 = get_token_overlap_candidates(name_sets, min_overlap=2)

        addr_sets = [idx_addr_tokens.get((c, t), set()) for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens]
        c5 = get_token_overlap_candidates(addr_sets, min_overlap=3)

        c_set = c1 | c2 | c3 | c4 | c5

        # Name recovery: DF <= 500
        rare_info = [idx_name_tokens[(c, t)] for t in p["info_name_tokens"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
        if rare_info and p["stop_name_tokens"]:
            s_union = set()
            for st in p["stop_name_tokens"]:
                s_union |= idx_name_stopwords.get((c, st), set())
            if s_union:
                for n_set in rare_info:
                    c_set |= (n_set & s_union)

        # Address number + token: len>=3, DF<=500, addr_cap<=1000
        valid_nums = [n for n in p["addr_numbers"] if len(n) >= 3 and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 500]
        valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
        if valid_nums and valid_addrs:
            a_union = set()
            for a_set in valid_addrs:
                a_union |= a_set
            for num in valid_nums:
                c_set |= (idx_addr_numbers[(c, num)] & a_union)

        # Two rare address tokens: DF <= 500
        rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
        if len(rare_addrs) >= 2:
            c_set |= get_token_overlap_candidates(rare_addrs, min_overlap=2)

        # Single rare name token: len>=5, DF<=50
        for t in p["info_name_tokens"]:
            if len(t) >= 5 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 50:
                c_set |= idx_name_tokens[(c, t)]

        # Prefix5: DF <= 50
        p5 = p["prefix5"]
        if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= 50:
            c_set |= idx_compact_prefix5[(c, p5)]

        combo3_cands[s1_id] = c_set

    combo3_time = time.time() - t_combo_start
    print(f"  Combo 3 candidates generated in {combo3_time:.1f}s", flush=True)

    # Step 5: Evaluate Secondary Blockers
    print("\n[Step 5] Evaluating secondary blockers for Experiments A, B, and C...", flush=True)

    # Secondary Blocker A: 2 shared address tokens, DF <= 2,000
    sec_a_cands: dict[str, set[str]] = {}
    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]
        rare_addrs_2000 = [
            idx_addr_tokens[(c, t)]
            for t in p["info_addr_tokens"]
            if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000
        ]
        if len(rare_addrs_2000) >= 2:
            sec_a_cands[s1_id] = get_token_overlap_candidates(rare_addrs_2000, min_overlap=2)
        else:
            sec_a_cands[s1_id] = set()

    # Secondary Blocker B: single informative name token, len >= 4, DF <= 100
    sec_b_cands: dict[str, set[str]] = {}
    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]
        b_set = set()
        for t in p["info_name_tokens"]:
            if len(t) >= 4 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 100:
                b_set |= idx_name_tokens[(c, t)]
        sec_b_cands[s1_id] = b_set

    # Define experiments
    experiments = [
        {
            "id": "Baseline (Combo 3)",
            "name": "Frozen Combo 3 Baseline",
            "cand_map": combo3_cands,
            "is_baseline": True,
        },
        {
            "id": "Experiment A",
            "name": "Combo 3 + Secondary Blocker A (2 Addr Tokens, DF<=2,000)",
            "cand_map": {s1_id: combo3_cands[s1_id] | sec_a_cands[s1_id] for s1_id in val_s1_ids},
            "is_baseline": False,
        },
        {
            "id": "Experiment B",
            "name": "Combo 3 + Secondary Blocker B (Single Name Token, len>=4, DF<=100)",
            "cand_map": {s1_id: combo3_cands[s1_id] | sec_b_cands[s1_id] for s1_id in val_s1_ids},
            "is_baseline": False,
        },
        {
            "id": "Experiment C",
            "name": "Combo 3 + Secondary Blocker A + Secondary Blocker B",
            "cand_map": {s1_id: combo3_cands[s1_id] | sec_a_cands[s1_id] | sec_b_cands[s1_id] for s1_id in val_s1_ids},
            "is_baseline": False,
        },
    ]

    results = []
    combo3_retrieved_set: set[tuple[str, str]] = set()
    combo3_total_cands = 0
    combo3_mean_cands = 0.0

    for exp in experiments:
        cand_map = exp["cand_map"]
        cand_counts = [len(cands) for cands in cand_map.values()]
        total_cands = sum(cand_counts)
        mean_cands = total_cands / len(val_s1_ids)
        median_cands = float(np.median(cand_counts))
        p90_cands = float(np.percentile(cand_counts, 90))
        p95_cands = float(np.percentile(cand_counts, 95))
        p99_cands = float(np.percentile(cand_counts, 99))
        max_cands = int(max(cand_counts))

        retrieved_pairs: set[tuple[str, str]] = set()
        for s1_id in val_s1_ids:
            found = set(val_gt[s1_id]) & cand_map[s1_id]
            for m in found:
                retrieved_pairs.add((s1_id, m))

        retrieved_count = len(retrieved_pairs)
        recall = retrieved_count / total_val_matches

        if exp["is_baseline"]:
            combo3_retrieved_set = retrieved_pairs
            combo3_total_cands = total_cands
            combo3_mean_cands = mean_cands
            newly_recovered = 0
            added_total_cands = 0
            added_mean_cands = 0.0
            efficiency = 0.0
            affected_s1_count = 0
        else:
            newly_recovered = len(retrieved_pairs - combo3_retrieved_set)
            added_total_cands = total_cands - combo3_total_cands
            added_mean_cands = mean_cands - combo3_mean_cands
            efficiency = (newly_recovered / added_mean_cands) if added_mean_cands > 0 else 0.0
            affected_s1_count = sum(1 for s1_id in val_s1_ids if len(cand_map[s1_id]) > len(combo3_cands[s1_id]))

        exp_result = {
            "experiment_id": exp["id"],
            "name": exp["name"],
            "retrieved_true_matches": retrieved_count,
            "total_val_matches": total_val_matches,
            "recall": recall,
            "newly_recovered_matches": newly_recovered,
            "total_candidates": total_cands,
            "mean_candidates_per_s1": mean_cands,
            "median_candidates": median_cands,
            "p90_candidates": p90_cands,
            "p95_candidates": p95_cands,
            "p99_candidates": p99_cands,
            "max_candidates": max_cands,
            "total_added_candidates": added_total_cands,
            "added_mean_candidates": added_mean_cands,
            "efficiency": efficiency,
            "affected_s1_count": affected_s1_count,
        }
        results.append(exp_result)

        print("\n" + "=" * 80, flush=True)
        print(f"RESULTS FOR {exp['id']}: {exp['name']}", flush=True)
        print("=" * 80, flush=True)
        print(f"  Retrieved True Matches:    {retrieved_count:,} / {total_val_matches:,} ({recall*100:.2f}%)", flush=True)
        print(f"  Newly Recovered over C3:   +{newly_recovered:,}", flush=True)
        print(f"  Total Candidates:          {total_cands:,}", flush=True)
        print(f"  Mean Candidates / S1:      {mean_cands:.1f}", flush=True)
        print(f"  Median:                    {median_cands:.0f}", flush=True)
        print(f"  p90:                       {p90_cands:.0f}", flush=True)
        print(f"  p95:                       {p95_cands:.0f}", flush=True)
        print(f"  p99:                       {p99_cands:.0f}", flush=True)
        print(f"  Max:                       {max_cands:,}", flush=True)
        if not exp["is_baseline"]:
            print(f"  Total Added Candidates:    +{added_total_cands:,}", flush=True)
            print(f"  Added Candidates / S1:     +{added_mean_cands:.2f}", flush=True)
            print(f"  Efficiency (Matches/Cands): {efficiency:.4f}", flush=True)
            print(f"  Affected S1 Entities:      {affected_s1_count:,} / {len(val_s1_ids):,} ({affected_s1_count/len(val_s1_ids)*100:.1f}%)", flush=True)

    # Save output JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\nSaved results to {output_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate targeted retrieval experiments on Combo 3")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-file", type=Path, default=Path("output/targeted_retrieval_experiments_results.json"))
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiments(
        data_dir=args.data_dir,
        output_file=args.output_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )
