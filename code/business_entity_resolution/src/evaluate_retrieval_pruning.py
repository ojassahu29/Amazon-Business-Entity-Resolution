"""
Retrieval Architecture Optimization & Pruning Experiment.

Evaluates structural redundancy, layer marginal contributions, single-layer ablations,
threshold pruning grids, and combined Pareto frontier across the current retrieval stack:
1. Combo 3 (Frozen Baseline B)
2. Secondary A: >=2 shared informative address tokens (DF <= cap)
3. Secondary B: single informative name token, len >= 4 (DF <= cap)
4. C2-A: same country, shared building/address number, >=1 address token (DF <= cap)

Current Full Union (Combo 3 + A + B + C2-A):
- 16,380 / 17,314 true matches (94.61% recall)
- 8,574,256 total candidates (1,714.85 / S1)

Validation: 5,000 S1 queries, seed=42, full S2+S3 candidate space.
Analysis-only script; output saved to output/retrieval_pruning_results.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
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
# pyrefly: ignore [missing-import]
from rapidfuzz import fuzz

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


def extract_pincodes(addr: str) -> set[str]:
    return set(re.findall(r"\b\d{5,6}\b", addr))


def is_synthetic_or_cross_script(text: str) -> bool:
    if any(ord(c) > 127 for c in text):
        return True
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 6:
        vowels = sum(1 for c in cleaned if c in "aeiou")
        vowel_ratio = vowels / len(cleaned)
        if vowel_ratio < 0.15 or vowel_ratio > 0.70:
            return True
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", cleaned):
            return True
    return False


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
    start_time = time.time()
    print("=" * 80)
    print("RETRIEVAL ARCHITECTURE PRUNING & PARETO OPTIMIZATION")
    print("=" * 80, flush=True)

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
    true_match_records = cache["true_match_records"]

    print("Loading Ground Truth and Source 1...", flush=True)
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

    # Step 1: Precompute candidate sets for Combo 3
    print("\nGenerating Combo 3 candidate sets...", flush=True)
    t0 = time.time()
    c3_cands: dict[str, set[str]] = {}
    c3_matches: dict[str, set[str]] = {}
    total_c3_matches = 0

    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]

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

        c3_cands[s1_id] = c_set
        gt_set = set(val_gt[s1_id])
        m_set = c_set & gt_set
        c3_matches[s1_id] = m_set
        total_c3_matches += len(m_set)

    c3_counts = [len(c3_cands[s]) for s in val_s1_ids]
    c3_stats = compute_distribution_stats(c3_counts)
    print(f"  Combo 3 generated in {time.time()-t0:.1f}s: {total_c3_matches:,} true matches "
          f"({total_c3_matches/total_val_positives*100:.2f}%), {c3_stats['total']:,} candidates "
          f"(mean {c3_stats['mean']:.2f}, median {c3_stats['median']:.1f}, p90 {c3_stats['p90']:.1f}, "
          f"p95 {c3_stats['p95']:.1f}, p99 {c3_stats['p99']:.1f}, max {c3_stats['max']:,})", flush=True)

    # -------------------------------------------------------------
    # PRECOMPUTE SECONDARY LAYERS FOR ALL THRESHOLDS
    # -------------------------------------------------------------
    # Thresholds to test:
    # Secondary A: >=2 shared address tokens, DF <= {500, 1000, 1500, 2000}
    # Secondary B: 1 name token len >= 4, DF <= {25, 50, 75, 100, 150, 200}
    # C2-A: shared bldg num & 1 addr token DF <= {250, 500, 750, 1000}

    a_thresholds = [500, 1000, 1500, 2000]
    b_thresholds = [25, 50, 75, 100, 150, 200]
    c2_thresholds = [250, 500, 750, 1000]

    print("\nPrecomputing Secondary A candidates for thresholds: ", a_thresholds, flush=True)
    # Store candidates disjoint from C3: diff_cands = raw_cands - c3_cands[s1_id]
    # And store full raw cands for provenance analysis
    a_raw_cands: dict[int, dict[str, set[str]]] = {th: {} for th in a_thresholds}
    a_diff_cands: dict[int, dict[str, set[str]]] = {th: {} for th in a_thresholds}
    for th in a_thresholds:
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= th]
            raw_s = get_token_overlap_candidates(rare_addrs, min_overlap=2) if len(rare_addrs) >= 2 else set()
            a_raw_cands[th][s1_id] = raw_s
            a_diff_cands[th][s1_id] = raw_s - c3_cands[s1_id]

    print("Precomputing Secondary B candidates for thresholds: ", b_thresholds, flush=True)
    b_raw_cands: dict[int, dict[str, set[str]]] = {th: {} for th in b_thresholds}
    b_diff_cands: dict[int, dict[str, set[str]]] = {th: {} for th in b_thresholds}
    for th in b_thresholds:
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            b_s = set()
            for t in p["info_name"]:
                if len(t) >= 4 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= th:
                    b_s |= idx_name_tokens[(c, t)]
            b_raw_cands[th][s1_id] = b_s
            b_diff_cands[th][s1_id] = b_s - c3_cands[s1_id]

    print("Precomputing C2-A candidates for thresholds: ", c2_thresholds, flush=True)
    c2_raw_cands: dict[int, dict[str, set[str]]] = {th: {} for th in c2_thresholds}
    c2_diff_cands: dict[int, dict[str, set[str]]] = {th: {} for th in c2_thresholds}
    for th in c2_thresholds:
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            b_nums = [n for n in p["bldg_nums"] if (c, n) in idx_addr_numbers]
            a_toks = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= th]
            c2_s = set()
            if b_nums and a_toks:
                a_u = set().union(*a_toks)
                for n in b_nums:
                    c2_s |= (idx_addr_numbers[(c, n)] & a_u)
            c2_raw_cands[th][s1_id] = c2_s
            c2_diff_cands[th][s1_id] = c2_s - c3_cands[s1_id]

    # Ground truth mapping
    gt_map = {s1_id: set(val_gt[s1_id]) for s1_id in val_s1_ids}

    # -------------------------------------------------------------
    # STEP 1 — MARGINAL CONTRIBUTION OF EACH LAYER IN CURRENT STACK
    # Current Stack: C3 + A (DF<=2000) + B (DF<=100) + C2-A (DF<=500)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: MARGINAL CONTRIBUTION & PROVENANCE OF EACH LAYER")
    print("=" * 80, flush=True)

    curr_a = a_raw_cands[2000]
    curr_b = b_raw_cands[100]
    curr_c2 = c2_raw_cands[500]

    # Full stack candidate sets per S1
    full_stack_cands: dict[str, set[str]] = {}
    full_stack_matches: set[tuple[str, str]] = set()

    # Track layer memberships for every candidate pair (s1, cand)
    # Provenance counts
    # Layers: 'C3', 'A', 'B', 'C2'
    layer_total_cands = {"C3": 0, "A": 0, "B": 0, "C2-A": 0}
    layer_unique_cands = {"C3": 0, "A": 0, "B": 0, "C2-A": 0}
    layer_overlap_2 = {"C3": 0, "A": 0, "B": 0, "C2-A": 0}
    layer_overlap_3 = {"C3": 0, "A": 0, "B": 0, "C2-A": 0}
    layer_overlap_4 = {"C3": 0, "A": 0, "B": 0, "C2-A": 0}

    # True match provenance
    layer_total_tm = {"C3": set(), "A": set(), "B": set(), "C2-A": set()}
    layer_unique_tm = {"C3": set(), "A": set(), "B": set(), "C2-A": set()}

    overlap_dist = Counter()  # distribution of candidate overlap counts: 1, 2, 3, 4

    for s1_id in val_s1_ids:
        s_c3 = c3_cands[s1_id]
        s_a = curr_a[s1_id]
        s_b = curr_b[s1_id]
        s_c2 = curr_c2[s1_id]
        union_all = s_c3 | s_a | s_b | s_c2
        full_stack_cands[s1_id] = union_all

        gt_s = gt_map[s1_id]
        for m in union_all & gt_s:
            full_stack_matches.add((s1_id, m))

        for cand in union_all:
            in_c3 = cand in s_c3
            in_a = cand in s_a
            in_b = cand in s_b
            in_c2 = cand in s_c2

            deg = int(in_c3) + int(in_a) + int(in_b) + int(in_c2)
            overlap_dist[deg] += 1

            if in_c3:
                layer_total_cands["C3"] += 1
                if deg == 1: layer_unique_cands["C3"] += 1
                elif deg == 2: layer_overlap_2["C3"] += 1
                elif deg == 3: layer_overlap_3["C3"] += 1
                elif deg == 4: layer_overlap_4["C3"] += 1

            if in_a:
                layer_total_cands["A"] += 1
                if deg == 1: layer_unique_cands["A"] += 1
                elif deg == 2: layer_overlap_2["A"] += 1
                elif deg == 3: layer_overlap_3["A"] += 1
                elif deg == 4: layer_overlap_4["A"] += 1

            if in_b:
                layer_total_cands["B"] += 1
                if deg == 1: layer_unique_cands["B"] += 1
                elif deg == 2: layer_overlap_2["B"] += 1
                elif deg == 3: layer_overlap_3["B"] += 1
                elif deg == 4: layer_overlap_4["B"] += 1

            if in_c2:
                layer_total_cands["C2-A"] += 1
                if deg == 1: layer_unique_cands["C2-A"] += 1
                elif deg == 2: layer_overlap_2["C2-A"] += 1
                elif deg == 3: layer_overlap_3["C2-A"] += 1
                elif deg == 4: layer_overlap_4["C2-A"] += 1

            if cand in gt_s:
                pair = (s1_id, cand)
                if in_c3: layer_total_tm["C3"].add(pair)
                if in_a: layer_total_tm["A"].add(pair)
                if in_b: layer_total_tm["B"].add(pair)
                if in_c2: layer_total_tm["C2-A"].add(pair)

                if deg == 1:
                    if in_c3: layer_unique_tm["C3"].add(pair)
                    elif in_a: layer_unique_tm["A"].add(pair)
                    elif in_b: layer_unique_tm["B"].add(pair)
                    elif in_c2: layer_unique_tm["C2-A"].add(pair)

    full_stack_counts = [len(full_stack_cands[s]) for s in val_s1_ids]
    full_stack_stats = compute_distribution_stats(full_stack_counts)
    total_full_tm = len(full_stack_matches)
    total_full_cands = full_stack_stats["total"]

    print(f"Current Full Stack (Combo 3 + A + B + C2-A): {total_full_tm:,} / {total_val_positives:,} "
          f"({total_full_tm/total_val_positives*100:.2f}%), {total_full_cands:,} candidates "
          f"(mean {full_stack_stats['mean']:.2f}, median {full_stack_stats['median']:.1f}, "
          f"p90 {full_stack_stats['p90']:.1f}, p95 {full_stack_stats['p95']:.1f}, "
          f"p99 {full_stack_stats['p99']:.1f}, max {full_stack_stats['max']:,})", flush=True)

    print("\nGlobal Candidate Overlap Distribution Across the 4 Layers:")
    for deg in [1, 2, 3, 4]:
        cnt = overlap_dist[deg]
        pct = cnt / total_full_cands * 100
        print(f"  Overlapping exactly {deg} layer(s): {cnt:,} ({pct:.2f}%)")

    # Marginal contributions table
    # Removal impacts:
    # Full - (without L)
    # Without C3: A + B + C2-A
    # Without A: C3 + B + C2-A
    # Without B: C3 + A + C2-A
    # Without C2-A: C3 + A + B
    step1_table = {}
    for lyr, name in [("C3", "Combo 3"), ("A", "Secondary A (DF<=2000)"), ("B", "Secondary B (DF<=100)"), ("C2-A", "C2-A (DF<=500)")]:
        tot_c = layer_total_cands[lyr]
        uniq_c = layer_unique_cands[lyr]
        ov2 = layer_overlap_2[lyr]
        ov3 = layer_overlap_3[lyr]
        ov4 = layer_overlap_4[lyr]
        tot_m = len(layer_total_tm[lyr])
        uniq_m = len(layer_unique_tm[lyr])  # true matches that disappear if layer removed
        recall_loss = uniq_m / total_val_positives
        cand_reduction = uniq_c  # candidates removed if layer removed

        step1_table[lyr] = {
            "layer_name": name,
            "total_candidates_contributed": tot_c,
            "candidates_unique_to_layer": uniq_c,
            "candidates_overlapping_2_layers": ov2,
            "candidates_overlapping_3_layers": ov3,
            "candidates_overlapping_4_layers": ov4,
            "true_matches_contributed_total": tot_m,
            "true_matches_unique_to_layer": uniq_m,
            "marginal_true_matches_lost_on_removal": uniq_m,
            "marginal_recall_loss_pct": float(recall_loss * 100),
            "marginal_candidate_reduction": cand_reduction,
            "marginal_candidate_reduction_per_s1": float(cand_reduction / len(val_s1_ids)),
        }

        print(f"\nLayer {lyr} ({name}):")
        print(f"  Total candidates contributed: {tot_c:,}")
        print(f"  Candidates unique to layer: {uniq_c:,} ({uniq_c/total_full_cands*100:.3f}% of total)")
        print(f"  Overlap breakdown: 2 layers={ov2:,}, 3 layers={ov3:,}, 4 layers={ov4:,}")
        print(f"  Total true matches covered: {tot_m:,}")
        print(f"  TRUE MATCHES UNIQUE TO LAYER: {uniq_m:,} (lost if layer removed)")
        print(f"  Marginal recall loss if removed: -{recall_loss*100:.3f}%")
        print(f"  Marginal candidate reduction if removed: -{cand_reduction:,} (-{cand_reduction/5000:.2f}/S1)")

    # -------------------------------------------------------------
    # STEP 2 — SINGLE-LAYER ABLATIONS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: SINGLE-LAYER ABLATIONS OVER FROZEN COMBO 3")
    print("=" * 80, flush=True)

    # 8 configurations:
    # 1. C3 only
    # 2. C3 + A
    # 3. C3 + B
    # 4. C3 + C2-A
    # 5. C3 + A + B
    # 6. C3 + A + C2-A
    # 7. C3 + B + C2-A
    # 8. C3 + A + B + C2-A (Full stack)

    ablation_configs = [
        ("C3 only", False, False, False),
        ("C3 + A", True, False, False),
        ("C3 + B", False, True, False),
        ("C3 + C2-A", False, False, True),
        ("C3 + A + B", True, True, False),
        ("C3 + A + C2-A", True, False, True),
        ("C3 + B + C2-A", False, True, True),
        ("C3 + A + B + C2-A (Full)", True, True, True),
    ]

    ablation_results = []
    print(f"{'Configuration':<26} | {'True Matches':<14} | {'Recall':<8} | {'Candidates':<11} | {'Mean/S1':<8} | {'p90':<7} | {'p95':<7} | {'p99':<8} | {'Max':<7} | {'Cand Red vs Full':<16} | {'Rec Loss':<8}")
    print("-" * 145)

    for cfg_name, use_a, use_b, use_c2 in ablation_configs:
        cand_counts = []
        retrieved_tm = 0

        for s1_id in val_s1_ids:
            # Efficient disjoint evaluation:
            diff_set = set()
            if use_a: diff_set |= a_diff_cands[2000][s1_id]
            if use_b: diff_set |= b_diff_cands[100][s1_id]
            if use_c2: diff_set |= c2_diff_cands[500][s1_id]

            # Candidate count is |C3| + |diff_set|
            cnt = len(c3_cands[s1_id]) + len(diff_set)
            cand_counts.append(cnt)

            # True matches: c3_matches | (diff_set & gt)
            gt_s = gt_map[s1_id]
            tm_s = len(c3_matches[s1_id]) + len(diff_set & gt_s)
            retrieved_tm += tm_s

        stats = compute_distribution_stats(cand_counts)
        rec = retrieved_tm / total_val_positives
        cand_red = total_full_cands - stats["total"]
        cand_red_per_s1 = cand_red / len(val_s1_ids)
        rec_loss = total_full_tm - retrieved_tm
        rec_loss_pct = (total_full_tm - retrieved_tm) / total_val_positives * 100

        res_entry = {
            "configuration": cfg_name,
            "use_A": use_a,
            "use_B": use_b,
            "use_C2A": use_c2,
            "retrieved_true_matches": retrieved_tm,
            "retrieval_recall": float(rec),
            "total_candidates": stats["total"],
            "mean_candidates_per_s1": stats["mean"],
            "median_candidates": stats["median"],
            "p90_candidates": stats["p90"],
            "p95_candidates": stats["p95"],
            "p99_candidates": stats["p99"],
            "max_candidates": stats["max"],
            "candidate_reduction_vs_full": cand_red,
            "candidate_reduction_per_s1": float(cand_red_per_s1),
            "true_matches_lost_vs_full": rec_loss,
            "recall_loss_pct_vs_full": float(rec_loss_pct),
        }
        ablation_results.append(res_entry)

        print(f"{cfg_name:<26} | {retrieved_tm:>6,} / 17,314 | {rec*100:>6.2f}% | {stats['total']:>11,} | {stats['mean']:>8.2f} | "
              f"{stats['p90']:>7.1f} | {stats['p95']:>7.1f} | {stats['p99']:>8.1f} | {stats['max']:>7,} | "
              f"-{cand_red:>7,} (-{cand_red_per_s1:>5.2f}/S1) | -{rec_loss:>3} (-{rec_loss_pct:.2f}%)")

    # -------------------------------------------------------------
    # STEP 3 & 4 — COMBINED THRESHOLD SWEEP & PARETO SEARCH
    # A in {500, 1000, 1500, 2000} (4 values)
    # B in {25, 50, 75, 100, 150, 200} (6 values)
    # C2 in {250, 500, 750, 1000} (4 values)
    # Total = 4 * 6 * 4 = 96 combinations + ablations
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3 & 4: EVALUATING 96 COMBINED THRESHOLD CONFIGURATIONS")
    print("=" * 80, flush=True)

    grid_results = []
    combo_idx = 0

    for a_th, b_th, c2_th in product(a_thresholds, b_thresholds, c2_thresholds):
        combo_idx += 1
        cfg_name = f"A{a_th}_B{b_th}_C2A{c2_th}"
        cand_counts = []
        retrieved_tm = 0

        a_diff = a_diff_cands[a_th]
        b_diff = b_diff_cands[b_th]
        c2_diff = c2_diff_cands[c2_th]

        for s1_id in val_s1_ids:
            diff_set = a_diff[s1_id] | b_diff[s1_id] | c2_diff[s1_id]
            cnt = len(c3_cands[s1_id]) + len(diff_set)
            cand_counts.append(cnt)

            gt_s = gt_map[s1_id]
            tm_s = len(c3_matches[s1_id]) + len(diff_set & gt_s)
            retrieved_tm += tm_s

        stats = compute_distribution_stats(cand_counts)
        rec = retrieved_tm / total_val_positives
        cand_red = total_full_cands - stats["total"]
        cand_red_per_s1 = cand_red / len(val_s1_ids)
        rec_loss = total_full_tm - retrieved_tm
        rec_loss_pct = (total_full_tm - retrieved_tm) / total_val_positives * 100

        grid_entry = {
            "id": cfg_name,
            "threshold_A": a_th,
            "threshold_B": b_th,
            "threshold_C2A": c2_th,
            "retrieved_true_matches": retrieved_tm,
            "retrieval_recall": float(rec),
            "total_candidates": stats["total"],
            "mean_candidates_per_s1": stats["mean"],
            "median_candidates": stats["median"],
            "p90_candidates": stats["p90"],
            "p95_candidates": stats["p95"],
            "p99_candidates": stats["p99"],
            "max_candidates": stats["max"],
            "candidate_reduction_vs_full": cand_red,
            "candidate_reduction_per_s1": float(cand_red_per_s1),
            "true_matches_lost_vs_full": rec_loss,
            "recall_loss_pct_vs_full": float(rec_loss_pct),
        }
        grid_results.append(grid_entry)

    print(f"Evaluated all {len(grid_results)} combinations.", flush=True)

    # -------------------------------------------------------------
    # STEP 5 — IDENTIFY PARETO FRONTIER
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: IDENTIFYING NON-DOMINATED PARETO FRONTIER")
    print("=" * 80, flush=True)

    # Include C3 only, ablations, and all grid results
    all_evaluated = list(ablation_results) + grid_results

    # Remove duplicates if any (e.g. A2000_B100_C2A500 is identical to C3+A+B+C2-A Full)
    unique_cfgs = {}
    for r in all_evaluated:
        key = (r["retrieved_true_matches"], r["total_candidates"])
        if key not in unique_cfgs:
            unique_cfgs[key] = r

    cfgs = list(unique_cfgs.values())

    pareto_frontier = []
    for candidate in cfgs:
        rec_c = candidate["retrieval_recall"]
        mean_c = candidate["mean_candidates_per_s1"]
        is_dominated = False
        for other in cfgs:
            rec_o = other["retrieval_recall"]
            mean_o = other["mean_candidates_per_s1"]
            # other dominates candidate if other.recall >= cand.recall AND other.mean <= cand.mean (with at least one strict)
            if (rec_o >= rec_c and mean_o <= mean_c) and (rec_o > rec_c or mean_o < mean_c):
                is_dominated = True
                break
        if not is_dominated:
            pareto_frontier.append(candidate)

    # Sort Pareto frontier by recall ascending
    pareto_frontier.sort(key=lambda x: (x["retrieval_recall"], x["mean_candidates_per_s1"]))

    print(f"Found {len(pareto_frontier)} non-dominated Pareto configurations:")
    print(f"{'Config ID':<26} | {'True Matches':<14} | {'Recall':<8} | {'Candidates':<11} | {'Mean/S1':<8} | {'p90':<7} | {'p95':<7} | {'p99':<8} | {'Max':<7} | {'Cand Red vs Full':<16} | {'Rec Loss':<8}")
    print("-" * 145)
    for p in pareto_frontier:
        name = p.get("id") or p.get("configuration")
        cand_red = p["candidate_reduction_vs_full"]
        cand_red_per_s1 = p["candidate_reduction_per_s1"]
        rec_loss = p["true_matches_lost_vs_full"]
        rec_loss_pct = p["recall_loss_pct_vs_full"]
        print(f"{name:<26} | {p['retrieved_true_matches']:>6,} / 17,314 | {p['retrieval_recall']*100:>6.2f}% | "
              f"{p['total_candidates']:>11,} | {p['mean_candidates_per_s1']:>8.2f} | "
              f"{p['p90_candidates']:>7.1f} | {p['p95_candidates']:>7.1f} | {p['p99_candidates']:>8.1f} | {p['max_candidates']:>7,} | "
              f"-{cand_red:>7,} (-{cand_red_per_s1:>5.2f}/S1) | -{rec_loss:>3} (-{rec_loss_pct:.2f}%)")

    # Specific Target Configurations:
    # A. Recall >= 94.50%
    # B. Recall >= 94.25%
    # C. Recall >= 94.00%
    target_thresholds = [
        ("A: Recall >= 94.50%", 0.9450),
        ("B: Recall >= 94.25%", 0.9425),
        ("C: Recall >= 94.00%", 0.9400),
    ]

    target_configs = {}
    print("\nLowest-Candidate Configurations Satisfying Recall Constraints:")
    for label, min_rec in target_thresholds:
        valid = [c for c in all_evaluated if c["retrieval_recall"] >= min_rec]
        valid.sort(key=lambda x: (x["mean_candidates_per_s1"], -x["retrieval_recall"]))
        best = valid[0]
        name = best.get("id") or best.get("configuration")
        target_configs[label] = best
        print(f"  {label:<22}: {name:<20} | Matches={best['retrieved_true_matches']:,} ({best['retrieval_recall']*100:.2f}%) | "
              f"Mean={best['mean_candidates_per_s1']:.2f} / S1 (Total={best['total_candidates']:,}) | "
              f"Cand Reduction vs Full: -{best['candidate_reduction_vs_full']:,} (-{best['candidate_reduction_per_s1']:.2f}/S1) | "
              f"Lost Matches: {best['true_matches_lost_vs_full']}")

    # -------------------------------------------------------------
    # STEP 6 — TAIL EXPLOSION ANALYSIS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: TAIL EXPLOSION RISK ANALYSIS")
    print("=" * 80, flush=True)

    base_p90 = full_stack_stats["p90"]
    base_p95 = full_stack_stats["p95"]
    base_p99 = full_stack_stats["p99"]
    base_max = full_stack_stats["max"]

    print(f"Baseline Tail Benchmarks: p90={base_p90:.1f}, p95={base_p95:.1f}, p99={base_p99:.1f}, Max={base_max:,}")

    tail_inspections = []
    for p in pareto_frontier:
        name = p.get("id") or p.get("configuration")
        dp90 = p["p90_candidates"] - base_p90
        dp95 = p["p95_candidates"] - base_p95
        dp99 = p["p99_candidates"] - base_p99
        dmax = p["max_candidates"] - base_max
        tail_entry = {
            "name": name,
            "p90": p["p90_candidates"],
            "p95": p["p95_candidates"],
            "p99": p["p99_candidates"],
            "max": p["max_candidates"],
            "delta_p90": float(dp90),
            "delta_p95": float(dp95),
            "delta_p99": float(dp99),
            "delta_max": int(dmax),
            "has_tail_explosion": (dp95 > 10.0 or dp99 > 10.0 or dmax > 0),
        }
        tail_inspections.append(tail_entry)
        flag = " [!] FLAGGED" if tail_entry["has_tail_explosion"] else " [OK]"
        print(f"  {name:<26}: p90={p['p90_candidates']:.1f} (dp={dp90:+.1f}), p95={p['p95_candidates']:.1f} (dp={dp95:+.1f}), "
              f"p99={p['p99_candidates']:.1f} (dp={dp99:+.1f}), max={p['max_candidates']:,} (dm={dmax:+d}){flag}")

    # -------------------------------------------------------------
    # STEP 7 — MATCH RECOVERY / LOSS ANALYSIS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7: FAILURE CATEGORIZATION OF LOST MATCHES")
    print("=" * 80, flush=True)

    # For the lowest-candidate config meeting 94.50% (and C3+A+C2-A or C3+B+C2-A)
    # Determine which true matches were lost compared to the full 16,380 stack
    # Categorize by:
    # - cross-script / synthetic
    # - spelling / OCR / fuzzy
    # - single-token
    # - address evidence
    # - name/address divergence
    # - other

    # Focus on the 94.50% target config and C3+B+C2-A (which drops A)
    selected_for_loss_analysis = [
        target_configs["A: Recall >= 94.50%"],
        next(c for c in ablation_results if c["configuration"] == "C3 + B + C2-A"),
        next(c for c in ablation_results if c["configuration"] == "C3 + A + C2-A"),
        next(c for c in ablation_results if c["configuration"] == "C3 + C2-A"),
    ]

    loss_analysis_results = {}

    for cfg in selected_for_loss_analysis:
        name = cfg.get("id") or cfg.get("configuration")
        # Compute exact retrieved matches for this config
        a_th = cfg.get("threshold_A")
        b_th = cfg.get("threshold_B")
        c2_th = cfg.get("threshold_C2A")

        # Handle ablation configs
        if a_th is None:
            a_th = 2000 if cfg["use_A"] else None
            b_th = 100 if cfg["use_B"] else None
            c2_th = 500 if cfg["use_C2A"] else None

        cfg_matches = set()
        for s1_id in val_s1_ids:
            diff_set = set()
            if a_th: diff_set |= a_diff_cands[a_th][s1_id]
            if b_th: diff_set |= b_diff_cands[b_th][s1_id]
            if c2_th: diff_set |= c2_diff_cands[c2_th][s1_id]

            for m in (c3_matches[s1_id] | (diff_set & gt_map[s1_id])):
                cfg_matches.add((s1_id, m))

        lost_matches = full_stack_matches - cfg_matches
        print(f"\nAnalyzing {len(lost_matches)} lost true match(es) in {name}:")

        lost_by_cat = Counter()
        lost_details = []

        for s1_id, mid in lost_matches:
            s1_info = s1_parsed[s1_id]
            m_rec = true_match_records.get(mid)
            if not m_rec:
                lost_by_cat["other"] += 1
                continue

            s1_n = s1_info["name_norm"]
            cand_n = normalize_basic(m_rec["business_name"])
            s1_a = normalize_basic(s1_info["raw_addr"])
            cand_a = normalize_basic(m_rec["business_address"])

            r_ratio = fuzz.ratio(s1_n, cand_n)
            toks_s1 = tokenize(s1_n)
            toks_cand = tokenize(cand_n)
            shared_exact_n = set(toks_s1) & set(toks_cand)
            shared_info_n = set(s1_info["info_name"]) & set([t for t in toks_cand if len(t) >= 3 and t not in NAME_STOPWORDS])
            shared_info_a = set(s1_info["info_addr"]) & set([t for t in tokenize(cand_a) if len(t) >= 3 and t not in ADDR_STOPWORDS])

            is_cross = is_synthetic_or_cross_script(s1_info["raw_name"]) or is_synthetic_or_cross_script(m_rec["business_name"])

            if is_cross:
                cat = "cross-script / synthetic"
            elif len(shared_info_n) == 1 and r_ratio < 70:
                cat = "single-token"
            elif r_ratio >= 70 or fuzz.token_sort_ratio(s1_n, cand_n) >= 75:
                cat = "spelling / OCR / fuzzy"
            elif len(shared_info_a) >= 2 and len(shared_exact_n) == 0:
                cat = "address evidence"
            elif r_ratio < 50 and len(shared_info_a) <= 1:
                cat = "name/address divergence"
            else:
                cat = "other"

            lost_by_cat[cat] += 1
            if len(lost_details) < 5:
                lost_details.append({
                    "s1_id": s1_id,
                    "s1_name": s1_info["raw_name"],
                    "matched_id": mid,
                    "matched_name": m_rec["business_name"],
                    "category": cat,
                    "sim_ratio": float(r_ratio),
                    "shared_name_tokens": list(shared_info_n),
                    "shared_addr_tokens": list(shared_info_a),
                })

        for cat, cnt in lost_by_cat.most_common():
            print(f"  - {cat:<26}: {cnt:>3} lost ({cnt/len(lost_matches)*100:.1f}%)")

        loss_analysis_results[name] = {
            "total_lost_matches": len(lost_matches),
            "lost_by_category": dict(lost_by_cat),
            "sample_lost_matches": lost_details,
        }

    # -------------------------------------------------------------
    # SAVE JSON RESULTS
    # -------------------------------------------------------------
    final_output = {
        "metadata": {
            "validation_sample": len(val_s1_ids),
            "seed": 42,
            "total_val_positives": total_val_positives,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - start_time, 2),
        },
        "full_stack_baseline": {
            "name": "Combo 3 + Secondary A + Secondary B + C2-A",
            "retrieved_positives": total_full_tm,
            "recall": float(total_full_tm / total_val_positives),
            "candidate_stats": full_stack_stats,
        },
        "step1_marginal_contributions": step1_table,
        "step2_ablations": ablation_results,
        "step3_and_4_threshold_grid": grid_results,
        "step5_pareto_frontier": pareto_frontier,
        "step5_target_configs": target_configs,
        "step6_tail_analysis": tail_inspections,
        "step7_lost_match_analysis": loss_analysis_results,
    }

    out_file = Path("output/retrieval_pruning_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nMachine-readable results successfully written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
