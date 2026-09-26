"""
Constrained Fuzzy / OCR Retrieval Experiments for Business Entity Resolution.

Evaluates targeted fuzzy name retrieval strategies on top of the current baseline:
Current Baseline: Combo 3 + Secondary A + Secondary B + C2-A
- Baseline Recall: 16,380 / 17,314 (94.61%)
- Baseline Candidates: 8,574,400 (mean 1,714.88 / S1)

Investigates recovering missed true matches in the name spelling / OCR / fuzzy distortion category:
- Characterizes remaining fuzzy misses (distributions of RapidFuzz ratio, token_sort_ratio,
  token_set_ratio, partial_ratio, name lengths, token counts, shared tokens, shared n-grams).
- Evaluates Variants F1, F2, F3, F4 across threshold sweep: 80, 85, 90, 95.
- Computes complete candidate distributions (mean, median, p90, p95, p99, max), efficiency,
  and error analysis.
- Saves machine-readable results to output/fuzzy_retrieval_experiments_results.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    print("EXPERIMENT: CONSTRAINED FUZZY / OCR RETRIEVAL")
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
        pins = extract_pincodes(addr)

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
            "pincodes": pins,
        }

    print("Computing Current Baseline retrieval (Combo 3 + Secondary A + Secondary B + C2-A)...", flush=True)
    baseline_cands: dict[str, set[str]] = {}
    baseline_retrieved_pairs: set[tuple[str, str]] = set()

    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]

        # 1. Combo 3
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

        # 2. Secondary A: 2 rare addr tokens, DF <= 2000
        rare_addrs_2000 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
        if len(rare_addrs_2000) >= 2:
            c_set |= get_token_overlap_candidates(rare_addrs_2000, min_overlap=2)

        # 3. Secondary B: single name token len >= 4, DF <= 100
        for t in p["info_name"]:
            if len(t) >= 4 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 100:
                c_set |= idx_name_tokens[(c, t)]

        # 4. C2-A: country + shared bldg num + at least 1 addr token DF <= 500
        b_nums = [n for n in p["bldg_nums"] if (c, n) in idx_addr_numbers]
        a_toks_500 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
        if b_nums and a_toks_500:
            a_u = set().union(*a_toks_500)
            for n in b_nums:
                c_set |= (idx_addr_numbers[(c, n)] & a_u)

        baseline_cands[s1_id] = c_set
        for m in set(val_gt[s1_id]) & c_set:
            baseline_retrieved_pairs.add((s1_id, m))

    base_candidate_counts = [len(baseline_cands[s]) for s in val_s1_ids]
    base_stats = compute_distribution_stats(base_candidate_counts)
    print(f"Validated Current Baseline Retrieved: {len(baseline_retrieved_pairs):,} / {total_val_positives:,} "
          f"({len(baseline_retrieved_pairs)/total_val_positives*100:.2f}%)", flush=True)
    print(f"Validated Current Baseline Candidates: {base_stats['total']:,} (mean {base_stats['mean']:.2f}, "
          f"median {base_stats['median']:.1f}, p90 {base_stats['p90']:.1f}, p95 {base_stats['p95']:.1f}, "
          f"p99 {base_stats['p99']:.1f}, max {base_stats['max']:,})", flush=True)

    # -------------------------------------------------------------
    # STEP 1 — CHARACTERIZE REMAINING FUZZY MISSES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: CHARACTERIZING REMAINING FUZZY / OCR MISSES")
    print("=" * 80, flush=True)

    remaining_misses = []
    for s1_id in val_s1_ids:
        missed_for_s1 = set(val_gt[s1_id]) - baseline_cands[s1_id]
        for mid in missed_for_s1:
            s1_info = s1_parsed[s1_id]
            m_rec = true_match_records.get(mid)
            if not m_rec:
                continue
            s1_n = s1_info["name_norm"]
            cand_n = normalize_basic(m_rec["business_name"])
            s1_a = normalize_basic(s1_info["raw_addr"])
            cand_a = normalize_basic(m_rec["business_address"])

            r_ratio = fuzz.ratio(s1_n, cand_n)
            r_tsort = fuzz.token_sort_ratio(s1_n, cand_n)
            r_tset = fuzz.token_set_ratio(s1_n, cand_n)
            r_part = fuzz.partial_ratio(s1_n, cand_n)

            toks_s1 = tokenize(s1_n)
            toks_cand = tokenize(cand_n)
            shared_exact_tokens = set(toks_s1) & set(toks_cand)
            shared_info_name = set(s1_info["info_name"]) & set([t for t in toks_cand if len(t) >= 3 and t not in NAME_STOPWORDS])

            shared_addr_tokens = set(s1_info["info_addr"]) & set([t for t in tokenize(cand_a) if len(t) >= 3 and t not in ADDR_STOPWORDS])
            m_nums = extract_address_numbers(m_rec["business_address"])
            shared_nums = set(s1_info["all_nums"]) & set(m_nums)

            s1_c = compact(s1_n)
            cand_c = compact(cand_n)
            ng3_s1 = {s1_c[i:i+3] for i in range(len(s1_c)-2)} if len(s1_c) >= 3 else set()
            ng3_cand = {cand_c[i:i+3] for i in range(len(cand_c)-2)} if len(cand_c) >= 3 else set()
            shared_3grams = ng3_s1 & ng3_cand

            ng4_s1 = {s1_c[i:i+4] for i in range(len(s1_c)-3)} if len(s1_c) >= 4 else set()
            ng4_cand = {cand_c[i:i+4] for i in range(len(cand_c)-3)} if len(cand_c) >= 4 else set()
            shared_4grams = ng4_s1 & ng4_cand

            is_cross = is_synthetic_or_cross_script(s1_info["raw_name"]) or is_synthetic_or_cross_script(m_rec["business_name"])

            remaining_misses.append({
                "s1_id": s1_id,
                "mid": mid,
                "s1_name": s1_info["raw_name"],
                "cand_name": m_rec["business_name"],
                "s1_addr": s1_info["raw_addr"],
                "cand_addr": m_rec["business_address"],
                "country": s1_info["country"],
                "r_ratio": float(r_ratio),
                "r_tsort": float(r_tsort),
                "r_tset": float(r_tset),
                "r_part": float(r_part),
                "s1_len": len(s1_n),
                "cand_len": len(cand_n),
                "s1_tok_cnt": len(toks_s1),
                "cand_tok_cnt": len(toks_cand),
                "shared_exact_tokens": len(shared_exact_tokens),
                "shared_info_name": len(shared_info_name),
                "shared_addr_tokens": len(shared_addr_tokens),
                "shared_nums": len(shared_nums),
                "shared_3grams": len(shared_3grams),
                "shared_4grams": len(shared_4grams),
                "is_cross": is_cross,
            })

    total_remaining = len(remaining_misses)
    cs_misses = [m for m in remaining_misses if m["is_cross"]]
    non_cs = [m for m in remaining_misses if not m["is_cross"]]
    fuzzy_misses = [m for m in non_cs if m["r_ratio"] >= 70 or m["r_tsort"] >= 75 or m["r_tset"] >= 75]
    other_misses = [m for m in non_cs if m not in fuzzy_misses]

    print(f"Total Remaining Missed Matches: {total_remaining}")
    print(f"  - Cross-script / Synthetic obfuscation: {len(cs_misses)}")
    print(f"  - Name spelling / OCR / Fuzzy distortion: {len(fuzzy_misses)}")
    print(f"  - Other / Severe divergence: {len(other_misses)}")

    ratios = [m["r_ratio"] for m in fuzzy_misses]
    tsorts = [m["r_tsort"] for m in fuzzy_misses]
    tsets = [m["r_tset"] for m in fuzzy_misses]
    parts = [m["r_part"] for m in fuzzy_misses]
    s1_lens = [m["s1_len"] for m in fuzzy_misses]
    cand_lens = [m["cand_len"] for m in fuzzy_misses]
    tok_counts_s1 = [m["s1_tok_cnt"] for m in fuzzy_misses]
    tok_counts_cand = [m["cand_tok_cnt"] for m in fuzzy_misses]

    step1_char = {
        "total_remaining_misses": total_remaining,
        "cross_script_misses": len(cs_misses),
        "fuzzy_ocr_misses": len(fuzzy_misses),
        "other_misses": len(other_misses),
        "similarity_distributions": {
            "rapidfuzz_ratio": {
                "mean": float(np.mean(ratios)),
                "median": float(np.median(ratios)),
                "min": float(np.min(ratios)),
                "max": float(np.max(ratios)),
                "counts_ge_95": int(sum(1 for x in ratios if x >= 95)),
                "counts_ge_90": int(sum(1 for x in ratios if x >= 90)),
                "counts_ge_85": int(sum(1 for x in ratios if x >= 85)),
                "counts_ge_80": int(sum(1 for x in ratios if x >= 80)),
                "counts_ge_75": int(sum(1 for x in ratios if x >= 75)),
            },
            "token_sort_ratio": {
                "mean": float(np.mean(tsorts)),
                "median": float(np.median(tsorts)),
                "min": float(np.min(tsorts)),
                "max": float(np.max(tsorts)),
                "counts_ge_95": int(sum(1 for x in tsorts if x >= 95)),
                "counts_ge_90": int(sum(1 for x in tsorts if x >= 90)),
                "counts_ge_85": int(sum(1 for x in tsorts if x >= 85)),
                "counts_ge_80": int(sum(1 for x in tsorts if x >= 80)),
                "counts_ge_75": int(sum(1 for x in tsorts if x >= 75)),
            },
            "token_set_ratio": {
                "mean": float(np.mean(tsets)),
                "median": float(np.median(tsets)),
                "min": float(np.min(tsets)),
                "max": float(np.max(tsets)),
                "counts_ge_95": int(sum(1 for x in tsets if x >= 95)),
                "counts_ge_90": int(sum(1 for x in tsets if x >= 90)),
                "counts_ge_85": int(sum(1 for x in tsets if x >= 85)),
                "counts_ge_80": int(sum(1 for x in tsets if x >= 80)),
                "counts_ge_75": int(sum(1 for x in tsets if x >= 75)),
            },
            "partial_ratio": {
                "mean": float(np.mean(parts)),
                "median": float(np.median(parts)),
                "min": float(np.min(parts)),
                "max": float(np.max(parts)),
                "counts_ge_95": int(sum(1 for x in parts if x >= 95)),
                "counts_ge_90": int(sum(1 for x in parts if x >= 90)),
                "counts_ge_85": int(sum(1 for x in parts if x >= 85)),
                "counts_ge_80": int(sum(1 for x in parts if x >= 80)),
                "counts_ge_75": int(sum(1 for x in parts if x >= 75)),
            },
        },
        "name_lengths": {
            "s1_length": {"mean": float(np.mean(s1_lens)), "median": float(np.median(s1_lens)), "min": int(np.min(s1_lens)), "max": int(np.max(s1_lens))},
            "cand_length": {"mean": float(np.mean(cand_lens)), "median": float(np.median(cand_lens)), "min": int(np.min(cand_lens)), "max": int(np.max(cand_lens))},
            "s1_token_count": {"mean": float(np.mean(tok_counts_s1)), "median": float(np.median(tok_counts_s1))},
            "cand_token_count": {"mean": float(np.mean(tok_counts_cand)), "median": float(np.median(tok_counts_cand))},
        },
        "token_and_ngram_overlap_counts": {
            "shared_exact_tokens": {str(k): v for k, v in sorted(Counter(m["shared_exact_tokens"] for m in fuzzy_misses).items())},
            "shared_info_name_tokens": {str(k): v for k, v in sorted(Counter(m["shared_info_name"] for m in fuzzy_misses).items())},
            "shared_address_tokens": {str(k): v for k, v in sorted(Counter(m["shared_addr_tokens"] for m in fuzzy_misses).items())},
            "shared_address_numbers": {str(k): v for k, v in sorted(Counter(m["shared_nums"] for m in fuzzy_misses).items())},
            "shared_4grams": {str(k): v for k, v in sorted(Counter(min(m["shared_4grams"], 10) for m in fuzzy_misses).items())},
        },
    }

    print("\nEmpirical Metrics for Remaining Fuzzy Misses:")
    for metric, vals in step1_char["similarity_distributions"].items():
        print(f"  {metric:<18}: Mean={vals['mean']:.1f}, Median={vals['median']:.1f}, "
              f">=95: {vals['counts_ge_95']}, >=90: {vals['counts_ge_90']}, >=85: {vals['counts_ge_85']}, >=80: {vals['counts_ge_80']}")

    # -------------------------------------------------------------
    # STEP 2 & 3 — CONSTRAINED RETRIEVAL STRATEGIES & THRESHOLD SWEEP
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2 & 3: FORMING CONSTRAINED BLOCKS & LOADING CANDIDATE NAMES")
    print("=" * 80, flush=True)

    # Candidate blocks per S1 query
    # Variant F1: Same country + 1 exact informative name token (DF <= 500)
    # Variant F2: Same country + shared rare compact prefix5 (DF <= 200) constrained by length
    # Variant F3: Same country + 1 informative name token (DF <= 1,000) & 1 addr token (DF <= 2,000)
    # Variant F4: Same country + shared bldg num & 1 addr token (DF <= 1,000)

    f1_blocks: dict[str, set[str]] = defaultdict(set)
    f2_blocks: dict[str, set[str]] = defaultdict(set)
    f3_blocks: dict[str, set[str]] = defaultdict(set)
    f4_blocks: dict[str, set[str]] = defaultdict(set)

    all_needed_cand_eids: set[str] = set()

    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]

        # F1
        for t in p["info_name"]:
            if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500:
                f1_blocks[s1_id] |= idx_name_tokens[(c, t)]

        # F2
        p5 = p["prefix5"]
        if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= 200:
            f2_blocks[s1_id] |= idx_compact_prefix5[(c, p5)]

        # F3
        n_sets_1000 = [idx_name_tokens[(c, t)] for t in p["info_name"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 1000]
        a_sets_2000 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
        if n_sets_1000 and a_sets_2000:
            f3_blocks[s1_id] |= (set().union(*n_sets_1000) & set().union(*a_sets_2000))

        # F4
        b_nums = [n for n in p["bldg_nums"] if (c, n) in idx_addr_numbers]
        a_sets_1000 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
        if b_nums and a_sets_1000:
            a_u = set().union(*a_sets_1000)
            for n in b_nums:
                f4_blocks[s1_id] |= (idx_addr_numbers[(c, n)] & a_u)

        all_needed_cand_eids |= f1_blocks[s1_id]
        all_needed_cand_eids |= f2_blocks[s1_id]
        all_needed_cand_eids |= f3_blocks[s1_id]
        all_needed_cand_eids |= f4_blocks[s1_id]

    print(f"Total Unique Candidate Records Needed Across All Blocks: {len(all_needed_cand_eids):,}", flush=True)

    # Load candidate business names from train_source2 and train_source3
    print("Loading candidate records (names and addresses) from Source 2 and Source 3...", flush=True)
    cand_records: dict[str, dict[str, str]] = {}

    for src_path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        t0 = time.time()
        print(f"  Scanning {src_path}...", flush=True)
        for chunk in pd.read_csv(src_path, sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
            mask = chunk["entity_id"].isin(all_needed_cand_eids)
            if mask.any():
                for eid, name, addr in zip(
                    chunk.loc[mask, "entity_id"],
                    chunk.loc[mask, "business_name"],
                    chunk.loc[mask, "business_address"],
                ):
                    cand_records[eid] = {
                        "name_norm": normalize_basic(name),
                        "raw_name": name,
                        "raw_addr": addr,
                    }
        print(f"    Loaded {len(cand_records):,} total matched records in {time.time()-t0:.1f}s", flush=True)

    fuzzy_miss_set = {(m["s1_id"], m["mid"]) for m in fuzzy_misses}
    thresholds = [80, 85, 90, 95]
    variants = [
        ("F1", "Country + 1 Info Name Token (DF<=500) + Fuzzy Name Sim", f1_blocks),
        ("F2", "Country + Rare Prefix5 (DF<=200) + Length Ratio >=0.7 + Fuzzy Name Sim", f2_blocks),
        ("F3", "Country + 1 Name Token (DF<=1000) & 1 Addr Token (DF<=2000) + Fuzzy Name Sim", f3_blocks),
        ("F4", "Country + Shared Bldg Num & 1 Addr Token (DF<=1000) + Fuzzy Name Sim", f4_blocks),
    ]

    experiment_results: dict[str, Any] = {}

    for var_id, var_desc, blocks in variants:
        print("\n" + "-" * 70)
        print(f"EVALUATING VARIANT {var_id}: {var_desc}")
        print("-" * 70, flush=True)

        experiment_results[var_id] = {
            "description": var_desc,
            "thresholds": {},
        }

        # Precompute candidate similarity scores for all candidate pairs in blocks
        s1_cand_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for s1_id in val_s1_ids:
            s1_nn = s1_parsed[s1_id]["name_norm"]
            s1_len = len(s1_nn)
            for eid in blocks[s1_id]:
                cand = cand_records.get(eid)
                if not cand:
                    continue
                cand_nn = cand["name_norm"]
                cand_len = len(cand_nn)

                # Variant F2 has length constraint
                if var_id == "F2":
                    if s1_len > 0 and cand_len > 0:
                        ratio = min(s1_len, cand_len) / max(s1_len, cand_len)
                        if ratio < 0.70:
                            continue

                sim = fuzz.ratio(s1_nn, cand_nn)
                if sim >= 80:  # store if reaches min sweep threshold
                    s1_cand_scores[s1_id].append((eid, sim))

        for th in thresholds:
            t0 = time.time()
            combined_retrieved_pairs: set[tuple[str, str]] = set(baseline_retrieved_pairs)
            newly_recovered_pairs: set[tuple[str, str]] = set()
            fuzzy_recovered_pairs: set[tuple[str, str]] = set()

            total_candidate_counts: list[int] = []
            added_candidate_counts: list[int] = []

            for s1_id in val_s1_ids:
                base_c = baseline_cands[s1_id]
                var_c = {eid for eid, sim in s1_cand_scores[s1_id] if sim >= th}
                union_c = base_c | var_c
                total_candidate_counts.append(len(union_c))
                added_candidate_counts.append(len(var_c - base_c))

                gt_set = set(val_gt[s1_id])
                for m in (var_c - base_c) & gt_set:
                    newly_recovered_pairs.add((s1_id, m))
                    if (s1_id, m) in fuzzy_miss_set:
                        fuzzy_recovered_pairs.add((s1_id, m))
                for m in union_c & gt_set:
                    combined_retrieved_pairs.add((s1_id, m))

            cand_stats = compute_distribution_stats(total_candidate_counts)
            added_stats = compute_distribution_stats(added_candidate_counts)
            total_retrieved = len(combined_retrieved_pairs)
            recall = total_retrieved / total_val_positives
            new_rec = len(newly_recovered_pairs)
            fuzz_rec = len(fuzzy_recovered_pairs)
            added_cands = added_stats["total"]
            added_per_s1 = added_stats["mean"]
            efficiency = (new_rec / added_cands) if added_cands > 0 else 0.0
            recall_gain_per_1k = (efficiency * 1000) / total_val_positives if added_cands > 0 else 0.0

            th_summary = {
                "threshold": th,
                "total_true_matches": total_retrieved,
                "retrieval_recall": float(recall),
                "newly_recovered_vs_baseline": new_rec,
                "fuzzy_category_recovered": fuzz_rec,
                "total_candidates": cand_stats["total"],
                "added_candidates_total": added_cands,
                "added_candidates_per_s1": float(added_per_s1),
                "mean_candidates_per_s1": cand_stats["mean"],
                "median_candidates": cand_stats["median"],
                "p90_candidates": cand_stats["p90"],
                "p95_candidates": cand_stats["p95"],
                "p99_candidates": cand_stats["p99"],
                "max_candidates": cand_stats["max"],
                "efficiency_positives_per_added_candidate": float(efficiency),
                "recall_gain_per_1000_added_cands_per_s1": float(recall_gain_per_1k),
            }
            experiment_results[var_id]["thresholds"][str(th)] = th_summary

            print(f"  Thresh >={th:<2}: Total={total_retrieved:,} ({recall*100:.2f}%) | "
                  f"New=+{new_rec:<3} (Fuzzy: +{fuzz_rec:<3}) | "
                  f"Added Cands=+{added_cands:<7,} (+{added_per_s1:.2f}/S1) | "
                  f"p95={cand_stats['p95']:.0f} p99={cand_stats['p99']:.0f} max={cand_stats['max']:,} | "
                  f"Eff={efficiency:.6f}", flush=True)

    # -------------------------------------------------------------
    # STEP 4 — ERROR ANALYSIS & EXAMPLES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: ERROR ANALYSIS & DETAILED EXAMPLES")
    print("=" * 80, flush=True)

    # Collect 10 recovered true matches from the best performing configuration (e.g. F3 @ 80 or F1 @ 85)
    recovered_examples: list[dict[str, Any]] = []
    # Identify true matches recovered by F1/F3
    for s1_id in val_s1_ids:
        gt_set = set(val_gt[s1_id])
        p = s1_parsed[s1_id]
        c = p["country"]
        s1_nn = p["name_norm"]
        for mid in gt_set - baseline_cands[s1_id]:
            m_rec = true_match_records.get(mid)
            if not m_rec:
                continue
            cand_nn = normalize_basic(m_rec["business_name"])
            sim = fuzz.ratio(s1_nn, cand_nn)
            tsort = fuzz.token_sort_ratio(s1_nn, cand_nn)
            if sim >= 80 or tsort >= 80:
                # Determine blocking evidence
                shared_n = set(p["info_name"]) & set(tokenize(cand_nn))
                shared_a = set(p["info_addr"]) & set(tokenize(m_rec["business_address"]))
                cand_nums = extract_address_numbers(m_rec["business_address"])
                shared_nums = p["all_nums"] & cand_nums

                distortion_type = "Spelling / Typo"
                if any(c in "0123456789" for c in s1_nn) != any(c in "0123456789" for c in cand_nn):
                    distortion_type = "Digit / Letter Substitution"
                elif compact(s1_nn) == compact(cand_nn):
                    distortion_type = "Token Concatenation / Split"
                elif fuzz.token_sort_ratio(s1_nn, cand_nn) >= 95:
                    distortion_type = "Word Order / Permutation"
                elif fuzz.partial_ratio(s1_nn, cand_nn) >= 95:
                    distortion_type = "Trunaction / Substring"
                elif sim >= 85:
                    distortion_type = "OCR / Character Substitution"

                recovered_examples.append({
                    "s1_id": s1_id,
                    "s1_name": p["raw_name"],
                    "matched_id": mid,
                    "matched_name": m_rec["business_name"],
                    "s1_address": p["raw_addr"],
                    "matched_address": m_rec["business_address"],
                    "fuzz_ratio": float(sim),
                    "fuzz_token_sort": float(tsort),
                    "shared_name_tokens": list(shared_n),
                    "shared_addr_tokens": list(shared_a),
                    "shared_nums": list(shared_nums),
                    "distortion_mechanism": distortion_type,
                })
                if len(recovered_examples) >= 10:
                    break
        if len(recovered_examples) >= 10:
            break

    # Collect 10 large-candidate / false-positive-style examples
    # (Candidates that passed fuzzy threshold >= 80 but are NOT in ground truth)
    fp_examples: list[dict[str, Any]] = []
    for s1_id in val_s1_ids:
        gt_set = set(val_gt[s1_id])
        p = s1_parsed[s1_id]
        s1_nn = p["name_norm"]
        for eid in f1_blocks[s1_id]:
            if eid in gt_set or eid in baseline_cands[s1_id]:
                continue
            cand = cand_records.get(eid)
            if not cand:
                continue
            cand_nn = cand["name_norm"]
            sim = fuzz.ratio(s1_nn, cand_nn)
            if sim >= 80:
                shared_n = set(p["info_name"]) & set(tokenize(cand_nn))
                fp_examples.append({
                    "s1_id": s1_id,
                    "s1_name": p["raw_name"],
                    "candidate_id": eid,
                    "candidate_name": cand["raw_name"],
                    "s1_address": p["raw_addr"],
                    "candidate_address": cand["raw_addr"],
                    "fuzz_ratio": float(sim),
                    "shared_blocking_tokens": list(shared_n),
                    "why_introduced": f"Shared token(s) {list(shared_n)} with fuzzy ratio {sim:.1f}",
                })
                if len(fp_examples) >= 10:
                    break
        if len(fp_examples) >= 10:
            break

    # Construct final results dictionary
    final_results = {
        "metadata": {
            "validation_sample": len(val_s1_ids),
            "seed": 42,
            "total_val_positives": total_val_positives,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - start_time, 2),
        },
        "baseline": {
            "name": "Combo 3 + Secondary A + Secondary B + C2-A",
            "retrieved_positives": len(baseline_retrieved_pairs),
            "recall": float(len(baseline_retrieved_pairs) / total_val_positives),
            "candidate_stats": base_stats,
        },
        "step1_characterization": step1_char,
        "step2_and_3_experiments": experiment_results,
        "step4_error_analysis": {
            "recovered_true_matches": recovered_examples,
            "false_positive_examples": fp_examples,
        },
    }

    out_file = Path("output/fuzzy_retrieval_experiments_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nMachine-readable results saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
