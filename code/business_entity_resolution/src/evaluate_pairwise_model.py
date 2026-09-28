"""
Pairwise Matching Model Evaluation for Business Entity Resolution.

Retrieval: Combo 3 (Retrieval Baseline B) frozen.
Split: Deterministic 5,000 S1 validation entities (seed=42),
       80/20 S1-level split (4,000 train S1, 1,000 test S1).
Model: LightGBM Binary Classifier on candidate pairs.
Evaluation: Macro S1-level F0.5 (competition metric), pairwise metrics,
            feature importance, threshold analysis, rule-based comparison,
            and error analysis.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import (
    ADDR_STOPWORDS,
    NAME_STOPWORDS,
)
from evaluation import f_beta_per_s1
from preprocessing import (
    compact,
    normalize_basic,
    sorted_tokens,
    tokenize,
)

FEATURE_NAMES = [
    # NAME features (15)
    "name_exact",
    "name_compact_exact",
    "name_sorted_exact",
    "name_char_len_ratio",
    "name_token_count_diff",
    "name_token_jaccard",
    "name_token_overlap",
    "name_fuzz_ratio",
    "name_fuzz_token_sort_ratio",
    "name_fuzz_token_set_ratio",
    "name_fuzz_partial_ratio",
    "name_shared_info_count",
    "name_shared_token_count",
    "name_rare_shared_count",
    "name_has_rare_shared",
    # ADDRESS features (11)
    "addr_exact",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_fuzz_ratio",
    "addr_fuzz_token_sort_ratio",
    "addr_fuzz_token_set_ratio",
    "addr_shared_tokens",
    "addr_shared_info_tokens",
    "addr_shared_numbers",
    "addr_pincode_overlap",
    "addr_char_len_ratio",
    # OTHER & BLOCKER features (13)
    "country_exact",
    "source_is_s2",
    "b_name_norm",
    "b_name_sorted",
    "b_name_compact",
    "b_name_overlap2",
    "b_addr_overlap3",
    "b_name_rare_stop",
    "b_addr_num_token",
    "b_addr_rare_overlap2",
    "b_name_rare_single",
    "b_name_prefix5",
    "num_blockers_fired",
]


def get_process_memory_mb() -> float:
    try:
        out = subprocess.check_output(
            ["powershell", "-c", f"(Get-Process -Id {os.getpid()}).WorkingSet64 / 1MB"],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return 0.0


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


def parse_record(name: str, addr: str, country: str) -> dict[str, Any]:
    norm_c = normalize_basic(country)
    nn = normalize_basic(name)
    ns = sorted_tokens(name)
    nc = compact(name)
    p5 = nc[:5] if len(nc) >= 5 else ""

    toks_name = tokenize(name)
    toks_name_set = set(toks_name)
    info_name = {t for t in toks_name if len(t) >= 3 and t not in NAME_STOPWORDS}
    stop_name = {t for t in toks_name if t in NAME_STOPWORDS}

    na = normalize_basic(addr)
    toks_addr = tokenize(addr)
    toks_addr_set = set(toks_addr)
    info_addr = {t for t in toks_addr if len(t) >= 3 and t not in ADDR_STOPWORDS}
    nums_addr = extract_address_numbers(addr)
    pincodes = extract_pincodes(addr)

    return {
        "country": norm_c,
        "name_norm": nn,
        "name_sorted": ns,
        "name_compact": nc,
        "prefix5": p5,
        "name_tokens": toks_name_set,
        "name_tokens_list": toks_name,
        "info_name_tokens": info_name,
        "stop_name_tokens": stop_name,
        "addr_norm": na,
        "addr_tokens": toks_addr_set,
        "info_addr_tokens": info_addr,
        "addr_numbers": nums_addr,
        "pincodes": pincodes,
    }


def compute_pair_features(
    s1_p: dict[str, Any],
    cand_p: dict[str, Any],
    cand_id: str,
    blocker_bits: int,
    rare_token_dfs: dict[tuple[str, str], int],
) -> list[float]:
    # NAME features
    s1_nn = s1_p["name_norm"]
    cand_nn = cand_p["name_norm"]
    name_exact = 1.0 if s1_nn == cand_nn and s1_nn else 0.0
    name_compact_exact = 1.0 if s1_p["name_compact"] == cand_p["name_compact"] and s1_p["name_compact"] else 0.0
    name_sorted_exact = 1.0 if s1_p["name_sorted"] == cand_p["name_sorted"] and s1_p["name_sorted"] else 0.0

    len_s1_n = len(s1_nn)
    len_cand_n = len(cand_nn)
    name_char_len_ratio = min(len_s1_n, len_cand_n) / max(max(len_s1_n, len_cand_n), 1)

    toks_s1 = s1_p["name_tokens"]
    toks_cand = cand_p["name_tokens"]
    name_token_count_diff = float(abs(len(toks_s1) - len(toks_cand)))

    n_intersect = len(toks_s1 & toks_cand)
    n_union = len(toks_s1 | toks_cand)
    name_token_jaccard = n_intersect / max(n_union, 1)
    name_token_overlap = n_intersect / max(min(len(toks_s1), len(toks_cand)), 1)

    name_fuzz_ratio = fuzz.ratio(s1_nn, cand_nn) / 100.0
    name_fuzz_token_sort_ratio = fuzz.token_sort_ratio(s1_nn, cand_nn) / 100.0
    name_fuzz_token_set_ratio = fuzz.token_set_ratio(s1_nn, cand_nn) / 100.0
    name_fuzz_partial_ratio = fuzz.partial_ratio(s1_nn, cand_nn) / 100.0

    name_shared_info_count = float(len(s1_p["info_name_tokens"] & cand_p["info_name_tokens"]))
    name_shared_token_count = float(n_intersect)

    c = s1_p["country"]
    rare_count = 0
    has_rare = 0.0
    for t in (toks_s1 & toks_cand):
        df = rare_token_dfs.get((c, t), 999999)
        if df <= 100:
            rare_count += 1
        if df <= 50:
            has_rare = 1.0
    name_rare_shared_count = float(rare_count)
    name_has_rare_shared = has_rare

    # ADDRESS features
    s1_an = s1_p["addr_norm"]
    cand_an = cand_p["addr_norm"]
    addr_exact = 1.0 if s1_an == cand_an and s1_an else 0.0

    a_toks_s1 = s1_p["addr_tokens"]
    a_toks_cand = cand_p["addr_tokens"]
    a_intersect = len(a_toks_s1 & a_toks_cand)
    a_union = len(a_toks_s1 | a_toks_cand)
    addr_token_jaccard = a_intersect / max(a_union, 1)
    addr_token_overlap = a_intersect / max(min(len(a_toks_s1), len(a_toks_cand)), 1)

    addr_fuzz_ratio = fuzz.ratio(s1_an, cand_an) / 100.0
    addr_fuzz_token_sort_ratio = fuzz.token_sort_ratio(s1_an, cand_an) / 100.0
    addr_fuzz_token_set_ratio = fuzz.token_set_ratio(s1_an, cand_an) / 100.0

    addr_shared_tokens = float(a_intersect)
    addr_shared_info_tokens = float(len(s1_p["info_addr_tokens"] & cand_p["info_addr_tokens"]))
    addr_shared_numbers = float(len(s1_p["addr_numbers"] & cand_p["addr_numbers"]))
    addr_pincode_overlap = 1.0 if (s1_p["pincodes"] & cand_p["pincodes"]) else 0.0

    len_s1_a = len(s1_an)
    len_cand_a = len(cand_an)
    addr_char_len_ratio = min(len_s1_a, len_cand_a) / max(max(len_s1_a, len_cand_a), 1)

    # OTHER & BLOCKER features
    country_exact = 1.0 if s1_p["country"] == cand_p["country"] and s1_p["country"] else 0.0
    source_is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    b0 = 1.0 if (blocker_bits & (1 << 0)) else 0.0
    b1 = 1.0 if (blocker_bits & (1 << 1)) else 0.0
    b2 = 1.0 if (blocker_bits & (1 << 2)) else 0.0
    b3 = 1.0 if (blocker_bits & (1 << 3)) else 0.0
    b4 = 1.0 if (blocker_bits & (1 << 4)) else 0.0
    b5 = 1.0 if (blocker_bits & (1 << 5)) else 0.0
    b6 = 1.0 if (blocker_bits & (1 << 6)) else 0.0
    b7 = 1.0 if (blocker_bits & (1 << 7)) else 0.0
    b8 = 1.0 if (blocker_bits & (1 << 8)) else 0.0
    b9 = 1.0 if (blocker_bits & (1 << 9)) else 0.0
    num_blockers = float(blocker_bits.bit_count())

    return [
        name_exact,
        name_compact_exact,
        name_sorted_exact,
        name_char_len_ratio,
        name_token_count_diff,
        name_token_jaccard,
        name_token_overlap,
        name_fuzz_ratio,
        name_fuzz_token_sort_ratio,
        name_fuzz_token_set_ratio,
        name_fuzz_partial_ratio,
        name_shared_info_count,
        name_shared_token_count,
        name_rare_shared_count,
        name_has_rare_shared,
        addr_exact,
        addr_token_jaccard,
        addr_token_overlap,
        addr_fuzz_ratio,
        addr_fuzz_token_sort_ratio,
        addr_fuzz_token_set_ratio,
        addr_shared_tokens,
        addr_shared_info_tokens,
        addr_shared_numbers,
        addr_pincode_overlap,
        addr_char_len_ratio,
        country_exact,
        source_is_s2,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        b8,
        b9,
        num_blockers,
    ]


def evaluate_predictions(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, list[str]],
) -> dict[str, Any]:
    scores = []
    precisions = []
    recalls = []
    singleton_scores = []
    non_singleton_scores = []

    s2_scores = []
    s3_scores = []

    pred_counts = []

    for s1_id, truth_list in ground_truth.items():
        truth = set(truth_list)
        pred = predictions.get(s1_id, set())
        pred_counts.append(len(pred))

        # Overall S1 metrics
        if not truth and not pred:
            f05 = 1.0
            p = 1.0
            r = 1.0
        elif not truth or not pred:
            f05 = 0.0
            p = 0.0
            r = 0.0
        else:
            tp = len(pred & truth)
            if tp == 0:
                f05 = 0.0
                p = 0.0
                r = 0.0
            else:
                p = tp / len(pred)
                r = tp / len(truth)
                beta_sq = 0.25
                f05 = (1 + beta_sq) * p * r / (beta_sq * p + r)

        scores.append(f05)
        precisions.append(p)
        recalls.append(r)

        if not truth:
            singleton_scores.append(f05)
        else:
            non_singleton_scores.append(f05)

        # S2 breakdown
        truth_s2 = {m for m in truth if m.startswith("S2-")}
        pred_s2 = {m for m in pred if m.startswith("S2-")}
        s2_scores.append(f_beta_per_s1(pred_s2, truth_s2, beta=0.5))

        # S3 breakdown
        truth_s3 = {m for m in truth if m.startswith("S3-")}
        pred_s3 = {m for m in pred if m.startswith("S3-")}
        s3_scores.append(f_beta_per_s1(pred_s3, truth_s3, beta=0.5))

    p_arr = np.array(pred_counts)
    return {
        "macro_f05": float(np.mean(scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "s2_macro_f05": float(np.mean(s2_scores)),
        "s3_macro_f05": float(np.mean(s3_scores)),
        "mean_pred_matches": float(np.mean(p_arr)),
        "median_pred_matches": float(np.median(p_arr)),
        "p95_pred_matches": float(np.percentile(p_arr, 95)),
        "max_pred_matches": int(np.max(p_arr)),
    }


def run_pairwise_pipeline(
    data_dir: Path,
    sample_size: int = 5_000,
    seed: int = 42,
    train_ratio: float = 0.8,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("PAIRWISE MATCHING PIPELINE (COMBO 3 RETRIEVAL + LIGHTGBM)", flush=True)
    print(f"Dataset dir: {data_dir}", flush=True)
    print(f"Validation sample size: {sample_size:,} S1 entities (seed={seed})", flush=True)
    print(f"S1-level split ratio: {train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% test", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # Step 1: Load Ground Truth and create S1 validation split
    t0 = time.time()
    print("\n[Step 1] Loading ground truth & creating deterministic validation split...", flush=True)
    gt = load_ground_truth(gt_path)
    all_s1_ids = sorted(gt.keys())
    rng = random.Random(seed)
    val_s1_ids = rng.sample(all_s1_ids, min(sample_size, len(all_s1_ids)))
    val_s1_set = set(val_s1_ids)

    n_train = int(len(val_s1_ids) * train_ratio)
    train_s1_ids = val_s1_ids[:n_train]
    test_s1_ids = val_s1_ids[n_train:]

    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}
    train_gt = {s1_id: val_gt[s1_id] for s1_id in train_s1_ids}
    test_gt = {s1_id: val_gt[s1_id] for s1_id in test_s1_ids}

    singletons_val = sum(1 for m in val_gt.values() if not m)
    singletons_train = sum(1 for m in train_gt.values() if not m)
    singletons_test = sum(1 for m in test_gt.values() if not m)

    print(f"  Total Validation S1: {len(val_s1_ids):,} (Singletons: {singletons_val:,} / {singletons_val/len(val_s1_ids)*100:.2f}%)", flush=True)
    print(f"  Train S1 split:      {len(train_s1_ids):,} (Singletons: {singletons_train:,} / {singletons_train/len(train_s1_ids)*100:.2f}%)", flush=True)
    print(f"  Test S1 split:       {len(test_s1_ids):,} (Singletons: {singletons_test:,} / {singletons_test/len(test_s1_ids)*100:.2f}%)", flush=True)

    # Step 2: Load S1 records & prepare query structures
    print("\n[Step 2] Loading S1 records & extracting Combo 3 query keys...", flush=True)
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
        if len(val_s1_records) >= len(val_s1_set):
            break

    # Build active query key structures
    active_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_addr_numbers: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_compact_prefix5: dict[tuple[str, str], set[str]] = defaultdict(set)

    s1_parsed: dict[str, dict] = {}

    for s1_id, rec in val_s1_records.items():
        parsed = parse_record(rec["business_name"], rec["business_address"], rec["country"])
        s1_parsed[s1_id] = parsed
        country = parsed["country"]

        active_name_norm[(country, parsed["name_norm"])].add(s1_id)
        active_name_sorted[(country, parsed["name_sorted"])].add(s1_id)
        active_name_compact[(country, parsed["name_compact"])].add(s1_id)
        if parsed["prefix5"]:
            active_compact_prefix5[(country, parsed["prefix5"])].add(s1_id)

        for t in parsed["info_name_tokens"]:
            active_name_tokens[(country, t)].add(s1_id)
        for t in parsed["stop_name_tokens"]:
            active_name_stopwords[(country, t)].add(s1_id)
        for t in parsed["info_addr_tokens"]:
            active_addr_tokens[(country, t)].add(s1_id)
        for num in parsed["addr_numbers"]:
            active_addr_numbers[(country, num)].add(s1_id)

    print(f"  Active keys: {len(active_name_norm):,} name_norm, {len(active_name_sorted):,} name_sorted, "
          f"{len(active_name_compact):,} name_compact, {len(active_compact_prefix5):,} prefix5, "
          f"{len(active_name_tokens):,} info_name_tokens, {len(active_name_stopwords):,} stop_name_tokens, "
          f"{len(active_addr_tokens):,} info_addr_tokens, {len(active_addr_numbers):,} addr_numbers", flush=True)

    # Step 3: Stream S2 & S3 (Pass 1 - Build Inverted Indices for Combo 3)
    print("\n[Step 3] Pass 1: Streaming S2 & S3 across full dataset (10.3M records) to populate Combo 3 index...", flush=True)
    t_stream_start = time.time()

    idx_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_compact_prefix5: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_numbers: dict[tuple[str, str], set[str]] = defaultdict(set)

    total_streamed = 0

    for source_label, source_file in [("S2", s2_path), ("S3", s3_path)]:
        t_src = time.time()
        count_src = 0
        print(f"  Streaming {source_label}...", flush=True)
        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            dtype="string",
            keep_default_na=False,
            chunksize=500_000,
        ):
            count_src += len(chunk)
            for eid, name, addr, country_str in zip(
                chunk["entity_id"],
                chunk["business_name"],
                chunk["business_address"],
                chunk["country"],
            ):
                c = normalize_basic(country_str)
                nn = normalize_basic(name)
                k_nn = (c, nn)
                if k_nn in active_name_norm:
                    idx_name_norm[k_nn].add(eid)

                ns = sorted_tokens(name)
                k_ns = (c, ns)
                if k_ns in active_name_sorted:
                    idx_name_sorted[k_ns].add(eid)

                nc = compact(name)
                k_nc = (c, nc)
                if k_nc in active_name_compact:
                    idx_name_compact[k_nc].add(eid)

                if len(nc) >= 5:
                    k_p5 = (c, nc[:5])
                    if k_p5 in active_compact_prefix5:
                        idx_compact_prefix5[k_p5].add(eid)

                for t in tokenize(name):
                    if len(t) >= 3:
                        if t not in NAME_STOPWORDS:
                            k_nt = (c, t)
                            if k_nt in active_name_tokens:
                                idx_name_tokens[k_nt].add(eid)
                        else:
                            k_ns = (c, t)
                            if k_ns in active_name_stopwords:
                                idx_name_stopwords[k_ns].add(eid)

                for t in tokenize(addr):
                    if len(t) >= 3 and t not in ADDR_STOPWORDS:
                        k_at = (c, t)
                        if k_at in active_addr_tokens:
                            idx_addr_tokens[k_at].add(eid)

                for num in extract_address_numbers(addr):
                    k_num = (c, num)
                    if k_num in active_addr_numbers:
                        idx_addr_numbers[k_num].add(eid)

        total_streamed += count_src
        print(f"  Finished {source_label}: {count_src:,} rows in {time.time()-t_src:.1f}s", flush=True)

    stream_time = time.time() - t_stream_start
    print(f"  Total streamed: {total_streamed:,} records in {stream_time:.1f}s", flush=True)
    print(f"  RAM working set: {get_process_memory_mb():.1f} MB", flush=True)

    # Step 4: Generate Combo 3 candidate pairs and track blockers fired
    print("\n[Step 4] Querying Combo 3 retrieval rules & tracking blocker keys...", flush=True)
    t_query_start = time.time()

    # rare token df map for feature extraction
    rare_token_dfs: dict[tuple[str, str], int] = {}
    for k, eid_set in idx_name_tokens.items():
        if len(eid_set) <= 500:
            rare_token_dfs[k] = len(eid_set)

    # s1_cand_blockers: s1_id -> {cand_id: bitmask}
    s1_cand_blockers: dict[str, dict[str, int]] = {}
    total_val_pairs = 0
    all_needed_cand_ids: set[str] = set()

    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]
        cands_dict: dict[str, int] = {}

        # 0: Name Norm
        c0 = idx_name_norm.get((c, p["name_norm"]), set())
        for cid in c0:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 0)

        # 1: Name Sorted
        c1 = idx_name_sorted.get((c, p["name_sorted"]), set())
        for cid in c1:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 1)

        # 2: Name Compact
        c2 = idx_name_compact.get((c, p["name_compact"]), set())
        for cid in c2:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 2)

        # 3: Name Token Overlap >= 2
        name_sets = [idx_name_tokens.get((c, t), set()) for t in p["info_name_tokens"] if (c, t) in idx_name_tokens]
        c3 = get_token_overlap_candidates(name_sets, min_overlap=2)
        for cid in c3:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 3)

        # 4: Addr Token Overlap >= 3
        addr_sets = [idx_addr_tokens.get((c, t), set()) for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens]
        c4 = get_token_overlap_candidates(addr_sets, min_overlap=3)
        for cid in c4:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 4)

        # 5: Name Rare Info (DF <= 500) + Stopword
        rare_info = [idx_name_tokens[(c, t)] for t in p["info_name_tokens"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
        if rare_info and p["stop_name_tokens"]:
            s_union = set()
            for st in p["stop_name_tokens"]:
                s_union |= idx_name_stopwords.get((c, st), set())
            if s_union:
                for n_set in rare_info:
                    for cid in (n_set & s_union):
                        cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 5)

        # 6: Addr Number (len>=3, DF<=500) + Addr Token (DF<=1000)
        valid_nums = [n for n in p["addr_numbers"] if len(n) >= 3 and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 500]
        valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
        if valid_nums and valid_addrs:
            a_union = set()
            for a_set in valid_addrs:
                a_union |= a_set
            for num in valid_nums:
                for cid in (idx_addr_numbers[(c, num)] & a_union):
                    cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 6)

        # 7: Two Rare Addr Tokens (DF <= 500)
        rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
        if len(rare_addrs) >= 2:
            c7 = get_token_overlap_candidates(rare_addrs, min_overlap=2)
            for cid in c7:
                cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 7)

        # 8: Single Rare Name Token (len>=5, DF<=50)
        for t in p["info_name_tokens"]:
            if len(t) >= 5 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 50:
                for cid in idx_name_tokens[(c, t)]:
                    cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 8)

        # 9: Compact Prefix 5 (DF <= 50)
        p5 = p["prefix5"]
        if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= 50:
            for cid in idx_compact_prefix5[(c, p5)]:
                cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 9)

        s1_cand_blockers[s1_id] = cands_dict
        total_val_pairs += len(cands_dict)
        all_needed_cand_ids.update(cands_dict.keys())

    print(f"  Combo 3 Query completed in {time.time()-t_query_start:.1f}s", flush=True)
    print(f"  Total candidate pairs: {total_val_pairs:,}", flush=True)
    print(f"  Unique candidate entities: {len(all_needed_cand_ids):,}", flush=True)

    # Free large inverted indices to reclaim RAM before Pass 2
    del idx_name_norm, idx_name_sorted, idx_name_compact, idx_compact_prefix5
    del idx_name_tokens, idx_name_stopwords, idx_addr_tokens, idx_addr_numbers
    gc.collect()
    print(f"  Freed indexing memory. RAM working set: {get_process_memory_mb():.1f} MB", flush=True)

    # =========================================================================
    # STEP 2: Analyze Hard Negatives
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 2 — HARD NEGATIVES ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    total_gt_matches_val = sum(len(m) for m in val_gt.values())
    total_positives_val = 0
    pos_per_s1 = []
    neg_per_s1 = []
    cands_per_s1 = []
    zero_pos_count = 0
    zero_pos_singletons = 0
    zero_pos_non_singletons = 0

    for s1_id in val_s1_ids:
        cands = set(s1_cand_blockers[s1_id].keys())
        tm = set(val_gt[s1_id])
        pos = len(cands & tm)
        neg = len(cands) - pos
        total_positives_val += pos
        pos_per_s1.append(pos)
        neg_per_s1.append(neg)
        cands_per_s1.append(len(cands))
        if pos == 0:
            zero_pos_count += 1
            if not tm:
                zero_pos_singletons += 1
            else:
                zero_pos_non_singletons += 1

    total_negatives_val = total_val_pairs - total_positives_val
    pos_rate_val = total_positives_val / total_val_pairs if total_val_pairs else 0.0

    cands_arr = np.array(cands_per_s1)
    pos_arr = np.array(pos_per_s1)
    neg_arr = np.array(neg_per_s1)

    hard_neg_stats = {
        "total_candidate_pairs": total_val_pairs,
        "total_positives": total_positives_val,
        "total_negatives": total_negatives_val,
        "positive_rate": pos_rate_val,
        "true_matches_total": total_gt_matches_val,
        "blocking_recall": total_positives_val / total_gt_matches_val if total_gt_matches_val else 0.0,
        "candidate_count_distribution": {
            "mean": float(np.mean(cands_arr)),
            "std": float(np.std(cands_arr)),
            "min": int(np.min(cands_arr)),
            "p25": float(np.percentile(cands_arr, 25)),
            "median": float(np.median(cands_arr)),
            "p75": float(np.percentile(cands_arr, 75)),
            "p90": float(np.percentile(cands_arr, 90)),
            "p95": float(np.percentile(cands_arr, 95)),
            "p99": float(np.percentile(cands_arr, 99)),
            "max": int(np.max(cands_arr)),
        },
        "positives_per_s1_distribution": {
            "mean": float(np.mean(pos_arr)),
            "min": int(np.min(pos_arr)),
            "median": float(np.median(pos_arr)),
            "p95": float(np.percentile(pos_arr, 95)),
            "max": int(np.max(pos_arr)),
        },
        "negatives_per_s1_distribution": {
            "mean": float(np.mean(neg_arr)),
            "min": int(np.min(neg_arr)),
            "median": float(np.median(neg_arr)),
            "p90": float(np.percentile(neg_arr, 90)),
            "p95": float(np.percentile(neg_arr, 95)),
            "p99": float(np.percentile(neg_arr, 99)),
            "max": int(np.max(neg_arr)),
        },
        "zero_positive_s1_entities": {
            "total_zero_pos": zero_pos_count,
            "pct_of_validation": zero_pos_count / len(val_s1_ids) * 100,
            "singletons_with_zero_pos": zero_pos_singletons,
            "non_singletons_with_zero_pos_missed_recall": zero_pos_non_singletons,
        },
    }

    print(f"Total Candidate Pairs:      {total_val_pairs:,}", flush=True)
    print(f"  Positives (True Matches): {total_positives_val:,} ({pos_rate_val*100:.3f}% of pairs, Blocking Recall: {total_positives_val/total_gt_matches_val*100:.2f}%)", flush=True)
    print(f"  Negatives (Hard Negatives): {total_negatives_val:,} ({(1-pos_rate_val)*100:.3f}% of pairs)", flush=True)
    print(f"  Negative-to-Positive Ratio: {total_negatives_val/max(total_positives_val, 1):.1f} : 1", flush=True)
    print("\nCandidate Count per S1 Distribution:", flush=True)
    print(f"  Mean: {np.mean(cands_arr):.1f}, Median: {np.median(cands_arr):.0f}, p25: {np.percentile(cands_arr, 25):.0f}, "
          f"p75: {np.percentile(cands_arr, 75):.0f}, p90: {np.percentile(cands_arr, 90):.0f}, p95: {np.percentile(cands_arr, 95):.0f}, Max: {np.max(cands_arr):,}", flush=True)
    print("\nPositives per S1 Distribution:", flush=True)
    print(f"  Mean: {np.mean(pos_arr):.2f}, Median: {np.median(pos_arr):.0f}, p95: {np.percentile(pos_arr, 95):.0f}, Max: {np.max(pos_arr)}", flush=True)
    print("\nNegatives per S1 Distribution:", flush=True)
    print(f"  Mean: {np.mean(neg_arr):.1f}, Median: {np.median(neg_arr):.0f}, p90: {np.percentile(neg_arr, 90):.0f}, p95: {np.percentile(neg_arr, 95):.0f}, Max: {np.max(neg_arr):,}", flush=True)
    print("\nS1 Entities with ZERO Positives among Retrieved Candidates:", flush=True)
    print(f"  Total with 0 positives: {zero_pos_count:,} ({zero_pos_count/len(val_s1_ids)*100:.2f}%)", flush=True)
    print(f"    - True Singletons (0 matches in GT): {zero_pos_singletons:,} ({zero_pos_singletons/len(val_s1_ids)*100:.2f}%)", flush=True)
    print(f"    - Non-Singletons (True matches missed by Combo 3): {zero_pos_non_singletons:,} ({zero_pos_non_singletons/len(val_s1_ids)*100:.2f}%)", flush=True)

    # Step 5: Pass 2 - Stream S2 & S3 to load candidate texts
    print(f"\n[Step 5] Pass 2: Streaming S2 & S3 to retrieve texts for {len(all_needed_cand_ids):,} candidate entities...", flush=True)
    t_pass2_start = time.time()
    cand_parsed: dict[str, dict[str, Any]] = {}

    for source_label, source_file in [("S2", s2_path), ("S3", s3_path)]:
        t_src = time.time()
        print(f"  Pass 2: Reading {source_label}...", flush=True)
        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            dtype="string",
            keep_default_na=False,
            chunksize=500_000,
        ):
            mask = chunk["entity_id"].isin(all_needed_cand_ids)
            if mask.any():
                for eid, name, addr, country_str in zip(
                    chunk.loc[mask, "entity_id"],
                    chunk.loc[mask, "business_name"],
                    chunk.loc[mask, "business_address"],
                    chunk.loc[mask, "country"],
                ):
                    cand_parsed[eid] = parse_record(name, addr, country_str)
        print(f"  Finished {source_label} in {time.time()-t_src:.1f}s, loaded {len(cand_parsed):,} candidate records", flush=True)

    print(f"  Pass 2 completed in {time.time()-t_pass2_start:.1f}s. Total candidate records loaded: {len(cand_parsed):,} / {len(all_needed_cand_ids):,}", flush=True)
    print(f"  RAM working set: {get_process_memory_mb():.1f} MB", flush=True)

    # =========================================================================
    # STEP 3: Feature Extraction (Train & Test)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 3 — FEATURE EXTRACTION", flush=True)
    print("=" * 80, flush=True)
    t_feat_start = time.time()

    # Pre-allocate train and test matrices
    n_train_pairs = sum(len(s1_cand_blockers[s1_id]) for s1_id in train_s1_ids)
    n_test_pairs = sum(len(s1_cand_blockers[s1_id]) for s1_id in test_s1_ids)

    print(f"  Train candidate pairs: {n_train_pairs:,} (from {len(train_s1_ids):,} train S1 entities)", flush=True)
    print(f"  Test candidate pairs:  {n_test_pairs:,} (from {len(test_s1_ids):,} test S1 entities)", flush=True)

    n_features = len(FEATURE_NAMES)
    X_train = np.empty((n_train_pairs, n_features), dtype=np.float32)
    y_train = np.empty(n_train_pairs, dtype=np.uint8)

    X_test = np.empty((n_test_pairs, n_features), dtype=np.float32)
    y_test = np.empty(n_test_pairs, dtype=np.uint8)

    # Extract Train Features
    print("  Extracting features for Train pairs...", flush=True)
    train_idx = 0
    for s1_id in train_s1_ids:
        s1_p = s1_parsed[s1_id]
        gt_set = set(train_gt[s1_id])
        for cand_id, bits in s1_cand_blockers[s1_id].items():
            cand_p = cand_parsed.get(cand_id)
            if cand_p is None:
                continue
            feats = compute_pair_features(s1_p, cand_p, cand_id, bits, rare_token_dfs)
            X_train[train_idx] = feats
            y_train[train_idx] = 1 if cand_id in gt_set else 0
            train_idx += 1

    if train_idx < n_train_pairs:
        X_train = X_train[:train_idx]
        y_train = y_train[:train_idx]

    # Extract Test Features
    print("  Extracting features for Test pairs...", flush=True)
    test_idx = 0
    # test_pair_meta: list of (s1_id, cand_id) to map row back
    test_pair_s1: list[str] = []
    test_pair_cand: list[str] = []

    for s1_id in test_s1_ids:
        s1_p = s1_parsed[s1_id]
        gt_set = set(test_gt[s1_id])
        for cand_id, bits in s1_cand_blockers[s1_id].items():
            cand_p = cand_parsed.get(cand_id)
            if cand_p is None:
                continue
            feats = compute_pair_features(s1_p, cand_p, cand_id, bits, rare_token_dfs)
            X_test[test_idx] = feats
            y_test[test_idx] = 1 if cand_id in gt_set else 0
            test_pair_s1.append(s1_id)
            test_pair_cand.append(cand_id)
            test_idx += 1

    if test_idx < n_test_pairs:
        X_test = X_test[:test_idx]
        y_test = y_test[:test_idx]

    feat_time = time.time() - t_feat_start
    print(f"  Feature extraction complete in {feat_time:.1f}s!", flush=True)
    print(f"  Train: {X_train.shape[0]:,} rows, {y_train.sum():,} positives ({y_train.sum()/len(y_train)*100:.3f}%)", flush=True)
    print(f"  Test:  {X_test.shape[0]:,} rows, {y_test.sum():,} positives ({y_test.sum()/len(y_test)*100:.3f}%)", flush=True)
    print(f"  RAM working set: {get_process_memory_mb():.1f} MB", flush=True)

    # =========================================================================
    # STEP 4: Train LightGBM Classifier
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 4 — TRAIN LIGHTGBM CLASSIFIER", flush=True)
    print("=" * 80, flush=True)
    t_train_start = time.time()

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES, free_raw_data=False)
    test_ds = lgb.Dataset(X_test, label=y_test, feature_name=FEATURE_NAMES, reference=train_ds, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.08,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 30,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": seed,
    }

    evals_result: dict[str, dict] = {}
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=300,
        valid_sets=[train_ds, test_ds],
        valid_names=["train", "test"],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )

    train_time = time.time() - t_train_start
    print(f"  LightGBM trained in {train_time:.1f}s", flush=True)

    # Feature Importance
    importance_gain = model.feature_importance(importance_type="gain")
    importance_split = model.feature_importance(importance_type="split")
    feature_imp_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "gain": importance_gain,
        "split": importance_split,
    }).sort_values(by="gain", ascending=False).reset_index(drop=True)

    total_gain = feature_imp_df["gain"].sum()
    feature_imp_df["gain_pct"] = feature_imp_df["gain"] / total_gain * 100

    print("\nTop 20 Features by Information Gain:")
    for rank, (feat, gain, split, gain_pct) in enumerate(feature_imp_df.head(20).itertuples(index=False), 1):
        print(f"  {rank:>2}. {feat:<30} Gain: {gain:>12.1f} ({gain_pct:>5.2f}%)  Splits: {int(split):>5}")

    # Pairwise evaluation on Test set
    print("\nEvaluating Pairwise Test Set Metrics...", flush=True)
    y_test_probs = np.asarray(model.predict(X_test), dtype=np.float64)
    pr_auc = float(average_precision_score(y_test, y_test_probs))

    # Evaluate pairwise metrics across multiple thresholds
    pairwise_metrics = {}
    for th in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]:
        pred_binary = (y_test_probs >= th).astype(np.uint8)
        tp = int(np.sum((pred_binary == 1) & (y_test == 1)))
        fp = int(np.sum((pred_binary == 1) & (y_test == 0)))
        fn = int(np.sum((pred_binary == 0) & (y_test == 1)))
        tn = int(np.sum((pred_binary == 0) & (y_test == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = 0.25
        f05 = (1 + beta_sq) * prec * rec / (beta_sq * prec + rec) if (beta_sq * prec + rec) > 0 else 0.0

        pairwise_metrics[str(th)] = {
            "threshold": th,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec,
            "recall": rec,
            "f05": f05,
        }

    print(f"  Test PR-AUC: {pr_auc:.4f}")
    print("\nPairwise Metrics by Probability Threshold (MICRO Pair Level):")
    print(f"  {'Thresh':>6} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Precision':>9} | {'Recall':>9} | {'Pair F0.5':>9}")
    print("  " + "-" * 65)
    for th_str, m in pairwise_metrics.items():
        print(f"  {float(th_str):>6.2f} | {m['tp']:>6} | {m['fp']:>6} | {m['fn']:>6} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f05']*100:>8.2f}%")

    # =========================================================================
    # STEP 5: S1-level Macro F0.5 Evaluation (The True Competition Metric)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 5 — EVALUATE S1-LEVEL MACRO F0.5 (COMPETITION METRIC)", flush=True)
    print("=" * 80, flush=True)

    # Group test predictions by S1
    test_s1_cand_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s1_id, cand_id, prob in zip(test_pair_s1, test_pair_cand, y_test_probs):
        test_s1_cand_scores[s1_id].append((cand_id, float(prob)))

    thresholds_to_test = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    macro_results_by_threshold = {}
    best_th = 0.50
    best_macro_f05 = -1.0

    print(f"  {'Thresh':>6} | {'Macro F0.5':>10} | {'Macro Prec':>10} | {'Macro Rec':>10} | {'Singletons':>10} | {'Non-Sing':>10} | {'S2 F0.5':>8} | {'S3 F0.5':>8} | {'Pred/S1':>8}")
    print("  " + "-" * 95)

    for th in thresholds_to_test:
        preds = {}
        for s1_id in test_s1_ids:
            cand_scores = test_s1_cand_scores[s1_id]
            preds[s1_id] = {cid for cid, p in cand_scores if p >= th}

        res = evaluate_predictions(preds, test_gt)
        macro_results_by_threshold[str(th)] = res
        if res["macro_f05"] > best_macro_f05:
            best_macro_f05 = res["macro_f05"]
            best_th = th

        print(f"  {th:>6.2f} | {res['macro_f05']*100:>9.2f}% | {res['macro_precision']*100:>9.2f}% | {res['macro_recall']*100:>9.2f}% | "
              f"{res['singleton_f05']*100:>9.2f}% | {res['non_singleton_f05']*100:>9.2f}% | {res['s2_macro_f05']*100:>7.2f}% | {res['s3_macro_f05']*100:>7.2f}% | "
              f"{res['mean_pred_matches']:>8.2f}")

    print(f"\nOptimal S1-level Threshold: {best_th:.2f} (Macro F0.5 = {best_macro_f05*100:.2f}%)", flush=True)

    # =========================================================================
    # STEP 6: Compare Against Simple Rules
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 6 — BENCHMARK AGAINST SIMPLE RULES", flush=True)
    print("=" * 80, flush=True)

    rule_results = {}

    # Rule 1: Exact Normalized Name + Country
    preds_r1 = {}
    for s1_id in test_s1_ids:
        s1_p = s1_parsed[s1_id]
        matches = set()
        for cand_id in s1_cand_blockers[s1_id]:
            cp = cand_parsed[cand_id]
            if s1_p["name_norm"] == cp["name_norm"] and s1_p["name_norm"] and s1_p["country"] == cp["country"]:
                matches.add(cand_id)
        preds_r1[s1_id] = matches
    res_r1 = evaluate_predictions(preds_r1, test_gt)
    rule_results["Rule 1: Exact Name + Country"] = res_r1

    # Rule 2: Exact Normalized Name + Exact Address + Country
    preds_r2 = {}
    for s1_id in test_s1_ids:
        s1_p = s1_parsed[s1_id]
        matches = set()
        for cand_id in s1_cand_blockers[s1_id]:
            cp = cand_parsed[cand_id]
            if (s1_p["name_norm"] == cp["name_norm"] and s1_p["name_norm"] and
                s1_p["addr_norm"] == cp["addr_norm"] and s1_p["addr_norm"] and
                s1_p["country"] == cp["country"]):
                matches.add(cand_id)
        preds_r2[s1_id] = matches
    res_r2 = evaluate_predictions(preds_r2, test_gt)
    rule_results["Rule 2: Exact Name + Exact Address + Country"] = res_r2

    # Rule 3: High Name Similarity (fuzz.ratio >= 0.90) + Country
    preds_r3 = {}
    for s1_id in test_s1_ids:
        s1_p = s1_parsed[s1_id]
        matches = set()
        for cand_id in s1_cand_blockers[s1_id]:
            cp = cand_parsed[cand_id]
            if fuzz.ratio(s1_p["name_norm"], cp["name_norm"]) >= 90 and s1_p["country"] == cp["country"]:
                matches.add(cand_id)
        preds_r3[s1_id] = matches
    res_r3 = evaluate_predictions(preds_r3, test_gt)
    rule_results["Rule 3: High Name Similarity (Ratio >= 90)"] = res_r3

    # Rule 4: Moderate Name Similarity (token_sort >= 85) + Country
    preds_r4 = {}
    for s1_id in test_s1_ids:
        s1_p = s1_parsed[s1_id]
        matches = set()
        for cand_id in s1_cand_blockers[s1_id]:
            cp = cand_parsed[cand_id]
            if fuzz.token_sort_ratio(s1_p["name_norm"], cp["name_norm"]) >= 85 and s1_p["country"] == cp["country"]:
                matches.add(cand_id)
        preds_r4[s1_id] = matches
    res_r4 = evaluate_predictions(preds_r4, test_gt)
    rule_results["Rule 4: Name Token-Sort >= 85 + Country"] = res_r4

    # Rule 5: Combined Name + Address (Name sort >= 85 AND Addr Jaccard >= 0.30)
    preds_r5 = {}
    for s1_id in test_s1_ids:
        s1_p = s1_parsed[s1_id]
        matches = set()
        for cand_id in s1_cand_blockers[s1_id]:
            cp = cand_parsed[cand_id]
            if s1_p["country"] == cp["country"] and fuzz.token_sort_ratio(s1_p["name_norm"], cp["name_norm"]) >= 85:
                a_inter = len(s1_p["addr_tokens"] & cp["addr_tokens"])
                a_union = len(s1_p["addr_tokens"] | cp["addr_tokens"])
                jacc = a_inter / max(a_union, 1)
                num_inter = len(s1_p["addr_numbers"] & cp["addr_numbers"])
                if jacc >= 0.25 or num_inter >= 1 or fuzz.ratio(s1_p["addr_norm"], cp["addr_norm"]) >= 75:
                    matches.add(cand_id)
        preds_r5[s1_id] = matches
    res_r5 = evaluate_predictions(preds_r5, test_gt)
    rule_results["Rule 5: Combined Name + Address Rule"] = res_r5

    # Print comparison table
    print(f"  {'Method':<42} | {'Macro F0.5':>10} | {'Macro Prec':>10} | {'Macro Rec':>10} | {'Singletons':>10} | {'Non-Sing':>10}")
    print("  " + "-" * 95)
    for r_name, r_res in rule_results.items():
        print(f"  {r_name:<42} | {r_res['macro_f05']*100:>9.2f}% | {r_res['macro_precision']*100:>9.2f}% | {r_res['macro_recall']*100:>9.2f}% | "
              f"{r_res['singleton_f05']*100:>9.2f}% | {r_res['non_singleton_f05']*100:>9.2f}%")

    lgb_best_res = macro_results_by_threshold[str(best_th)]
    print(f"  {'LightGBM Pairwise Model (Thresh=' + f'{best_th:.2f})':<42} | "
          f"{lgb_best_res['macro_f05']*100:>9.2f}% | {lgb_best_res['macro_precision']*100:>9.2f}% | {lgb_best_res['macro_recall']*100:>9.2f}% | "
          f"{lgb_best_res['singleton_f05']*100:>9.2f}% | {lgb_best_res['non_singleton_f05']*100:>9.2f}%")

    # =========================================================================
    # STEP 8: Error Analysis (False Positives and False Negatives)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 8 — ERROR ANALYSIS (FALSE POSITIVES & FALSE NEGATIVES)", flush=True)
    print("=" * 80, flush=True)

    # Collect predictions at best threshold
    best_preds = {}
    for s1_id in test_s1_ids:
        best_preds[s1_id] = {cid for cid, p in test_s1_cand_scores[s1_id] if p >= best_th}

    # Find Top False Positives
    false_positives = []
    false_negatives = []

    for s1_id, cand_id, prob in zip(test_pair_s1, test_pair_cand, y_test_probs):
        is_true = cand_id in test_gt[s1_id]
        if prob >= best_th and not is_true:
            false_positives.append((float(prob), s1_id, cand_id))
        elif is_true and prob < best_th:
            false_negatives.append((float(prob), s1_id, cand_id))

    false_positives.sort(key=lambda x: x[0], reverse=True)
    false_negatives.sort(key=lambda x: x[0])

    print(f"\nTotal False Positive Pairs at threshold {best_th:.2f}: {len(false_positives):,}")
    print("Top 8 False Positives (Highest Model Confidence Non-Matches):")
    top_fp_details = []
    for rank, (p, s1_id, cand_id) in enumerate(false_positives[:8], 1):
        s1_r = val_s1_records[s1_id]
        c_r = cand_parsed[cand_id]
        fp_item = {
            "rank": rank,
            "predicted_prob": round(p, 4),
            "s1_id": s1_id,
            "cand_id": cand_id,
            "s1_name": s1_r["business_name"],
            "cand_name": c_r["name_norm"],
            "s1_addr": s1_r["business_address"],
            "cand_addr": c_r["addr_norm"],
            "s1_country": s1_r["country"],
            "cand_country": c_r["country"],
        }
        top_fp_details.append(fp_item)
        print(f"  [{rank}] Prob: {p:.4f} | S1: {s1_id} vs Cand: {cand_id}")
        print(f"      S1 Name:   {s1_r['business_name']}")
        print(f"      Cand Name: {c_r['name_norm']}")
        print(f"      S1 Addr:   {s1_r['business_address']}")
        print(f"      Cand Addr: {c_r['addr_norm']}")

    print(f"\nTotal False Negative Pairs (Recovered by Combo 3 but prob < {best_th:.2f}): {len(false_negatives):,}")
    print("Top 8 False Negatives (Lowest Model Confidence True Matches):")
    top_fn_details = []
    for rank, (p, s1_id, cand_id) in enumerate(false_negatives[:8], 1):
        s1_r = val_s1_records[s1_id]
        c_r = cand_parsed[cand_id]
        fn_item = {
            "rank": rank,
            "predicted_prob": round(p, 4),
            "s1_id": s1_id,
            "cand_id": cand_id,
            "s1_name": s1_r["business_name"],
            "cand_name": c_r["name_norm"],
            "s1_addr": s1_r["business_address"],
            "cand_addr": c_r["addr_norm"],
            "s1_country": s1_r["country"],
            "cand_country": c_r["country"],
        }
        top_fn_details.append(fn_item)
        print(f"  [{rank}] Prob: {p:.4f} | S1: {s1_id} vs Cand: {cand_id}")
        print(f"      S1 Name:   {s1_r['business_name']}")
        print(f"      Cand Name: {c_r['name_norm']}")
        print(f"      S1 Addr:   {s1_r['business_address']}")
        print(f"      Cand Addr: {c_r['addr_norm']}")

    # Compile all outputs
    full_output = {
        "metadata": {
            "validation_sample_size": sample_size,
            "seed": seed,
            "train_ratio": train_ratio,
            "train_s1_count": len(train_s1_ids),
            "test_s1_count": len(test_s1_ids),
            "total_candidate_pairs": total_val_pairs,
            "train_candidate_pairs": n_train_pairs,
            "test_candidate_pairs": n_test_pairs,
            "optimal_threshold": best_th,
            "optimal_macro_f05": best_macro_f05,
            "test_pr_auc": pr_auc,
            "total_runtime_seconds": time.time() - t0,
        },
        "hard_negative_analysis": hard_neg_stats,
        "feature_importance": feature_imp_df.to_dict(orient="records"),
        "pairwise_metrics_by_threshold": pairwise_metrics,
        "macro_metrics_by_threshold": macro_results_by_threshold,
        "rule_based_comparisons": rule_results,
        "error_analysis": {
            "total_false_positives": len(false_positives),
            "total_false_negatives": len(false_negatives),
            "top_false_positives": top_fp_details,
            "top_false_negatives": top_fn_details,
        },
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "pairwise_model_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2)
        print(f"\nSaved full results to {out_file}", flush=True)

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pairwise LightGBM Model on Combo 3 Candidates")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    run_pairwise_pipeline(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
    )
