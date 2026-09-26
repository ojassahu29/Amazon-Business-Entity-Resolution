"""
Frequency-Aware Blocker Experiments for Business Entity Resolution.

Evaluates Experiments A, B, C, D, and E on the SAME validation split:
- 5,000 S1 validation entities (seed=42)
- Full S2 (5.03M rows) + S3 (5.29M rows) dataset
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any, TypedDict

# pyrefly: ignore [missing-import]
import numpy as np
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


class ComboConfig(TypedDict):
    name: str
    use_name_df: int | None
    use_addr_num: tuple[int, int, int] | None
    use_rare_addr: int | None
    use_single: tuple[int, int] | None
    use_prefix: int | None


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


def run_frequency_aware_experiments(
    data_dir: Path,
    sample_size: int = 5_000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("FREQUENCY-AWARE BLOCKER EXPERIMENTS (EXPERIMENTS A, B, C, D, E)", flush=True)
    print(f"Dataset dir: {data_dir}", flush=True)
    print(f"Validation sample size: {sample_size:,} S1 entities (seed={seed})", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # Step 1: Load Ground Truth and create validation split
    t0 = time.time()
    print("\n[Step 1] Loading ground truth & creating validation split...", flush=True)
    gt = load_ground_truth(gt_path)
    all_s1_ids = sorted(gt.keys())
    rng = random.Random(seed)
    val_s1_ids = rng.sample(all_s1_ids, min(sample_size, len(all_s1_ids)))
    val_s1_set = set(val_s1_ids)

    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}
    singletons = sum(1 for s1_id, matches in val_gt.items() if not matches)
    val_s2_matches = sum(sum(1 for m in matches if m.startswith("S2-")) for matches in val_gt.values())
    val_s3_matches = sum(sum(1 for m in matches if m.startswith("S3-")) for matches in val_gt.values())
    total_val_matches = val_s2_matches + val_s3_matches

    all_needed_match_ids = {m for matches in val_gt.values() for m in matches}
    needed_s2_ids = {m for m in all_needed_match_ids if m.startswith("S2-")}
    needed_s3_ids = {m for m in all_needed_match_ids if m.startswith("S3-")}

    print(f"  Validation S1 entities: {len(val_s1_ids):,}", flush=True)
    print(f"    Singletons: {singletons:,} ({singletons/len(val_s1_ids)*100:.2f}%)", flush=True)
    print(f"    Total true matches: {total_val_matches:,} (S2: {val_s2_matches:,}, S3: {val_s3_matches:,})", flush=True)

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

        active_name_norm[(country, nn)].add(s1_id)
        active_name_sorted[(country, ns)].add(s1_id)
        active_name_compact[(country, nc)].add(s1_id)
        if p5:
            active_compact_prefix5[(country, p5)].add(s1_id)

        for t in info_name:
            active_name_tokens[(country, t)].add(s1_id)

        for t in stop_name:
            active_name_stopwords[(country, t)].add(s1_id)

        for t in info_addr:
            active_addr_tokens[(country, t)].add(s1_id)

        for num in nums_addr:
            active_addr_numbers[(country, num)].add(s1_id)

    print(f"  Active keys: {len(active_name_norm):,} name_norm, {len(active_name_sorted):,} name_sorted, "
          f"{len(active_name_compact):,} name_compact, {len(active_compact_prefix5):,} prefix5, "
          f"{len(active_name_tokens):,} info_name_tokens, {len(active_name_stopwords):,} stop_name_tokens, "
          f"{len(active_addr_tokens):,} info_addr_tokens, {len(active_addr_numbers):,} addr_numbers", flush=True)

    # Step 3: Stream S2 & S3 across full dataset
    print("\n[Step 3] Streaming S2 & S3 across full dataset (10.3M records)...", flush=True)
    t_stream_start = time.time()

    idx_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_compact_prefix5: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_numbers: dict[tuple[str, str], set[str]] = defaultdict(set)

    match_records: dict[str, dict] = {}
    total_streamed = 0

    for source_label, source_file, needed_ids in [
        ("S2", s2_path, needed_s2_ids),
        ("S3", s3_path, needed_s3_ids),
    ]:
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
                if eid in needed_ids:
                    match_records[eid] = {
                        "entity_id": eid,
                        "business_name": name,
                        "business_address": addr,
                        "country": country_str,
                    }

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
    print(f"  Cached match records: {len(match_records):,} / {len(all_needed_match_ids):,}", flush=True)
    print(f"  RAM working set: {get_process_memory_mb():.1f} MB", flush=True)

    gc.collect()

    # Step 4: Run Experiments A, B, C, D, E
    print("\n[Step 4] Querying & Evaluating Frequency-Aware Blockers...", flush=True)
    t_eval_start = time.time()

    # Precompute Baseline A candidates
    baseline_a_candidates: dict[str, set[str]] = {}
    baseline_retrieved = 0
    baseline_retrieved_s2 = 0
    baseline_retrieved_s3 = 0

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

        c_base = c1 | c2 | c3 | c4 | c5
        baseline_a_candidates[s1_id] = c_base

        tm = set(val_gt[s1_id])
        found = tm & c_base
        baseline_retrieved += len(found)
        baseline_retrieved_s2 += sum(1 for m in found if m.startswith("S2-"))
        baseline_retrieved_s3 += sum(1 for m in found if m.startswith("S3-"))

    base_cands_arr = np.array([len(c) for c in baseline_a_candidates.values()])

    print(f"  Baseline A Union verified: {baseline_retrieved:,} / {total_val_matches:,} "
          f"({baseline_retrieved/total_val_matches*100:.2f}%), Mean cands: {np.mean(base_cands_arr):.1f}", flush=True)

    # Helper function to evaluate any candidate map
    def score_configuration(cand_map: dict[str, set[str]], name: str) -> dict[str, Any]:
        ret_total = 0
        ret_s2 = 0
        ret_s3 = 0
        counts = []
        for s1_id in val_s1_ids:
            c_set = cand_map[s1_id]
            counts.append(len(c_set))
            tm = set(val_gt[s1_id])
            found = tm & c_set
            ret_total += len(found)
            ret_s2 += sum(1 for m in found if m.startswith("S2-"))
            ret_s3 += sum(1 for m in found if m.startswith("S3-"))

        c_arr = np.array(counts)
        rec = ret_total / total_val_matches if total_val_matches else 0.0
        return {
            "name": name,
            "overall_recall": rec,
            "total_retrieved": ret_total,
            "newly_recovered": ret_total - baseline_retrieved,
            "s2_recall": ret_s2 / val_s2_matches if val_s2_matches else 0.0,
            "s3_recall": ret_s3 / val_s3_matches if val_s3_matches else 0.0,
            "mean_cands": float(np.mean(c_arr)),
            "median_cands": float(np.median(c_arr)),
            "p90_cands": float(np.percentile(c_arr, 90)),
            "p95_cands": float(np.percentile(c_arr, 95)),
            "p99_cands": float(np.percentile(c_arr, 99)),
            "max_cands": int(np.max(c_arr)),
            "total_cands": int(np.sum(c_arr)),
        }

    # =========================================================================
    # EXPERIMENT A: Frequency-aware name recovery
    # Variant 1A: 1 info token (with DF <= threshold) + >= 1 secondary token
    # =========================================================================
    print("\n--- Running Experiment A: Frequency-aware name recovery ---", flush=True)
    exp_a_results = []
    name_df_thresholds = [10, 50, 100, 500, 1000]

    exp_a_cand_maps: dict[int, dict[str, set[str]]] = {}

    for df_thresh in name_df_thresholds:
        cand_map_a: dict[str, set[str]] = {}
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            c_set = set(baseline_a_candidates[s1_id])

            # Filter informative tokens by DF <= df_thresh
            rare_info_sets = [
                idx_name_tokens[(c, t)] for t in p["info_name_tokens"]
                if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= df_thresh
            ]

            if rare_info_sets and p["stop_name_tokens"]:
                stop_union = set()
                for st in p["stop_name_tokens"]:
                    stop_union |= idx_name_stopwords.get((c, st), set())
                if stop_union:
                    for n_set in rare_info_sets:
                        c_set |= (n_set & stop_union)

            cand_map_a[s1_id] = c_set

        exp_a_cand_maps[df_thresh] = cand_map_a
        res = score_configuration(cand_map_a, f"Exp A: Name DF <= {df_thresh}")
        exp_a_results.append(res)
        print(f"  DF <= {df_thresh:>4}: Recall={res['overall_recall']*100:.2f}% (+{res['newly_recovered']:,}), "
              f"Mean cands={res['mean_cands']:.1f}, Median={res['median_cands']:.0f}, p95={res['p95_cands']:.0f}, Max={res['max_cands']:,}", flush=True)

    # =========================================================================
    # EXPERIMENT B: Frequency-aware address recovery
    # 1. Address number (len >= 3 or 4, DF <= cap) + rare address token (DF <= cap)
    # 2. Address number (DF <= cap) + rare locality/street token
    # 3. Two rare address tokens (both DF <= cap)
    # 4. Postal / pincode (5 or 6 digits) + rare address token
    # =========================================================================
    print("\n--- Running Experiment B: Frequency-aware address recovery ---", flush=True)
    exp_b_results = []
    exp_b_cand_maps: dict[str, dict[str, set[str]]] = {}

    # Test B1: Number length >= 3, Number DF <= 200, Addr token DF <= 500
    for num_len, num_cap, addr_cap in [
        (3, 100, 200),
        (3, 200, 500),
        (3, 500, 1000),
        (4, 500, 1000),
    ]:
        b_name = f"Exp B: Number(len>={num_len}, DF<={num_cap}) + AddrToken(DF<={addr_cap})"
        cand_map_b: dict[str, set[str]] = {}
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            c_set = set(baseline_a_candidates[s1_id])

            valid_nums = [
                n for n in p["addr_numbers"]
                if len(n) >= num_len and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= num_cap
            ]
            valid_addr_tokens = [
                idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"]
                if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= addr_cap
            ]

            if valid_nums and valid_addr_tokens:
                addr_union = set()
                for a_set in valid_addr_tokens:
                    addr_union |= a_set
                for n in valid_nums:
                    c_set |= (idx_addr_numbers[(c, n)] & addr_union)

            cand_map_b[s1_id] = c_set

        exp_b_cand_maps[b_name] = cand_map_b
        res = score_configuration(cand_map_b, b_name)
        exp_b_results.append(res)
        print(f"  {b_name}: Recall={res['overall_recall']*100:.2f}% (+{res['newly_recovered']:,}), "
              f"Mean cands={res['mean_cands']:.1f}, Median={res['median_cands']:.0f}, p95={res['p95_cands']:.0f}, Max={res['max_cands']:,}", flush=True)

    # Test B2: Two rare address tokens (min overlap 2 where both have DF <= cap)
    for cap_addr in [100, 200, 500]:
        b_name = f"Exp B: Two Rare Addr Tokens (both DF<={cap_addr})"
        cand_map_b = {}
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            c_set = set(baseline_a_candidates[s1_id])

            rare_addr_sets = [
                idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"]
                if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= cap_addr
            ]
            if len(rare_addr_sets) >= 2:
                c_set |= get_token_overlap_candidates(rare_addr_sets, min_overlap=2)

            cand_map_b[s1_id] = c_set

        exp_b_cand_maps[b_name] = cand_map_b
        res = score_configuration(cand_map_b, b_name)
        exp_b_results.append(res)
        print(f"  {b_name}: Recall={res['overall_recall']*100:.2f}% (+{res['newly_recovered']:,}), "
              f"Mean cands={res['mean_cands']:.1f}, Median={res['median_cands']:.0f}, p95={res['p95_cands']:.0f}, Max={res['max_cands']:,}", flush=True)

    # Test B3: Postal / Pin code (5 or 6 digits) + Addr token (DF <= 500)
    b_name = "Exp B: Postal/PinCode(5-6 digits) + AddrToken(DF<=500)"
    cand_map_b = {}
    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]
        c_set = set(baseline_a_candidates[s1_id])

        pincodes = [
            n for n in p["addr_numbers"]
            if re.match(r"^\d{5,6}$", n) and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 2000
        ]
        valid_addr_tokens = [
            idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"]
            if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500
        ]
        if pincodes and valid_addr_tokens:
            addr_union = set()
            for a_set in valid_addr_tokens:
                addr_union |= a_set
            for pin in pincodes:
                c_set |= (idx_addr_numbers[(c, pin)] & addr_union)

        cand_map_b[s1_id] = c_set

    exp_b_cand_maps[b_name] = cand_map_b
    res = score_configuration(cand_map_b, b_name)
    exp_b_results.append(res)
    print(f"  {b_name}: Recall={res['overall_recall']*100:.2f}% (+{res['newly_recovered']:,}), "
          f"Mean cands={res['mean_cands']:.1f}, Median={res['median_cands']:.0f}, p95={res['p95_cands']:.0f}, Max={res['max_cands']:,}", flush=True)

    # =========================================================================
    # EXPERIMENT C: Targeted single-token recovery
    # Single token length >= 5 or >= 6, with DF <= 10, 50, 100
    # =========================================================================
    print("\n--- Running Experiment C: Targeted single-token recovery ---", flush=True)
    exp_c_results = []
    exp_c_cand_maps: dict[str, dict[str, set[str]]] = {}

    for min_len in [5, 6]:
        for cap_df in [10, 50, 100]:
            c_name = f"Exp C: Single Token (len>={min_len}, DF<={cap_df})"
            cand_map_c = {}
            for s1_id in val_s1_ids:
                p = s1_parsed[s1_id]
                c = p["country"]
                c_set = set(baseline_a_candidates[s1_id])

                rare_single = [
                    idx_name_tokens[(c, t)] for t in p["info_name_tokens"]
                    if len(t) >= min_len and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= cap_df
                ]
                for s in rare_single:
                    c_set |= s

                cand_map_c[s1_id] = c_set

            exp_c_cand_maps[c_name] = cand_map_c
            res = score_configuration(cand_map_c, c_name)
            exp_c_results.append(res)
            print(f"  {c_name}: Recall={res['overall_recall']*100:.2f}% (+{res['newly_recovered']:,}), "
                  f"Mean cands={res['mean_cands']:.1f}, Median={res['median_cands']:.0f}, p95={res['p95_cands']:.0f}, Max={res['max_cands']:,}", flush=True)

    # =========================================================================
    # EXPERIMENT D: Compact prefix recovery
    # Compact name 5-char prefix with DF <= cap
    # =========================================================================
    print("\n--- Running Experiment D: Compact prefix recovery ---", flush=True)
    exp_d_results = []
    exp_d_cand_maps: dict[str, dict[str, set[str]]] = {}

    for cap_df in [10, 50, 100, 500]:
        d_name = f"Exp D: Compact Prefix 5 (DF<={cap_df})"
        cand_map_d = {}
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            c_set = set(baseline_a_candidates[s1_id])

            p5 = p["prefix5"]
            if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= cap_df:
                c_set |= idx_compact_prefix5[(c, p5)]

            cand_map_d[s1_id] = c_set

        exp_d_cand_maps[d_name] = cand_map_d
        res = score_configuration(cand_map_d, d_name)
        exp_d_results.append(res)
        print(f"  {d_name}: Recall={res['overall_recall']*100:.2f}% (+{res['newly_recovered']:,}), "
              f"Mean cands={res['mean_cands']:.1f}, Median={res['median_cands']:.0f}, p95={res['p95_cands']:.0f}, Max={res['max_cands']:,}", flush=True)

    # =========================================================================
    # EXPERIMENT E: Combine the best frequency-aware mechanisms (Pareto frontier)
    # =========================================================================
    print("\n--- Running Experiment E: Combined frequency-aware configurations ---", flush=True)
    exp_e_results = []

    # Selected component candidates
    # Best Name: DF <= 50, DF <= 100
    # Best Address: Number(len>=3, DF<=200)+Addr(DF<=500) OR TwoRare(DF<=200)
    # Best Single Token: len>=5, DF<=50 OR len>=6, DF<=50
    # Best Prefix: Prefix5(DF<=50)

    combos: list[ComboConfig] = [
        {
            "name": "Combo 1 (Conservative): Base A + Name(DF<=50) + AddrNum(len>=3,DF<=200)+Addr(DF<=500)",
            "use_name_df": 50,
            "use_addr_num": (3, 200, 500),
            "use_rare_addr": None,
            "use_single": None,
            "use_prefix": None,
        },
        {
            "name": "Combo 2 (Moderate): Base A + Name(DF<=100) + AddrNum(len>=3,DF<=200) + TwoRareAddr(DF<=200) + Single(len>=5,DF<=50)",
            "use_name_df": 100,
            "use_addr_num": (3, 200, 500),
            "use_rare_addr": 200,
            "use_single": (5, 50),
            "use_prefix": None,
        },
        {
            "name": "Combo 3 (High-Recall): Base A + Name(DF<=500) + AddrNum(len>=3,DF<=500)+Addr(DF<=1000) + TwoRareAddr(DF<=500) + Single(len>=5,DF<=50) + Prefix5(DF<=50)",
            "use_name_df": 500,
            "use_addr_num": (3, 500, 1000),
            "use_rare_addr": 500,
            "use_single": (5, 50),
            "use_prefix": 50,
        },
        {
            "name": "Combo 4 (Ultra-High-Recall): Base A + Name(DF<=1000) + AddrNum(len>=3,DF<=500) + TwoRareAddr(DF<=500) + Single(len>=5,DF<=100) + Prefix5(DF<=100)",
            "use_name_df": 1000,
            "use_addr_num": (3, 500, 1000),
            "use_rare_addr": 500,
            "use_single": (5, 100),
            "use_prefix": 100,
        },
    ]

    for combo in combos:
        cand_map_combo = {}
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            c_set = set(baseline_a_candidates[s1_id])

            # Name recovery
            if combo["use_name_df"] is not None:
                thresh = combo["use_name_df"]
                rare_info = [idx_name_tokens[(c, t)] for t in p["info_name_tokens"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= thresh]
                if rare_info and p["stop_name_tokens"]:
                    s_union = set()
                    for st in p["stop_name_tokens"]:
                        s_union |= idx_name_stopwords.get((c, st), set())
                    if s_union:
                        for n_set in rare_info:
                            c_set |= (n_set & s_union)

            # Address number + token
            if combo["use_addr_num"] is not None:
                n_len, n_cap, a_cap = combo["use_addr_num"]
                valid_nums = [n for n in p["addr_numbers"] if len(n) >= n_len and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= n_cap]
                valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= a_cap]
                if valid_nums and valid_addrs:
                    a_union = set()
                    for a_set in valid_addrs:
                        a_union |= a_set
                    for num in valid_nums:
                        c_set |= (idx_addr_numbers[(c, num)] & a_union)

            # Two rare address tokens
            if combo["use_rare_addr"] is not None:
                cap_a = combo["use_rare_addr"]
                rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= cap_a]
                if len(rare_addrs) >= 2:
                    c_set |= get_token_overlap_candidates(rare_addrs, min_overlap=2)

            # Single token
            if combo["use_single"] is not None:
                s_len, s_cap = combo["use_single"]
                for t in p["info_name_tokens"]:
                    if len(t) >= s_len and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= s_cap:
                        c_set |= idx_name_tokens[(c, t)]

            # Prefix
            if combo["use_prefix"] is not None:
                p_cap = combo["use_prefix"]
                p5 = p["prefix5"]
                if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= p_cap:
                    c_set |= idx_compact_prefix5[(c, p5)]

            cand_map_combo[s1_id] = c_set

        res = score_configuration(cand_map_combo, combo["name"])
        exp_e_results.append(res)
        print(f"  {combo['name']}:\n    Recall={res['overall_recall']*100:.2f}% (S2: {res['s2_recall']*100:.2f}%, S3: {res['s3_recall']*100:.2f}%) | "
              f"+{res['newly_recovered']:,} matches\n    Mean cands={res['mean_cands']:.1f}, Median={res['median_cands']:.0f}, "
              f"p90={res['p90_cands']:.0f}, p95={res['p95_cands']:.0f}, p99={res['p99_cands']:.0f}, Max={res['max_cands']:,}", flush=True)

    eval_time = time.time() - t_eval_start
    peak_mem = get_process_memory_mb()

    # Compile all results into structured output
    full_output = {
        "metadata": {
            "validation_sample_size": sample_size,
            "seed": seed,
            "total_val_matches": total_val_matches,
            "val_s2_matches": val_s2_matches,
            "val_s3_matches": val_s3_matches,
            "streaming_time_seconds": round(stream_time, 2),
            "evaluation_time_seconds": round(eval_time, 2),
            "peak_memory_mb": round(peak_mem, 1),
        },
        "baseline_a": score_configuration(baseline_a_candidates, "Baseline A Union"),
        "experiment_a_name_frequency": exp_a_results,
        "experiment_b_address_frequency": exp_b_results,
        "experiment_c_single_token": exp_c_results,
        "experiment_d_compact_prefix": exp_d_results,
        "experiment_e_combos": exp_e_results,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "frequency_aware_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2)
        print(f"\nSaved frequency-aware results to {out_path}", flush=True)

    return full_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency-Aware Blocker Experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    results = run_frequency_aware_experiments(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
