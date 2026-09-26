"""
Targeted Retrieval Experiments for Business Entity Resolution.

Runs Experiments 1, 2, 3, and 4 on the SAME 5,000 S1 validation split (seed=42):
- Experiment 1: Relax stopword blocking safely (Variants 1A, 1B, 1C, and Union)
- Experiment 2: Cross-script failure analysis on multilingual missed true matches
- Experiment 3: Targeted fuzzy retrieval feasibility on missed matches
- Experiment 4: Hard-negative and candidate quality / precision check
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
from typing import Any

import numpy as np
import pandas as pd
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

# Geographic tokens extracted from ADDR_STOPWORDS
GEOGRAPHIC_TOKENS: frozenset[str] = frozenset({
    "delhi", "maharashtra", "mumbai", "bangalore",
    "karnataka", "kolkata", "pradesh", "uttar",
    "tamil", "nadu", "bengal", "gujarat", "pune",
    "telangana", "chennai", "hyderabad", "महाराष्ट्र",
})

STRUCTURAL_ADDR_STOPWORDS: frozenset[str] = ADDR_STOPWORDS - GEOGRAPHIC_TOKENS


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
    """Extract distinctive address numbers (flat, plot, building, pin code)."""
    raw_nums = re.findall(r"\b\d+[/\w-]*\b", addr)
    nums = set()
    for n in raw_nums:
        clean_n = n.strip(" ,.-/#")
        if len(clean_n) >= 2 and clean_n not in {"2019", "2020", "2021", "2022", "2023", "2024", "2025"}:
            nums.add(clean_n.lower())
    return nums


def detect_script(text: str) -> str:
    """Detect dominant non-Latin script or Latin."""
    counts: Counter[str] = Counter()
    for ch in text:
        cp = ord(ch)
        if 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            counts["Latin"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["Devanagari"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["Tamil"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["Telugu"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["Kannada"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["Bengali"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["Gujarati"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["Malayalam"] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["Gurmukhi"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["Arabic"] += 1

    if not counts:
        return "Unknown"
    for s, c in counts.most_common():
        if s != "Latin" and c >= 2:
            return s
    return counts.most_common(1)[0][0]


def get_token_overlap_candidates(
    token_sets: list[set[str]],
    min_overlap: int,
) -> set[str]:
    """Fast candidate set intersection for >= min_overlap shared tokens."""
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


def run_targeted_experiments(
    data_dir: Path,
    sample_size: int = 5_000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("TARGETED RETRIEVAL EXPERIMENTS (EXPERIMENTS 1, 2, 3, 4)", flush=True)
    print(f"Dataset dir: {data_dir}", flush=True)
    print(f"Validation sample size: {sample_size:,} S1 entities (seed={seed})", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # 1. Load Ground Truth and create deterministic validation split
    t0 = time.time()
    print("\n[Step 1] Loading ground truth & creating validation split...", flush=True)
    gt = load_ground_truth(gt_path)
    all_s1_ids = sorted(gt.keys())
    rng = random.Random(seed)
    val_s1_ids = rng.sample(all_s1_ids, min(sample_size, len(all_s1_ids)))
    val_s1_set = set(val_s1_ids)

    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}
    singletons = sum(1 for s1_id, matches in val_gt.items() if not matches)
    entities_with_matches = len(val_gt) - singletons
    val_s2_matches = sum(sum(1 for m in matches if m.startswith("S2-")) for matches in val_gt.values())
    val_s3_matches = sum(sum(1 for m in matches if m.startswith("S3-")) for matches in val_gt.values())
    total_val_matches = val_s2_matches + val_s3_matches

    all_needed_match_ids = {m for matches in val_gt.values() for m in matches}
    needed_s2_ids = {m for m in all_needed_match_ids if m.startswith("S2-")}
    needed_s3_ids = {m for m in all_needed_match_ids if m.startswith("S3-")}

    print(f"  Validation S1 entities: {len(val_s1_ids):,}", flush=True)
    print(f"    Singletons: {singletons:,} ({singletons/len(val_s1_ids)*100:.2f}%)", flush=True)
    print(f"    Total true matches: {total_val_matches:,} (S2: {val_s2_matches:,}, S3: {val_s3_matches:,})", flush=True)

    # 2. Load validation S1 records & prepare query key structures
    t0 = time.time()
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

    # Build active keys for Baseline A + Experiment 1
    active_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_geo_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_addr_numbers: dict[tuple[str, str], set[str]] = defaultdict(set)

    s1_parsed: dict[str, dict] = {}

    for s1_id, rec in val_s1_records.items():
        country = normalize_basic(rec["country"])
        name = rec["business_name"]
        addr = rec["business_address"]

        nn = normalize_basic(name)
        ns = sorted_tokens(name)
        nc = compact(name)

        toks_name = tokenize(name)
        toks_addr = tokenize(addr)
        nums_addr = extract_address_numbers(addr)

        info_name = [t for t in toks_name if len(t) >= 3 and t not in NAME_STOPWORDS]
        stop_name = [t for t in toks_name if t in NAME_STOPWORDS]

        info_addr = [t for t in toks_addr if len(t) >= 3 and t not in ADDR_STOPWORDS]
        geo_addr = [t for t in toks_addr if t in GEOGRAPHIC_TOKENS]

        s1_parsed[s1_id] = {
            "country": country,
            "name_norm": nn,
            "name_sorted": ns,
            "name_compact": nc,
            "info_name_tokens": info_name,
            "stop_name_tokens": stop_name,
            "info_addr_tokens": info_addr,
            "geo_addr_tokens": geo_addr,
            "addr_numbers": nums_addr,
        }

        active_name_norm[(country, nn)].add(s1_id)
        active_name_sorted[(country, ns)].add(s1_id)
        active_name_compact[(country, nc)].add(s1_id)

        for t in info_name:
            active_name_tokens[(country, t)].add(s1_id)

        for t in stop_name:
            active_name_stopwords[(country, t)].add(s1_id)

        for t in info_addr:
            active_addr_tokens[(country, t)].add(s1_id)

        for t in geo_addr:
            active_geo_tokens[(country, t)].add(s1_id)

        for num in nums_addr:
            active_addr_numbers[(country, num)].add(s1_id)

    print(f"  Active keys: {len(active_name_norm):,} name_norm, {len(active_name_sorted):,} name_sorted, "
          f"{len(active_name_compact):,} name_compact, {len(active_name_tokens):,} info_name_tokens, "
          f"{len(active_name_stopwords):,} stop_name_tokens, {len(active_addr_tokens):,} info_addr_tokens, "
          f"{len(active_geo_tokens):,} geo_tokens, {len(active_addr_numbers):,} addr_numbers", flush=True)

    # 3. Stream S2 & S3 across full dataset
    print("\n[Step 3] Streaming S2 & S3 across full dataset (10.3M records)...", flush=True)
    t_stream_start = time.time()

    idx_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_geo_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
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
                    if len(t) >= 3:
                        if t not in ADDR_STOPWORDS:
                            k_at = (c, t)
                            if k_at in active_addr_tokens:
                                idx_addr_tokens[k_at].add(eid)
                        elif t in GEOGRAPHIC_TOKENS:
                            k_geo = (c, t)
                            if k_geo in active_geo_tokens:
                                idx_geo_tokens[k_geo].add(eid)

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

    # 4. EXPERIMENT 1: Query Baseline A & Safe Stopword-Relaxed Variants
    print("\n[Step 4] Querying Baseline A & Safe Stopword-Relaxed Variants (Experiment 1)...", flush=True)
    t_query_start = time.time()

    variant_names = [
        "Baseline A Union",
        "Variant 1A: Relaxed Name (1 info + >=1 secondary)",
        "Variant 1B: Relaxed Address (number+1 token OR >=2 tokens with >=1 info)",
        "Variant 1C: Geo Secondary Evidence",
        "Combined Relaxed Blocker (Baseline A + 1A + 1B + 1C)",
    ]

    retrieved: dict[str, int] = {v: 0 for v in variant_names}
    retrieved_s2: dict[str, int] = {v: 0 for v in variant_names}
    retrieved_s3: dict[str, int] = {v: 0 for v in variant_names}

    cand_counts: dict[str, list[int]] = {v: [] for v in variant_names}
    cand_counts_s2: dict[str, list[int]] = {v: [] for v in variant_names}
    cand_counts_s3: dict[str, list[int]] = {v: [] for v in variant_names}

    baseline_found_pairs: set[tuple[str, str]] = set()
    variant_found_pairs: dict[str, set[tuple[str, str]]] = {v: set() for v in variant_names}
    s1_affected: dict[str, int] = {v: 0 for v in variant_names}

    baseline_missed_pairs: list[dict] = []
    baseline_all_candidates_by_s1: dict[str, set[str]] = {}

    for i, s1_id in enumerate(val_s1_ids):
        if (i + 1) % 1000 == 0:
            print(f"  Evaluated {i+1:,} / {len(val_s1_ids):,} entities ({time.time()-t_query_start:.1f}s)...", flush=True)

        parsed = s1_parsed[s1_id]
        c = parsed["country"]
        nn = parsed["name_norm"]
        ns = parsed["name_sorted"]
        nc = parsed["name_compact"]
        s1_info_name = parsed["info_name_tokens"]
        s1_stop_name = parsed["stop_name_tokens"]
        s1_info_addr = parsed["info_addr_tokens"]
        s1_geo_addr = parsed["geo_addr_tokens"]
        s1_nums = parsed["addr_numbers"]

        true_matches = set(val_gt[s1_id])

        # --- Baseline A candidate sets ---
        c1 = idx_name_norm.get((c, nn), set())
        c2 = idx_name_sorted.get((c, ns), set())
        c3 = idx_name_compact.get((c, nc), set())

        # Baseline A Name Tokens (>= 2 shared informative tokens)
        name_sets = [idx_name_tokens.get((c, t), set()) for t in s1_info_name if (c, t) in idx_name_tokens]
        c4 = get_token_overlap_candidates(name_sets, min_overlap=2)

        # Baseline A Address Tokens (>= 3 shared informative tokens)
        addr_sets = [idx_addr_tokens.get((c, t), set()) for t in s1_info_addr if (c, t) in idx_addr_tokens]
        c5 = get_token_overlap_candidates(addr_sets, min_overlap=3)

        c_base = c1 | c2 | c3 | c4 | c5
        baseline_all_candidates_by_s1[s1_id] = c_base

        # --- Variant 1A: Relaxed Name ---
        # >= 1 info token AND >= 1 secondary token (informative or legal/generic)
        # Fast C set intersection: intersect each info token bucket with secondary stopword buckets
        c_1a = set(c_base)
        if s1_stop_name and name_sets:
            # Union of all stopword sets for this S1
            stop_union = set()
            for st in s1_stop_name:
                stop_union |= idx_name_stopwords.get((c, st), set())
            if stop_union:
                for n_set in name_sets:
                    c_1a |= (n_set & stop_union)

        # --- Variant 1B: Relaxed Address ---
        # Part 1: shared address number AND >= 1 shared address token (informative or geo)
        # Part 2: >= 2 address tokens where at least 1 is informative
        c_1b = set(c_base)
        # Part 2: >= 2 address tokens among informative tokens
        c_addr_2info = get_token_overlap_candidates(addr_sets, min_overlap=2)
        c_1b |= c_addr_2info

        # Address tokens union for this S1 (informative + geo)
        all_addr_union = set()
        for a_set in addr_sets:
            all_addr_union |= a_set
        for gt in s1_geo_addr:
            all_addr_union |= idx_geo_tokens.get((c, gt), set())

        # Part 1: Number + >=1 address token
        if s1_nums and all_addr_union:
            for num in s1_nums:
                num_set = idx_addr_numbers.get((c, num), set())
                if num_set:
                    c_1b |= (num_set & all_addr_union)

        # --- Variant 1C: Geographic Secondary Evidence ---
        # Informative token or address number + matching geographic token
        c_1c = set(c_base)
        if s1_geo_addr:
            geo_union = set()
            for gt in s1_geo_addr:
                geo_union |= idx_geo_tokens.get((c, gt), set())

            if geo_union:
                for a_set in addr_sets:
                    c_1c |= (a_set & geo_union)
                if s1_nums:
                    for num in s1_nums:
                        num_set = idx_addr_numbers.get((c, num), set())
                        if num_set:
                            c_1c |= (num_set & geo_union)

        # --- Combined Relaxed Blocker ---
        c_combined = c_base | c_1a | c_1b | c_1c

        variant_cand_map = {
            "Baseline A Union": c_base,
            "Variant 1A: Relaxed Name (1 info + >=1 secondary)": c_1a,
            "Variant 1B: Relaxed Address (number+1 token OR >=2 tokens with >=1 info)": c_1b,
            "Variant 1C: Geo Secondary Evidence": c_1c,
            "Combined Relaxed Blocker (Baseline A + 1A + 1B + 1C)": c_combined,
        }

        for v_name, c_set in variant_cand_map.items():
            s2_cands = {eid for eid in c_set if eid.startswith("S2-")}
            s3_cands = {eid for eid in c_set if eid.startswith("S3-")}

            cand_counts[v_name].append(len(c_set))
            cand_counts_s2[v_name].append(len(s2_cands))
            cand_counts_s3[v_name].append(len(s3_cands))

            found = true_matches & c_set
            f_s2 = sum(1 for m in found if m.startswith("S2-"))
            f_s3 = sum(1 for m in found if m.startswith("S3-"))

            retrieved[v_name] += len(found)
            retrieved_s2[v_name] += f_s2
            retrieved_s3[v_name] += f_s3

            for m in found:
                variant_found_pairs[v_name].add((s1_id, m))
                if v_name == "Baseline A Union":
                    baseline_found_pairs.add((s1_id, m))

            if len(c_set) > len(c_base):
                s1_affected[v_name] += 1

        # Track Baseline A missed matches
        for m in true_matches:
            if m not in c_base:
                baseline_missed_pairs.append({
                    "s1_id": s1_id,
                    "matched_id": m,
                    "source": "S2" if m.startswith("S2-") else "S3",
                    "s1_record": val_s1_records[s1_id],
                    "match_record": match_records.get(m),
                })

    exp1_runtime = time.time() - t_query_start
    print(f"  Completed queries for all variants in {exp1_runtime:.1f}s", flush=True)

    # Compile Experiment 1 results
    exp1_rows = []
    base_retrieved = retrieved["Baseline A Union"]
    for v_name in variant_names:
        c_arr = np.array(cand_counts[v_name])
        n_ret = retrieved[v_name]
        rec = n_ret / total_val_matches if total_val_matches else 0.0
        newly_recovered = n_ret - base_retrieved

        row = {
            "variant": v_name,
            "overall_recall": rec,
            "total_retrieved": n_ret,
            "newly_recovered_matches": newly_recovered,
            "s2_recall": retrieved_s2[v_name] / val_s2_matches if val_s2_matches else 0.0,
            "s3_recall": retrieved_s3[v_name] / val_s3_matches if val_s3_matches else 0.0,
            "mean_candidates": float(np.mean(c_arr)),
            "median_candidates": float(np.median(c_arr)),
            "p95_candidates": float(np.percentile(c_arr, 95)),
            "max_candidates": int(np.max(c_arr)),
            "s1_affected": s1_affected[v_name],
            "runtime_seconds": round(exp1_runtime, 2),
        }
        exp1_rows.append(row)

    # 5. EXPERIMENT 2: Cross-Script Failure Analysis
    print(f"\n[Step 5] Characterizing Cross-Script / Multilingual Missed Matches (Experiment 2)...", flush=True)
    cross_script_cases = []
    for item in baseline_missed_pairs:
        s1_rec = item["s1_record"]
        m_rec = item["match_record"]
        if not m_rec:
            continue
        n1 = s1_rec["business_name"]
        n2 = m_rec["business_name"]
        a1 = s1_rec["business_address"]
        a2 = m_rec["business_address"]

        s_n1 = detect_script(n1)
        s_n2 = detect_script(n2)
        s_a1 = detect_script(a1)
        s_a2 = detect_script(a2)

        if s_n1 != s_n2 or s_a1 != s_a2 or s_n2 != "Latin" or s_n1 != "Latin":
            nums1 = extract_address_numbers(a1)
            nums2 = extract_address_numbers(a2)
            shared_nums = nums1 & nums2

            toks_a1 = set(tokenize(a1))
            toks_a2 = set(tokenize(a2))
            shared_addr_tokens = toks_a1 & toks_a2

            if s_n1 == s_n2 and s_n1 != "Latin":
                nature = "2. same-script spelling variation (Indic script)"
            elif (s_n1 != s_n2) and (shared_nums or len(shared_addr_tokens) >= 2):
                nature = "3. different scripts with useful address evidence"
            elif (s_n1 != s_n2):
                nature = "1. true transliteration"
            else:
                nature = "4. something else"

            cross_script_cases.append({
                "s1_id": item["s1_id"],
                "matched_id": item["matched_id"],
                "source": item["source"],
                "country": s1_rec["country"],
                "s1_name": n1,
                "match_name": n2,
                "s1_name_script": s_n1,
                "match_name_script": s_n2,
                "script_pair": f"{s_n1} -> {s_n2}",
                "s1_addr": a1,
                "match_addr": a2,
                "s1_addr_script": s_a1,
                "match_addr_script": s_a2,
                "address_same_script": (s_a1 == s_a2),
                "shared_address_numbers": list(shared_nums),
                "shared_address_tokens": list(shared_addr_tokens),
                "nature": nature,
            })

    print(f"  Identified {len(cross_script_cases):,} cross-script / multilingual cases", flush=True)

    script_pair_counts = Counter(c["script_pair"] for c in cross_script_cases)
    nature_counts = Counter(c["nature"] for c in cross_script_cases)
    addr_same_script_count = sum(1 for c in cross_script_cases if c["address_same_script"])
    addr_num_overlap_count = sum(1 for c in cross_script_cases if c["shared_address_numbers"])
    addr_token_overlap_count = sum(1 for c in cross_script_cases if len(c["shared_address_tokens"]) >= 2)
    country_counts = Counter(c["country"] for c in cross_script_cases)

    # 6. EXPERIMENT 3: Targeted Fuzzy Retrieval Feasibility
    print("\n[Step 6] Evaluating Targeted Fuzzy Retrieval Feasibility (Experiment 3)...", flush=True)

    cat_spelling: list[dict] = []
    cat_multilingual: list[dict] = []
    cat_addr_similar: list[dict] = []
    cat_single_token: list[dict] = []

    for item in baseline_missed_pairs:
        s1_rec = item["s1_record"]
        m_rec = item["match_record"]
        if not m_rec:
            continue
        n1 = s1_rec["business_name"]
        n2 = m_rec["business_name"]
        a1 = s1_rec["business_address"]
        a2 = m_rec["business_address"]

        nn1 = normalize_basic(n1)
        nn2 = normalize_basic(n2)
        nc1 = compact(n1)
        nc2 = compact(n2)
        ns1 = sorted_tokens(n1)
        ns2 = sorted_tokens(n2)

        t_n1 = set(tokenize(n1))
        t_n2 = set(tokenize(n2))
        info_n1 = set(t for t in t_n1 if len(t) >= 3 and t not in NAME_STOPWORDS)
        info_n2 = set(t for t in t_n2 if len(t) >= 3 and t not in NAME_STOPWORDS)
        shared_info_n = info_n1 & info_n2

        t_a1 = set(tokenize(a1))
        t_a2 = set(tokenize(a2))
        info_a1 = set(t for t in t_a1 if len(t) >= 3 and t not in ADDR_STOPWORDS)
        info_a2 = set(t for t in t_a2 if len(t) >= 3 and t not in ADDR_STOPWORDS)
        shared_info_a = info_a1 & info_a2

        s_n1 = detect_script(n1)
        s_n2 = detect_script(n2)
        is_cross_script = (s_n1 != s_n2 or s_n1 != "Latin" or s_n2 != "Latin")

        n_ratio = fuzz.ratio(nn1, nn2)
        n_tsort = fuzz.token_sort_ratio(nn1, nn2)
        a_ratio = fuzz.ratio(normalize_basic(a1), normalize_basic(a2))
        a_tsort = fuzz.token_sort_ratio(normalize_basic(a1), normalize_basic(a2))

        pair_data = {
            "s1_id": item["s1_id"],
            "matched_id": item["matched_id"],
            "nn1": nn1, "nn2": nn2,
            "nc1": nc1, "nc2": nc2,
            "ns1": ns1, "ns2": ns2,
            "n_ratio": n_ratio, "n_tsort": n_tsort,
            "a_ratio": a_ratio, "a_tsort": a_tsort,
            "shared_info_n": shared_info_n,
            "shared_info_a": shared_info_a,
            "shared_nums": extract_address_numbers(a1) & extract_address_numbers(a2),
        }

        if is_cross_script:
            cat_multilingual.append(pair_data)
        elif len(shared_info_n) == 1:
            cat_single_token.append(pair_data)
        elif n_ratio >= 70 or n_tsort >= 75:
            cat_spelling.append(pair_data)
        elif a_ratio >= 60 or a_tsort >= 60 or len(pair_data["shared_nums"]) >= 2:
            cat_addr_similar.append(pair_data)

    print(f"  Categorized missed true matches: {len(cat_spelling)} spelling/OCR, {len(cat_multilingual)} multilingual, "
          f"{len(cat_addr_similar)} name-diff/addr-similar, {len(cat_single_token)} single-shared-token", flush=True)

    rec_spelling_prefix5 = sum(1 for p in cat_spelling if p["nc1"][:5] == p["nc2"][:5] and len(p["nc1"]) >= 5)
    rec_spelling_tsort80 = sum(1 for p in cat_spelling if p["n_tsort"] >= 80)
    rec_spelling_tsort70 = sum(1 for p in cat_spelling if p["n_tsort"] >= 70)

    rec_single_len5 = sum(1 for p in cat_single_token if any(len(t) >= 5 for t in p["shared_info_n"]))
    rec_single_len6 = sum(1 for p in cat_single_token if any(len(t) >= 6 for t in p["shared_info_n"]))

    rec_multi_addr = sum(1 for p in cat_multilingual if (p["shared_nums"] or len(p["shared_info_a"]) >= 1))
    rec_addr_sim = sum(1 for p in cat_addr_similar if (p["shared_nums"] or len(p["shared_info_a"]) >= 1))

    fuzzy_strategies = [
        {
            "name": "Targeted Fuzzy: Token Sort Ratio >= 80% (Spelling/OCR)",
            "target_category": "Spelling/OCR/Typo",
            "category_size": len(cat_spelling),
            "recovered": rec_spelling_tsort80,
            "recovered_pct": (rec_spelling_tsort80 / len(cat_spelling) * 100) if cat_spelling else 0.0,
            "indexing_mechanism": "Length-constrained 3-gram index or BK-tree over informative tokens",
            "estimated_cand_expansion": "+15-25 candidates / S1",
        },
        {
            "name": "Targeted Fuzzy: Compact Name Prefix (5 chars) (Spelling/OCR)",
            "target_category": "Spelling/OCR/Typo",
            "category_size": len(cat_spelling),
            "recovered": rec_spelling_prefix5,
            "recovered_pct": (rec_spelling_prefix5 / len(cat_spelling) * 100) if cat_spelling else 0.0,
            "indexing_mechanism": "country + compact(name)[:5]",
            "estimated_cand_expansion": "+45-70 candidates / S1",
        },
        {
            "name": "Single Informative Token (Length >= 5) (Single-Token)",
            "target_category": "Single Shared Name Token",
            "category_size": len(cat_single_token),
            "recovered": rec_single_len5,
            "recovered_pct": (rec_single_len5 / len(cat_single_token) * 100) if cat_single_token else 0.0,
            "indexing_mechanism": "country + distinctive token (len >= 5, non-stopword)",
            "estimated_cand_expansion": "+30-60 candidates / S1",
        },
        {
            "name": "Cross-Script Address Evidence (Number + Shared Addr Token)",
            "target_category": "Multilingual / Transliteration",
            "category_size": len(cat_multilingual),
            "recovered": rec_multi_addr,
            "recovered_pct": (rec_multi_addr / len(cat_multilingual) * 100) if cat_multilingual else 0.0,
            "indexing_mechanism": "country + address number + secondary Latin address token",
            "estimated_cand_expansion": "+20-40 candidates / S1",
        },
    ]

    # 7. EXPERIMENT 4: Hard-Negative / Candidate Precision Analysis
    print("\n[Step 7] Analyzing Candidate Precision & Worst-Case Buckets (Experiment 4)...", flush=True)
    base_cands_arr = np.array(cand_counts["Baseline A Union"])
    total_base_cands = int(np.sum(base_cands_arr))
    base_retrieved_matches = retrieved["Baseline A Union"]

    overall_candidate_precision = (base_retrieved_matches / total_base_cands) if total_base_cands else 0.0

    p25 = float(np.percentile(base_cands_arr, 25))
    p50 = float(np.percentile(base_cands_arr, 50))
    p75 = float(np.percentile(base_cands_arr, 75))
    p90 = float(np.percentile(base_cands_arr, 90))
    p95 = float(np.percentile(base_cands_arr, 95))
    p99 = float(np.percentile(base_cands_arr, 99))
    max_cands = int(np.max(base_cands_arr))

    n_gt_1k = int(np.sum(base_cands_arr > 1000))
    n_gt_5k = int(np.sum(base_cands_arr > 5000))
    n_gt_10k = int(np.sum(base_cands_arr > 10000))
    n_gt_20k = int(np.sum(base_cands_arr > 20000))

    s1_cand_pairs = sorted(
        [(s1_id, len(baseline_all_candidates_by_s1[s1_id])) for s1_id in val_s1_ids],
        key=lambda x: x[1],
        reverse=True,
    )

    worst_case_s1s = []
    for s1_id, cnt in s1_cand_pairs[:10]:
        rec = val_s1_records[s1_id]
        true_cnt = len(val_gt[s1_id])
        worst_case_s1s.append({
            "s1_id": s1_id,
            "candidate_count": cnt,
            "true_matches": true_cnt,
            "business_name": rec["business_name"],
            "business_address": rec["business_address"],
            "country": rec["country"],
        })

    # Summary table for final report
    exp_report = {
        "metadata": {
            "validation_sample_size": sample_size,
            "seed": seed,
            "total_val_matches": total_val_matches,
            "val_s2_matches": val_s2_matches,
            "val_s3_matches": val_s3_matches,
        },
        "experiment_1_variants": exp1_rows,
        "experiment_2_cross_script": {
            "total_cross_script_cases": len(cross_script_cases),
            "nature_breakdown": dict(nature_counts),
            "script_pairs": dict(script_pair_counts.most_common(10)),
            "country_distribution": dict(country_counts),
            "address_same_script_count": addr_same_script_count,
            "address_number_overlap_count": addr_num_overlap_count,
            "address_token_overlap_count": addr_token_overlap_count,
            "examples": cross_script_cases[:10],
        },
        "experiment_3_fuzzy": fuzzy_strategies,
        "experiment_4_hard_negatives": {
            "total_candidates": total_base_cands,
            "total_retrieved_true_matches": base_retrieved_matches,
            "candidate_precision": overall_candidate_precision,
            "candidate_percentiles": {
                "p25": p25, "p50": p50, "p75": p75, "p90": p90, "p95": p95, "p99": p99, "max": max_cands,
            },
            "bucket_counts": {
                ">1,000 candidates": n_gt_1k,
                ">5,000 candidates": n_gt_5k,
                ">10,000 candidates": n_gt_10k,
                ">20,000 candidates": n_gt_20k,
            },
            "worst_case_entities": worst_case_s1s,
        },
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "targeted_retrieval_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(exp_report, f, indent=2)
        print(f"\nSaved targeted retrieval results to {out_file}", flush=True)

    return exp_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted Retrieval Experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    results = run_targeted_experiments(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
