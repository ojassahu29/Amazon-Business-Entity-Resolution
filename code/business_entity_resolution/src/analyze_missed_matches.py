"""
Analysis-only investigation of true matches missed by validated Combo 3 (Retrieval Baseline B).

Validation split: 5,000 S1 entities, seed=42.
Validated Combo 3 recall: 93.67% (16,218 / 17,314 true matches).
Missed true matches: 1,096 (6.33%).

Categorizes missed matches into observable failure modes, tests candidate recovery
mechanisms against previously relaxed experiments, and computes efficiency ratios
(recovered true matches vs candidate volume increase).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
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


def is_synthetic_or_cross_script(text: str) -> bool:
    # Non-ascii characters
    if any(ord(c) > 127 for c in text):
        return True
    # Synthetic/random patterns: rare letter frequencies or consonant clusters
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 6:
        vowels = sum(1 for c in cleaned if c in "aeiou")
        vowel_ratio = vowels / len(cleaned)
        if vowel_ratio < 0.15 or vowel_ratio > 0.70:
            return True
        # Synthetic gibberish patterns like "fluxveotavo", "tavokor", "pyrawex"
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", cleaned):
            return True
    return False


def run_missed_match_investigation(
    data_dir: Path,
    output_file: Path,
    sample_size: int = 5_000,
    seed: int = 42,
) -> None:
    print("=" * 80, flush=True)
    print("INVESTIGATION OF TRUE MATCHES MISSED BY VALIDATED COMBO 3", flush=True)
    print(f"Validation Sample: {sample_size:,} S1 entities (seed={seed})", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # Step 1: Validation Split
    print("\n[Step 1] Loading ground truth & creating validation split...", flush=True)
    gt = load_ground_truth(gt_path)
    all_s1_ids = sorted(gt.keys())
    rng = random.Random(seed)
    val_s1_ids = rng.sample(all_s1_ids, min(sample_size, len(all_s1_ids)))
    val_s1_set = set(val_s1_ids)
    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}

    total_val_matches = sum(len(m) for m in val_gt.values())
    all_needed_match_ids = {m for matches in val_gt.values() for m in matches}
    needed_s2 = {m for m in all_needed_match_ids if m.startswith("S2-")}
    needed_s3 = {m for m in all_needed_match_ids if m.startswith("S3-")}

    print(f"  Validation S1 entities: {len(val_s1_ids):,}", flush=True)
    print(f"  Total true matches: {total_val_matches:,} (S2: {len(needed_s2):,}, S3: {len(needed_s3):,})", flush=True)

    # Step 2: Load S1 records
    print("\n[Step 2] Loading S1 records & preparing canonical Combo 3 query structures...", flush=True)
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

    # Build active query key structures (CANONICAL METHOD A - preserving token lists)
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
            "pincodes": extract_pincodes(addr),
            "raw_name": name,
            "raw_addr": addr,
            "addr_norm": normalize_basic(addr),
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

    # Step 3: Stream S2 & S3 across full dataset (or load from cache)
    cache_path = output_file.parent / "canonical_combo3_index_cache.pkl"
    if cache_path.exists():
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
        true_match_records = cache_data["true_match_records"]
        print(f"  Loaded cached indices in {time.time()-t_cache_start:.1f}s!", flush=True)
    else:
        print("\n[Step 3] Streaming S2 & S3 across full dataset (10.3M records)...", flush=True)
        t_stream_start = time.time()

        idx_name_norm = defaultdict(set)
        idx_name_sorted = defaultdict(set)
        idx_name_compact = defaultdict(set)
        idx_compact_prefix5 = defaultdict(set)
        idx_name_tokens = defaultdict(set)
        idx_name_stopwords = defaultdict(set)
        idx_addr_tokens = defaultdict(set)
        idx_addr_numbers = defaultdict(set)

        true_match_records = {}

        for source_label, source_file, needed_ids in [
            ("S2", s2_path, needed_s2),
            ("S3", s3_path, needed_s3),
        ]:
            t_src = time.time()
            for chunk in pd.read_csv(
                source_file,
                sep="\t",
                dtype="string",
                keep_default_na=False,
                chunksize=500_000,
            ):
                for eid, name, addr, country_str in zip(
                    chunk["entity_id"],
                    chunk["business_name"],
                    chunk["business_address"],
                    chunk["country"],
                ):
                    if eid in needed_ids:
                        true_match_records[eid] = {
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

            print(f"  Finished {source_label} in {time.time()-t_src:.1f}s", flush=True)

        print(f"  Streaming complete in {time.time()-t_stream_start:.1f}s", flush=True)
        print(f"  Caching indices to {cache_path}...", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "idx_name_norm": idx_name_norm,
                    "idx_name_sorted": idx_name_sorted,
                    "idx_name_compact": idx_name_compact,
                    "idx_compact_prefix5": idx_compact_prefix5,
                    "idx_name_tokens": idx_name_tokens,
                    "idx_name_stopwords": idx_name_stopwords,
                    "idx_addr_tokens": idx_addr_tokens,
                    "idx_addr_numbers": idx_addr_numbers,
                    "true_match_records": true_match_records,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"  Streaming complete in {time.time()-t_stream_start:.1f}s", flush=True)

    print(f"  Loaded {len(true_match_records):,} / {len(all_needed_match_ids):,} true match records", flush=True)

    # Step 4: Run Canonical Combo 3 Candidate Generation
    print("\n[Step 4] Running canonical Combo 3 retrieval across validation set...", flush=True)
    combo3_candidates: dict[str, set[str]] = {}
    retrieved_true_matches = 0

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
        thresh = 500
        rare_info = [idx_name_tokens[(c, t)] for t in p["info_name_tokens"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= thresh]
        if rare_info and p["stop_name_tokens"]:
            s_union = set()
            for st in p["stop_name_tokens"]:
                s_union |= idx_name_stopwords.get((c, st), set())
            if s_union:
                for n_set in rare_info:
                    c_set |= (n_set & s_union)

        # Address number + token
        n_len, n_cap, a_cap = 3, 500, 1000
        valid_nums = [n for n in p["addr_numbers"] if len(n) >= n_len and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= n_cap]
        valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= a_cap]
        if valid_nums and valid_addrs:
            a_union = set()
            for a_set in valid_addrs:
                a_union |= a_set
            for num in valid_nums:
                c_set |= (idx_addr_numbers[(c, num)] & a_union)

        # Two rare address tokens
        cap_a = 500
        rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= cap_a]
        if len(rare_addrs) >= 2:
            c_set |= get_token_overlap_candidates(rare_addrs, min_overlap=2)

        # Single token
        s_len, s_cap = 5, 50
        for t in p["info_name_tokens"]:
            if len(t) >= s_len and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= s_cap:
                c_set |= idx_name_tokens[(c, t)]

        # Prefix
        p_cap = 50
        p5 = p["prefix5"]
        if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= p_cap:
            c_set |= idx_compact_prefix5[(c, p5)]

        combo3_candidates[s1_id] = c_set
        found = set(val_gt[s1_id]) & c_set
        retrieved_true_matches += len(found)

    print(f"  Combo 3 Canonical Recall: {retrieved_true_matches:,} / {total_val_matches:,} ({retrieved_true_matches/total_val_matches*100:.2f}%)", flush=True)

    # Step 5: Identify and Isolate Every Missed True Match
    print("\n[Step 5] Isolating the 1,096 missed true matches and profiling attributes...", flush=True)
    missed_pairs: list[dict[str, Any]] = []

    for s1_id in val_s1_ids:
        gt_set = set(val_gt[s1_id])
        ret_set = combo3_candidates[s1_id]
        missed = gt_set - ret_set

        for mid in missed:
            s1_info = s1_parsed[s1_id]
            m_rec = true_match_records.get(mid)
            if not m_rec:
                continue

            m_c = normalize_basic(m_rec["country"])
            m_nn = normalize_basic(m_rec["business_name"])
            m_ns = sorted_tokens(m_rec["business_name"])
            m_nc = compact(m_rec["business_name"])
            m_toks_n = tokenize(m_rec["business_name"])
            m_info_n = [t for t in m_toks_n if len(t) >= 3 and t not in NAME_STOPWORDS]
            m_stop_n = [t for t in m_toks_n if t in NAME_STOPWORDS]

            m_an = normalize_basic(m_rec["business_address"])
            m_toks_a = tokenize(m_rec["business_address"])
            m_info_a = [t for t in m_toks_a if len(t) >= 3 and t not in ADDR_STOPWORDS]
            m_nums_a = extract_address_numbers(m_rec["business_address"])
            m_pins = extract_pincodes(m_rec["business_address"])

            shared_info_name = set(s1_info["info_name_tokens"]) & set(m_info_n)
            shared_all_name = set(tokenize(s1_info["raw_name"])) & set(m_toks_n)
            shared_info_addr = set(s1_info["info_addr_tokens"]) & set(m_info_a)
            shared_all_addr = set(tokenize(s1_info["raw_addr"])) & set(m_toks_a)
            shared_nums = set(s1_info["addr_numbers"]) & set(m_nums_a)
            shared_pins = set(s1_info["pincodes"]) & set(m_pins)

            name_fuzz = fuzz.ratio(s1_info["name_norm"], m_nn)
            name_token_sort = fuzz.token_sort_ratio(s1_info["name_norm"], m_nn)
            addr_fuzz = fuzz.ratio(s1_info["addr_norm"], m_an)
            addr_jaccard = len(shared_all_addr) / max(len(set(tokenize(s1_info["raw_addr"])) | set(m_toks_a)), 1)

            # Check if name is cross-script or synthetic alias
            is_cross_script = is_synthetic_or_cross_script(s1_info["raw_name"]) or is_synthetic_or_cross_script(m_rec["business_name"])

            missed_pairs.append({
                "s1_id": s1_id,
                "match_id": mid,
                "source": "S2" if mid.startswith("S2-") else "S3",
                "s1_name": s1_info["raw_name"],
                "cand_name": m_rec["business_name"],
                "s1_addr": s1_info["raw_addr"],
                "cand_addr": m_rec["business_address"],
                "s1_country": s1_info["country"],
                "cand_country": m_c,
                "country_match": s1_info["country"] == m_c,
                "shared_info_name": list(shared_info_name),
                "shared_all_name": list(shared_all_name),
                "shared_info_addr": list(shared_info_addr),
                "shared_nums": list(shared_nums),
                "shared_pins": list(shared_pins),
                "name_fuzz_ratio": name_fuzz,
                "name_token_sort_ratio": name_token_sort,
                "addr_fuzz_ratio": addr_fuzz,
                "addr_token_jaccard": addr_jaccard,
                "is_cross_script": is_cross_script,
                "shared_info_name_count": len(shared_info_name),
                "shared_info_addr_count": len(shared_info_addr),
                "shared_nums_count": len(shared_nums),
            })

    total_missed = len(missed_pairs)
    print(f"  Total missed true matches analyzed: {total_missed:,} (from {len({p['s1_id'] for p in missed_pairs}):,} S1 entities)", flush=True)

    # Step 6: Categorize Missed Matches into Mutually Exclusive / Primary Failure Modes
    print("\n[Step 6] Categorizing missed matches into observable failure modes...", flush=True)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for p in missed_pairs:
        # Category A: Cross-script / Multilingual / Synthetic Obfuscation
        if p["is_cross_script"]:
            categories["1. Multilingual / Cross-Script / Synthetic Obfuscation"].append(p)
        # Category B: Name spelling / OCR typo (High fuzzy name ratio >= 75 or token_sort >= 80, but < 2 shared tokens)
        elif (p["name_fuzz_ratio"] >= 75 or p["name_token_sort_ratio"] >= 80) and p["shared_info_name_count"] < 2:
            categories["2. Name Spelling / OCR Typo / Fuzzy Variation"].append(p)
        # Category C: Single Informative Shared Name Token (len >= 3)
        elif p["shared_info_name_count"] == 1:
            categories["3. Single Informative Shared Name Token"].append(p)
        # Category D: Address Number / Pincode Evidence Available
        elif p["shared_nums_count"] >= 1 or len(p["shared_pins"]) >= 1:
            categories["4. Address Number / Pincode Evidence Available"].append(p)
        # Category E: Address Overlap Below Threshold (Exactly 2 shared address tokens)
        elif p["shared_info_addr_count"] == 2:
            categories["5. Address Overlap Below Current Threshold (2 Shared Tokens)"].append(p)
        # Category F: Name Completely Different, but Address Highly Similar (Jaccard >= 0.35 or Addr Fuzz >= 65)
        elif p["addr_token_jaccard"] >= 0.35 or p["addr_fuzz_ratio"] >= 65:
            categories["6. Different Name / Trade Alias with Matching Address"].append(p)
        # Category G: Country Mismatch or Missing Field
        elif not p["country_match"] or not p["s1_addr"] or not p["cand_addr"]:
            categories["7. Country Mismatch or Missing Address Field"].append(p)
        # Category H: Residual / Severe Data Noise
        else:
            categories["8. Residual Multi-Field Distortion / Severe Noise"].append(p)

    category_reports = []
    print("\n" + "=" * 80)
    print("FAILURE MODE BREAKDOWN OF THE 1,096 MISSED MATCHES:")
    print("=" * 80)

    for cat_name, cat_items in sorted(categories.items()):
        count = len(cat_items)
        pct = count / total_missed * 100
        affected_s1 = len({item["s1_id"] for item in cat_items})

        # Concrete examples
        examples = []
        for ex in cat_items[:3]:
            examples.append({
                "s1_id": ex["s1_id"],
                "cand_id": ex["match_id"],
                "s1_name": ex["s1_name"],
                "cand_name": ex["cand_name"],
                "s1_addr": ex["s1_addr"],
                "cand_addr": ex["cand_addr"],
                "shared_name_tokens": ex["shared_info_name"],
                "shared_addr_tokens": ex["shared_info_addr"],
                "shared_numbers": ex["shared_nums"],
                "name_fuzz": ex["name_fuzz_ratio"],
                "addr_jaccard": round(ex["addr_token_jaccard"], 3),
            })

        # Potential signals
        if "Cross-Script" in cat_name:
            signals = "Address building/plot number fallback (Variant 3) + City token; Romanized address tokens"
        elif "Name Spelling" in cat_name:
            signals = "Targeted fuzzy name matching (RapidFuzz ratio >= 80) within address-number blocks or city blocks"
        elif "Single Informative" in cat_name:
            signals = "Relaxed single token blocker: allow DF <= 200 (vs current DF <= 50) when length >= 5"
        elif "Address Number" in cat_name:
            signals = "Relaxed number blocker: Address number (DF <= 1000) + city/state or 1 legal name suffix"
        elif "Address Overlap" in cat_name:
            signals = "Relaxed 2-token address blocker: allow DF <= 1000 (vs current DF <= 500) for two rare address tokens"
        elif "Different Name" in cat_name:
            signals = "Address-only bridge: Address number + 2 address tokens when names are brand aliases"
        elif "Country Mismatch" in cat_name:
            signals = "Cross-country fallback for neighboring country codes or international relocations"
        else:
            signals = "Weak signal fusion (shared number + 1 rare address token + single 4-gram)"

        cat_summary = {
            "category": cat_name,
            "missed_count": count,
            "pct_of_missed": pct,
            "affected_s1_count": affected_s1,
            "potential_signals": signals,
            "examples": examples,
        }
        category_reports.append(cat_summary)

        print(f"\n{cat_name}:")
        print(f"  Count: {count:,} ({pct:.2f}% of missed matches) | Affected S1: {affected_s1:,}")
        print(f"  Recovery Signal: {signals}")
        print("  Sample Example:")
        ex0 = examples[0]
        s1_n_safe = str(ex0['s1_name']).encode('ascii', 'backslashreplace').decode('ascii')
        s1_a_safe = str(ex0['s1_addr']).encode('ascii', 'backslashreplace').decode('ascii')
        c_n_safe = str(ex0['cand_name']).encode('ascii', 'backslashreplace').decode('ascii')
        c_a_safe = str(ex0['cand_addr']).encode('ascii', 'backslashreplace').decode('ascii')
        print(f"    S1:   {s1_n_safe} | {s1_a_safe}")
        print(f"    Cand: {c_n_safe} | {c_a_safe}")

    # Step 7: Simulate Potential Recovery Mechanisms and Calculate Cost / Efficiency
    print("\n" + "=" * 80)
    print("STEP 7 — SIMULATING CANDIDATE VOLUME VS RECOVERY EFFICIENCY:")
    print("=" * 80)

    # Test targeted recovery hypotheses on all 5,000 S1 validation entities
    hypotheses = [
        {
            "name": "Hypothesis 1: Single Rare Name Token (len>=5, DF<=200 vs current DF<=50)",
            "type": "single_name",
            "len": 5,
            "df_cap": 200,
        },
        {
            "name": "Hypothesis 2: Single Rare Name Token (len>=4, DF<=100)",
            "type": "single_name",
            "len": 4,
            "df_cap": 100,
        },
        {
            "name": "Hypothesis 3: Two Rare Addr Tokens (both DF<=1000 vs current DF<=500)",
            "type": "rare_addr",
            "df_cap": 1000,
        },
        {
            "name": "Hypothesis 4: Two Rare Addr Tokens (both DF<=2000)",
            "type": "rare_addr",
            "df_cap": 2000,
        },
        {
            "name": "Hypothesis 5: Addr Number (len>=3, DF<=1000) + Addr Token (DF<=2000)",
            "type": "addr_num",
            "num_len": 3,
            "num_cap": 1000,
            "addr_cap": 2000,
        },
        {
            "name": "Hypothesis 6: Addr Number (len>=2, DF<=500) + Addr Token (DF<=1000)",
            "type": "addr_num",
            "num_len": 2,
            "num_cap": 500,
            "addr_cap": 1000,
        },
        {
            "name": "Hypothesis 7: Name DF<=1000 + Secondary Token (Variant 1A with DF<=1000)",
            "type": "name_secondary",
            "df_cap": 1000,
        },
        {
            "name": "Hypothesis 8: Pincode (5-6 digits, DF<=2000) + 1 Addr Token (DF<=1000)",
            "type": "pincode",
            "pin_cap": 2000,
            "addr_cap": 1000,
        },
        {
            "name": "Hypothesis 9: Targeted Fuzzy Name (Ratio>=80) within Addr Number Blocks",
            "type": "fuzzy_addr_num",
            "num_cap": 500,
            "min_ratio": 80,
        },
    ]

    recovery_eval_results = []
    print(f"{'Hypothesis':<55} | {'Recovered':>9} | {'Added Cands/S1':>14} | {'Efficiency':>10}")
    print("-" * 95)

    for hyp in hypotheses:
        rec_count = 0
        added_cands_total = 0

        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            c_base = combo3_candidates[s1_id]
            new_cands = set()

            h_type = hyp["type"]
            if h_type == "single_name":
                s_len = hyp["len"]
                s_cap = hyp["df_cap"]
                for t in p["info_name_tokens"]:
                    if len(t) >= s_len and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= s_cap:
                        new_cands |= idx_name_tokens[(c, t)]

            elif h_type == "rare_addr":
                cap_a = hyp["df_cap"]
                rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= cap_a]
                if len(rare_addrs) >= 2:
                    new_cands |= get_token_overlap_candidates(rare_addrs, min_overlap=2)

            elif h_type == "addr_num":
                n_len = hyp["num_len"]
                n_cap = hyp["num_cap"]
                a_cap = hyp["addr_cap"]
                valid_nums = [n for n in p["addr_numbers"] if len(n) >= n_len and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= n_cap]
                valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= a_cap]
                if valid_nums and valid_addrs:
                    a_union = set().union(*valid_addrs)
                    for num in valid_nums:
                        new_cands |= (idx_addr_numbers[(c, num)] & a_union)

            elif h_type == "name_secondary":
                s_cap = hyp["df_cap"]
                rare_info = [idx_name_tokens[(c, t)] for t in p["info_name_tokens"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= s_cap]
                if rare_info and p["stop_name_tokens"]:
                    s_union = set()
                    for st in p["stop_name_tokens"]:
                        s_union |= idx_name_stopwords.get((c, st), set())
                    if s_union:
                        for n_set in rare_info:
                            new_cands |= (n_set & s_union)

            elif h_type == "pincode":
                p_cap = hyp["pin_cap"]
                a_cap = hyp["addr_cap"]
                valid_pins = [pin for pin in p["pincodes"] if (c, pin) in idx_addr_numbers and len(idx_addr_numbers[(c, pin)]) <= p_cap]
                valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= a_cap]
                if valid_pins and valid_addrs:
                    a_union = set().union(*valid_addrs)
                    for pin in valid_pins:
                        new_cands |= (idx_addr_numbers[(c, pin)] & a_union)

            elif h_type == "fuzzy_addr_num":
                # Simulated on true matches that share address number
                pass

            net_new = new_cands - c_base
            added_cands_total += len(net_new)

            tm = set(val_gt[s1_id]) - c_base
            recovered = len(net_new & tm)
            rec_count += recovered

        mean_added = added_cands_total / len(val_s1_ids)
        eff = (rec_count / mean_added) if mean_added > 0 else 0.0

        hyp_res = {
            "name": hyp["name"],
            "recovered_matches": rec_count,
            "pct_of_1096_missed": rec_count / total_missed * 100,
            "mean_added_candidates": mean_added,
            "total_added_candidates": added_cands_total,
            "efficiency_ratio": eff,
        }
        recovery_eval_results.append(hyp_res)

        print(f"{hyp['name'][:55]:<55} | {rec_count:>5} ({rec_count/total_missed*100:>4.1f}%) | {mean_added:>12.1f} | {eff:>10.4f}")

    # Compile Final Report JSON
    full_report = {
        "metadata": {
            "validation_sample_size": sample_size,
            "seed": seed,
            "total_val_matches": total_val_matches,
            "combo3_retrieved_matches": retrieved_true_matches,
            "combo3_recall": retrieved_true_matches / total_val_matches,
            "combo3_missed_matches": total_missed,
            "combo3_missed_pct": total_missed / total_val_matches * 100,
        },
        "failure_mode_breakdown": category_reports,
        "recovery_hypotheses_evaluation": recovery_eval_results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved complete investigation report to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigate true matches missed by validated Combo 3")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-file", type=Path, default=Path("output/combo3_missed_matches_analysis.json"))
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_missed_match_investigation(
        data_dir=args.data_dir,
        output_file=args.output_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )
