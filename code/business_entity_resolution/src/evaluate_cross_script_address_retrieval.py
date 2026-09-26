"""
Controlled Retrieval Experiment: Cross-Script / Address-Structure Retrieval (Experiment C2).

Goal:
Recover missed true matches whose names have little or no lexical overlap because of
script differences, transliteration, synthetic obfuscation, or severe name distortion,
using structured address evidence.

Evaluates 5 variants independently on top of frozen Combo 3:
- Variant C2-A: Country + shared address number + at least 1 address token (DF <= 500)
- Variant C2-B: Country + shared address number + at least 1 address token (DF <= 1,000)
- Variant C2-C: Country + shared address number + at least 1 address token (DF <= 2,000)
- Variant C2-D: Country + shared pincode + at least 1 address token (DF <= 2,000)
- Variant C2-E: Country + (shared address number OR shared pincode) + at least 1 address token (DF <= 2,000)

Validation split: 5,000 S1 entities, seed=42, full S2 + S3 space (10.3M records).
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


def run_cross_script_retrieval_evaluation(
    data_dir: Path,
    output_file: Path,
    sample_size: int = 5_000,
    seed: int = 42,
) -> None:
    print("=" * 80, flush=True)
    print("EXPERIMENT C2: CROSS-SCRIPT / ADDRESS-STRUCTURE RETRIEVAL", flush=True)
    print(f"Validation sample size: {sample_size:,} S1 entities (seed={seed})", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

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
        all_nums = extract_address_numbers(addr)
        pincodes = extract_pincodes(addr)
        bldg_nums = {n for n in all_nums if not re.match(r"^\d{5,6}$", n)}

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
            "all_nums": all_nums,
            "bldg_nums": bldg_nums,
            "pincodes": pincodes,
            "raw_name": name,
            "raw_addr": addr,
        }

    # Step 3: Load cached indices & true match records
    cache_path = output_file.parent / "canonical_combo3_index_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at {cache_path}.")

    print(f"\n[Step 3] Loading precomputed indices from {cache_path}...", flush=True)
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

    # Step 4: Generate Baseline Combo 3 Candidates and Experiment A/B Secondary Layers
    print("\n[Step 4] Generating baseline Combo 3 candidates & secondary layers A/B...", flush=True)
    t_combo_start = time.time()
    combo3_cands: dict[str, set[str]] = {}
    sec_a_cands: dict[str, set[str]] = {}
    sec_b_cands: dict[str, set[str]] = {}

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
        valid_nums = [n for n in p["all_nums"] if len(n) >= 3 and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 500]
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

        # Experiment A secondary: 2 shared address tokens, DF <= 2,000
        rare_addrs_2000 = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
        if len(rare_addrs_2000) >= 2:
            sec_a_cands[s1_id] = get_token_overlap_candidates(rare_addrs_2000, min_overlap=2)
        else:
            sec_a_cands[s1_id] = set()

        # Experiment B secondary: single name token, len >= 4, DF <= 100
        b_set = set()
        for t in p["info_name_tokens"]:
            if len(t) >= 4 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 100:
                b_set |= idx_name_tokens[(c, t)]
        sec_b_cands[s1_id] = b_set

    combo3_time = time.time() - t_combo_start
    print(f"  Combo 3 baseline candidates generated in {combo3_time:.1f}s", flush=True)

    # Combined Combo 3 + A + B
    combo3_plus_ab_cands: dict[str, set[str]] = {
        s1_id: combo3_cands[s1_id] | sec_a_cands[s1_id] | sec_b_cands[s1_id] for s1_id in val_s1_ids
    }

    # Step 5: Identify the 1,096 Missed Matches & the 594 Cross-Script Misses
    print("\n[Step 5] Profiling the 1,096 missed matches and isolating 594 cross-script pairs...", flush=True)
    missed_all_pairs: list[dict[str, Any]] = []
    cross_script_pairs: list[dict[str, Any]] = []
    other_missed_pairs: list[dict[str, Any]] = []

    for s1_id in val_s1_ids:
        gt_set = set(val_gt[s1_id])
        c3_set = combo3_cands[s1_id]
        missed = gt_set - c3_set

        for mid in missed:
            s1_info = s1_parsed[s1_id]
            m_rec = true_match_records.get(mid)
            if not m_rec:
                continue

            m_c = normalize_basic(m_rec["country"])
            m_name = m_rec["business_name"]
            m_addr = m_rec["business_address"]

            m_nums = extract_address_numbers(m_addr)
            m_pins = extract_pincodes(m_addr)
            m_bldg = {n for n in m_nums if not re.match(r"^\d{5,6}$", n)}

            m_toks_n = tokenize(m_name)
            m_info_n = [t for t in m_toks_n if len(t) >= 3 and t not in NAME_STOPWORDS]
            m_toks_a = tokenize(m_addr)
            m_info_a = [t for t in m_toks_a if len(t) >= 3 and t not in ADDR_STOPWORDS]

            shared_info_name = set(s1_info["info_name_tokens"]) & set(m_info_n)
            shared_all_name = set(tokenize(s1_info["raw_name"])) & set(m_toks_n)
            shared_bldg_nums = set(s1_info["bldg_nums"]) & set(m_bldg)
            shared_pins = set(s1_info["pincodes"]) & set(m_pins)
            shared_all_nums = set(s1_info["all_nums"]) & set(m_nums)
            shared_info_addr = set(s1_info["info_addr_tokens"]) & set(m_info_a)
            shared_all_addr = set(tokenize(s1_info["raw_addr"])) & set(m_toks_a)

            name_fuzz = fuzz.ratio(s1_info["name_norm"], normalize_basic(m_name))
            is_cross = is_synthetic_or_cross_script(s1_info["raw_name"]) or is_synthetic_or_cross_script(m_name)

            pair_info = {
                "s1_id": s1_id,
                "match_id": mid,
                "source": "S2" if mid.startswith("S2-") else "S3",
                "s1_name": s1_info["raw_name"],
                "cand_name": m_name,
                "s1_addr": s1_info["raw_addr"],
                "cand_addr": m_addr,
                "s1_country": s1_info["country"],
                "cand_country": m_c,
                "shared_info_name": list(shared_info_name),
                "shared_all_name": list(shared_all_name),
                "shared_bldg_nums": list(shared_bldg_nums),
                "shared_pins": list(shared_pins),
                "shared_all_nums": list(shared_all_nums),
                "shared_info_addr": list(shared_info_addr),
                "shared_all_addr": list(shared_all_addr),
                "name_fuzz": name_fuzz,
                "is_cross_script": is_cross,
            }
            missed_all_pairs.append(pair_info)
            if is_cross:
                cross_script_pairs.append(pair_info)
            else:
                other_missed_pairs.append(pair_info)

    print(f"  Total missed matches analyzed: {len(missed_all_pairs):,}", flush=True)
    print(f"  Cross-script / synthetic missed matches: {len(cross_script_pairs):,} (54.20%)", flush=True)
    print(f"  Other missed matches: {len(other_missed_pairs):,} (45.80%)", flush=True)

    # Step 6: Generate Candidate Sets for Variants C2-A, C2-B, C2-C, C2-D, C2-E
    print("\n[Step 6] Generating candidate sets for C2 variants...", flush=True)

    variant_specs = [
        {
            "id": "Variant C2-A",
            "name": "Country + Shared Address Number + 1 Addr Token (DF <= 500)",
            "num_type": "bldg_nums",
            "addr_df_cap": 500,
        },
        {
            "id": "Variant C2-B",
            "name": "Country + Shared Address Number + 1 Addr Token (DF <= 1,000)",
            "num_type": "bldg_nums",
            "addr_df_cap": 1000,
        },
        {
            "id": "Variant C2-C",
            "name": "Country + Shared Address Number + 1 Addr Token (DF <= 2,000)",
            "num_type": "bldg_nums",
            "addr_df_cap": 2000,
        },
        {
            "id": "Variant C2-D",
            "name": "Country + Shared Pincode + 1 Addr Token (DF <= 2,000)",
            "num_type": "pincodes",
            "addr_df_cap": 2000,
        },
        {
            "id": "Variant C2-E",
            "name": "Country + (Shared Addr Number OR Pincode) + 1 Addr Token (DF <= 2,000)",
            "num_type": "all_nums",
            "addr_df_cap": 2000,
        },
    ]

    variant_cands_map: dict[str, dict[str, set[str]]] = {}

    for v in variant_specs:
        t_v_start = time.time()
        v_id = v["id"]
        num_field = v["num_type"]
        addr_cap = v["addr_df_cap"]

        cand_map_v: dict[str, set[str]] = {}
        for s1_id in val_s1_ids:
            p = s1_parsed[s1_id]
            c = p["country"]
            nums = [n for n in p[num_field] if (c, n) in idx_addr_numbers]
            addr_tokens = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= addr_cap]

            if nums and addr_tokens:
                a_union = set().union(*addr_tokens)
                v_set = set()
                for n in nums:
                    v_set |= (idx_addr_numbers[(c, n)] & a_union)
                cand_map_v[s1_id] = v_set
            else:
                cand_map_v[s1_id] = set()

        variant_cands_map[v_id] = cand_map_v
        print(f"  {v_id} generated in {time.time()-t_v_start:.1f}s", flush=True)

    # Step 7: Evaluate Each Variant Independently Unioned with Combo 3
    print("\n[Step 7] Evaluating each variant independently unioned with Combo 3...", flush=True)

    # Precompute baseline sets
    c3_retrieved_pairs: set[tuple[str, str]] = set()
    for s1_id in val_s1_ids:
        for m in set(val_gt[s1_id]) & combo3_cands[s1_id]:
            c3_retrieved_pairs.add((s1_id, m))

    c3_ab_retrieved_pairs: set[tuple[str, str]] = set()
    for s1_id in val_s1_ids:
        for m in set(val_gt[s1_id]) & combo3_plus_ab_cands[s1_id]:
            c3_ab_retrieved_pairs.add((s1_id, m))

    c3_total_cands = sum(len(c) for c in combo3_cands.values())
    c3_mean_cands = c3_total_cands / len(val_s1_ids)

    cross_script_set = {(p["s1_id"], p["match_id"]) for p in cross_script_pairs}
    other_missed_set = {(p["s1_id"], p["match_id"]) for p in other_missed_pairs}

    results = []

    for v in variant_specs:
        v_id = v["id"]
        v_name = v["name"]
        v_cands = variant_cands_map[v_id]

        # Union with Combo 3
        union_cands: dict[str, set[str]] = {
            s1_id: combo3_cands[s1_id] | v_cands[s1_id] for s1_id in val_s1_ids
        }

        cand_counts = [len(cands) for cands in union_cands.values()]
        total_cands = sum(cand_counts)
        mean_cands = total_cands / len(val_s1_ids)
        median_cands = float(np.median(cand_counts))
        p90_cands = float(np.percentile(cand_counts, 90))
        p95_cands = float(np.percentile(cand_counts, 95))
        p99_cands = float(np.percentile(cand_counts, 99))
        max_cands = int(max(cand_counts))

        union_retrieved_pairs: set[tuple[str, str]] = set()
        for s1_id in val_s1_ids:
            found = set(val_gt[s1_id]) & union_cands[s1_id]
            for m in found:
                union_retrieved_pairs.add((s1_id, m))

        retrieved_count = len(union_retrieved_pairs)
        recall = retrieved_count / total_val_matches

        newly_recovered_c3 = union_retrieved_pairs - c3_retrieved_pairs
        newly_recovered_c3_count = len(newly_recovered_c3)

        newly_recovered_c3ab = union_retrieved_pairs - c3_ab_retrieved_pairs
        newly_recovered_c3ab_count = len(newly_recovered_c3ab)

        added_total_cands = total_cands - c3_total_cands
        added_mean_cands = mean_cands - c3_mean_cands
        efficiency = (newly_recovered_c3_count / added_total_cands) if added_total_cands > 0 else 0.0
        efficiency_mean = (newly_recovered_c3_count / added_mean_cands) if added_mean_cands > 0 else 0.0
        recall_gain_per_1000 = (newly_recovered_c3_count / total_val_matches * 100) / (added_mean_cands / 1000) if added_mean_cands > 0 else 0.0
        affected_s1_count = sum(1 for s1_id in val_s1_ids if len(union_cands[s1_id]) > len(combo3_cands[s1_id]))

        # Breakdown of newly recovered matches
        rec_cross_script = newly_recovered_c3 & cross_script_set
        rec_other_missed = newly_recovered_c3 & other_missed_set

        # Detailed profiling of recovered cross-script matches
        rec_cs_pairs = [p for p in cross_script_pairs if (p["s1_id"], p["match_id"]) in rec_cross_script]
        cs_zero_name_tokens = sum(1 for p in rec_cs_pairs if len(p["shared_all_name"]) == 0)
        cs_zero_name_char = sum(1 for p in rec_cs_pairs if p["name_fuzz"] < 25)
        cs_depend_addr_num = sum(1 for p in rec_cs_pairs if len(p["shared_bldg_nums"]) > 0)
        cs_depend_pincode = sum(1 for p in rec_cs_pairs if len(p["shared_pins"]) > 0)
        cs_have_shared_addr_tok = sum(1 for p in rec_cs_pairs if len(p["shared_info_addr"]) > 0)

        # 10 Representative Recovered Examples
        recovered_examples = []
        for p in rec_cs_pairs[:10]:
            recovered_examples.append({
                "s1_id": p["s1_id"],
                "cand_id": p["match_id"],
                "s1_name": p["s1_name"],
                "cand_name": p["cand_name"],
                "s1_addr": p["s1_addr"],
                "cand_addr": p["cand_addr"],
                "shared_numbers": p["shared_bldg_nums"] or p["shared_pins"],
                "shared_addr_tokens": p["shared_info_addr"],
                "source": p["source"],
                "retrieval_evidence": f"Shared number: {p['shared_bldg_nums'] or p['shared_pins']} + Shared Addr Token: {p['shared_info_addr']}",
            })

        # 10 False Positive Examples from largest candidate additions
        s1_added_counts = [(s1_id, len(v_cands[s1_id] - combo3_cands[s1_id])) for s1_id in val_s1_ids]
        s1_added_counts.sort(key=lambda x: x[1], reverse=True)

        false_positive_ids = []
        for s1_id, added_cnt in s1_added_counts:
            fp_cands = (v_cands[s1_id] - combo3_cands[s1_id]) - set(val_gt[s1_id])
            for cid in fp_cands:
                false_positive_ids.append((s1_id, cid))
                if len(false_positive_ids) >= 10:
                    break
            if len(false_positive_ids) >= 10:
                break

        res_dict = {
            "variant_id": v_id,
            "name": v_name,
            "retrieved_true_matches": retrieved_count,
            "total_val_matches": total_val_matches,
            "recall": recall,
            "newly_recovered_over_c3": newly_recovered_c3_count,
            "newly_recovered_over_c3ab": newly_recovered_c3ab_count,
            "total_candidates": total_cands,
            "mean_candidates_per_s1": mean_cands,
            "median_candidates": median_cands,
            "p90_candidates": p90_cands,
            "p95_candidates": p95_cands,
            "p99_candidates": p99_cands,
            "max_candidates": max_cands,
            "total_added_candidates": added_total_cands,
            "added_mean_candidates": added_mean_cands,
            "efficiency_total": efficiency,
            "efficiency_mean": efficiency_mean,
            "recall_gain_per_1000_cands_s1": recall_gain_per_1000,
            "affected_s1_count": affected_s1_count,
            "recovered_cross_script_count": len(rec_cross_script),
            "recovered_other_missed_count": len(rec_other_missed),
            "cross_script_zero_name_tokens": cs_zero_name_tokens,
            "cross_script_zero_name_char_overlap": cs_zero_name_char,
            "cross_script_depend_addr_num": cs_depend_addr_num,
            "cross_script_depend_pincode": cs_depend_pincode,
            "cross_script_have_shared_addr_tok": cs_have_shared_addr_tok,
            "recovered_examples": recovered_examples,
            "false_positive_pairs": false_positive_ids,
        }
        results.append(res_dict)

        print("\n" + "=" * 80, flush=True)
        print(f"RESULTS FOR {v_id}: {v_name}", flush=True)
        print("=" * 80, flush=True)
        print(f"  Retrieved True Matches:      {retrieved_count:,} / {total_val_matches:,} ({recall*100:.2f}%)", flush=True)
        print(f"  Newly Recovered over C3:     +{newly_recovered_c3_count:,}", flush=True)
        print(f"  Newly Recovered over C3+A+B: +{newly_recovered_c3ab_count:,}", flush=True)
        print(f"    - From Cross-Script (594): {len(rec_cross_script):,} / 594 ({len(rec_cross_script)/594*100:.1f}%)", flush=True)
        print(f"    - From Other Misses (502): {len(rec_other_missed):,} / 502 ({len(rec_other_missed)/502*100:.1f}%)", flush=True)
        print(f"  Total Candidates:            {total_cands:,}", flush=True)
        print(f"  Mean Candidates / S1:        {mean_cands:.1f}", flush=True)
        print(f"  Median:                      {median_cands:.0f}", flush=True)
        print(f"  p90:                         {p90_cands:.0f}", flush=True)
        print(f"  p95:                         {p95_cands:.0f}", flush=True)
        print(f"  p99:                         {p99_cands:.0f}", flush=True)
        print(f"  Max:                         {max_cands:,}", flush=True)
        print(f"  Total Added Candidates:      +{added_total_cands:,}", flush=True)
        print(f"  Added Candidates / S1:       +{added_mean_cands:.2f}", flush=True)
        print(f"  Efficiency (Matches / Cand): {efficiency:.6f}", flush=True)
        print(f"  Recall Gain per +1k Cands/S1:{recall_gain_per_1000:.4f}%", flush=True)
        print(f"  Affected S1 Entities:        {affected_s1_count:,} / {len(val_s1_ids):,} ({affected_s1_count/len(val_s1_ids)*100:.1f}%)", flush=True)

    # Fetch textual records for the 10 false-positive examples from S2/S3
    print("\n[Step 8] Resolving text records for false positive examples...", flush=True)
    needed_fp_eids = {cid for r in results for s1_id, cid in r["false_positive_pairs"][:10]}
    fp_records: dict[str, dict[str, str]] = {}

    for s_path in [s2_path, s3_path]:
        if len(fp_records) >= len(needed_fp_eids):
            break
        for chunk in pd.read_csv(
            s_path,
            sep="\t",
            dtype="string",
            keep_default_na=False,
            chunksize=500_000,
        ):
            mask = chunk["entity_id"].isin(needed_fp_eids)
            for eid, name, addr, country in zip(
                chunk.loc[mask, "entity_id"],
                chunk.loc[mask, "business_name"],
                chunk.loc[mask, "business_address"],
                chunk.loc[mask, "country"],
            ):
                fp_records[eid] = {
                    "entity_id": eid,
                    "business_name": name,
                    "business_address": addr,
                    "country": country,
                }
            if len(fp_records) >= len(needed_fp_eids):
                break

    for r in results:
        detailed_fps = []
        for s1_id, cid in r["false_positive_pairs"][:10]:
            s1_info = s1_parsed[s1_id]
            cand_info = fp_records.get(cid, {"business_name": "UNKNOWN", "business_address": "UNKNOWN"})
            detailed_fps.append({
                "s1_id": s1_id,
                "cand_id": cid,
                "s1_name": s1_info["raw_name"],
                "cand_name": cand_info["business_name"],
                "s1_addr": s1_info["raw_addr"],
                "cand_addr": cand_info["business_address"],
            })
        r["false_positive_examples"] = detailed_fps

    # Save output JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved complete results to {output_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Experiment C2: Cross-Script / Address-Structure Retrieval")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-file", type=Path, default=Path("output/cross_script_address_retrieval_results.json"))
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cross_script_retrieval_evaluation(
        data_dir=args.data_dir,
        output_file=args.output_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )
