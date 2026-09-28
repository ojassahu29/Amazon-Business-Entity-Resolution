"""
Audit script to diagnose the exact reproducibility discrepancy between
evaluate_frequency_aware.py (Experiment A) and evaluate_pairwise_model.py (Experiment B)
on the SAME 5,000 S1 validation entities (seed=42).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

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


def run_audit(data_dir: Path, output_file: Path) -> None:
    print("=" * 80, flush=True)
    print("COMBO 3 REPRODUCIBILITY AUDIT: EXPERIMENT A vs EXPERIMENT B", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # Step 1: Load Ground Truth and 5,000 S1 validation split
    gt = load_ground_truth(gt_path)
    all_s1_ids = sorted(gt.keys())
    rng = random.Random(42)
    val_s1_ids = rng.sample(all_s1_ids, 5_000)
    val_s1_set = set(val_s1_ids)
    val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_ids}
    total_val_matches = sum(len(m) for m in val_gt.values())

    print(f"Validation S1 entities: {len(val_s1_ids):,}", flush=True)
    print(f"Total True Matches: {total_val_matches:,}", flush=True)

    # Step 2: Load S1 records
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

    # Build S1 representations for both methods
    # Method A: exactly as in evaluate_frequency_aware.py (lists)
    s1_parsed_a = {}
    # Method B: exactly as in evaluate_pairwise_model.py (sets)
    s1_parsed_b = {}

    active_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_addr_numbers: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_compact_prefix5: dict[tuple[str, str], set[str]] = defaultdict(set)

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

        # Method A: LISTS
        info_name_a = [t for t in toks_name if len(t) >= 3 and t not in NAME_STOPWORDS]
        stop_name_a = [t for t in toks_name if t in NAME_STOPWORDS]
        info_addr_a = [t for t in toks_addr if len(t) >= 3 and t not in ADDR_STOPWORDS]

        s1_parsed_a[s1_id] = {
            "country": country,
            "name_norm": nn,
            "name_sorted": ns,
            "name_compact": nc,
            "prefix5": p5,
            "info_name_tokens": info_name_a,
            "stop_name_tokens": stop_name_a,
            "info_addr_tokens": info_addr_a,
            "addr_numbers": nums_addr,
        }

        # Method B: SETS
        info_name_b = {t for t in toks_name if len(t) >= 3 and t not in NAME_STOPWORDS}
        stop_name_b = {t for t in toks_name if t in NAME_STOPWORDS}
        info_addr_b = {t for t in toks_addr if len(t) >= 3 and t not in ADDR_STOPWORDS}

        s1_parsed_b[s1_id] = {
            "country": country,
            "name_norm": nn,
            "name_sorted": ns,
            "name_compact": nc,
            "prefix5": p5,
            "info_name_tokens": info_name_b,
            "stop_name_tokens": stop_name_b,
            "info_addr_tokens": info_addr_b,
            "addr_numbers": nums_addr,
        }

        active_name_norm[(country, nn)].add(s1_id)
        active_name_sorted[(country, ns)].add(s1_id)
        active_name_compact[(country, nc)].add(s1_id)
        if p5:
            active_compact_prefix5[(country, p5)].add(s1_id)

        for t in info_name_a:
            active_name_tokens[(country, t)].add(s1_id)
        for t in stop_name_a:
            active_name_stopwords[(country, t)].add(s1_id)
        for t in info_addr_a:
            active_addr_tokens[(country, t)].add(s1_id)
        for num in nums_addr:
            active_addr_numbers[(country, num)].add(s1_id)

    # Step 3: Stream S2 & S3 to populate indices
    print("Streaming S2 & S3 across full dataset...", flush=True)
    idx_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_compact_prefix5: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_numbers: dict[tuple[str, str], set[str]] = defaultdict(set)

    for source_label, source_file in [("S2", s2_path), ("S3", s3_path)]:
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

    # Step 4: Run Combo 3 Candidate Generation for both Method A and Method B
    print("\nGenerating candidate sets using Method A (Frequency-Aware script)...", flush=True)
    cands_a: dict[str, set[str]] = {}
    for s1_id in val_s1_ids:
        p = s1_parsed_a[s1_id]
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

        cands_a[s1_id] = c_set

    print("Generating candidate sets using Method B (Pairwise script)...", flush=True)
    cands_b: dict[str, set[str]] = {}
    for s1_id in val_s1_ids:
        p = s1_parsed_b[s1_id]
        c = p["country"]
        cands_dict: dict[str, int] = {}

        c0 = idx_name_norm.get((c, p["name_norm"]), set())
        for cid in c0:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 0)

        c1 = idx_name_sorted.get((c, p["name_sorted"]), set())
        for cid in c1:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 1)

        c2 = idx_name_compact.get((c, p["name_compact"]), set())
        for cid in c2:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 2)

        name_sets = [idx_name_tokens.get((c, t), set()) for t in p["info_name_tokens"] if (c, t) in idx_name_tokens]
        c3 = get_token_overlap_candidates(name_sets, min_overlap=2)
        for cid in c3:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 3)

        addr_sets = [idx_addr_tokens.get((c, t), set()) for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens]
        c4 = get_token_overlap_candidates(addr_sets, min_overlap=3)
        for cid in c4:
            cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 4)

        rare_info = [idx_name_tokens[(c, t)] for t in p["info_name_tokens"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
        if rare_info and p["stop_name_tokens"]:
            s_union = set()
            for st in p["stop_name_tokens"]:
                s_union |= idx_name_stopwords.get((c, st), set())
            if s_union:
                for n_set in rare_info:
                    for cid in (n_set & s_union):
                        cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 5)

        valid_nums = [n for n in p["addr_numbers"] if len(n) >= 3 and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 500]
        valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
        if valid_nums and valid_addrs:
            a_union = set()
            for a_set in valid_addrs:
                a_union |= a_set
            for num in valid_nums:
                for cid in (idx_addr_numbers[(c, num)] & a_union):
                    cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 6)

        rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr_tokens"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
        if len(rare_addrs) >= 2:
            c7 = get_token_overlap_candidates(rare_addrs, min_overlap=2)
            for cid in c7:
                cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 7)

        for t in p["info_name_tokens"]:
            if len(t) >= 5 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 50:
                for cid in idx_name_tokens[(c, t)]:
                    cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 8)

        p5 = p["prefix5"]
        if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= 50:
            for cid in idx_compact_prefix5[(c, p5)]:
                cands_dict[cid] = cands_dict.get(cid, 0) | (1 << 9)

        cands_b[s1_id] = set(cands_dict.keys())

    # Step 5: Compare Candidate Sets for every S1
    print("\nComparing candidate sets across all 5,000 S1 entities...", flush=True)

    total_cands_a = sum(len(s) for s in cands_a.values())
    total_cands_b = sum(len(s) for s in cands_b.values())

    total_intersection = 0
    total_only_a = 0
    total_only_b = 0
    s1_with_diff = 0

    tm_in_a = 0
    tm_in_b = 0
    tm_only_a = 0
    tm_only_b = 0

    diff_examples = []

    for s1_id in val_s1_ids:
        set_a = cands_a[s1_id]
        set_b = cands_b[s1_id]
        tm = set(val_gt[s1_id])

        inter = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a

        total_intersection += len(inter)
        total_only_a += len(only_a)
        total_only_b += len(only_b)

        tm_a = set_a & tm
        tm_b = set_b & tm
        tm_in_a += len(tm_a)
        tm_in_b += len(tm_b)
        tm_only_a += len(tm_a - tm_b)
        tm_only_b += len(tm_b - tm_a)

        if only_a or only_b:
            s1_with_diff += 1
            if len(diff_examples) < 10:
                diff_examples.append({
                    "s1_id": s1_id,
                    "s1_name": val_s1_records[s1_id]["business_name"],
                    "s1_addr": val_s1_records[s1_id]["business_address"],
                    "s1_info_name_a_list": s1_parsed_a[s1_id]["info_name_tokens"],
                    "s1_info_name_b_set": list(s1_parsed_b[s1_id]["info_name_tokens"]),
                    "s1_info_addr_a_list": s1_parsed_a[s1_id]["info_addr_tokens"],
                    "s1_info_addr_b_set": list(s1_parsed_b[s1_id]["info_addr_tokens"]),
                    "count_a": len(set_a),
                    "count_b": len(set_b),
                    "diff_only_a_count": len(only_a),
                    "diff_only_b_count": len(only_b),
                    "tm_only_a": list(tm_a - tm_b),
                    "tm_only_b": list(tm_b - tm_a),
                    "sample_only_a_cands": list(only_a)[:5],
                })

    global_union = total_intersection + total_only_a + total_only_b
    jaccard = total_intersection / global_union if global_union else 1.0

    results = {
        "total_cands_method_a": total_cands_a,
        "total_cands_method_b": total_cands_b,
        "mean_cands_method_a": total_cands_a / len(val_s1_ids),
        "mean_cands_method_b": total_cands_b / len(val_s1_ids),
        "total_intersection": total_intersection,
        "total_only_a": total_only_a,
        "total_only_b": total_only_b,
        "jaccard_similarity": jaccard,
        "s1_with_diff_count": s1_with_diff,
        "s1_with_diff_pct": s1_with_diff / len(val_s1_ids) * 100,
        "true_matches_in_gt": total_val_matches,
        "true_matches_in_a": tm_in_a,
        "true_matches_in_b": tm_in_b,
        "true_matches_recall_a": tm_in_a / total_val_matches,
        "true_matches_recall_b": tm_in_b / total_val_matches,
        "true_matches_only_in_a": tm_only_a,
        "true_matches_only_in_b": tm_only_b,
        "examples_of_diff": diff_examples,
    }

    print("\n" + "=" * 80)
    print("AUDIT RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Total Candidates in Method A (Frequency-Aware): {total_cands_a:,} (Mean: {total_cands_a/len(val_s1_ids):.1f})")
    print(f"Total Candidates in Method B (Pairwise):        {total_cands_b:,} (Mean: {total_cands_b/len(val_s1_ids):.1f})")
    print(f"Candidate Set Intersection:                     {total_intersection:,}")
    print(f"Candidates ONLY in Method A:                    {total_only_a:,}")
    print(f"Candidates ONLY in Method B:                    {total_only_b:,}")
    print(f"Pairwise Candidate Jaccard:                     {jaccard*100:.2f}%")
    print(f"S1 Entities with Differing Candidate Sets:      {s1_with_diff:,} / 5,000 ({s1_with_diff/5000*100:.2f}%)")
    print(f"True Matches Retrieved by Method A:             {tm_in_a:,} ({tm_in_a/total_val_matches*100:.2f}%)")
    print(f"True Matches Retrieved by Method B:             {tm_in_b:,} ({tm_in_b/total_val_matches*100:.2f}%)")
    print(f"True Matches ONLY in Method A:                  {tm_only_a:,}")
    print(f"True Matches ONLY in Method B:                  {tm_only_b:,}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed audit results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Combo 3 reproducibility discrepancy")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-file", type=Path, default=Path("output/combo3_audit_results.json"))
    args = parser.parse_args()

    run_audit(args.data_dir, args.output_file)
