"""
Evaluation and failure mode analysis for Baseline A.

Freezes and measures the five-key candidate blocker:
1. country + name_norm
2. country + name_sorted
3. country + name_compact
4. country + informative name tokens (min overlap 2, length >= 3)
5. country + informative address tokens (min overlap 3, length >= 3)
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


def get_process_memory_mb() -> float:
    """Return current process working set memory in MB."""
    try:
        out = subprocess.check_output(
            ["powershell", "-c", f"(Get-Process -Id {os.getpid()}).WorkingSet64 / 1MB"],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def is_non_latin(text: str) -> bool:
    """Check if string contains non-Latin scripts (Devanagari, Tamil, etc.)."""
    for ch in text:
        if ord(ch) > 0x024F:
            return True
    return False


def get_numeric_tokens(text: str) -> set[str]:
    """Extract numeric/digit tokens from text."""
    return set(re.findall(r"\b\d+\b", text))


def token_jaccard(tokens_a: tuple[str, ...], tokens_b: tuple[str, ...]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    sa, sb = set(tokens_a), set(tokens_b)
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0


def overlap_coefficient(tokens_a: tuple[str, ...], tokens_b: tuple[str, ...]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    sa, sb = set(tokens_a), set(tokens_b)
    min_size = min(len(sa), len(sb))
    return len(sa & sb) / min_size if min_size > 0 else 0.0


def is_abbreviation(s1: str, s2: str) -> bool:
    """Check if one string is an abbreviation or acronym of the other."""
    t1 = [t for t in tokenize(s1) if t not in NAME_STOPWORDS]
    t2 = [t for t in tokenize(s2) if t not in NAME_STOPWORDS]
    if not t1 or not t2:
        return False
    short, long_tokens = (t1, t2) if len(t1) < len(t2) else (t2, t1)
    if len(short) == 1 and 2 <= len(short[0]) <= len(long_tokens):
        acronym = "".join(t[0] for t in long_tokens if t)
        if short[0] == acronym[: len(short[0])]:
            return True
    return False


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


def get_token_overlap_candidates(
    token_sets: list[set[str]],
    min_overlap: int,
) -> set[str]:
    """Fast candidate set intersection for >= min_overlap shared tokens."""
    if len(token_sets) < min_overlap:
        return set()

    # Sort token sets by size so smaller sets are processed first
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


def run_baseline_evaluation(
    data_dir: Path,
    sample_size: int = 2_500,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("BASELINE A: BLOCKING RECALL & FAILURE MODE EVALUATION", flush=True)
    print(f"Dataset dir: {data_dir}", flush=True)
    print(f"Validation sample size: {sample_size:,} S1 entities", flush=True)
    print(f"Random seed: {seed}", flush=True)
    print("=" * 80, flush=True)

    train_dir = data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # Step 1: Load Ground Truth and create deterministic S1 validation split
    t0 = time.time()
    print("\n[Step 1] Loading ground truth & creating validation split...", flush=True)
    gt = load_ground_truth(gt_path)
    all_s1_ids = sorted(gt.keys())
    print(f"  Loaded {len(all_s1_ids):,} total S1 entities in {time.time()-t0:.1f}s", flush=True)

    rng = random.Random(seed)
    val_s1_ids = rng.sample(all_s1_ids, min(sample_size, len(all_s1_ids)))
    val_s1_set = set(val_s1_ids)

    # Ground truth for validation set
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
    print(f"    Singletons (0 matches): {singletons:,} ({singletons/len(val_s1_ids)*100:.2f}%)", flush=True)
    print(f"    Entities with matches:  {entities_with_matches:,} ({entities_with_matches/len(val_s1_ids)*100:.2f}%)", flush=True)
    print(f"    Total true matches:     {total_val_matches:,} (S2: {val_s2_matches:,}, S3: {val_s3_matches:,})", flush=True)

    # Step 2: Load S1 records for validation set and extract active query keys
    t0 = time.time()
    print("\n[Step 2] Loading validation S1 records & extracting active query keys...", flush=True)
    val_s1_records: dict[str, dict] = {}
    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        mask = chunk["entity_id"].isin(val_s1_set)
        for row in chunk[mask].itertuples(index=False):
            val_s1_records[row.entity_id] = {
                "entity_id": row.entity_id,
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }
        if len(val_s1_records) >= len(val_s1_set):
            break

    # Build active query key sets for S1
    active_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)

    for s1_id, rec in val_s1_records.items():
        country = normalize_basic(rec["country"])
        nn = normalize_basic(rec["business_name"])
        ns = sorted_tokens(rec["business_name"])
        nc = compact(rec["business_name"])

        active_name_norm[(country, nn)].add(s1_id)
        active_name_sorted[(country, ns)].add(s1_id)
        active_name_compact[(country, nc)].add(s1_id)

        for t in tokenize(rec["business_name"]):
            if len(t) >= 3 and t not in NAME_STOPWORDS:
                active_name_tokens[(country, t)].add(s1_id)

        for t in tokenize(rec["business_address"]):
            if len(t) >= 3 and t not in ADDR_STOPWORDS:
                active_addr_tokens[(country, t)].add(s1_id)

    print(f"  Active keys: {len(active_name_norm):,} name_norm, {len(active_name_sorted):,} name_sorted, "
          f"{len(active_name_compact):,} name_compact, {len(active_name_tokens):,} name_tokens, "
          f"{len(active_addr_tokens):,} addr_tokens", flush=True)
    print(f"  Loaded validation S1 records in {time.time()-t0:.1f}s", flush=True)

    # Step 3: Stream S2 and S3, populating candidate hits and caching match records
    print("\n[Step 3] Streaming S2 & S3 across full dataset (10.3M records)...", flush=True)
    t_index_start = time.time()
    mem_start = get_process_memory_mb()

    idx_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    idx_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)

    match_records: dict[str, dict] = {}
    total_s2_rows = 0
    total_s3_rows = 0

    # Stream S2
    print("  Streaming S2 (5.03M rows)...", flush=True)
    t_source = time.time()
    for chunk in pd.read_csv(
        s2_path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        total_s2_rows += len(chunk)
        for row in chunk.itertuples(index=False):
            eid = row.entity_id
            if eid in needed_s2_ids:
                match_records[eid] = {
                    "entity_id": eid,
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country,
                }
            country = normalize_basic(row.country)
            nn = normalize_basic(row.business_name)
            k_nn = (country, nn)
            if k_nn in active_name_norm:
                idx_name_norm[k_nn].add(eid)

            ns = sorted_tokens(row.business_name)
            k_ns = (country, ns)
            if k_ns in active_name_sorted:
                idx_name_sorted[k_ns].add(eid)

            nc = compact(row.business_name)
            k_nc = (country, nc)
            if k_nc in active_name_compact:
                idx_name_compact[k_nc].add(eid)

            for t in tokenize(row.business_name):
                if len(t) >= 3 and t not in NAME_STOPWORDS:
                    k_nt = (country, t)
                    if k_nt in active_name_tokens:
                        idx_name_tokens[k_nt].add(eid)

            for t in tokenize(row.business_address):
                if len(t) >= 3 and t not in ADDR_STOPWORDS:
                    k_at = (country, t)
                    if k_at in active_addr_tokens:
                        idx_addr_tokens[k_at].add(eid)
    print(f"  Processed {total_s2_rows:,} S2 rows in {time.time()-t_source:.1f}s", flush=True)

    # Stream S3
    print("  Streaming S3 (5.29M rows)...", flush=True)
    t_source = time.time()
    for chunk in pd.read_csv(
        s3_path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        total_s3_rows += len(chunk)
        for row in chunk.itertuples(index=False):
            eid = row.entity_id
            if eid in needed_s3_ids:
                match_records[eid] = {
                    "entity_id": eid,
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country,
                }
            country = normalize_basic(row.country)
            nn = normalize_basic(row.business_name)
            k_nn = (country, nn)
            if k_nn in active_name_norm:
                idx_name_norm[k_nn].add(eid)

            ns = sorted_tokens(row.business_name)
            k_ns = (country, ns)
            if k_ns in active_name_sorted:
                idx_name_sorted[k_ns].add(eid)

            nc = compact(row.business_name)
            k_nc = (country, nc)
            if k_nc in active_name_compact:
                idx_name_compact[k_nc].add(eid)

            for t in tokenize(row.business_name):
                if len(t) >= 3 and t not in NAME_STOPWORDS:
                    k_nt = (country, t)
                    if k_nt in active_name_tokens:
                        idx_name_tokens[k_nt].add(eid)

            for t in tokenize(row.business_address):
                if len(t) >= 3 and t not in ADDR_STOPWORDS:
                    k_at = (country, t)
                    if k_at in active_addr_tokens:
                        idx_addr_tokens[k_at].add(eid)
    print(f"  Processed {total_s3_rows:,} S3 rows in {time.time()-t_source:.1f}s", flush=True)

    total_indexed = total_s2_rows + total_s3_rows
    index_time = time.time() - t_index_start
    mem_after_index = get_process_memory_mb()
    print(f"  Total records streamed: {total_indexed:,} in {index_time:.1f}s", flush=True)
    print(f"  RAM: start={mem_start:.1f} MB, after_index={mem_after_index:.1f} MB", flush=True)
    print(f"  Cached true match records: {len(match_records):,} / {len(all_needed_match_ids):,}", flush=True)

    gc.collect()

    # Step 4: Evaluate Each Blocker Independently and the Union
    print("\n[Step 4] Querying candidates & evaluating recall for each blocker and union...", flush=True)
    t_eval_start = time.time()

    blocker_keys = [
        "1: country + name_norm",
        "2: country + name_sorted",
        "3: country + name_compact",
        "4: country + informative name tokens",
        "5: country + informative address tokens",
        "UNION (all 5 blockers)",
    ]

    cand_counts_s2: dict[str, list[int]] = {k: [] for k in blocker_keys}
    cand_counts_s3: dict[str, list[int]] = {k: [] for k in blocker_keys}
    cand_counts_all: dict[str, list[int]] = {k: [] for k in blocker_keys}

    retrieved_s2: dict[str, int] = {k: 0 for k in blocker_keys}
    retrieved_s3: dict[str, int] = {k: 0 for k in blocker_keys}
    retrieved_all: dict[str, int] = {k: 0 for k in blocker_keys}

    missed_pairs: list[dict] = []
    found_by_which: dict[str, set[str]] = defaultdict(set)

    for i, s1_id in enumerate(val_s1_ids):
        if (i + 1) % 500 == 0:
            print(f"  Queried {i+1:,} / {len(val_s1_ids):,} entities ({time.time()-t_eval_start:.1f}s)...", flush=True)

        s1_rec = val_s1_records[s1_id]
        country = normalize_basic(s1_rec["country"])
        nn = normalize_basic(s1_rec["business_name"])
        ns = sorted_tokens(s1_rec["business_name"])
        nc = compact(s1_rec["business_name"])

        true_matches = val_gt[s1_id]
        true_s2 = {m for m in true_matches if m.startswith("S2-")}
        true_s3 = {m for m in true_matches if m.startswith("S3-")}

        # Blocker 1
        c1 = idx_name_norm.get((country, nn), set())

        # Blocker 2
        c2 = idx_name_sorted.get((country, ns), set())

        # Blocker 3
        c3 = idx_name_compact.get((country, nc), set())

        # Blocker 4: >= 2 shared name tokens
        name_toks = [t for t in tokenize(s1_rec["business_name"]) if len(t) >= 3 and t not in NAME_STOPWORDS]
        name_sets = [idx_name_tokens.get((country, t), set()) for t in name_toks if (country, t) in idx_name_tokens]
        c4 = get_token_overlap_candidates(name_sets, min_overlap=2)

        # Blocker 5: >= 3 shared address tokens
        addr_toks = [t for t in tokenize(s1_rec["business_address"]) if len(t) >= 3 and t not in ADDR_STOPWORDS]
        addr_sets = [idx_addr_tokens.get((country, t), set()) for t in addr_toks if (country, t) in idx_addr_tokens]
        c5 = get_token_overlap_candidates(addr_sets, min_overlap=3)

        # Union
        c_union = c1 | c2 | c3 | c4 | c5

        candidates_map = {
            "1: country + name_norm": c1,
            "2: country + name_sorted": c2,
            "3: country + name_compact": c3,
            "4: country + informative name tokens": c4,
            "5: country + informative address tokens": c5,
            "UNION (all 5 blockers)": c_union,
        }

        for b_name, c_set in candidates_map.items():
            s2_cands = {c for c in c_set if c.startswith("S2-")}
            s3_cands = {c for c in c_set if c.startswith("S3-")}

            cand_counts_s2[b_name].append(len(s2_cands))
            cand_counts_s3[b_name].append(len(s3_cands))
            cand_counts_all[b_name].append(len(c_set))

            s2_found = len(true_s2 & s2_cands)
            s3_found = len(true_s3 & s3_cands)

            retrieved_s2[b_name] += s2_found
            retrieved_s3[b_name] += s3_found
            retrieved_all[b_name] += (s2_found + s3_found)

            if b_name != "UNION (all 5 blockers)":
                for m in (true_s2 & s2_cands) | (true_s3 & s3_cands):
                    found_by_which[m].add(b_name)

        # Track missed matches by the union
        for tm in true_matches:
            if tm not in c_union:
                missed_pairs.append({
                    "s1_id": s1_id,
                    "matched_id": tm,
                    "source": "S2" if tm.startswith("S2-") else "S3",
                    "s1_record": s1_rec,
                    "match_record": match_records.get(tm),
                })

    eval_time = time.time() - t_eval_start
    peak_mem = get_process_memory_mb()
    print(f"  Evaluated {len(val_s1_ids):,} entities in {eval_time:.1f}s", flush=True)
    print(f"  Peak memory during evaluation: {peak_mem:.1f} MB", flush=True)

    # Step 5: Compute summary table
    summary_results = []
    for b_name in blocker_keys:
        s2_arr = np.array(cand_counts_s2[b_name])
        s3_arr = np.array(cand_counts_s3[b_name])
        all_arr = np.array(cand_counts_all[b_name])

        res = {
            "blocker": b_name,
            # S2 metrics
            "s2_true_matches": val_s2_matches,
            "s2_retrieved": retrieved_s2[b_name],
            "s2_recall": retrieved_s2[b_name] / val_s2_matches if val_s2_matches else 0.0,
            "s2_mean_cands": float(np.mean(s2_arr)),
            "s2_median_cands": float(np.median(s2_arr)),
            "s2_p95_cands": float(np.percentile(s2_arr, 95)),
            "s2_max_cands": int(np.max(s2_arr)),
            # S3 metrics
            "s3_true_matches": val_s3_matches,
            "s3_retrieved": retrieved_s3[b_name],
            "s3_recall": retrieved_s3[b_name] / val_s3_matches if val_s3_matches else 0.0,
            "s3_mean_cands": float(np.mean(s3_arr)),
            "s3_median_cands": float(np.median(s3_arr)),
            "s3_p95_cands": float(np.percentile(s3_arr, 95)),
            "s3_max_cands": int(np.max(s3_arr)),
            # Overall union metrics
            "overall_true_matches": total_val_matches,
            "overall_retrieved": retrieved_all[b_name],
            "overall_recall": retrieved_all[b_name] / total_val_matches if total_val_matches else 0.0,
            "overall_mean_cands": float(np.mean(all_arr)),
            "overall_median_cands": float(np.median(all_arr)),
            "overall_p95_cands": float(np.percentile(all_arr, 95)),
            "overall_max_cands": int(np.max(all_arr)),
        }
        summary_results.append(res)

    # Step 6: Deep Dive into Missed Matches (Task 2 & Task 3)
    print(f"\n[Step 5] Analyzing {len(missed_pairs):,} missed true matches (out of {total_val_matches:,})...", flush=True)
    missed_features: list[dict] = []
    failure_mode_counts: Counter[str] = Counter()
    stopword_audit_counts: Counter[str] = Counter()
    stopword_blocked_examples: list[dict] = []

    for item in missed_pairs:
        s1_rec = item["s1_record"]
        m_rec = item["match_record"]
        if not m_rec:
            failure_mode_counts["missing_match_record"] += 1
            continue

        n1 = s1_rec["business_name"]
        n2 = m_rec["business_name"]
        a1 = s1_rec["business_address"]
        a2 = m_rec["business_address"]
        c1 = s1_rec["country"]
        c2 = m_rec["country"]

        nn1 = normalize_basic(n1)
        nn2 = normalize_basic(n2)
        ns1 = sorted_tokens(n1)
        ns2 = sorted_tokens(n2)
        nc1 = compact(n1)
        nc2 = compact(n2)

        an1 = normalize_basic(a1)
        an2 = normalize_basic(a2)

        toks_n1 = tokenize(n1)
        toks_n2 = tokenize(n2)
        toks_a1 = tokenize(a1)
        toks_a2 = tokenize(a2)

        info_n1 = set(t for t in toks_n1 if len(t) >= 3 and t not in NAME_STOPWORDS)
        info_n2 = set(t for t in toks_n2 if len(t) >= 3 and t not in NAME_STOPWORDS)
        info_a1 = set(t for t in toks_a1 if len(t) >= 3 and t not in ADDR_STOPWORDS)
        info_a2 = set(t for t in toks_a2 if len(t) >= 3 and t not in ADDR_STOPWORDS)

        all_n1 = set(t for t in toks_n1 if len(t) >= 3)
        all_n2 = set(t for t in toks_n2 if len(t) >= 3)
        all_a1 = set(t for t in toks_a1 if len(t) >= 3)
        all_a2 = set(t for t in toks_a2 if len(t) >= 3)

        shared_info_name = info_n1 & info_n2
        shared_info_addr = info_a1 & info_a2
        shared_all_name = all_n1 & all_n2
        shared_all_addr = all_a1 & all_a2

        # Check Task 3: Stopword Audit
        name_blocked_by_stopwords = (len(shared_info_name) < 2 and len(shared_all_name) >= 2)
        addr_blocked_by_stopwords = (len(shared_info_addr) < 3 and len(shared_all_addr) >= 3)
        unretrievable_due_to_stopwords = name_blocked_by_stopwords or addr_blocked_by_stopwords

        stopped_name_tokens = (shared_all_name - shared_info_name)
        stopped_addr_tokens = (shared_all_addr - shared_info_addr)

        if unretrievable_due_to_stopwords:
            stopword_audit_counts["total_blocked_by_stopwords"] += 1
            if name_blocked_by_stopwords:
                stopword_audit_counts["blocked_by_name_stopwords"] += 1
                for st in stopped_name_tokens:
                    stopword_audit_counts[f"name_stopword:{st}"] += 1
            if addr_blocked_by_stopwords:
                stopword_audit_counts["blocked_by_addr_stopwords"] += 1
                for st in stopped_addr_tokens:
                    stopword_audit_counts[f"addr_stopword:{st}"] += 1

            if len(stopword_blocked_examples) < 20:
                stopword_blocked_examples.append({
                    "s1_name": n1,
                    "match_name": n2,
                    "s1_addr": a1,
                    "match_addr": a2,
                    "stopped_name": list(stopped_name_tokens),
                    "stopped_addr": list(stopped_addr_tokens),
                })

        name_exact = (nn1 == nn2)
        name_sorted_exact = (ns1 == ns2)
        name_compact_exact = (nc1 == nc2)
        n_jacc = token_jaccard(toks_n1, toks_n2)
        n_overlap = overlap_coefficient(toks_n1, toks_n2)
        n_ratio = fuzz.ratio(nn1, nn2) / 100.0
        n_tsort = fuzz.token_sort_ratio(nn1, nn2) / 100.0
        n_tset = fuzz.token_set_ratio(nn1, nn2) / 100.0

        addr_exact = (an1 == an2 and an1 != "")
        a_jacc = token_jaccard(toks_a1, toks_a2)
        a_ratio = fuzz.ratio(an1, an2) / 100.0 if (an1 and an2) else 0.0
        a_tsort = fuzz.token_sort_ratio(an1, an2) / 100.0 if (an1 and an2) else 0.0

        num1 = get_numeric_tokens(a1)
        num2 = get_numeric_tokens(a2)
        shared_nums = len(num1 & num2)
        shared_addr_cnt = len(set(toks_a1) & set(toks_a2))

        # Categorize Failure Mode
        mode = "other"
        if normalize_basic(c1) != normalize_basic(c2):
            mode = "country mismatch"
        elif is_non_latin(n1) != is_non_latin(n2) or is_non_latin(a1) != is_non_latin(a2):
            mode = "multilingual / transliteration"
        elif unretrievable_due_to_stopwords:
            mode = "overly aggressive stopword filtering"
        elif is_abbreviation(n1, n2):
            mode = "abbreviation"
        elif n_ratio >= 0.70 or n_tsort >= 0.75:
            mode = "spelling variation"
        elif len(shared_info_name) == 1:
            mode = "token variation (single shared name token)"
        elif n_jacc < 0.20 and (a_jacc >= 0.40 or a_tsort >= 0.60 or shared_nums >= 2):
            mode = "name substantially different but address similar"
        elif n_jacc < 0.25 and a_jacc < 0.25:
            mode = "name/address both substantially different"
        elif a_jacc < 0.20 and n_jacc >= 0.30:
            mode = "address variation"
        else:
            mode = "other"

        failure_mode_counts[mode] += 1

        missed_features.append({
            "s1_id": item["s1_id"],
            "matched_id": item["matched_id"],
            "source": item["source"],
            "country": c1,
            "failure_mode": mode,
            "name_exact": name_exact,
            "name_sorted_exact": name_sorted_exact,
            "name_compact_exact": name_compact_exact,
            "name_jaccard": round(n_jacc, 4),
            "name_overlap": round(n_overlap, 4),
            "name_ratio": round(n_ratio, 4),
            "name_token_sort_ratio": round(n_tsort, 4),
            "name_token_set_ratio": round(n_tset, 4),
            "addr_exact": addr_exact,
            "addr_jaccard": round(a_jacc, 4),
            "addr_ratio": round(a_ratio, 4),
            "addr_token_sort_ratio": round(a_tsort, 4),
            "shared_numeric_tokens": shared_nums,
            "shared_address_tokens": shared_addr_cnt,
            "s1_name_tokens": len(toks_n1),
            "match_name_tokens": len(toks_n2),
            "s1_addr_tokens": len(toks_a1),
            "match_addr_tokens": len(toks_a2),
            "s1_name": n1,
            "match_name": n2,
            "s1_addr": a1,
            "match_addr": a2,
        })

    metrics_record = {
        "metadata": {
            "validation_sample_size": len(val_s1_ids),
            "random_seed": seed,
            "singletons": singletons,
            "entities_with_matches": entities_with_matches,
            "total_val_matches": total_val_matches,
            "val_s2_matches": val_s2_matches,
            "val_s3_matches": val_s3_matches,
            "index_runtime_seconds": round(index_time, 2),
            "eval_runtime_seconds": round(eval_time, 2),
            "peak_memory_mb": round(peak_mem, 1),
        },
        "summary": summary_results,
        "failure_modes": dict(failure_mode_counts.most_common()),
        "stopword_audit": dict(stopword_audit_counts.most_common()),
        "missed_sample": missed_features[:50],
        "stopword_blocked_examples": stopword_blocked_examples[:10],
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "baseline_a_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics_record, f, indent=2)

    return metrics_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Baseline A Blocker.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--sample-size", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    results = run_baseline_evaluation(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 80, flush=True)
    for row in results["summary"]:
        print(f"\nConfiguration: {row['blocker']}", flush=True)
        print(f"  S2 Recall:      {row['s2_recall']*100:>6.2f}% ({row['s2_retrieved']:,} / {row['s2_true_matches']:,}) | "
              f"Candidates/S1: mean={row['s2_mean_cands']:>6.1f}, median={row['s2_median_cands']:>4.0f}, "
              f"p95={row['s2_p95_cands']:>4.0f}, max={row['s2_max_cands']:,}", flush=True)
        print(f"  S3 Recall:      {row['s3_recall']*100:>6.2f}% ({row['s3_retrieved']:,} / {row['s3_true_matches']:,}) | "
              f"Candidates/S1: mean={row['s3_mean_cands']:>6.1f}, median={row['s3_median_cands']:>4.0f}, "
              f"p95={row['s3_p95_cands']:>4.0f}, max={row['s3_max_cands']:,}", flush=True)
        print(f"  Overall Recall: {row['overall_recall']*100:>6.2f}% ({row['overall_retrieved']:,} / {row['overall_true_matches']:,}) | "
              f"Candidates/S1: mean={row['overall_mean_cands']:>6.1f}, median={row['overall_median_cands']:>4.0f}, "
              f"p95={row['overall_p95_cands']:>4.0f}, max={row['overall_max_cands']:,}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("FAILURE MODE DISTRIBUTION (Missed True Matches)", flush=True)
    print("=" * 80, flush=True)
    total_missed = sum(results["failure_modes"].values())
    for mode, cnt in results["failure_modes"].items():
        pct = (cnt / total_missed) * 100 if total_missed else 0.0
        print(f"  {mode:<45} {cnt:>6,} ({pct:>5.1f}%)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("STOPWORD AUDIT (Unretrievable True Matches)", flush=True)
    print("=" * 80, flush=True)
    for key, val in results["stopword_audit"].items():
        print(f"  {key:<40}: {val:,}", flush=True)


if __name__ == "__main__":
    main()
