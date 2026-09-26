"""
Analyze hard negatives to understand what separates true matches
from plausible non-matches.

Constructs hard negatives using the same blocking keys we plan to use,
then compares their feature distributions against true positives.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
# pyrefly: ignore [missing-import]
from rapidfuzz import fuzz


sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import (
    normalize_basic,
    compact,
    sorted_tokens,
    tokenize,
)
from analyze_true_matches import (
    compute_pair_features,
    load_ground_truth,
    token_jaccard,
)


def build_inverted_indexes(
    records: dict[str, dict],
) -> dict[str, dict[str, set[str]]]:
    """
    Build multiple inverted indexes from entity records.

    Returns a dict of index_name -> {key -> set of entity_ids}.
    """

    idx_name_norm: dict[str, set[str]] = defaultdict(set)
    idx_name_sorted: dict[str, set[str]] = defaultdict(set)
    idx_name_compact: dict[str, set[str]] = defaultdict(set)
    idx_name_tokens: dict[str, set[str]] = defaultdict(set)
    idx_addr_tokens: dict[str, set[str]] = defaultdict(set)

    for eid, rec in records.items():
        country = normalize_basic(rec["country"])
        name = rec["business_name"]
        addr = rec["business_address"]

        nn = normalize_basic(name)
        ns = sorted_tokens(name)
        nc = compact(name)

        # Country-scoped name indexes.
        idx_name_norm[(country, nn)].add(eid)
        idx_name_sorted[(country, ns)].add(eid)
        idx_name_compact[(country, nc)].add(eid)

        # Token-level indexes.
        for token in tokenize(name):
            if len(token) >= 3:  # Skip very short tokens.
                idx_name_tokens[(country, token)].add(eid)

        for token in tokenize(addr):
            if len(token) >= 3:
                idx_addr_tokens[(country, token)].add(eid)

    return {
        "name_norm": idx_name_norm,
        "name_sorted": idx_name_sorted,
        "name_compact": idx_name_compact,
        "name_tokens": idx_name_tokens,
        "addr_tokens": idx_addr_tokens,
    }


def find_hard_negatives_for_s1(
    s1_id: str,
    s1_rec: dict,
    indexes: dict[str, dict],
    true_match_ids: set[str],
    max_per_source: int = 10,
) -> list[tuple[str, str]]:
    """
    Find hard negative entity IDs for a given S1 entity.

    Returns list of (entity_id, source_of_retrieval).
    Hard negatives are entities retrieved by our blocking keys
    but NOT in the ground truth.
    """

    country = normalize_basic(s1_rec["country"])
    name = s1_rec["business_name"]
    addr = s1_rec["business_address"]

    nn = normalize_basic(name)
    ns = sorted_tokens(name)
    nc = compact(name)

    candidates: dict[str, str] = {}  # eid -> retrieval_source

    # 1. Same normalized name.
    for eid in indexes["name_norm"].get((country, nn), set()):
        if eid not in true_match_ids and eid != s1_id:
            candidates[eid] = "name_norm"

    # 2. Same sorted name.
    for eid in indexes["name_sorted"].get((country, ns), set()):
        if eid not in true_match_ids and eid != s1_id:
            candidates.setdefault(eid, "name_sorted")

    # 3. Same compact name.
    for eid in indexes["name_compact"].get((country, nc), set()):
        if eid not in true_match_ids and eid != s1_id:
            candidates.setdefault(eid, "name_compact")

    # 4. Shared name tokens (at least 2 shared tokens of length >= 3).
    name_toks = [t for t in tokenize(name) if len(t) >= 3]
    token_hits: dict[str, int] = defaultdict(int)

    for token in name_toks:
        for eid in indexes["name_tokens"].get((country, token), set()):
            if eid not in true_match_ids and eid != s1_id:
                token_hits[eid] += 1

    for eid, count in token_hits.items():
        if count >= 2:
            candidates.setdefault(eid, f"name_tokens({count})")

    # 5. Shared address tokens (at least 3 shared).
    addr_toks = [t for t in tokenize(addr) if len(t) >= 3]
    addr_hits: dict[str, int] = defaultdict(int)

    for token in addr_toks:
        for eid in indexes["addr_tokens"].get((country, token), set()):
            if eid not in true_match_ids and eid != s1_id:
                addr_hits[eid] += 1

    for eid, count in addr_hits.items():
        if count >= 3:
            candidates.setdefault(eid, f"addr_tokens({count})")

    # Limit output size.
    result = list(candidates.items())

    if len(result) > max_per_source:
        random.shuffle(result)
        result = result[:max_per_source]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze hard negatives vs. true matches."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of S1 entities to sample.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    train_dir = args.data_dir / "train"

    # ------------------------------------------------------------------
    # 1. Load ground truth.
    # ------------------------------------------------------------------

    print("Loading ground truth...")
    t0 = time.time()
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    print(f"  Loaded {len(gt):,} S1 entities in {time.time()-t0:.1f}s")

    s1_with_matches = [s1 for s1, m in gt.items() if m]
    sample_size = min(args.sample_size, len(s1_with_matches))
    sampled_s1 = random.sample(s1_with_matches, sample_size)
    print(f"  Sampled {sample_size} S1 entities")

    # ------------------------------------------------------------------
    # 2. Load a manageable subset of S2/S3 for building indexes.
    #    We load the first 500K from each source to keep memory bounded
    #    while still having enough data for meaningful hard negatives.
    # ------------------------------------------------------------------

    print("\nLoading S2 sample for index building...")
    t0 = time.time()
    s2_df = pd.read_csv(
        train_dir / "train_source2.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        keep_default_na=False,
        nrows=500_000,
    )
    s2_records = {
        row.entity_id: {
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country,
        }
        for row in s2_df.itertuples(index=False)
    }
    del s2_df
    print(f"  Loaded {len(s2_records):,} S2 records in {time.time()-t0:.1f}s")

    print("Loading S3 sample for index building...")
    t0 = time.time()
    s3_df = pd.read_csv(
        train_dir / "train_source3.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        keep_default_na=False,
        nrows=500_000,
    )
    s3_records = {
        row.entity_id: {
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country,
        }
        for row in s3_df.itertuples(index=False)
    }
    del s3_df
    print(f"  Loaded {len(s3_records):,} S3 records in {time.time()-t0:.1f}s")

    all_match_records = {**s2_records, **s3_records}

    # ------------------------------------------------------------------
    # 3. Load S1 records for sampled entities.
    # ------------------------------------------------------------------

    print("\nLoading S1 records for sampled entities...")
    t0 = time.time()
    s1_ids_needed = set(sampled_s1)

    s1_records: dict[str, dict] = {}

    for chunk in pd.read_csv(
        train_dir / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        mask = chunk["entity_id"].isin(s1_ids_needed)

        for row in chunk[mask].itertuples(index=False):
            s1_records[row.entity_id] = {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }

        if len(s1_records) >= len(s1_ids_needed):
            break

    print(f"  Loaded {len(s1_records):,} S1 records in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Build inverted indexes on S2/S3 records.
    # ------------------------------------------------------------------

    print("\nBuilding inverted indexes on S2/S3...")
    t0 = time.time()
    indexes = build_inverted_indexes(all_match_records)
    print(f"  Built indexes in {time.time()-t0:.1f}s")

    for name, idx in indexes.items():
        print(f"    {name}: {len(idx):,} keys")

    # ------------------------------------------------------------------
    # 5. Generate true-positive and hard-negative features.
    # ------------------------------------------------------------------

    print("\nGenerating features for true positives and hard negatives...")
    t0 = time.time()

    tp_features: list[dict] = []
    hn_features: list[dict] = []
    s1_with_no_hard_neg = 0
    s1_processed = 0

    for s1_id in sampled_s1:
        s1_rec = s1_records.get(s1_id)

        if s1_rec is None:
            continue

        s1_processed += 1
        true_match_ids = set(gt.get(s1_id, []))

        # True positives: only score those in our loaded records.
        for mid in true_match_ids:
            mrec = all_match_records.get(mid)

            if mrec is None:
                continue

            feats = compute_pair_features(s1_rec, mrec)
            feats["label"] = "true_positive"
            tp_features.append(feats)

        # Hard negatives.
        hard_negs = find_hard_negatives_for_s1(
            s1_id, s1_rec, indexes, true_match_ids, max_per_source=10
        )

        if not hard_negs:
            s1_with_no_hard_neg += 1

        for eid, source in hard_negs:
            neg_rec = all_match_records.get(eid)

            if neg_rec is None:
                continue

            feats = compute_pair_features(s1_rec, neg_rec)
            feats["label"] = "hard_negative"
            feats["retrieval_source"] = source
            hn_features.append(feats)

    elapsed = time.time() - t0
    print(f"  S1 processed: {s1_processed:,}")
    print(f"  True positive pairs: {len(tp_features):,}")
    print(f"  Hard negative pairs: {len(hn_features):,}")
    print(f"  S1 with no hard negatives: {s1_with_no_hard_neg:,}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 6. Compare distributions.
    # ------------------------------------------------------------------

    if not tp_features or not hn_features:
        print("\nInsufficient data for comparison.")
        return

    tp_df = pd.DataFrame(tp_features)
    hn_df = pd.DataFrame(hn_features)

    numeric_cols = [
        "name_jaccard",
        "name_overlap_coeff",
        "name_ratio",
        "name_partial_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_char_len_ratio",
        "addr_jaccard",
        "addr_overlap_coeff",
        "addr_ratio",
        "addr_token_sort_ratio",
    ]

    bool_cols = [
        "country_match",
        "name_exact",
        "name_sorted_exact",
        "name_compact_exact",
        "name_prefix_match",
        "addr_exact",
        "addr_missing_both",
        "addr_missing_one",
    ]

    print("\n" + "=" * 90)
    print("TRUE POSITIVE vs HARD NEGATIVE COMPARISON")
    print("=" * 90)

    print(f"\n  True positives:  {len(tp_df):,}")
    print(f"  Hard negatives:  {len(hn_df):,}")

    # Boolean features.
    print("\n--- Boolean Feature Rates ---")
    print(f"{'Feature':<25} {'TP Rate':>10} {'HN Rate':>10} {'Separation':>12}")
    print("-" * 60)

    for col in bool_cols:
        if col in tp_df.columns and col in hn_df.columns:
            tp_rate = tp_df[col].mean() * 100
            hn_rate = hn_df[col].mean() * 100
            diff = abs(tp_rate - hn_rate)
            sep = "HIGH" if diff > 30 else "MEDIUM" if diff > 10 else "LOW"
            print(f"{col:<25} {tp_rate:>9.1f}% {hn_rate:>9.1f}% {sep:>12}")

    # Numeric features.
    print("\n--- Numeric Feature Medians ---")
    print(f"{'Feature':<25} {'TP p50':>8} {'HN p50':>8} {'TP p25':>8} {'HN p75':>8} {'Separation':>12}")
    print("-" * 75)

    for col in numeric_cols:
        if col in tp_df.columns and col in hn_df.columns:
            tp_p50 = tp_df[col].median()
            hn_p50 = hn_df[col].median()
            tp_p25 = tp_df[col].quantile(0.25)
            hn_p75 = hn_df[col].quantile(0.75)

            # Separation: does TP p25 > HN p75? If so, very separable.
            if tp_p25 > hn_p75:
                sep = "VERY HIGH"
            elif tp_p50 > hn_p75:
                sep = "HIGH"
            elif abs(tp_p50 - hn_p50) > 0.2:
                sep = "MEDIUM"
            else:
                sep = "LOW"

            print(
                f"{col:<25} {tp_p50:>8.3f} {hn_p50:>8.3f} "
                f"{tp_p25:>8.3f} {hn_p75:>8.3f} {sep:>12}"
            )

    # ------------------------------------------------------------------
    # 7. Retrieval source breakdown for hard negatives.
    # ------------------------------------------------------------------

    print("\n--- Hard Negative Retrieval Sources ---")

    if "retrieval_source" in hn_df.columns:
        for source, group in hn_df.groupby("retrieval_source"):
            n = len(group)
            print(f"\n  Source: {source} ({n:,} pairs)")
            print(f"    name_jaccard p50:      {group['name_jaccard'].median():.3f}")
            print(f"    name_ratio p50:        {group['name_ratio'].median():.3f}")
            print(f"    name_token_set p50:    {group['name_token_set_ratio'].median():.3f}")
            print(f"    addr_jaccard p50:      {group['addr_jaccard'].median():.3f}")

    # ------------------------------------------------------------------
    # 8. Feature combination analysis.
    # ------------------------------------------------------------------

    print("\n--- Feature Combination Analysis ---")
    print("Which feature combinations best separate TP from HN?\n")

    thresholds = [
        ("name_ratio >= 0.8", tp_df["name_ratio"] >= 0.8, hn_df["name_ratio"] >= 0.8),
        ("name_token_set >= 0.9", tp_df["name_token_set_ratio"] >= 0.9, hn_df["name_token_set_ratio"] >= 0.9),
        ("name_jacc >= 0.5 AND addr_jacc >= 0.3",
         (tp_df["name_jaccard"] >= 0.5) & (tp_df["addr_jaccard"] >= 0.3),
         (hn_df["name_jaccard"] >= 0.5) & (hn_df["addr_jaccard"] >= 0.3)),
        ("name_ratio >= 0.7 AND addr_ratio >= 0.5",
         (tp_df["name_ratio"] >= 0.7) & (tp_df["addr_ratio"] >= 0.5),
         (hn_df["name_ratio"] >= 0.7) & (hn_df["addr_ratio"] >= 0.5)),
        ("(name_ratio >= 0.8) OR (addr_jacc >= 0.6 AND name_jacc >= 0.3)",
         (tp_df["name_ratio"] >= 0.8) | ((tp_df["addr_jaccard"] >= 0.6) & (tp_df["name_jaccard"] >= 0.3)),
         (hn_df["name_ratio"] >= 0.8) | ((hn_df["addr_jaccard"] >= 0.6) & (hn_df["name_jaccard"] >= 0.3))),
    ]

    print(f"{'Condition':<55} {'TP pass':>8} {'HN pass':>8} {'TP%':>6} {'HN%':>6}")
    print("-" * 90)

    for label, tp_mask, hn_mask in thresholds:
        tp_count = tp_mask.sum()
        hn_count = hn_mask.sum()
        tp_pct = tp_count / len(tp_df) * 100
        hn_pct = hn_count / len(hn_df) * 100
        print(f"{label:<55} {int(tp_count):>8} {int(hn_count):>8} {tp_pct:>5.1f}% {hn_pct:>5.1f}%")

    # ------------------------------------------------------------------
    # 9. Show example hard negatives.
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXAMPLE HARD NEGATIVES (highest name similarity)")
    print("=" * 80)

    hn_sorted = hn_df.sort_values("name_ratio", ascending=False)

    for i, (_, row) in enumerate(hn_sorted.head(10).iterrows()):
        print(f"\n--- Hard Negative {i+1} ---")
        print(f"  name_jaccard={row['name_jaccard']:.3f}  "
              f"name_ratio={row['name_ratio']:.3f}  "
              f"addr_jaccard={row['addr_jaccard']:.3f}  "
              f"name_token_set={row['name_token_set_ratio']:.3f}  "
              f"country_match={row['country_match']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
