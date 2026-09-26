"""
Analyze true match pairs to understand feature distributions.

Samples N S1 entities from training ground truth, retrieves their
matched S2/S3 records, and computes pairwise similarity features.
Outputs distributional statistics that inform blocking and feature design.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import (
    normalize_basic,
    compact,
    sorted_tokens,
    tokenize,
)


# ===================================================================
# Data loading helpers
# ===================================================================

def load_ground_truth(path: Path) -> dict[str, list[str]]:
    """Load ground truth as {s1_id: [matched_ids]}."""

    gt: dict[str, list[str]] = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        for row in chunk.itertuples(index=False):
            s1_id = row.source1_entity_id
            matched = row.matched_entity_ids.strip()

            if matched:
                gt[s1_id] = [
                    mid.strip()
                    for mid in matched.split(",")
                    if mid.strip()
                ]
            else:
                gt[s1_id] = []

    return gt


def load_entities_by_id(
    path: Path,
    entity_ids: set[str],
) -> dict[str, dict]:
    """
    Stream through a source TSV and extract records for given IDs.

    Returns {entity_id: {business_name, business_address, country}}.
    """

    found: dict[str, dict] = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        keep_default_na=False,
        chunksize=500_000,
    ):
        mask = chunk["entity_id"].isin(entity_ids)

        for row in chunk[mask].itertuples(index=False):
            found[row.entity_id] = {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }

        # Early stop if we found everything.
        if len(found) >= len(entity_ids):
            break

    return found


# ===================================================================
# Feature computation
# ===================================================================

def token_jaccard(tokens_a: tuple[str, ...], tokens_b: tuple[str, ...]) -> float:
    """Jaccard similarity between two token tuples."""

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def overlap_coefficient(
    tokens_a: tuple[str, ...],
    tokens_b: tuple[str, ...],
) -> float:
    """Overlap coefficient: |intersection| / min(|A|, |B|)."""

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))

    return intersection / min_size if min_size > 0 else 0.0


def compute_pair_features(
    s1_record: dict,
    match_record: dict,
) -> dict:
    """Compute similarity features for a single (S1, matched) pair."""

    name_a = s1_record["business_name"]
    name_b = match_record["business_name"]
    addr_a = s1_record["business_address"]
    addr_b = match_record["business_address"]
    country_a = s1_record["country"]
    country_b = match_record["country"]

    # Normalized representations.
    name_norm_a = normalize_basic(name_a)
    name_norm_b = normalize_basic(name_b)
    name_sorted_a = sorted_tokens(name_a)
    name_sorted_b = sorted_tokens(name_b)
    name_compact_a = compact(name_a)
    name_compact_b = compact(name_b)
    addr_norm_a = normalize_basic(addr_a)
    addr_norm_b = normalize_basic(addr_b)

    # Tokens.
    name_tokens_a = tokenize(name_a)
    name_tokens_b = tokenize(name_b)
    addr_tokens_a = tokenize(addr_a)
    addr_tokens_b = tokenize(addr_b)

    # Country.
    country_match = (
        normalize_basic(country_a) == normalize_basic(country_b)
    )

    # Name features.
    name_exact = (name_norm_a == name_norm_b)
    name_sorted_exact = (name_sorted_a == name_sorted_b)
    name_compact_exact = (name_compact_a == name_compact_b)

    name_jacc = token_jaccard(name_tokens_a, name_tokens_b)
    name_overlap = overlap_coefficient(name_tokens_a, name_tokens_b)

    # rapidfuzz operates on raw strings; feed normalized versions.
    name_ratio = fuzz.ratio(name_norm_a, name_norm_b) / 100.0
    name_partial = fuzz.partial_ratio(name_norm_a, name_norm_b) / 100.0
    name_token_sort = fuzz.token_sort_ratio(name_norm_a, name_norm_b) / 100.0
    name_token_set = fuzz.token_set_ratio(name_norm_a, name_norm_b) / 100.0

    name_len_diff = abs(len(name_tokens_a) - len(name_tokens_b))

    max_name_len = max(len(name_norm_a), len(name_norm_b), 1)
    min_name_len = min(len(name_norm_a), len(name_norm_b))
    name_char_len_ratio = min_name_len / max_name_len

    # Address features.
    addr_a_empty = (addr_norm_a == "")
    addr_b_empty = (addr_norm_b == "")
    addr_missing_both = addr_a_empty and addr_b_empty
    addr_missing_one = addr_a_empty != addr_b_empty

    addr_exact = (
        addr_norm_a == addr_norm_b
        and not addr_a_empty
    )

    addr_jacc = token_jaccard(addr_tokens_a, addr_tokens_b)
    addr_overlap = overlap_coefficient(addr_tokens_a, addr_tokens_b)

    if addr_norm_a and addr_norm_b:
        addr_ratio = fuzz.ratio(addr_norm_a, addr_norm_b) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(
            addr_norm_a, addr_norm_b
        ) / 100.0
    else:
        addr_ratio = 0.0
        addr_token_sort = 0.0

    # Name prefix.
    prefix_len = 5
    name_prefix_match = (
        name_norm_a[:prefix_len] == name_norm_b[:prefix_len]
        if len(name_norm_a) >= prefix_len and len(name_norm_b) >= prefix_len
        else False
    )

    return {
        "country_match": country_match,
        "name_exact": name_exact,
        "name_sorted_exact": name_sorted_exact,
        "name_compact_exact": name_compact_exact,
        "name_jaccard": name_jacc,
        "name_overlap_coeff": name_overlap,
        "name_ratio": name_ratio,
        "name_partial_ratio": name_partial,
        "name_token_sort_ratio": name_token_sort,
        "name_token_set_ratio": name_token_set,
        "name_len_diff": name_len_diff,
        "name_char_len_ratio": name_char_len_ratio,
        "name_prefix_match": name_prefix_match,
        "addr_exact": addr_exact,
        "addr_jaccard": addr_jacc,
        "addr_overlap_coeff": addr_overlap,
        "addr_ratio": addr_ratio,
        "addr_token_sort_ratio": addr_token_sort,
        "addr_missing_both": addr_missing_both,
        "addr_missing_one": addr_missing_one,
    }


# ===================================================================
# Main analysis
# ===================================================================

def print_feature_distributions(all_features: list[dict]) -> None:
    """Print percentile distributions for all numeric features."""

    df = pd.DataFrame(all_features)

    print("\n" + "=" * 80)
    print("TRUE MATCH FEATURE DISTRIBUTIONS")
    print(f"Total pairs analyzed: {len(df):,}")
    print("=" * 80)

    # Boolean features: report rates.
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

    print("\n--- Boolean Feature Rates ---")
    print(f"{'Feature':<25} {'True%':>8} {'Count':>8}")
    print("-" * 45)

    for col in bool_cols:
        if col in df.columns:
            rate = df[col].mean() * 100
            count = df[col].sum()
            print(f"{col:<25} {rate:>7.1f}% {int(count):>8,}")

    # Numeric features: report percentiles.
    numeric_cols = [
        "name_jaccard",
        "name_overlap_coeff",
        "name_ratio",
        "name_partial_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_char_len_ratio",
        "name_len_diff",
        "addr_jaccard",
        "addr_overlap_coeff",
        "addr_ratio",
        "addr_token_sort_ratio",
    ]

    print("\n--- Numeric Feature Percentiles ---")
    header = f"{'Feature':<25} {'p5':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'p95':>6} {'mean':>6}"
    print(header)
    print("-" * len(header))

    for col in numeric_cols:
        if col in df.columns:
            vals = df[col]
            p5 = vals.quantile(0.05)
            p25 = vals.quantile(0.25)
            p50 = vals.quantile(0.50)
            p75 = vals.quantile(0.75)
            p95 = vals.quantile(0.95)
            mean = vals.mean()
            print(
                f"{col:<25} {p5:>6.3f} {p25:>6.3f} "
                f"{p50:>6.3f} {p75:>6.3f} {p95:>6.3f} {mean:>6.3f}"
            )

    # Name Jaccard bucketing.
    print("\n--- Name Jaccard Similarity Buckets ---")
    buckets = [
        (1.0, 1.0, "Exact (1.0)"),
        (0.8, 1.0, "[0.8, 1.0)"),
        (0.5, 0.8, "[0.5, 0.8)"),
        (0.2, 0.5, "[0.2, 0.5)"),
        (0.0, 0.2, "[0.0, 0.2)"),
    ]

    name_jacc = df["name_jaccard"]

    for lo, hi, label in buckets:
        if lo == hi:
            count = (name_jacc == lo).sum()
        else:
            count = ((name_jacc >= lo) & (name_jacc < hi)).sum()
        pct = count / len(df) * 100
        print(f"  {label:<20} {int(count):>8,} ({pct:>5.1f}%)")

    # Source breakdown.
    print("\n--- Source Breakdown ---")

    if "match_source" in df.columns:
        for source, group in df.groupby("match_source"):
            n = len(group)
            print(f"\n  Source: {source} ({n:,} pairs)")
            print(f"    name_exact rate:    {group['name_exact'].mean()*100:.1f}%")
            print(f"    name_jaccard p50:   {group['name_jaccard'].median():.3f}")
            print(f"    addr_jaccard p50:   {group['addr_jaccard'].median():.3f}")
            print(f"    addr_missing_one:   {group['addr_missing_one'].mean()*100:.1f}%")

    # Country breakdown.
    print("\n--- Country Breakdown ---")

    if "s1_country" in df.columns:
        for country, group in df.groupby("s1_country"):
            n = len(group)
            print(f"\n  Country: {country} ({n:,} pairs)")
            print(f"    name_exact rate:    {group['name_exact'].mean()*100:.1f}%")
            print(f"    name_jaccard p50:   {group['name_jaccard'].median():.3f}")
            print(f"    name_sorted_exact:  {group['name_sorted_exact'].mean()*100:.1f}%")
            print(f"    addr_jaccard p50:   {group['addr_jaccard'].median():.3f}")


def print_example_pairs(
    examples: list[dict],
    label: str,
    n: int = 10,
) -> None:
    """Print example match pairs for inspection."""

    print(f"\n{'=' * 80}")
    print(f"EXAMPLE PAIRS: {label} (showing {min(n, len(examples))})")
    print("=" * 80)

    for i, ex in enumerate(examples[:n]):
        print(f"\n--- Pair {i+1} ---")
        print(f"  S1 name:    {ex['s1_name']!r}")
        print(f"  Match name: {ex['match_name']!r}")
        print(f"  S1 addr:    {ex['s1_addr']!r}")
        print(f"  Match addr: {ex['match_addr']!r}")
        print(f"  Country:    {ex['s1_country']} / {ex['match_country']}")
        print(f"  name_jaccard={ex['features']['name_jaccard']:.3f}  "
              f"name_ratio={ex['features']['name_ratio']:.3f}  "
              f"addr_jaccard={ex['features']['addr_jaccard']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze true match feature distributions."
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
        default=1000,
        help="Number of S1 entities to sample.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    train_dir = args.data_dir / "train"
    gt_path = train_dir / "train_ground_truth.tsv"

    # ------------------------------------------------------------------
    # 1. Load ground truth and sample S1 entities with non-empty matches.
    # ------------------------------------------------------------------

    print("Loading ground truth...")
    t0 = time.time()
    gt = load_ground_truth(gt_path)
    print(f"  Loaded {len(gt):,} S1 entities in {time.time()-t0:.1f}s")

    # Only sample S1 entities that have at least one match.
    s1_with_matches = [
        s1_id for s1_id, matches in gt.items()
        if matches
    ]
    print(f"  S1 with matches: {len(s1_with_matches):,}")

    sample_size = min(args.sample_size, len(s1_with_matches))
    sampled_s1 = random.sample(s1_with_matches, sample_size)
    print(f"  Sampled {sample_size:,} S1 entities")

    # ------------------------------------------------------------------
    # 2. Collect all entity IDs we need to load.
    # ------------------------------------------------------------------

    s1_ids_needed: set[str] = set(sampled_s1)
    s2_ids_needed: set[str] = set()
    s3_ids_needed: set[str] = set()

    for s1_id in sampled_s1:
        for mid in gt[s1_id]:
            if mid.startswith("S2-"):
                s2_ids_needed.add(mid)
            elif mid.startswith("S3-"):
                s3_ids_needed.add(mid)

    print(f"  Need to load: {len(s1_ids_needed):,} S1, "
          f"{len(s2_ids_needed):,} S2, {len(s3_ids_needed):,} S3 records")

    # ------------------------------------------------------------------
    # 3. Load entity records.
    # ------------------------------------------------------------------

    print("\nLoading S1 records...")
    t0 = time.time()
    s1_records = load_entities_by_id(
        train_dir / "train_source1.tsv", s1_ids_needed
    )
    print(f"  Loaded {len(s1_records):,} S1 records in {time.time()-t0:.1f}s")

    print("Loading S2 records...")
    t0 = time.time()
    s2_records = load_entities_by_id(
        train_dir / "train_source2.tsv", s2_ids_needed
    )
    print(f"  Loaded {len(s2_records):,} S2 records in {time.time()-t0:.1f}s")

    print("Loading S3 records...")
    t0 = time.time()
    s3_records = load_entities_by_id(
        train_dir / "train_source3.tsv", s3_ids_needed
    )
    print(f"  Loaded {len(s3_records):,} S3 records in {time.time()-t0:.1f}s")

    all_match_records = {**s2_records, **s3_records}

    # ------------------------------------------------------------------
    # 4. Compute features for all true-match pairs.
    # ------------------------------------------------------------------

    print("\nComputing pair features...")
    t0 = time.time()

    all_features: list[dict] = []
    exact_examples: list[dict] = []
    fuzzy_examples: list[dict] = []
    hard_examples: list[dict] = []
    missing_match_records = 0

    for s1_id in sampled_s1:
        s1_rec = s1_records.get(s1_id)

        if s1_rec is None:
            continue

        for match_id in gt[s1_id]:
            match_rec = all_match_records.get(match_id)

            if match_rec is None:
                missing_match_records += 1
                continue

            features = compute_pair_features(s1_rec, match_rec)

            # Metadata for breakdowns.
            features["match_source"] = (
                "S2" if match_id.startswith("S2-") else "S3"
            )
            features["s1_country"] = s1_rec["country"].strip()

            all_features.append(features)

            # Collect examples by difficulty.
            example = {
                "s1_id": s1_id,
                "match_id": match_id,
                "s1_name": s1_rec["business_name"],
                "match_name": match_rec["business_name"],
                "s1_addr": s1_rec["business_address"],
                "match_addr": match_rec["business_address"],
                "s1_country": s1_rec["country"],
                "match_country": match_rec["country"],
                "features": features,
            }

            nj = features["name_jaccard"]

            if features["name_exact"]:
                exact_examples.append(example)
            elif nj >= 0.5:
                fuzzy_examples.append(example)
            else:
                hard_examples.append(example)

    elapsed = time.time() - t0
    print(f"  Computed {len(all_features):,} pair features in {elapsed:.1f}s")

    if missing_match_records > 0:
        print(f"  WARNING: {missing_match_records:,} match records "
              f"not found in source files")

    # ------------------------------------------------------------------
    # 5. Print results.
    # ------------------------------------------------------------------

    print_feature_distributions(all_features)

    # Show examples from each difficulty tier.
    random.shuffle(exact_examples)
    random.shuffle(fuzzy_examples)
    random.shuffle(hard_examples)

    print_example_pairs(exact_examples, "EXACT NAME MATCHES", n=5)
    print_example_pairs(fuzzy_examples, "FUZZY MATCHES (jaccard >= 0.5)", n=10)
    print_example_pairs(hard_examples, "HARD MATCHES (jaccard < 0.5)", n=10)

    # ------------------------------------------------------------------
    # 6. Blocking-relevant statistics.
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BLOCKING-RELEVANT INSIGHTS")
    print("=" * 80)

    df = pd.DataFrame(all_features)

    # What fraction would each simple blocker catch?
    print("\n--- Blocker Catch Rates (upper bound) ---")

    blockers = {
        "country + name_norm":    df["name_exact"] & df["country_match"],
        "country + name_sorted":  df["name_sorted_exact"] & df["country_match"],
        "country + name_compact": df["name_compact_exact"] & df["country_match"],
        "name_norm (no country)": df["name_exact"],
        "name_sorted (no ctry)":  df["name_sorted_exact"],
        "country + prefix5":      df["name_prefix_match"] & df["country_match"],
        "country_match only":     df["country_match"],
    }

    for name, mask in blockers.items():
        caught = mask.sum()
        rate = caught / len(df) * 100
        print(f"  {name:<30} {int(caught):>6,} / {len(df):>6,}  ({rate:>5.1f}%)")

    # Token overlap analysis.
    print("\n--- Name Token Overlap Analysis ---")

    token_overlaps = []

    for feat in all_features:
        s1_tokens = set()  # We need raw tokens, recompute from jaccard context.

    # Instead use the precomputed name_jaccard and overlap stats.
    print(f"  Pairs with name_jaccard >= 0.5:  "
          f"{(df['name_jaccard'] >= 0.5).sum():,} / {len(df):,}  "
          f"({(df['name_jaccard'] >= 0.5).mean()*100:.1f}%)")
    print(f"  Pairs with name_jaccard >= 0.33: "
          f"{(df['name_jaccard'] >= 0.33).sum():,} / {len(df):,}  "
          f"({(df['name_jaccard'] >= 0.33).mean()*100:.1f}%)")
    print(f"  Pairs with name_overlap >= 0.5:  "
          f"{(df['name_overlap_coeff'] >= 0.5).sum():,} / {len(df):,}  "
          f"({(df['name_overlap_coeff'] >= 0.5).mean()*100:.1f}%)")
    print(f"  Pairs with name_ratio >= 0.6:    "
          f"{(df['name_ratio'] >= 0.6).sum():,} / {len(df):,}  "
          f"({(df['name_ratio'] >= 0.6).mean()*100:.1f}%)")
    print(f"  Pairs with name_token_sort >= 0.6: "
          f"{(df['name_token_sort_ratio'] >= 0.6).sum():,} / {len(df):,}  "
          f"({(df['name_token_sort_ratio'] >= 0.6).mean()*100:.1f}%)")

    # What the hardest cases look like.
    print(f"\n--- Hardest True Matches (name_jaccard < 0.3) ---")
    hardest = df[df["name_jaccard"] < 0.3]
    print(f"  Count: {len(hardest):,} / {len(df):,} ({len(hardest)/len(df)*100:.1f}%)")

    if len(hardest) > 0:
        print(f"  These have name_ratio p50: {hardest['name_ratio'].median():.3f}")
        print(f"  addr_jaccard p50: {hardest['addr_jaccard'].median():.3f}")
        print(f"  addr_missing_one rate: {hardest['addr_missing_one'].mean()*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
