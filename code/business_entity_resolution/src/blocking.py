"""
Blocking (candidate generation) for Business Entity Resolution.

Multi-key inverted-index blocking with union-of-blocks strategy.
Designed for high recall — missed true matches can never be recovered.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import (
    normalize_basic,
    compact,
    sorted_tokens,
    tokenize,
)


# ===================================================================
# Stopwords: tokens too common to be useful for blocking
# Derived from empirical frequency analysis on 500K S2 records.
# These appear in 10-50% of all business names/addresses and
# provide no discriminative signal for blocking.
# ===================================================================

# Legal suffixes and extremely common business name tokens.
NAME_STOPWORDS: frozenset[str] = frozenset({
    # English legal suffixes
    "limited", "ltd", "llc", "inc", "corp", "corporation",
    "incorporated", "pvt", "private", "public", "company",
    "llp", "partners", "holdings", "co",
    # Hindi/Devanagari legal suffixes
    "लिमिटेड", "प्राइवेट", "प्रा", "लि",
    # Telugu legal suffixes
    "లిమిటెడ్", "ప్రైవేట్",
    # Kannada legal suffixes
    "ಲಿಮಿಟೆಡ್", "ಪ್ರೈವೇಟ್",
    # Tamil legal suffixes
    "லிமிடெட்", "பிரைவேட்",
    # Bengali legal suffixes
    "লিমিটেড",
    # Common generic terms
    "the", "and", "com", "www", "services", "service",
    "group", "enterprises", "industries", "associates",
    "solutions", "international", "ventures", "trading",
    "global", "india", "center",
})

# Extremely common address tokens that don't help blocking.
ADDR_STOPWORDS: frozenset[str] = frozenset({
    "road", "street", "avenue", "ave", "lane", "drive",
    "floor", "plot", "flat", "block", "sector", "colony",
    "nagar", "near", "door", "house", "main", "cross",
    "new", "old", "east", "west", "north", "south",
    "city", "null",
    # Very common Indian cities/states (too many addresses share these).
    "delhi", "maharashtra", "mumbai", "bangalore",
    "karnataka", "kolkata", "pradesh", "uttar",
    "tamil", "nadu", "bengal", "gujarat", "pune",
    "telangana", "chennai", "hyderabad",
    # Hindi/regional state names
    "महाराष्ट्र",
})



class InvertedIndexBlocker:
    """
    Multi-key inverted-index blocker.

    Indexes S2/S3 entities using multiple blocking keys.
    At query time, retrieves candidates via the union of all keys.
    """

    def __init__(self) -> None:
        # Each index maps (country, key_value) -> set of entity IDs.
        self.idx_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)

        self.n_indexed = 0

    def index_record(self, entity_id: str, record: dict) -> None:
        """Add a single entity to all indexes."""

        country = normalize_basic(record["country"])
        name = record["business_name"]
        addr = record["business_address"]

        nn = normalize_basic(name)
        ns = sorted_tokens(name)
        nc = compact(name)

        self.idx_name_norm[(country, nn)].add(entity_id)
        self.idx_name_sorted[(country, ns)].add(entity_id)
        self.idx_name_compact[(country, nc)].add(entity_id)

        # Index informative name tokens (skip stopwords).
        for token in tokenize(name):
            if len(token) >= 3 and token not in NAME_STOPWORDS:
                self.idx_name_tokens[(country, token)].add(entity_id)

        # Index informative address tokens (skip stopwords).
        for token in tokenize(addr):
            if len(token) >= 3 and token not in ADDR_STOPWORDS:
                self.idx_addr_tokens[(country, token)].add(entity_id)

        self.n_indexed += 1

    def index_dataframe_chunk(self, chunk: pd.DataFrame) -> None:
        """Index all records in a DataFrame chunk."""

        for row in chunk.itertuples(index=False):
            self.index_record(
                row.entity_id,
                {
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country,
                },
            )

    def query(
        self,
        record: dict,
        min_name_token_overlap: int = 2,
        min_addr_token_overlap: int = 3,
    ) -> dict[str, set[str]]:
        """
        Retrieve candidate entity IDs for a query record.

        Returns dict mapping blocker_name -> set of candidate IDs.
        The union of all sets is the full candidate set.
        """

        country = normalize_basic(record["country"])
        name = record["business_name"]
        addr = record["business_address"]

        nn = normalize_basic(name)
        ns = sorted_tokens(name)
        nc = compact(name)

        candidates: dict[str, set[str]] = {}

        # Key 1: Exact normalized name match.
        key = (country, nn)
        if key in self.idx_name_norm:
            candidates["name_norm"] = set(self.idx_name_norm[key])

        # Key 2: Token-sorted name match.
        key = (country, ns)
        if key in self.idx_name_sorted:
            candidates["name_sorted"] = set(self.idx_name_sorted[key])

        # Key 3: Compact name match.
        key = (country, nc)
        if key in self.idx_name_compact:
            candidates["name_compact"] = set(self.idx_name_compact[key])

        # Key 4: Shared name tokens (stopwords filtered).
        name_toks = [
            t for t in tokenize(name)
            if len(t) >= 3 and t not in NAME_STOPWORDS
        ]
        token_hits: dict[str, int] = defaultdict(int)

        for token in name_toks:
            key = (country, token)
            if key in self.idx_name_tokens:
                for eid in self.idx_name_tokens[key]:
                    token_hits[eid] += 1

        shared_name = {
            eid for eid, count in token_hits.items()
            if count >= min_name_token_overlap
        }

        if shared_name:
            candidates["name_tokens"] = shared_name

        # Key 5: Shared address tokens (stopwords filtered).
        addr_toks = [
            t for t in tokenize(addr)
            if len(t) >= 3 and t not in ADDR_STOPWORDS
        ]

        if addr_toks:
            addr_hits: dict[str, int] = defaultdict(int)

            for token in addr_toks:
                key = (country, token)
                if key in self.idx_addr_tokens:
                    for eid in self.idx_addr_tokens[key]:
                        addr_hits[eid] += 1

            shared_addr = {
                eid for eid, count in addr_hits.items()
                if count >= min_addr_token_overlap
            }

            if shared_addr:
                candidates["addr_tokens"] = shared_addr

        return candidates

    def query_union(
        self,
        record: dict,
        min_name_token_overlap: int = 2,
        min_addr_token_overlap: int = 3,
    ) -> set[str]:
        """Return the union of all blocker candidate sets."""

        by_source = self.query(
            record,
            min_name_token_overlap=min_name_token_overlap,
            min_addr_token_overlap=min_addr_token_overlap,
        )

        result: set[str] = set()

        for cands in by_source.values():
            result |= cands

        return result

    def stats(self) -> dict:
        """Return index size statistics."""

        return {
            "n_indexed": self.n_indexed,
            "name_norm_keys": len(self.idx_name_norm),
            "name_sorted_keys": len(self.idx_name_sorted),
            "name_compact_keys": len(self.idx_name_compact),
            "name_token_keys": len(self.idx_name_tokens),
            "addr_token_keys": len(self.idx_addr_tokens),
        }


def build_blocker_from_source(
    path: Path,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> InvertedIndexBlocker:
    """Build an InvertedIndexBlocker from a source TSV file."""

    blocker = InvertedIndexBlocker()
    rows_read = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        keep_default_na=False,
        chunksize=chunksize,
    ):
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        blocker.index_dataframe_chunk(chunk)
        rows_read += len(chunk)

        if max_rows is not None and rows_read >= max_rows:
            break

    return blocker


def build_combined_blocker(
    s2_path: Path,
    s3_path: Path,
    chunksize: int = 250_000,
    max_rows_per_source: int | None = None,
) -> InvertedIndexBlocker:
    """Build a single blocker indexing both S2 and S3."""

    blocker = InvertedIndexBlocker()
    total = 0

    for source_path in [s2_path, s3_path]:
        rows_read = 0

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            usecols=[
                "entity_id", "business_name",
                "business_address", "country",
            ],
            dtype="string",
            keep_default_na=False,
            chunksize=chunksize,
        ):
            if max_rows_per_source is not None:
                remaining = max_rows_per_source - rows_read
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)

            blocker.index_dataframe_chunk(chunk)
            rows_read += len(chunk)
            total += len(chunk)

            if max_rows_per_source is not None and rows_read >= max_rows_per_source:
                break

    return blocker


# ===================================================================
# Recall evaluation
# ===================================================================

def evaluate_blocking_recall(
    blocker: InvertedIndexBlocker,
    s1_path: Path,
    ground_truth: dict[str, list[str]],
    sample_s1_ids: list[str] | None = None,
    min_name_token_overlap: int = 2,
    min_addr_token_overlap: int = 3,
) -> dict:
    """
    Evaluate blocking recall on sampled S1 entities.

    Reports:
    - Overall recall
    - Per-source recall (S2 vs S3)
    - Per-blocker-key recall contribution
    - Candidate count distribution
    - Analysis of missed matches
    """

    # Load S1 records for sampled entities.
    if sample_s1_ids is None:
        s1_with_matches = [
            s1 for s1, matches in ground_truth.items()
            if matches
        ]
        sample_s1_ids = s1_with_matches

    s1_ids_needed = set(sample_s1_ids)
    s1_records: dict[str, dict] = {}

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        usecols=[
            "entity_id", "business_name",
            "business_address", "country",
        ],
        dtype="string",
        keep_default_na=False,
        chunksize=250_000,
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

    # Evaluate.
    total_true_matches = 0
    total_found = 0
    s2_true = 0
    s2_found = 0
    s3_true = 0
    s3_found = 0

    # Per-blocker-key contribution tracking.
    found_by_key: dict[str, int] = defaultdict(int)
    found_only_by_key: dict[str, int] = defaultdict(int)

    candidate_counts: list[int] = []
    missed_examples: list[dict] = []

    # Per-country tracking.
    country_stats: dict[str, dict] = defaultdict(
        lambda: {"true": 0, "found": 0}
    )

    for s1_id in sample_s1_ids:
        s1_rec = s1_records.get(s1_id)

        if s1_rec is None:
            continue

        true_matches = set(ground_truth.get(s1_id, []))

        if not true_matches:
            continue

        country = s1_rec["country"].strip()

        # Query blocker with per-key breakdown.
        by_key = blocker.query(
            s1_rec,
            min_name_token_overlap=min_name_token_overlap,
            min_addr_token_overlap=min_addr_token_overlap,
        )

        all_candidates = set()

        for cands in by_key.values():
            all_candidates |= cands

        candidate_counts.append(len(all_candidates))

        # Count found matches.
        found_matches = true_matches & all_candidates
        missed_matches = true_matches - all_candidates

        total_true_matches += len(true_matches)
        total_found += len(found_matches)

        country_stats[country]["true"] += len(true_matches)
        country_stats[country]["found"] += len(found_matches)

        for mid in true_matches:
            if mid.startswith("S2-"):
                s2_true += 1
                if mid in found_matches:
                    s2_found += 1
            elif mid.startswith("S3-"):
                s3_true += 1
                if mid in found_matches:
                    s3_found += 1

        # Track which key found which match.
        for mid in found_matches:
            keys_that_found = [
                key for key, cands in by_key.items()
                if mid in cands
            ]

            for key in keys_that_found:
                found_by_key[key] += 1

            if len(keys_that_found) == 1:
                found_only_by_key[keys_that_found[0]] += 1

        # Collect missed examples.
        if missed_matches and len(missed_examples) < 50:
            for mid in list(missed_matches)[:3]:
                missed_examples.append({
                    "s1_id": s1_id,
                    "s1_name": s1_rec["business_name"],
                    "s1_addr": s1_rec["business_address"],
                    "s1_country": country,
                    "missed_id": mid,
                })

    # Compile results.
    overall_recall = total_found / total_true_matches if total_true_matches else 0.0

    cand_series = pd.Series(candidate_counts) if candidate_counts else pd.Series([0])

    return {
        "overall_recall": overall_recall,
        "total_true_matches": total_true_matches,
        "total_found": total_found,
        "total_missed": total_true_matches - total_found,
        "s1_entities_evaluated": len(candidate_counts),
        "s2_recall": s2_found / s2_true if s2_true else 0.0,
        "s2_true": s2_true,
        "s2_found": s2_found,
        "s3_recall": s3_found / s3_true if s3_true else 0.0,
        "s3_true": s3_true,
        "s3_found": s3_found,
        "found_by_key": dict(found_by_key),
        "found_only_by_key": dict(found_only_by_key),
        "candidate_count_stats": {
            "mean": cand_series.mean(),
            "p25": cand_series.quantile(0.25),
            "p50": cand_series.quantile(0.50),
            "p75": cand_series.quantile(0.75),
            "p95": cand_series.quantile(0.95),
            "p99": cand_series.quantile(0.99),
            "max": cand_series.max(),
        },
        "country_stats": {
            c: {
                "recall": s["found"] / s["true"] if s["true"] else 0.0,
                **s,
            }
            for c, s in country_stats.items()
        },
        "missed_examples": missed_examples,
    }


def print_recall_report(result: dict) -> None:
    """Print a formatted blocking recall report."""

    print("\n" + "=" * 80)
    print("BLOCKING RECALL REPORT")
    print("=" * 80)

    print(f"\n  S1 entities evaluated:   {result['s1_entities_evaluated']:,}")
    print(f"  Total true matches:      {result['total_true_matches']:,}")
    print(f"  Total found:             {result['total_found']:,}")
    print(f"  Total missed:            {result['total_missed']:,}")
    print(f"  Overall recall:          {result['overall_recall']:.4f} "
          f"({result['overall_recall']*100:.2f}%)")

    print(f"\n--- Source Breakdown ---")
    print(f"  S2 recall: {result['s2_recall']:.4f} "
          f"({result['s2_found']:,} / {result['s2_true']:,})")
    print(f"  S3 recall: {result['s3_recall']:.4f} "
          f"({result['s3_found']:,} / {result['s3_true']:,})")

    print(f"\n--- Country Breakdown ---")
    for country, stats in sorted(result["country_stats"].items()):
        print(f"  {country}: recall={stats['recall']:.4f} "
              f"({stats['found']:,} / {stats['true']:,})")

    print(f"\n--- Per-Blocker-Key Contribution ---")
    print(f"  {'Key':<20} {'Found':>8} {'Only by this key':>18}")
    print("  " + "-" * 50)

    for key in sorted(result["found_by_key"].keys()):
        found = result["found_by_key"][key]
        only = result["found_only_by_key"].get(key, 0)
        print(f"  {key:<20} {found:>8,} {only:>18,}")

    print(f"\n--- Candidate Count Distribution ---")
    stats = result["candidate_count_stats"]
    print(f"  Mean:  {stats['mean']:>8.1f}")
    print(f"  p25:   {stats['p25']:>8.0f}")
    print(f"  p50:   {stats['p50']:>8.0f}")
    print(f"  p75:   {stats['p75']:>8.0f}")
    print(f"  p95:   {stats['p95']:>8.0f}")
    print(f"  p99:   {stats['p99']:>8.0f}")
    print(f"  Max:   {stats['max']:>8.0f}")

    if result["missed_examples"]:
        print(f"\n--- Sample Missed Matches (first 10) ---")

        for i, ex in enumerate(result["missed_examples"][:10]):
            print(f"\n  Missed {i+1}:")
            print(f"    S1: {ex['s1_id']}  name={ex['s1_name']!r}")
            print(f"    S1 addr: {ex['s1_addr']!r}  country={ex['s1_country']}")
            print(f"    Missed: {ex['missed_id']}")


# ===================================================================
# CLI for standalone blocking recall evaluation
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate blocking recall."
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
        default=2000,
        help="Number of S1 entities to evaluate.",
    )

    parser.add_argument(
        "--max-s2s3-rows",
        type=int,
        default=None,
        help="Max rows to index per source (None = all).",
    )

    parser.add_argument(
        "--min-name-tokens",
        type=int,
        default=2,
        help="Min shared name tokens for token blocker.",
    )

    parser.add_argument(
        "--min-addr-tokens",
        type=int,
        default=3,
        help="Min shared address tokens for address blocker.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    random.seed(args.seed)

    train_dir = args.data_dir / "train"

    # Load ground truth.
    print("Loading ground truth...")
    t0 = time.time()

    gt: dict[str, list[str]] = {}

    for chunk in pd.read_csv(
        train_dir / "train_ground_truth.tsv",
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

    print(f"  Loaded {len(gt):,} entries in {time.time()-t0:.1f}s")

    # Sample S1 entities.
    s1_with_matches = [s1 for s1, m in gt.items() if m]
    sample_size = min(args.sample_size, len(s1_with_matches))
    sampled = random.sample(s1_with_matches, sample_size)
    print(f"  Sampled {sample_size:,} S1 entities for evaluation")

    # Figure out which S2/S3 IDs are true matches for our sample.
    needed_ids: set[str] = set()

    for s1_id in sampled:
        needed_ids.update(gt[s1_id])

    print(f"  True match IDs to find: {len(needed_ids):,}")

    # Build blocker.
    print("\nBuilding blocker indexes...")
    t0 = time.time()

    blocker = build_combined_blocker(
        train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv",
        max_rows_per_source=args.max_s2s3_rows,
    )

    build_time = time.time() - t0
    stats = blocker.stats()
    print(f"  Indexed {stats['n_indexed']:,} records in {build_time:.1f}s")

    for key, val in stats.items():
        if key != "n_indexed":
            print(f"    {key}: {val:,}")

    # Evaluate recall.
    print("\nEvaluating blocking recall...")
    t0 = time.time()

    result = evaluate_blocking_recall(
        blocker,
        train_dir / "train_source1.tsv",
        gt,
        sample_s1_ids=sampled,
        min_name_token_overlap=args.min_name_tokens,
        min_addr_token_overlap=args.min_addr_tokens,
    )

    eval_time = time.time() - t0
    print(f"  Evaluation completed in {eval_time:.1f}s")

    print_recall_report(result)
    print(f"\n  Build time: {build_time:.1f}s")
    print(f"  Eval time:  {eval_time:.1f}s")


if __name__ == "__main__":
    main()
