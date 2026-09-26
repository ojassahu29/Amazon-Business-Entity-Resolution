from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


def profile_source_file(
    path: Path,
    chunksize: int = 250_000,
) -> dict:
    """Profile a source TSV without loading the entire file into memory."""

    total_rows = 0
    duplicate_ids = 0

    empty_name = 0
    empty_address = 0
    empty_country = 0

    country_counts: Counter[str] = Counter()

    seen_ids: set[str] = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=SOURCE_COLUMNS,
        dtype="string",
        chunksize=chunksize,
        keep_default_na=False,
    ):
        total_rows += len(chunk)

        ids = chunk["entity_id"]

        # Duplicate IDs within the file.
        duplicate_ids += int(ids.duplicated().sum())

        # Cross-chunk duplicate detection.
        for entity_id in ids:
            if entity_id in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(entity_id)

        empty_name += int(
            chunk["business_name"].str.strip().eq("").sum()
        )

        empty_address += int(
            chunk["business_address"].str.strip().eq("").sum()
        )

        empty_country += int(
            chunk["country"].str.strip().eq("").sum()
        )

        country_counts.update(
            chunk["country"].str.strip().tolist()
        )

    return {
        "file": str(path),
        "rows": total_rows,
        "unique_entity_ids": len(seen_ids),
        "duplicate_entity_ids": duplicate_ids,
        "empty_business_name": empty_name,
        "empty_business_address": empty_address,
        "empty_country": empty_country,
        "countries": dict(country_counts.most_common()),
    }


def profile_ground_truth(
    path: Path,
    chunksize: int = 250_000,
) -> dict:
    """Profile the training ground-truth mapping."""

    total_rows = 0
    empty_match_lists = 0
    nonempty_match_lists = 0

    total_matches = 0
    match_count_distribution: Counter[int] = Counter()

    source2_matches = 0
    source3_matches = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=GROUND_TRUTH_COLUMNS,
        dtype="string",
        chunksize=chunksize,
        keep_default_na=False,
    ):
        total_rows += len(chunk)

        for value in chunk["matched_entity_ids"]:
            value = value.strip()

            if not value:
                empty_match_lists += 1
                match_count_distribution[0] += 1
                continue

            ids = [x for x in value.split(",") if x]

            nonempty_match_lists += 1
            total_matches += len(ids)
            match_count_distribution[len(ids)] += 1

            source2_matches += sum(
                entity_id.startswith("S2-")
                for entity_id in ids
            )

            source3_matches += sum(
                entity_id.startswith("S3-")
                for entity_id in ids
            )

    return {
        "file": str(path),
        "rows": total_rows,
        "empty_match_lists": empty_match_lists,
        "nonempty_match_lists": nonempty_match_lists,
        "total_matches": total_matches,
        "average_matches_per_s1": (
            total_matches / total_rows if total_rows else 0.0
        ),
        "source2_matches": source2_matches,
        "source3_matches": source3_matches,
        "match_count_distribution": dict(
            sorted(match_count_distribution.items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the Amazon Business Entity Resolution dataset."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the challenge dataset directory.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Number of rows processed per chunk.",
    )

    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"

    source_files = [
        train_dir / "train_source1.tsv",
        train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv",
        test_dir / "test_source1.tsv",
        test_dir / "test_source2.tsv",
        test_dir / "test_source3.tsv",
    ]

    ground_truth = train_dir / "train_ground_truth.tsv"

    print("\n" + "=" * 80)
    print("SOURCE DATA PROFILE")
    print("=" * 80)

    for path in source_files:
        print(f"\nProfiling: {path}")

        result = profile_source_file(
            path,
            chunksize=args.chunksize,
        )

        print(f"Rows:                 {result['rows']:,}")
        print(f"Unique entity IDs:    {result['unique_entity_ids']:,}")
        print(f"Duplicate IDs:        {result['duplicate_entity_ids']:,}")
        print(f"Empty business names:  {result['empty_business_name']:,}")
        print(f"Empty addresses:      {result['empty_business_address']:,}")
        print(f"Empty countries:      {result['empty_country']:,}")

        print("Countries:")
        for country, count in result["countries"].items():
            print(f"  {country!r}: {count:,}")

    print("\n" + "=" * 80)
    print("GROUND TRUTH PROFILE")
    print("=" * 80)

    result = profile_ground_truth(
        ground_truth,
        chunksize=args.chunksize,
    )

    print(f"Rows:                 {result['rows']:,}")
    print(f"Empty match lists:    {result['empty_match_lists']:,}")
    print(f"Non-empty match lists:{result['nonempty_match_lists']:,}")
    print(f"Total matches:        {result['total_matches']:,}")
    print(
        f"Average matches/S1:  "
        f"{result['average_matches_per_s1']:.4f}"
    )
    print(f"S2 matches:           {result['source2_matches']:,}")
    print(f"S3 matches:           {result['source3_matches']:,}")

    print("\nMatch-count distribution:")

    for count, frequency in result["match_count_distribution"].items():
        print(
            f"  {count:>3} matches: "
            f"{frequency:,} S1 entities"
        )


if __name__ == "__main__":
    main()