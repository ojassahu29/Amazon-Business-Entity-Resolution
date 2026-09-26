from __future__ import annotations

from pathlib import Path
from typing import Iterator

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


class DatasetPaths:
    """Paths to all files in the challenge dataset."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"

        self.train_source1 = self.train_dir / "train_source1.tsv"
        self.train_source2 = self.train_dir / "train_source2.tsv"
        self.train_source3 = self.train_dir / "train_source3.tsv"
        self.train_ground_truth = (
            self.train_dir / "train_ground_truth.tsv"
        )

        self.test_source1 = self.test_dir / "test_source1.tsv"
        self.test_source2 = self.test_dir / "test_source2.tsv"
        self.test_source3 = self.test_dir / "test_source3.tsv"

    def validate(self) -> None:
        required = [
            self.train_source1,
            self.train_source2,
            self.train_source3,
            self.train_ground_truth,
            self.test_source1,
            self.test_source2,
            self.test_source3,
        ]

        missing = [path for path in required if not path.exists()]

        if missing:
            message = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Missing required dataset files:\n{message}"
            )


def iter_source(
    path: str | Path,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """
    Iterate over a source TSV without loading the entire file into memory.
    """

    yield from pd.read_csv(
        path,
        sep="\t",
        usecols=SOURCE_COLUMNS,
        dtype="string",
        keep_default_na=False,
        chunksize=chunksize,
    )


def iter_ground_truth(
    path: str | Path,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """
    Iterate over the ground-truth TSV.
    """

    yield from pd.read_csv(
        path,
        sep="\t",
        usecols=GROUND_TRUTH_COLUMNS,
        dtype="string",
        keep_default_na=False,
        chunksize=chunksize,
    )


def load_source_sample(
    path: str | Path,
    nrows: int = 10_000,
) -> pd.DataFrame:
    """
    Load a small source sample for development/debugging.
    """

    return pd.read_csv(
        path,
        sep="\t",
        usecols=SOURCE_COLUMNS,
        dtype="string",
        keep_default_na=False,
        nrows=nrows,
    )


def load_ground_truth_sample(
    path: str | Path,
    nrows: int = 10_000,
) -> pd.DataFrame:
    """
    Load a small ground-truth sample.
    """

    return pd.read_csv(
        path,
        sep="\t",
        usecols=GROUND_TRUTH_COLUMNS,
        dtype="string",
        keep_default_na=False,
        nrows=nrows,
    )


def parse_match_ids(value: str) -> list[str]:
    """
    Convert the comma-separated ground-truth field into entity IDs.

    Empty ground-truth fields become [].
    """

    value = str(value).strip()

    if not value:
        return []

    return [
        entity_id.strip()
        for entity_id in value.split(",")
        if entity_id.strip()
    ]