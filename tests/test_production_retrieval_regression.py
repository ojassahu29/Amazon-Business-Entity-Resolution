"""
Automated Production Retrieval Regression Test.

Validates that `code.business_entity_resolution.src.retrieval` produces
the exact bit-for-bit candidate-pair identity set recorded in the frozen
Phase 2 golden reference on seed=42 against the full S2+S3 universe.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import random
import sys

import pandas as pd
import pytest

# Ensure source package is in path
SRC_DIR = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import (
    ProductionRetrievalIndex,
    retrieve_candidates_batch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_PATH = REPO_ROOT / "output" / "golden_reference_seed42.pkl"
CACHE_PATH = REPO_ROOT / "output" / "canonical_combo3_index_cache.pkl"
GT_PATH = REPO_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
S1_PATH = REPO_ROOT / "dataset" / "train" / "train_source1.tsv"

EXPECTED_SHA256 = "873c791862d91ae0c91f26047c4787af50de93e32d06db878e0d5802956f2c5c"
EXPECTED_CANDIDATE_PAIRS = 8_555_167
EXPECTED_TRUE_MATCHES = 16_370
TOTAL_GT_MATCHES = 17_314


@pytest.mark.skipif(
    not (REF_PATH.exists() and CACHE_PATH.exists() and GT_PATH.exists() and S1_PATH.exists()),
    reason="Full regression requires local dataset, cache, and golden reference files.",
)
def test_production_candidate_pair_exact_regression():
    """Verify bit-for-bit candidate set equality and SHA-256 fingerprint."""
    with open(REF_PATH, "rb") as f:
        golden_candidates: dict[str, list[str]] = pickle.load(f)

    # Load 5000 S1 sample
    gt: dict[str, list[str]] = {}
    for chunk in pd.read_csv(GT_PATH, sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
        for row in chunk.itertuples(index=False):
            m = row.matched_entity_ids.strip()
            gt[row.source1_entity_id] = [x.strip() for x in m.split(",") if x.strip()] if m else []

    val_s1_ids = random.Random(42).sample(sorted(gt.keys()), 5000)
    val_s1_set = set(val_s1_ids)

    val_s1_records: dict[str, dict[str, str]] = {}
    for chunk in pd.read_csv(S1_PATH, sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
        mask = chunk["entity_id"].isin(val_s1_set)
        for eid, name, addr, country in zip(
            chunk.loc[mask, "entity_id"],
            chunk.loc[mask, "business_name"],
            chunk.loc[mask, "business_address"],
            chunk.loc[mask, "country"],
        ):
            val_s1_records[eid] = {
                "business_name": name,
                "business_address": addr,
                "country": country,
            }

    index = ProductionRetrievalIndex.load_cache(CACHE_PATH)
    ordered_s1_records = [
        (s1_id, val_s1_records[s1_id]["business_name"], val_s1_records[s1_id]["business_address"], val_s1_records[s1_id]["country"])
        for s1_id in val_s1_ids
    ]
    new_candidates = retrieve_candidates_batch(ordered_s1_records, index)

    hasher = hashlib.sha256()
    new_total_pairs = 0
    total_matches = 0
    missing = 0
    extra = 0

    for s1_id in val_s1_ids:
        old_set = set(golden_candidates[s1_id])
        new_set = new_candidates[s1_id]

        missing += len(old_set - new_set)
        extra += len(new_set - old_set)

        gt_matches = new_set & set(gt[s1_id])
        total_matches += len(gt_matches)

        sorted_cands = sorted(new_set)
        new_total_pairs += len(sorted_cands)
        for cand_id in sorted_cands:
            hasher.update(f"{s1_id}\t{cand_id}\n".encode("utf-8"))

    new_sha256 = hasher.hexdigest()

    assert missing == 0, f"Candidate regression failed: {missing} missing candidate pairs!"
    assert extra == 0, f"Candidate regression failed: {extra} unexpected extra candidate pairs!"
    assert new_total_pairs == EXPECTED_CANDIDATE_PAIRS
    assert total_matches == EXPECTED_TRUE_MATCHES
    assert new_sha256 == EXPECTED_SHA256
