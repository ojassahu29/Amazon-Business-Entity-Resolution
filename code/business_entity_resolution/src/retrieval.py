"""
Production Retrieval Engine for Business Entity Resolution.

FROZEN RETRIEVAL ARCHITECTURE:
This module implements the exact frozen multi-layer retrieval architecture
validated against independent splits (seed=42 and seed=123):
  1. Combo 3: Frequency-aware multi-index primary retrieval.
  2. Secondary A: Informative address token overlap (>= 2 tokens, DF <= 2,000).
  3. C2-A: Country-constrained building/address number + address token (DF <= 500).

NOTE: Secondary B (single rare name token) was evaluated on independent validation
splits and eliminated because it contributed only 9-10 matches at the cost of
~20,000 extra candidates. Secondary B is NOT executed in production retrieval.

ENTRY POINTS:
  - `retrieve_candidates_for_record(...)`: Single-entity candidate retrieval.
  - `retrieve_candidates_batch(...)`: Batch candidate retrieval for S1 query entities.
  - `ProductionRetrievalIndex`: Fast in-memory inverted index substrate.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import heapq
from pathlib import Path
import pickle
import re
from typing import Any, Iterable

import pandas as pd

try:
    from .preprocessing import (
        compact,
        normalize_basic,
        sorted_tokens,
        tokenize,
    )
except ImportError:
    from preprocessing import (
        compact,
        normalize_basic,
        sorted_tokens,
        tokenize,
    )


# ===================================================================
# FROZEN DOMAIN STOPWORDS
# Empirical stopwords derived from 500k entity frequency analysis.
# ===================================================================

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

ADDR_STOPWORDS: frozenset[str] = frozenset({
    "road", "street", "avenue", "ave", "lane", "drive",
    "floor", "plot", "flat", "block", "sector", "colony",
    "nagar", "near", "door", "house", "main", "cross",
    "new", "old", "east", "west", "north", "south",
    "city", "null",
    # Very common Indian cities/states
    "delhi", "maharashtra", "mumbai", "bangalore",
    "karnataka", "kolkata", "pradesh", "uttar",
    "tamil", "nadu", "bengal", "gujarat", "pune",
    "telangana", "chennai", "hyderabad",
    # Hindi/regional state names
    "महाराष्ट्र",
})


# ===================================================================
# FROZEN RETRIEVAL THRESHOLDS (EXPLICIT AND DISCOVERABLE)
# DO NOT ALTER WITHOUT FORMAL EXPERIMENTAL VALIDATION.
# ===================================================================

# 1. Primary Combo 3 thresholds
COMBO3_NAME_OVERLAP_MIN: int = 2
COMBO3_ADDR_OVERLAP_MIN: int = 3
COMBO3_RARE_NAME_TOKEN_DF_MAX: int = 500
COMBO3_ADDR_NUM_LEN_MIN: int = 3
COMBO3_RARE_ADDR_NUM_DF_MAX: int = 500
COMBO3_RARE_ADDR_TOKEN_DF_MAX: int = 1000
COMBO3_RARE_ADDR_OVERLAP_MIN: int = 2
COMBO3_RARE_ADDR_OVERLAP_DF_MAX: int = 500
COMBO3_SINGLE_RARE_NAME_LEN_MIN: int = 4
COMBO3_SINGLE_RARE_NAME_DF_MAX: int = 200
COMBO3_PREFIX5_LEN_MIN: int = 5
COMBO3_PREFIX5_DF_MAX: int = 50

# 2. Secondary A thresholds
SECONDARY_A_ADDR_OVERLAP_MIN: int = 2
SECONDARY_A_ADDR_TOKEN_DF_MAX: int = 2000

# 3. C2-A thresholds
C2A_ADDR_TOKEN_DF_MAX: int = 500

# Keep strong candidates; cap only lower-evidence matches.
MAX_LOW_EVIDENCE_CANDIDATES_PER_QUERY: int = 17500
MIN_PRESERVED_EVIDENCE_WEIGHT: int = 4
EXACT_EVIDENCE_WEIGHT: int = 4
NAME_OVERLAP_EVIDENCE_WEIGHT: int = 1
ADDRESS_OVERLAP_EVIDENCE_WEIGHT: int = 3
NAME_STOPWORD_EVIDENCE_WEIGHT: int = 2
ADDRESS_NUMBER_TOKEN_EVIDENCE_WEIGHT: int = 5
RARE_ADDRESS_EVIDENCE_WEIGHT: int = 2
SINGLE_RARE_NAME_EVIDENCE_WEIGHT: int = 1
PREFIX_EVIDENCE_WEIGHT: int = 3
SECONDARY_A_EVIDENCE_WEIGHT: int = 1
C2A_EVIDENCE_WEIGHT: int = 5


# ===================================================================
# RECORD PARSING & NUMBER EXTRACTION
# ===================================================================

def extract_address_numbers(addr: str) -> set[str]:
    """
    Extract alphanumeric address/building numbers while filtering out
    recent calendar years (2019-2025).
    """
    raw_nums = re.findall(r"\b\d+[/\w-]*\b", addr)
    nums = set()
    for n in raw_nums:
        clean_n = n.strip(" ,.-/#")
        if len(clean_n) >= 2 and clean_n not in {"2019", "2020", "2021", "2022", "2023", "2024", "2025"}:
            nums.add(clean_n.lower())
    return nums


def parse_entity_record(
    business_name: str,
    business_address: str,
    country: str,
) -> dict[str, Any]:
    """
    Extract parsed, normalized, and tokenized features from an entity record.
    Preserves exact list semantics for token sequences to match validated behavior.
    """
    c = normalize_basic(country)
    name = business_name or ""
    addr = business_address or ""

    nn = normalize_basic(name)
    ns = sorted_tokens(name)
    nc = compact(name)
    p5 = nc[:COMBO3_PREFIX5_LEN_MIN] if len(nc) >= COMBO3_PREFIX5_LEN_MIN else ""

    toks_n = tokenize(name)
    toks_a = tokenize(addr)

    info_n = [t for t in toks_n if len(t) >= 3 and t not in NAME_STOPWORDS]
    stop_n = [t for t in toks_n if t in NAME_STOPWORDS]
    info_a = [t for t in toks_a if len(t) >= 3 and t not in ADDR_STOPWORDS]

    nums = extract_address_numbers(addr)
    bldg = {n for n in nums if not re.match(r"^\d{5,6}$", n)}

    return {
        "country": c,
        "raw_name": name,
        "raw_addr": addr,
        "name_norm": nn,
        "name_sorted": ns,
        "name_compact": nc,
        "prefix5": p5,
        "info_name": info_n,
        "stop_name": stop_n,
        "info_addr": info_a,
        "all_nums": nums,
        "bldg_nums": bldg,
    }


def get_token_overlap_candidates(token_sets: list[set[str]], min_overlap: int) -> set[str]:
    """
    Fast candidate set intersection requiring at least `min_overlap` shared tokens.
    Uses combinatorial intersection for small list lengths and frequency counting
    for larger lists.
    """
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


# ===================================================================
# INVERTED INDEX DATA STRUCTURE
# ===================================================================

class ProductionRetrievalIndex:
    """
    Multi-index inverted index storing candidates from the candidate space (S2 + S3).
    """

    def __init__(self, active_keys: dict[str, set[tuple[str, str]]] | None = None) -> None:
        self.active_keys = active_keys
        self.idx_name_norm: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_name_sorted: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_name_compact: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_compact_prefix5: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_name_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_name_stopwords: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_addr_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.idx_addr_numbers: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.n_indexed: int = 0

    @classmethod
    def from_cache_dict(cls, cache: dict[str, Any]) -> ProductionRetrievalIndex:
        """Construct index instance directly from a deserialized cache dict."""
        instance = cls()
        instance.idx_name_norm = cache["idx_name_norm"]
        instance.idx_name_sorted = cache["idx_name_sorted"]
        instance.idx_name_compact = cache["idx_name_compact"]
        instance.idx_compact_prefix5 = cache["idx_compact_prefix5"]
        instance.idx_name_tokens = cache["idx_name_tokens"]
        instance.idx_name_stopwords = cache["idx_name_stopwords"]
        instance.idx_addr_tokens = cache["idx_addr_tokens"]
        instance.idx_addr_numbers = cache["idx_addr_numbers"]
        instance.n_indexed = cache.get("n_indexed", sum(len(s) for s in instance.idx_name_norm.values()))
        return instance

    @classmethod
    def load_cache(cls, path: str | Path) -> ProductionRetrievalIndex:
        """Load precomputed inverted index cache from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls.from_cache_dict(data)

    def index_record(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str,
    ) -> None:
        """Index one entity, optionally retaining only keys active in the query set."""
        c = normalize_basic(country)
        name = business_name or ""
        addr = business_address or ""
        nn = normalize_basic(name)
        ns = sorted_tokens(name)
        nc = compact(name)

        key = (c, nn)
        if self.active_keys is None or key in self.active_keys["idx_name_norm"]:
            self.idx_name_norm[key].add(entity_id)
        key = (c, ns)
        if self.active_keys is None or key in self.active_keys["idx_name_sorted"]:
            self.idx_name_sorted[key].add(entity_id)
        key = (c, nc)
        if self.active_keys is None or key in self.active_keys["idx_name_compact"]:
            self.idx_name_compact[key].add(entity_id)

        if len(nc) >= COMBO3_PREFIX5_LEN_MIN:
            key = (c, nc[:COMBO3_PREFIX5_LEN_MIN])
            if self.active_keys is None or key in self.active_keys["idx_compact_prefix5"]:
                self.idx_compact_prefix5[key].add(entity_id)

        for t in tokenize(name):
            if len(t) >= 3:
                index_name = "idx_name_tokens" if t not in NAME_STOPWORDS else "idx_name_stopwords"
                index = self.idx_name_tokens if t not in NAME_STOPWORDS else self.idx_name_stopwords
                key = (c, t)
                if self.active_keys is None or key in self.active_keys[index_name]:
                    index[key].add(entity_id)

        for t in tokenize(addr):
            if len(t) >= 3 and t not in ADDR_STOPWORDS:
                key = (c, t)
                if self.active_keys is None or key in self.active_keys["idx_addr_tokens"]:
                    self.idx_addr_tokens[key].add(entity_id)

        for num in extract_address_numbers(addr):
            key = (c, num)
            if self.active_keys is None or key in self.active_keys["idx_addr_numbers"]:
                self.idx_addr_numbers[key].add(entity_id)

        self.n_indexed += 1


# ===================================================================
# CANONICAL PRODUCTION RETRIEVAL ENTRY POINTS
# ===================================================================

def _limit_candidates(
    candidates: set[str],
    evidence_sets: list[tuple[set[str], int]],
    limit: int,
) -> set[str]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates
    support: dict[str, int] = defaultdict(int)
    preserved: set[str] = set()
    for evidence, weight in evidence_sets:
        for entity_id in evidence:
            if entity_id in candidates:
                support[entity_id] += weight
                if weight >= MIN_PRESERVED_EVIDENCE_WEIGHT:
                    preserved.add(entity_id)
    weak_candidates = (entity_id for entity_id in candidates if entity_id not in preserved)
    ranked_weak = heapq.nsmallest(limit, weak_candidates, key=lambda entity_id: (-support.get(entity_id, 0), entity_id))
    return preserved | set(ranked_weak)


def retrieve_candidates_for_record(
    parsed: dict[str, Any],
    index: ProductionRetrievalIndex,
) -> set[str]:
    """
    Retrieve candidate entity IDs for a single parsed query record.

    Executes exactly:
      1. Primary Combo 3
      2. Secondary A (DF <= 2000)
      3. C2-A (DF <= 500)
    """
    c = parsed["country"]
    cap = MAX_LOW_EVIDENCE_CANDIDATES_PER_QUERY
    evidence_sets: list[tuple[set[str], int]] | None = [] if cap > 0 else None

    # -------------------------------------------------------------
    # 1. COMBO 3: Primary Frequency-Aware Multi-Index Retrieval
    # -------------------------------------------------------------
    c1 = index.idx_name_norm.get((c, parsed["name_norm"]), set())
    c2 = index.idx_name_sorted.get((c, parsed["name_sorted"]), set())
    c3 = index.idx_name_compact.get((c, parsed["name_compact"]), set())

    name_sets = [index.idx_name_tokens.get((c, t), set()) for t in parsed["info_name"] if (c, t) in index.idx_name_tokens]
    c4 = get_token_overlap_candidates(name_sets, min_overlap=COMBO3_NAME_OVERLAP_MIN)
    addr_sets = [index.idx_addr_tokens.get((c, t), set()) for t in parsed["info_addr"] if (c, t) in index.idx_addr_tokens]
    c5 = get_token_overlap_candidates(addr_sets, min_overlap=COMBO3_ADDR_OVERLAP_MIN)
    exact = c1 | c2 | c3
    c_set: set[str] = exact | c4 | c5
    if evidence_sets is not None:
        evidence_sets.extend((
            (exact, EXACT_EVIDENCE_WEIGHT),
            (c4, NAME_OVERLAP_EVIDENCE_WEIGHT),
            (c5, ADDRESS_OVERLAP_EVIDENCE_WEIGHT),
        ))

    rare_info = [
        index.idx_name_tokens[(c, t)]
        for t in parsed["info_name"]
        if (c, t) in index.idx_name_tokens and len(index.idx_name_tokens[(c, t)]) <= COMBO3_RARE_NAME_TOKEN_DF_MAX
    ]
    if rare_info and parsed["stop_name"]:
        s_union = set()
        for st in parsed["stop_name"]:
            s_union |= index.idx_name_stopwords.get((c, st), set())
        if s_union:
            if evidence_sets is None:
                for n_set in rare_info:
                    c_set |= n_set & s_union
            else:
                signal = set()
                for n_set in rare_info:
                    signal |= n_set & s_union
                c_set |= signal
                if signal:
                    evidence_sets.append((signal, NAME_STOPWORD_EVIDENCE_WEIGHT))

    valid_nums = [
        n for n in parsed["all_nums"]
        if len(n) >= COMBO3_ADDR_NUM_LEN_MIN and (c, n) in index.idx_addr_numbers and len(index.idx_addr_numbers[(c, n)]) <= COMBO3_RARE_ADDR_NUM_DF_MAX
    ]
    valid_addrs = [
        index.idx_addr_tokens[(c, t)]
        for t in parsed["info_addr"]
        if (c, t) in index.idx_addr_tokens and len(index.idx_addr_tokens[(c, t)]) <= COMBO3_RARE_ADDR_TOKEN_DF_MAX
    ]
    if valid_nums and valid_addrs:
        a_union = set().union(*valid_addrs)
        if evidence_sets is None:
            for num in valid_nums:
                c_set |= index.idx_addr_numbers[(c, num)] & a_union
        else:
            signal = set()
            for num in valid_nums:
                signal |= index.idx_addr_numbers[(c, num)] & a_union
            c_set |= signal
            if signal:
                evidence_sets.append((signal, ADDRESS_NUMBER_TOKEN_EVIDENCE_WEIGHT))

    rare_addrs = [
        index.idx_addr_tokens[(c, t)]
        for t in parsed["info_addr"]
        if (c, t) in index.idx_addr_tokens and len(index.idx_addr_tokens[(c, t)]) <= COMBO3_RARE_ADDR_OVERLAP_DF_MAX
    ]
    if len(rare_addrs) >= COMBO3_RARE_ADDR_OVERLAP_MIN:
        signal = get_token_overlap_candidates(rare_addrs, min_overlap=COMBO3_RARE_ADDR_OVERLAP_MIN)
        c_set |= signal
        if evidence_sets is not None and signal:
            evidence_sets.append((signal, RARE_ADDRESS_EVIDENCE_WEIGHT))

    if evidence_sets is None:
        for t in parsed["info_name"]:
            if (
                len(t) >= COMBO3_SINGLE_RARE_NAME_LEN_MIN
                and (c, t) in index.idx_name_tokens
                and len(index.idx_name_tokens[(c, t)]) <= COMBO3_SINGLE_RARE_NAME_DF_MAX
            ):
                c_set |= index.idx_name_tokens[(c, t)]
    else:
        signal = set()
        for t in parsed["info_name"]:
            if (
                len(t) >= COMBO3_SINGLE_RARE_NAME_LEN_MIN
                and (c, t) in index.idx_name_tokens
                and len(index.idx_name_tokens[(c, t)]) <= COMBO3_SINGLE_RARE_NAME_DF_MAX
            ):
                signal |= index.idx_name_tokens[(c, t)]
        c_set |= signal
        if signal:
            evidence_sets.append((signal, SINGLE_RARE_NAME_EVIDENCE_WEIGHT))

    p5 = parsed["prefix5"]
    if p5 and (c, p5) in index.idx_compact_prefix5 and len(index.idx_compact_prefix5[(c, p5)]) <= COMBO3_PREFIX5_DF_MAX:
        signal = index.idx_compact_prefix5[(c, p5)]
        c_set |= signal
        if evidence_sets is not None:
            evidence_sets.append((signal, PREFIX_EVIDENCE_WEIGHT))

    # -------------------------------------------------------------
    # 2. SECONDARY A: >= 2 shared informative address tokens (DF <= 2000)
    # -------------------------------------------------------------
    sec_a_rare = [
        index.idx_addr_tokens[(c, t)]
        for t in parsed["info_addr"]
        if (c, t) in index.idx_addr_tokens and len(index.idx_addr_tokens[(c, t)]) <= SECONDARY_A_ADDR_TOKEN_DF_MAX
    ]
    if len(sec_a_rare) >= SECONDARY_A_ADDR_OVERLAP_MIN:
        signal = get_token_overlap_candidates(sec_a_rare, min_overlap=SECONDARY_A_ADDR_OVERLAP_MIN)
        c_set |= signal
        if evidence_sets is not None and signal:
            evidence_sets.append((signal, SECONDARY_A_EVIDENCE_WEIGHT))

    # -------------------------------------------------------------
    # 3. C2-A: Country + Building Number + >= 1 Address Token (DF <= 500)
    # -------------------------------------------------------------
    b_nums = [n for n in parsed["bldg_nums"] if (c, n) in index.idx_addr_numbers]
    a_toks = [
        index.idx_addr_tokens[(c, t)]
        for t in parsed["info_addr"]
        if (c, t) in index.idx_addr_tokens and len(index.idx_addr_tokens[(c, t)]) <= C2A_ADDR_TOKEN_DF_MAX
    ]
    if b_nums and a_toks:
        a_u = set().union(*a_toks)
        if evidence_sets is None:
            for n in b_nums:
                c_set |= index.idx_addr_numbers[(c, n)] & a_u
        else:
            signal = set()
            for n in b_nums:
                signal |= index.idx_addr_numbers[(c, n)] & a_u
            c_set |= signal
            if signal:
                evidence_sets.append((signal, C2A_EVIDENCE_WEIGHT))

    return c_set if evidence_sets is None else _limit_candidates(c_set, evidence_sets, cap)


def retrieve_candidates_batch(
    records: dict[str, dict[str, str]] | Iterable[tuple[str, str, str, str]],
    index: ProductionRetrievalIndex,
) -> dict[str, set[str]]:
    """
    Batch retrieval entry point.
    Maps query entity_id -> set of retrieved candidate entity IDs.
    """
    candidates_by_id: dict[str, set[str]] = {}

    if isinstance(records, dict):
        iterable = (
            (eid, d["business_name"], d["business_address"], d["country"])
            for eid, d in records.items()
        )
    else:
        iterable = records

    for eid, name, addr, country in iterable:
        parsed = parse_entity_record(name, addr, country)
        candidates_by_id[eid] = retrieve_candidates_for_record(parsed, index)

    return candidates_by_id
