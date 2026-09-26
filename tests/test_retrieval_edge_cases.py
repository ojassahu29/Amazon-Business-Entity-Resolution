"""
Deterministic Edge-Case Regression Tests for Production Retrieval and Preprocessing.

Tests cover:
  1. Empty address fields and missing address info
  2. Empty / absent token collections
  3. Repeated tokens and set-based overlap semantics
  4. Duplicate blocker candidates and candidate deduplication
  5. Non-Latin scripts (Devanagari, Telugu, Tamil, etc.)
  6. Mixed-script text
  7. Punctuation-heavy names and addresses
  8. Legal suffixes and stopword interactions
  9. Country normalization and country partitioning
 10. Address-number extraction (calendar year exclusion, pincode separation)
 11. S1 entities with zero retrieved candidates
 12. S1 entities with multiple candidates spanning S2 and S3
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure source package is in path
SRC_DIR = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import (
    compact,
    normalize_basic,
    normalize_country,
    sorted_tokens,
    tokenize,
)
from retrieval import (
    ADDR_STOPWORDS,
    NAME_STOPWORDS,
    ProductionRetrievalIndex,
    extract_address_numbers,
    get_token_overlap_candidates,
    parse_entity_record,
    retrieve_candidates_for_record,
)


# ===================================================================
# 1. EMPTY ADDRESS FIELDS & MISSING ADDRESS INFO
# ===================================================================

def test_empty_address_field():
    """Verify that an empty or None address does not raise and queries name blockers only."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_101", "Acme Industrial Solutions", "123 Main Road", "India")
    index.index_record("s3_102", "Globex Global", "", "India")

    # S1 with empty address
    parsed = parse_entity_record("Acme Industrial Solutions", "", "India")
    assert parsed["info_addr"] == []
    assert parsed["all_nums"] == set()
    assert parsed["bldg_nums"] == set()

    cands = retrieve_candidates_for_record(parsed, index)
    assert "s2_101" in cands


def test_missing_address_none():
    """Verify that None address is treated gracefully as empty string."""
    parsed = parse_entity_record("Acme Corp", None, "India")
    assert parsed["raw_addr"] == ""
    assert parsed["info_addr"] == []
    assert parsed["all_nums"] == set()


# ===================================================================
# 2. EMPTY / ABSENT TOKEN COLLECTIONS
# ===================================================================

def test_empty_record():
    """Verify completely empty name and address returns empty candidate set."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_1", "Beta Corp", "Sector 5", "India")

    parsed = parse_entity_record("", "", "India")
    cands = retrieve_candidates_for_record(parsed, index)
    assert cands == set()


def test_short_tokens_only():
    """Tokens shorter than 3 characters should not enter informative token lists."""
    parsed = parse_entity_record("A B CD", "X Y Z 1", "India")
    assert parsed["info_name"] == []
    assert parsed["info_addr"] == []


def test_only_stopwords_record():
    """Records consisting solely of stopwords should not fire token-overlap blockers."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_99", "Limited Private Company", "Road Street", "India")

    # Name is only stopwords
    parsed = parse_entity_record("Company Limited", "Street Road", "India")
    assert parsed["info_name"] == []
    assert parsed["info_addr"] == []
    assert "limited" in parsed["stop_name"]
    assert "company" in parsed["stop_name"]


# ===================================================================
# 3. REPEATED TOKENS & SET-BASED OVERLAP SEMANTICS
# ===================================================================

def test_repeated_name_tokens_semantics():
    """
    Verify exact set semantics when an entity has repeated tokens.
    In the validated implementation, tokenize preserves duplicate tokens in lists.
    When passed to get_token_overlap_candidates with min_overlap=2,
    pairwise set intersection (a & b) where a == b evaluates to a.
    """
    index = ProductionRetrievalIndex()
    # s2_1 has only 1 occurrence of 'solaris'
    index.index_record("s2_1", "Solaris Tech", "Tech Park", "India")

    # Query with repeated token 'solaris solaris'
    parsed = parse_entity_record("Solaris Solaris Systems", "Tech Park", "India")
    assert parsed["info_name"].count("solaris") == 2

    # Query execution
    cands = retrieve_candidates_for_record(parsed, index)
    assert "s2_1" in cands


def test_get_token_overlap_combinatorial_vs_counting():
    """Verify that get_token_overlap_candidates returns consistent results across branches."""
    set_a = {"e1", "e2", "e3"}
    set_b = {"e2", "e3", "e4"}
    set_c = {"e3", "e4", "e5"}

    overlap_2 = get_token_overlap_candidates([set_a, set_b, set_c], min_overlap=2)
    assert overlap_2 == {"e2", "e3", "e4"}

    overlap_3 = get_token_overlap_candidates([set_a, set_b, set_c], min_overlap=3)
    assert overlap_3 == {"e3"}


# ===================================================================
# 4. DUPLICATE BLOCKER CANDIDATES & DEDUPLICATION
# ===================================================================

def test_candidate_deduplication():
    """
    When an entity matches on multiple blockers (exact norm, sorted, compact, token overlap),
    the returned set must contain the candidate ID exactly once.
    """
    index = ProductionRetrievalIndex()
    # Matches exact, sorted, compact, and tokens
    index.index_record("s2_dup", "Apex Dynamics International", "Plot 10 Industrial Area", "India")

    parsed = parse_entity_record("Apex Dynamics International", "Plot 10 Industrial Area", "India")
    cands = retrieve_candidates_for_record(parsed, index)
    assert isinstance(cands, set)
    assert len(cands) == 1
    assert "s2_dup" in cands


# ===================================================================
# 5. NON-LATIN SCRIPTS (INDIC LANGUAGES)
# ===================================================================

def test_devanagari_script_retrieval():
    """Verify Unicode preservation and stopword recognition for Hindi/Devanagari."""
    index = ProductionRetrievalIndex()
    # रिलायंस इंडस्ट्रीज प्राइवेट लिमिटेड (Reliance Industries Private Limited)
    index.index_record("s2_hin_1", "रिलायंस इंडस्ट्रीज लिमिटेड", "मुंबई महाराष्ट्र", "India")

    parsed = parse_entity_record("रिलायंस इंडस्ट्रीज", "मुंबई", "India")
    assert "रिलायंस" in parsed["info_name"]
    assert "इंडस्ट्रीज" in parsed["info_name"]

    cands = retrieve_candidates_for_record(parsed, index)
    assert "s2_hin_1" in cands


def test_devanagari_stopword_filtering():
    """Verify Devanagari legal suffix stopwords are properly filtered."""
    assert "लिमिटेड" in NAME_STOPWORDS
    assert "प्राइवेट" in NAME_STOPWORDS

    parsed = parse_entity_record("रिलायंस लिमिटेड", "", "India")
    assert "लिमिटेड" in parsed["stop_name"]
    assert "लिमिटेड" not in parsed["info_name"]


def test_telugu_script_normalization():
    """Verify Telugu characters and legal suffixes."""
    assert "లిమిటెడ్" in NAME_STOPWORDS
    parsed = parse_entity_record("హైదరాబాద్ ఎంటర్‌ప్రైజెస్ లిమిటెడ్", "", "India")
    assert "లిమిటెడ్" in parsed["stop_name"]
    assert "లిమిటెడ్" not in parsed["info_name"]


# ===================================================================
# 6. MIXED-SCRIPT TEXT
# ===================================================================

def test_mixed_script_handling():
    """Verify mixed Latin and Indic text in name and address."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_mix", "Tata टाटा Motors मोटर्स", "Pune पुणे Maharashtra", "India")

    parsed = parse_entity_record("Tata Motors", "Pune", "India")
    cands = retrieve_candidates_for_record(parsed, index)
    assert "s2_mix" in cands


# ===================================================================
# 7. PUNCTUATION-HEAVY NAMES & ADDRESSES
# ===================================================================

def test_punctuation_heavy_normalization():
    """Punctuation should be replaced by spaces and collapsed without mangling tokens."""
    raw = "A.B.C. & Co., (India) Pvt. Ltd. / #404-B, 2nd Cross, St. John's Rd."
    norm = normalize_basic(raw)
    assert "." not in norm
    assert "&" not in norm
    assert "#" not in norm
    assert "/" not in norm
    assert "'" not in norm
    assert "(" not in norm
    assert ")" not in norm
    assert "a b c co india pvt ltd 404 b 2nd cross st john s rd" == norm


def test_punctuation_heavy_retrieval():
    """Entity with punctuation-heavy representation matches clean query."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_punct", "O'Connor & Sons (Holdings) Ltd.", "#12/A, Park-Way Blvd.", "India")

    parsed = parse_entity_record("O Connor Sons Holdings Ltd", "12 A Park Way Blvd", "India")
    cands = retrieve_candidates_for_record(parsed, index)
    assert "s2_punct" in cands


# ===================================================================
# 8. LEGAL SUFFIXES & STOPWORD INTERACTIONS
# ===================================================================

def test_rare_name_plus_stopword_rule():
    """
    Test the Combo 3 rule: rare informative name token (DF <= 500)
    co-occurring with a name stopword.
    """
    index = ProductionRetrievalIndex()
    # 'zyxwv' is unique (DF=1 <= 500) and 'limited' is a stopword
    index.index_record("s2_rare_stop", "Zyxwv Enterprises Limited", "Address 1", "India")

    parsed = parse_entity_record("Zyxwv Limited", "Address 2", "India")
    assert "zyxwv" in parsed["info_name"]
    assert "limited" in parsed["stop_name"]

    cands = retrieve_candidates_for_record(parsed, index)
    assert "s2_rare_stop" in cands


# ===================================================================
# 9. COUNTRY NORMALIZATION & PARTITIONING
# ===================================================================

def test_country_normalization_variants():
    """Country field case and whitespace normalization."""
    assert normalize_country("INDIA") == "india"
    assert normalize_country("  India  ") == "india"
    assert normalize_country("inDia") == "india"


def test_cross_country_isolation():
    """Entities in different countries must never match."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_us", "Acme Corporation", "100 Broadway New York", "United States")
    index.index_record("s2_in", "Acme Corporation", "100 Broadway Mumbai", "India")

    # Query in India
    parsed_in = parse_entity_record("Acme Corporation", "100 Broadway", "India")
    cands_in = retrieve_candidates_for_record(parsed_in, index)
    assert "s2_in" in cands_in
    assert "s2_us" not in cands_in


# ===================================================================
# 10. ADDRESS-NUMBER EXTRACTION
# ===================================================================

def test_address_number_extraction_calendar_years():
    """Years 2019-2025 should be excluded from address numbers."""
    addr = "Registered in 2021, Plot 42, Floor 3, Year 2024, Suite 105"
    nums = extract_address_numbers(addr)
    assert "2021" not in nums
    assert "2024" not in nums
    assert "42" in nums
    assert "105" in nums


def test_address_number_pincode_separation_for_c2a():
    """
    5-6 digit postal codes are kept in all_nums (for Combo 3)
    but excluded from bldg_nums (for C2-A).
    """
    addr = "Building 45, Sector 12, Pincode 560001"
    parsed = parse_entity_record("Sample Corp", addr, "India")

    assert "560001" in parsed["all_nums"]
    assert "560001" not in parsed["bldg_nums"]
    assert "45" in parsed["bldg_nums"]
    assert "12" in parsed["bldg_nums"]


def test_c2a_building_number_matching():
    """
    Test C2-A retrieval: country + building number + shared address token (DF <= 500).
    """
    index = ProductionRetrievalIndex()
    # Different business names, but shared building number '88-b' and address token 'chambers'
    index.index_record("s2_c2a", "Old Firm", "88-B Windsor Chambers MG Road", "India")

    parsed = parse_entity_record("New Firm", "Suite 88-B Windsor Chambers", "India")
    cands = retrieve_candidates_for_record(parsed, index)
    assert "s2_c2a" in cands


# ===================================================================
# 11. S1 ENTITIES WITH ZERO RETRIEVED CANDIDATES
# ===================================================================

def test_zero_candidate_retrieval():
    """Completely novel entity query returns an empty set cleanly."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_existing", "Alpha Beta Gamma", "Delta Epsilon", "India")

    parsed = parse_entity_record("Zeta Theta Iota", "Kappa Lambda", "India")
    cands = retrieve_candidates_for_record(parsed, index)
    assert cands == set()


# ===================================================================
# 12. S1 ENTITIES WITH MULTIPLE CANDIDATES SPANNING S2 AND S3
# ===================================================================

def test_s2_and_s3_multi_candidate_retrieval():
    """Verify that retrieval simultaneously collects candidates from both S2 and S3."""
    index = ProductionRetrievalIndex()
    index.index_record("s2_target_1", "Zenith Power Solutions", "Industrial Area", "India")
    index.index_record("s3_target_2", "Zenith Power Systems", "Industrial Area", "India")
    index.index_record("s2_target_3", "Zenith Power Corp", "City Center", "India")

    parsed = parse_entity_record("Zenith Power Limited", "Industrial Area", "India")
    cands = retrieve_candidates_for_record(parsed, index)

    assert "s2_target_1" in cands
    assert "s3_target_2" in cands
    assert "s2_target_3" in cands
