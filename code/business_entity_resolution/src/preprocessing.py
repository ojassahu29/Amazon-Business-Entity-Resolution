from __future__ import annotations

import unicodedata

import pandas as pd


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text while preserving multilingual characters.

    NFKC handles compatibility variants without transliterating
    non-Latin scripts.
    """

    return unicodedata.normalize("NFKC", str(text))


def is_punctuation(char: str) -> bool:
    """
    Return True for Unicode punctuation characters.

    Unlike regex \\w, this preserves combining marks used by
    Indic and other writing systems.
    """

    return unicodedata.category(char).startswith("P")


def normalize_basic(text: str) -> str:
    """
    Conservative Unicode-safe normalization.

    Preserves:
    - letters
    - numbers
    - combining marks
    - characters from non-Latin scripts

    Removes:
    - punctuation

    Also:
    - case-folds text
    - normalizes whitespace
    """

    text = normalize_unicode(text)
    text = text.casefold()

    chars = []

    for char in text:
        if is_punctuation(char):
            chars.append(" ")
        else:
            chars.append(char)

    text = "".join(chars)

    # Unicode-aware whitespace normalization.
    text = " ".join(text.split())

    return text.strip()


def compact(text: str) -> str:
    """
    Remove spaces from an already normalized representation.
    """

    return normalize_basic(text).replace(" ", "")


def tokenize(text: str) -> tuple[str, ...]:
    """
    Return normalized whitespace-separated tokens.
    """

    normalized = normalize_basic(text)

    if not normalized:
        return ()

    return tuple(normalized.split())


def sorted_tokens(text: str) -> str:
    """
    Create a token-order-independent representation.

    Example:
        Alpha Technologies Inc
        -> alpha inc technologies
    """

    return " ".join(sorted(tokenize(text)))


def normalize_country(text: str) -> str:
    """
    Normalize a country label without assuming a fixed vocabulary.
    """

    return normalize_basic(text)


def add_normalized_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add conservative normalized representations to a source dataframe.
    """

    result = df.copy()

    result["name_norm"] = (
        result["business_name"]
        .fillna("")
        .map(normalize_basic)
    )

    result["name_compact"] = (
        result["business_name"]
        .fillna("")
        .map(compact)
    )

    result["name_sorted"] = (
        result["business_name"]
        .fillna("")
        .map(sorted_tokens)
    )

    result["address_norm"] = (
        result["business_address"]
        .fillna("")
        .map(normalize_basic)
    )

    result["address_compact"] = (
        result["business_address"]
        .fillna("")
        .map(compact)
    )

    result["country_norm"] = (
        result["country"]
        .fillna("")
        .map(normalize_country)
    )

    result["address_missing"] = (
        result["address_norm"].eq("")
    )

    return result