from __future__ import annotations

from collections import defaultdict
from math import fsum


Group = tuple[str, str]


def cosine_score(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    return fsum(a * b for a, b in zip(left, right, strict=True))


def rank_probe_rows(rows: list[dict[str, str]], query_vectors: dict[str, list[float]], target_vectors: dict[str, list[float]]) -> dict[Group, list[dict[str, object]]]:
    grouped: dict[Group, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        s1_id, target_id = row["s1_id"], row["target_id"]
        grouped[(row["bucket"], s1_id)].append({
            "target_id": target_id,
            "label": int(row["label"]),
            "similarity": cosine_score(query_vectors[s1_id], target_vectors[target_id]),
        })
    for candidates in grouped.values():
        candidates.sort(key=lambda candidate: (-candidate["similarity"], candidate["target_id"]))
    return dict(grouped)


def rank_lexical_rows(rows: list[dict[str, str]]) -> dict[Group, list[dict[str, object]]]:
    grouped: dict[Group, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["bucket"], row["s1_id"])].append({
            "target_id": row["target_id"],
            "label": int(row["label"]),
            "lexical_rank": int(row["lexical_rank"]),
        })
    for candidates in grouped.values():
        candidates.sort(key=lambda candidate: (candidate["lexical_rank"], candidate["target_id"]))
    return dict(grouped)


def ranking_metrics(ranked: dict[Group, list[dict[str, object]]], top_ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, object]:
    positive_pairs = sum(sum(int(row["label"]) for row in candidates) for candidates in ranked.values())
    negative_pairs = sum(len(candidates) for candidates in ranked.values()) - positive_pairs
    positive_groups = [candidates for candidates in ranked.values() if any(int(row["label"]) for row in candidates)]
    metrics: dict[str, object] = {
        "query_count": len(ranked),
        "positive_query_count": len(positive_groups),
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "recall_at_k": {},
        "positive_row_coverage_at_k": {},
        "mean_reciprocal_rank": None,
    }
    for k in top_ks:
        found = sum(int(row["label"]) for candidates in ranked.values() for row in candidates[:k])
        covered = sum(any(int(row["label"]) for row in candidates[:k]) for candidates in positive_groups)
        metrics["recall_at_k"][str(k)] = found / positive_pairs if positive_pairs else None
        metrics["positive_row_coverage_at_k"][str(k)] = covered / len(positive_groups) if positive_groups else None
    if positive_groups:
        reciprocal_ranks = [1 / next(index for index, row in enumerate(candidates, 1) if int(row["label"])) for candidates in positive_groups]
        metrics["mean_reciprocal_rank"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return metrics
