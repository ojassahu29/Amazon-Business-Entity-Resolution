"""
Evaluation metrics for Business Entity Resolution.

Implements the exact competition metric: macro-averaged F_0.5 per S1 entity.
"""

from __future__ import annotations


def f_beta_per_s1(
    predicted: set[str],
    truth: set[str],
    beta: float = 0.5,
) -> float:
    """
    Compute F_beta for a single S1 entity.

    Special cases:
        - truth empty AND predicted empty  -> 1.0 (correct singleton)
        - truth empty AND predicted non-empty -> 0.0 (false positives)
        - truth non-empty AND predicted empty -> 0.0 (false negatives)
    """

    if not truth and not predicted:
        return 1.0

    if not truth or not predicted:
        return 0.0

    tp = len(predicted & truth)

    if tp == 0:
        return 0.0

    precision = tp / len(predicted)
    recall = tp / len(truth)
    beta_sq = beta ** 2

    return (
        (1 + beta_sq) * precision * recall
        / (beta_sq * precision + recall)
    )


def macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """
    Macro-averaged F_0.5 over all S1 entities in the ground truth.

    Every S1 entity in ground_truth is scored.  If predictions
    is missing an S1, it is treated as an empty prediction set.
    """

    if not ground_truth:
        return 0.0

    total = 0.0

    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        total += f_beta_per_s1(pred, truth, beta=0.5)

    return total / len(ground_truth)


def macro_f05_detailed(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> dict:
    """
    Compute macro F_0.5 with per-bucket breakdowns.

    Returns a dict with overall score plus breakdowns by
    ground-truth match count.
    """

    bucket_scores: dict[str, list[float]] = {}
    all_scores: list[float] = []

    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        score = f_beta_per_s1(pred, truth, beta=0.5)
        all_scores.append(score)

        n_true = len(truth)

        if n_true == 0:
            bucket = "0_matches"
        elif n_true == 1:
            bucket = "1_match"
        elif n_true <= 3:
            bucket = "2-3_matches"
        elif n_true <= 5:
            bucket = "4-5_matches"
        else:
            bucket = "6+_matches"

        bucket_scores.setdefault(bucket, []).append(score)

    result = {
        "macro_f05": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "n_s1_entities": len(all_scores),
        "buckets": {},
    }

    for bucket, scores in sorted(bucket_scores.items()):
        result["buckets"][bucket] = {
            "count": len(scores),
            "mean_f05": sum(scores) / len(scores),
        }

    return result
