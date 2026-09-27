from __future__ import annotations

import heapq
import argparse
import csv
import json
import os
import resource
import shutil
import sqlite3
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from math import ceil
from typing import Callable
from rapidfuzz import fuzz

import baseline
from blocking import ADDR_STOPWORDS, NAME_STOPWORDS
from data_loader import DatasetPaths, iter_ground_truth, iter_source, parse_match_ids
from evaluation import f_beta_per_s1
from preprocessing import normalize_basic, normalize_country

PROBE_COLUMNS = (
    "bucket", "s1_id", "s1_name", "s1_address", "target_id",
    "target_name", "target_address", "label", "lexical_rank",
)
METHODS = ("control", "exact", "name_tokens", "name_address_tokens", "tokens_trigram")
MIN_FREE_BYTES = 10 * 1024**3


def _tokens(text: str, stopwords: set[str]) -> list[str]:
    return sorted({word for word in text.split() if len(word) >= 3 and word not in stopwords}, key=lambda x: (-len(x), x))[:3]


def _query(name: str, address: str, country: str) -> dict[str, str]:
    return {
        "name_norm": normalize_basic(name),
        "address_norm": normalize_basic(address),
        "country": normalize_country(country),
    }


def exact_candidates(conn: sqlite3.Connection, query: dict[str, str]) -> dict[str, dict[str, str]]:
    name = query["name_norm"]
    country = query["country"]
    if not name or not country:
        return {}
    keys = (name, " ".join(sorted(name.split())), name.replace(" ", ""))
    rows = conn.execute(
        "SELECT entity_id, country, name_norm, address_norm FROM targets "
        "WHERE country = ? AND (name_norm = ? OR name_sorted = ? OR name_compact = ?)",
        (country, *keys),
    )
    return {row[0]: {"country": row[1], "name_norm": row[2], "address_norm": row[3]} for row in rows}


def fts_candidates(conn: sqlite3.Connection, query: dict[str, str], fields: tuple[str, ...] = ("name_norm",)) -> dict[str, dict[str, str]]:
    """Fetch bounded country-scoped FTS hits; the special field name_trigram selects the trigram index."""
    country = query["country"]
    if not country:
        return {}
    hits: dict[str, dict[str, str]] = {}
    for field in fields:
        if field == "name_trigram":
            table, column, tokens = "target_name_trigram", "name_norm", [query["name_norm"][i:i + 3] for i in range(max(0, len(query["name_norm"]) - 2)) if " " not in query["name_norm"][i:i + 3]]
            unique = list(dict.fromkeys(tokens))
            tokens = [unique[0], unique[len(unique) // 2], unique[-1]] if unique else []
        elif field == "name_norm":
            table, column, tokens = "target_name_fts", "name_norm", _tokens(query["name_norm"], NAME_STOPWORDS)
        elif field == "address_norm":
            table, column, tokens = "target_name_fts", "address_norm", _tokens(query["address_norm"], ADDR_STOPWORDS)
        else:
            raise ValueError(f"Unsupported FTS field: {field}")
        if not tokens:
            continue
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        rows = conn.execute(
            f"SELECT t.entity_id, t.country, t.name_norm, t.address_norm "
            f"FROM {table} AS f JOIN targets AS t ON t.rowid = f.rowid "
            f"WHERE {table} MATCH ? AND t.country = ? ORDER BY bm25({table}), t.entity_id LIMIT 150",
            (f"{column} : ({match})", country),
        )
        hits.update((row[0], {"country": row[1], "name_norm": row[2], "address_norm": row[3]}) for row in rows)
    return hits


def rank_candidates(candidates: dict[str, dict[str, str]], query: dict[str, str], k: int = 150) -> list[dict[str, object]]:
    ranked = []
    for entity_id, target in candidates.items():
        score = 0.6 * fuzz.ratio(query["name_norm"], target["name_norm"]) / 100 + 0.4 * baseline.pair_score(query["address_norm"], target["address_norm"])
        ranked.append((entity_id, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [{"entity_id": entity_id, "score": score} for entity_id, score in ranked[:k]]


def model_probe_pairs(*, s1_id: str, truth_ids: set[str], baseline_ids: set[str], raw_candidate_ids: set[str], target_rows: dict[str, dict[str, str]]) -> list[str]:
    """Return only truth IDs in the eligible raw candidate pool that baseline missed."""
    return sorted((truth_ids & raw_candidate_ids) - baseline_ids)

def _write_probe_pairs(conn: sqlite3.Connection, writer: csv.writer, bucket: int, s1_id: str, s1_name: str, s1_address: str, query: dict[str, str], truth: set[str], baseline_ids: set[str]) -> bool:
    raw_truth = set()
    for start in range(0, len(truth), 500):
        batch = sorted(truth)[start:start + 500]
        if batch:
            marks = ",".join("?" for _ in batch)
            raw_truth.update(row[0] for row in conn.execute(
                f"SELECT entity_id FROM study_candidate_ids WHERE entity_id IN ({marks})", batch
            ))
    if not (raw_truth - baseline_ids):
        return False
    target_rows = {}
    for start in range(0, len(raw_truth), 500):
        batch = sorted(raw_truth)[start:start + 500]
        marks = ",".join("?" for _ in batch)
        target_rows.update({
            row[0]: {"name_norm": row[1], "address_norm": row[2]}
            for row in conn.execute(
                f"SELECT entity_id,name_norm,address_norm FROM targets WHERE entity_id IN ({marks})",
                batch,
            )
        })
    truth_keys = {
        entity_id: (
            -(0.6 * fuzz.ratio(query["name_norm"], target["name_norm"]) / 100 + 0.4 * baseline.pair_score(query["address_norm"], target["address_norm"])),
            entity_id,
        )
        for entity_id, target in target_rows.items()
    }
    def nonmatches():
        for row in conn.execute(
            "SELECT t.entity_id,t.name_norm,t.address_norm FROM targets AS t "
            "JOIN study_candidate_ids AS c ON c.entity_id=t.entity_id"
        ):
            entity_id, name, address = row
            if entity_id not in truth:
                score_key = (
                    -(0.6 * fuzz.ratio(query["name_norm"], name) / 100 + 0.4 * baseline.pair_score(query["address_norm"], address)),
                    entity_id,
                )
                yield score_key, entity_id, name, address
    negatives = heapq.nsmallest(20, nonmatches(), key=lambda item: item[0])
    selected_keys = dict(truth_keys)
    selected_keys.update({entity_id: score_key for score_key, entity_id, _, _ in negatives})
    ranks = {entity_id: 1 for entity_id in selected_keys}
    for entity_id, name, address in conn.execute(
        "SELECT t.entity_id,t.name_norm,t.address_norm FROM targets AS t "
        "JOIN study_candidate_ids AS c ON c.entity_id=t.entity_id"
    ):
        score_key = (
            -(0.6 * fuzz.ratio(query["name_norm"], name) / 100 + 0.4 * baseline.pair_score(query["address_norm"], address)),
            entity_id,
        )
        for selected_id, selected_key in selected_keys.items():
            if score_key < selected_key:
                ranks[selected_id] += 1
    for entity_id in sorted(target_rows, key=lambda item: (ranks[item], item)):
        target = target_rows[entity_id]
        writer.writerow((bucket, s1_id, s1_name, s1_address, entity_id, target["name_norm"], target["address_norm"], 1, ranks[entity_id]))
    for _, entity_id, name, address in sorted(negatives, key=lambda item: (ranks[item[1]], item[1])):
        if entity_id not in truth_keys:
            writer.writerow((bucket, s1_id, s1_name, s1_address, entity_id, name, address, 0, ranks[entity_id]))
    return True


def _raw_true_candidate_ids(conn: sqlite3.Connection, truth: set[str]) -> set[str]:
    found = set()
    ordered = sorted(truth)
    for start in range(0, len(ordered), 500):
        batch = ordered[start:start + 500]
        marks = ",".join("?" for _ in batch)
        found.update(row[0] for row in conn.execute(
            f"SELECT entity_id FROM study_candidate_ids WHERE entity_id IN ({marks})", batch
        ))
    return found
def _candidate_pool(conn: sqlite3.Connection, query: dict[str, str], method: str) -> tuple[dict[str, dict[str, str]], int]:
    """Build the raw ID union in SQLite and retain only the deterministic top 150 in Python."""
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS study_candidate_ids (entity_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM study_candidate_ids")
    if method == "control":
        candidates, _ = baseline.get_candidates(conn, query["country"], query["name_norm"])
        conn.executemany("INSERT OR IGNORE INTO study_candidate_ids VALUES (?)", ((eid,) for eid in candidates))
    else:
        name = query["name_norm"]
        if name and query["country"]:
            keys = (name, " ".join(sorted(name.split())), name.replace(" ", ""))
            conn.execute(
                "INSERT OR IGNORE INTO study_candidate_ids "
                "SELECT entity_id FROM targets WHERE country=? AND "
                "(name_norm=? OR name_sorted=? OR name_compact=?)",
                (query["country"], *keys),
            )
        if method in ("name_tokens", "name_address_tokens", "tokens_trigram"):
            conn.executemany("INSERT OR IGNORE INTO study_candidate_ids VALUES (?)", ((eid,) for eid in fts_candidates(conn, query, ("name_norm",))))
        if method in ("name_address_tokens", "tokens_trigram"):
            conn.executemany("INSERT OR IGNORE INTO study_candidate_ids VALUES (?)", ((eid,) for eid in fts_candidates(conn, query, ("address_norm",))))
        if method == "tokens_trigram":
            conn.executemany("INSERT OR IGNORE INTO study_candidate_ids VALUES (?)", ((eid,) for eid in fts_candidates(conn, query, ("name_trigram",))))
        if method not in ("exact", "name_tokens", "name_address_tokens", "tokens_trigram"):
            raise ValueError(f"Unknown candidate method: {method}")
    raw_count = conn.execute("SELECT COUNT(*) FROM study_candidate_ids").fetchone()[0]
    query_rows = (
        (row[0], {"country": row[1], "name_norm": row[2], "address_norm": row[3]})
        for row in conn.execute(
            "SELECT t.entity_id,t.country,t.name_norm,t.address_norm FROM targets AS t "
            "JOIN study_candidate_ids AS c ON c.entity_id=t.entity_id"
        )
    )
    top = heapq.nsmallest(
        150,
        query_rows,
        key=lambda item: (
            -(0.6 * fuzz.ratio(query["name_norm"], item[1]["name_norm"]) / 100 + 0.4 * baseline.pair_score(query["address_norm"], item[1]["address_norm"])),
            item[0],
        ),
    )
    return {entity_id: target for entity_id, target in top}, raw_count


def _shared_tokens(left: str, right: str, stopwords: set[str]) -> int:
    return len({token for token in left.split() if len(token) >= 3 and token not in stopwords} & {token for token in right.split() if len(token) >= 3 and token not in stopwords})


def analyze_missed_pairs(conn: sqlite3.Connection, pairs: list[dict[str, object]], raw_target_rows: dict[str, dict[str, str]] | None = None) -> dict[str, object]:
    """Attribute missed true pairs in the specified exclusive priority order."""
    raw_target_rows = raw_target_rows or {}
    causes: Counter[str] = Counter()
    for name in ("absent_from_raw_targets", "blank_s1_country_or_name", "blank_target_country_or_name", "normalized_country_mismatch", "broad_name_key_discarded", "no_shared_name_key"):
        causes[name] += 0
    details = Counter()
    classified_causes = []
    columns = ("name_norm", "name_sorted", "name_compact")
    for pair in pairs:
        if pair.get("target_id") is None:
            details["singleton_rows"] += 1
            continue
        target_id = str(pair["target_id"])
        s1_name = str(pair.get("name_norm", ""))
        s1_country = str(pair.get("country", ""))
        if target_id not in raw_target_rows:
            cause = "absent_from_raw_targets"
        else:
            target = raw_target_rows[target_id]
            target_name, target_country = target["name_norm"], target["country"]
            if not s1_country or not s1_name:
                cause = "blank_s1_country_or_name"
            elif not target_country or not target_name:
                cause = "blank_target_country_or_name"
            elif s1_country != target_country:
                cause = "normalized_country_mismatch"
            else:
                s1_keys = (s1_name, " ".join(sorted(s1_name.split())), s1_name.replace(" ", ""))
                target_keys = (target_name, " ".join(sorted(target_name.split())), target_name.replace(" ", ""))
                shared = [(column, key) for column, key, target_key in zip(columns, s1_keys, target_keys) if key and key == target_key]
                if shared:
                    sizes = [conn.execute(f"SELECT COUNT(*) FROM targets WHERE country=? AND {column}=?", (s1_country, key)).fetchone()[0] for column, key in shared]
                    if all(size >= 51 for size in sizes):
                        cause = "broad_name_key_discarded"
                    else:
                        raise ValueError(f"Missed target {target_id} shares a non-broad exact key; baseline retrieval invariant failed")
                else:
                    cause = "no_shared_name_key"
                    s1_address = str(pair.get("address_norm", ""))
                    target_address = target.get("address_norm", "")
                    details[f"no_shared_name_name_similarity_{_band(fuzz.ratio(s1_name, target_name) / 100)}"] += 1
                    address_score = fuzz.ratio(s1_address, target_address) / 100 if s1_address and target_address else None
                    details[f"no_shared_name_address_similarity_{_band(address_score)}"] += 1
                    details[f"shared_informative_name_tokens_{_token_band(_shared_tokens(s1_name, target_name, NAME_STOPWORDS))}"] += 1
                    details[f"shared_informative_address_tokens_{_token_band(_shared_tokens(s1_address, target_address, ADDR_STOPWORDS))}"] += 1
        causes[cause] += 1
        classified_causes.append(cause)
    return {"causes": dict(causes), "details": dict(details), "missed_true_pairs": sum(causes.values()), "singleton_rows": details["singleton_rows"], "classified_causes": classified_causes}


def _band(score: float | None) -> str:
    if score is None:
        return "missing"
    return "0_0.5" if score < 0.5 else "0.5_0.85" if score < 0.85 else "0.85_1"


def _token_band(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2_plus"


def _disk_guard(path: Path) -> int:
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise OSError(f"free disk space below required 10 GiB at {path}: {free} bytes available")
    return free


def _load_selected(paths: DatasetPaths) -> tuple[dict[int, dict[str, set[str]]], dict[int, dict[str, tuple[str, str, str]]], int, int]:
    truths: dict[int, dict[str, set[str]]] = {0: {}, 1: {}}
    gt_count = 0
    for frame in iter_ground_truth(paths.train_ground_truth):
        gt_count += len(frame)
        for s1_id, value in frame.itertuples(index=False, name=None):
            bucket = zlib.crc32(str(s1_id).encode("utf-8")) % 200
            if bucket in truths:
                truths[bucket][str(s1_id)] = set(parse_match_ids(value))
    selected = set(truths[0]) | set(truths[1])
    rows: dict[int, dict[str, tuple[str, str, str]]] = {0: {}, 1: {}}
    source_count = 0
    for frame in iter_source(paths.train_source1):
        source_count += len(frame)
        for s1_id, name, address, country in frame.itertuples(index=False, name=None):
            sid = str(s1_id)
            if sid in selected:
                rows[zlib.crc32(sid.encode("utf-8")) % 200][sid] = (str(name), str(address), str(country))
    missing = selected - set(rows[0]) - set(rows[1])
    if missing:
        raise ValueError(f"Selected ground-truth S1 IDs missing from train_source1: {min(missing)}")
    return truths, rows, gt_count, source_count


def _target_truth_rows(paths: DatasetPaths, selected: set[str]) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, int]]:
    raw: dict[str, dict[str, str]] = {}
    sources: dict[str, str] = {}
    counts = {"s2_rows": 0, "s3_rows": 0}
    for source, path in (("S2", paths.train_source2), ("S3", paths.train_source3)):
        for frame in iter_source(path):
            counts[f"s{source[1]}_rows"] += len(frame)
            for eid, name, address, country in frame.itertuples(index=False, name=None):
                entity_id = str(eid)
                if entity_id in selected:
                    raw[entity_id] = {"name_norm": normalize_basic(str(name)), "address_norm": normalize_basic(str(address)), "country": normalize_country(str(country))}
                    sources[entity_id] = source
    return raw, sources, counts


def _truth_rows(conn: sqlite3.Connection, ids: set[str]) -> dict[str, dict[str, str]]:
    rows = {}
    ordered = sorted(ids)
    for start in range(0, len(ordered), 500):
        batch = ordered[start:start + 500]
        marks = ",".join("?" for _ in batch)
        rows.update({row[0]: {"country": row[1], "name_norm": row[2], "address_norm": row[3]} for row in conn.execute(f"SELECT entity_id,country,name_norm,address_norm FROM targets WHERE entity_id IN ({marks})", batch)})
    return rows


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {"mean": sum(values) / len(values), "p95": ordered[min(len(ordered) - 1, ceil(0.95 * len(values)) - 1)], "max": max(values)}


def _evaluate_bucket(conn: sqlite3.Connection, bucket: int, truths: dict[str, set[str]], rows: dict[str, tuple[str, str, str]], target_raw: dict[str, dict[str, str]], sources: dict[str, str], out: csv.writer, progress: Callable[[dict[str, object]], None] | None = None, saved_results: dict[str, dict[str, object]] | None = None) -> dict[str, object]:
    result = {}
    saved_results = saved_results or {}
    total_truths = sum(map(len, truths.values()))
    total_rows = len(truths)
    positive_rows = sum(bool(truth) for truth in truths.values())
    if "control" in saved_results:
        result["control"] = saved_results["control"]
        result["baseline_miss_causes"] = saved_results["control"]["miss_causes"]
    for method in METHODS:
        if method in saved_results:
            result[method] = saved_results[method]
            continue
        if progress:
            progress({"stage": "evaluating", "bucket": bucket, "method": method, "completed_s1": 0, "total_s1": total_rows})
        started = time.perf_counter()
        candidate_pairs = raw_candidate_pairs = covered_rows = ranking_misses = 0
        marginal_gains = marginal_losses = 0
        query_elapsed_seconds = 0.0
        recalls = {25: 0, 50: 0, 150: 0}
        macro_totals = {25: 0.0, 50: 0.0, 150: 0.0}
        cohort_counts: dict[str, dict[str, list[int]]] = {"positive": {"raw": [], "capped": []}, "singleton": {"raw": [], "capped": []}}
        breakdown = {"source": defaultdict(lambda: [0, 0]), "country": defaultdict(lambda: [0, 0])}
        retrieval_misses = []
        probe_samples = 0
        for row_number, (sid, truth) in enumerate(sorted(truths.items()), 1):
            query_started = time.perf_counter()
            name, address, country = rows[sid]
            query = _query(name, address, country)
            pool, raw_count = _candidate_pool(conn, query, method)
            raw_truth_ids = _raw_true_candidate_ids(conn, truth) if truth else set()
            ranked = rank_candidates(pool, query, 150)
            ranked_ids = [str(row["entity_id"]) for row in ranked]
            top_ids = set(ranked_ids)
            cohort = "positive" if truth else "singleton"
            cohort_counts[cohort]["raw"].append(raw_count)
            cohort_counts[cohort]["capped"].append(len(ranked))
            if truth:
                raw_candidate_pairs += len(raw_truth_ids)
                retrieval_misses.extend({"s1_id": sid, "target_id": target_id, **query} for target_id in truth - raw_truth_ids)
                ranking_misses += len(raw_truth_ids - top_ids)
            for k in recalls:
                ids_at_k = {str(row["entity_id"]) for row in ranked[:k]}
                found = truth & ids_at_k
                recalls[k] += len(found)
                if k == 150:
                    candidate_pairs += len(found)
                    covered_rows += bool(found)
                predicted = {eid for eid in ids_at_k if baseline.pair_score(query["address_norm"], pool[eid]["address_norm"]) >= 0.5}
                macro_totals[k] += f_beta_per_s1(predicted, truth)
            if truth:
                for target_id in truth:
                    if target_id not in top_ids:
                        breakdown["source"][sources.get(target_id, "unknown")][1] += 1
                        breakdown["country"][query["country"] or "missing"][1] += 1
                    else:
                        breakdown["source"][sources.get(target_id, "unknown")][0] += 1
                        breakdown["country"][query["country"] or "missing"][0] += 1
            query_elapsed_seconds += time.perf_counter() - query_started
            control_ids = set(ranked_ids) if method == "control" else set(baseline.get_candidates(conn, query["country"], query["name_norm"])[0])
            if truth:
                method_hits = truth & top_ids
                control_hits = truth & control_ids
                marginal_gains += len(method_hits - control_hits)
                marginal_losses += len(control_hits - method_hits)
            if method == "tokens_trigram" and truth and probe_samples < 200:
                if _write_probe_pairs(conn, out, bucket, sid, name, address, query, truth, control_ids):
                    probe_samples += 1
            if progress and (row_number % 500 == 0 or row_number == total_rows):
                progress({"stage": "evaluating", "bucket": bucket, "method": method, "completed_s1": row_number, "total_s1": total_rows})
        method_cause_report = analyze_missed_pairs(conn, retrieval_misses, target_raw)
        if sum(method_cause_report["causes"].values()) != len(retrieval_misses):
            raise ValueError(f"Cause counts do not match retrieval misses for {method}")
        method_causes_by_source: dict[str, Counter[str]] = defaultdict(Counter)
        for missed, cause in zip(retrieval_misses, method_cause_report["classified_causes"]):
            method_causes_by_source[sources.get(str(missed["target_id"]), "unknown")][cause] += 1
        method_cause_report.pop("classified_causes")
        result[method] = {
            "raw_true_pair_candidate_recall": raw_candidate_pairs / total_truths if total_truths else None,
            "true_pair_candidate_recall": candidate_pairs / total_truths if total_truths else None,
            "positive_row_coverage": covered_rows / positive_rows if positive_rows else None,
            "recall_at_k": {str(k): count / total_truths if total_truths else None for k, count in recalls.items()},
            "cohort_row_counts": {cohort: len(by_type["raw"]) for cohort, by_type in cohort_counts.items()},
            "candidate_counts": {cohort: {kind: _stats(values) for kind, values in by_type.items()} for cohort, by_type in cohort_counts.items()},
            "macro_f05_at_k": {str(k): score / len(truths) if truths else None for k, score in macro_totals.items()},
            "missed_true_pairs": total_truths - candidate_pairs,
            "retrieval_misses": len(retrieval_misses),
            "ranked_below_150": ranking_misses,
            "query_elapsed_seconds": query_elapsed_seconds,
            "marginal_true_link_gains": marginal_gains,
            "marginal_true_link_losses": marginal_losses,
            "miss_causes": {**method_cause_report, "by_source": {key: dict(value) for key, value in sorted(method_causes_by_source.items())}, "by_bucket": {str(bucket): dict(method_cause_report["causes"])}},
            "restricted_pool_probe_s1_count": probe_samples if method == "tokens_trigram" else None,
            "elapsed_seconds": time.perf_counter() - started,
            "s2_s3_breakdown": {key: {"found": value[0], "missed": value[1]} for key, value in sorted(breakdown["source"].items())},
            "country_breakdown": {key: {"found": value[0], "missed": value[1]} for key, value in sorted(breakdown["country"].items())},
        }
        if method == "control":
            result["baseline_miss_causes"] = result[method]["miss_causes"]
        if progress:
            progress({"stage": "method_complete", "bucket": bucket, "method": method, "completed_s1": total_rows, "total_s1": total_rows, "metrics": result[method]})
    return result


def _build_fts(conn: sqlite3.Connection, table: str, definition: str) -> None:
    conn.execute(f"CREATE VIRTUAL TABLE {table} USING fts5({definition}, content='targets', content_rowid='rowid')")
    conn.execute(f"INSERT INTO {table}({table}) VALUES ('rebuild')")
    conn.commit()


def _control_recall(conn: sqlite3.Connection, truths: dict[str, set[str]], rows: dict[str, tuple[str, str, str]], progress: Callable[[dict[str, object]], None] | None = None) -> float | None:
    total = sum(map(len, truths.values()))
    if not total:
        return None
    found = 0
    for row_number, (sid, truth) in enumerate(truths.items(), 1):
        candidates, _ = baseline.get_candidates(conn, rows[sid][2], rows[sid][0])
        found += len(truth & candidates.keys())
        if progress and (row_number % 500 == 0 or row_number == len(truths)):
            progress({"stage": "validating_control", "bucket": 1, "completed_s1": row_number, "total_s1": len(truths)})
    return found / total

def _choose_recommendation(bucket0: dict[str, object], index_build_seconds: dict[str, float | None], peak_rss_bytes: int, free_disk_bytes: int, dev_s1_count: int) -> dict[str, object]:
    test_s1_count = 1_732_544
    estimates = {}
    for method, metrics in bucket0.items():
        if method == "baseline_miss_causes":
            continue
        build = index_build_seconds[method]
        query_per_row = metrics["query_elapsed_seconds"] / dev_s1_count if dev_s1_count else float("inf")
        estimate = build + query_per_row * test_s1_count if build is not None else None
        counts = metrics["candidate_counts"]
        rows = metrics["cohort_row_counts"]
        total_rows = sum(rows.values())
        candidates = sum(counts[cohort]["raw"]["mean"] * rows[cohort] for cohort in rows) / total_rows if total_rows else 0.0
        feasible = build is not None and peak_rss_bytes <= 16 * 1024**3 and free_disk_bytes >= MIN_FREE_BYTES and estimate <= 24 * 60 * 60
        estimates[method] = {
            "index_build_seconds": build,
            "bucket0_mean_query_seconds_per_s1": query_per_row,
            "estimated_full_test_seconds": estimate,
            "mean_raw_candidates": candidates,
            "feasible": feasible,
            "deterministic_id_tiebreak": True,
        }
    feasible = [method for method in estimates if estimates[method]["feasible"]]
    selected = max(
        feasible,
        key=lambda method: (
            bucket0[method]["recall_at_k"]["150"],
            bucket0[method]["positive_row_coverage"],
            -estimates[method]["mean_raw_candidates"],
            -estimates[method]["estimated_full_test_seconds"],
        ),
        default=None,
    )
    return {
        "selected_method": selected,
        "selection_split": "bucket 0 only; frozen before bucket 1",
        "full_test_s1_rows": test_s1_count,
        "limits": {"ram_bytes": 16 * 1024**3, "free_disk_bytes": MIN_FREE_BYTES, "runtime_seconds": 24 * 60 * 60},
        "observed_peak_rss_bytes": peak_rss_bytes,
        "observed_free_disk_bytes": free_disk_bytes,
        "estimates": estimates,
        "basis": "bucket-0 recall@150, then positive-row coverage, then fewer raw candidates, then lower estimated runtime",
    }


def _save_progress(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    fd, temporary_name = tempfile.mkstemp(prefix="shortlist-progress-", suffix=".json.tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    checkpoint = state.get("last_checkpoint", {})
    fields = ("stage", "bucket", "method", "completed_s1", "total_s1")
    detail = " ".join(f"{key}={checkpoint[key]}" for key in fields if key in checkpoint)
    print(f"[shortlist-study] {detail or 'checkpoint updated'}", flush=True)


def _publish_outputs(report_tmp: Path, probe_tmp: Path, output_dir: Path) -> dict[Path, Path | None]:
    destinations = (
        (probe_tmp, output_dir / "shortlist-probe.tsv"),
        (report_tmp, output_dir / "shortlist-study.json"),
    )
    backups: dict[Path, Path | None] = {}
    try:
        for _, destination in destinations:
            if not destination.exists():
                backups[destination] = None
                continue
            fd, name = tempfile.mkstemp(prefix=f"{destination.name}-", suffix=".backup.tmp", dir=output_dir)
            os.close(fd)
            backup = Path(name)
            backup.unlink()
            backups[destination] = backup
            try:
                os.link(destination, backup)
            except OSError:
                shutil.copy2(destination, backup)
    except Exception:
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)
        raise
    published: list[Path] = []
    try:
        for source, destination in destinations:
            os.replace(source, destination)
            published.append(destination)
    except OSError as exc:
        try:
            for _, destination in reversed(destinations):
                backup = backups[destination]
                if backup is not None:
                    os.replace(backup, destination)
                elif destination in published:
                    destination.unlink(missing_ok=True)
        except OSError as rollback_exc:
            raise RuntimeError(f"Output publication failed and rollback failed; backup retained: {rollback_exc}") from exc
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)
        raise RuntimeError(f"Output publication failed; previous outputs restored: {exc}") from exc
    return backups

def _restore_outputs(backups: dict[Path, Path | None]) -> None:
    for destination, backup in backups.items():
        if backup is None:
            destination.unlink(missing_ok=True)
        else:
            os.replace(backup, destination)


def _open_resume_index(workdir: Path) -> tuple[sqlite3.Connection, int, float | None]:
    db_path = workdir / "index.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"Resume index does not exist: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        required = {"targets", "target_name_fts", "target_name_trigram"}
        if not required <= tables:
            raise RuntimeError(f"Resume index missing tables: {sorted(required - tables)}")
        integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Resume index failed SQLite quick_check: {integrity}")
        target_rows = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        if not target_rows:
            raise RuntimeError("Resume index has no target rows")
        for table in ("target_name_fts", "target_name_trigram"):
            try:
                conn.execute(f"INSERT INTO {table}({table}, rank) VALUES ('integrity-check', 1)")
                conn.execute(f"SELECT rowid FROM {table} WHERE {table} MATCH ? LIMIT 1", ('name_norm : "zzqnonexistenttoken"',)).fetchall()
            except sqlite3.DatabaseError as exc:
                raise RuntimeError(f"Resume index FTS content integrity failed for {table}: {exc}") from exc
        conn.commit()
        workdir_stat = workdir.stat()
        db_stat = db_path.stat()
        created_at = getattr(workdir_stat, "st_birthtime", None)
        preparation_seconds = max(0.0, db_stat.st_mtime - created_at) if created_at is not None else None
        return conn, target_rows, preparation_seconds
    except Exception:
        conn.close()
        raise


def run_study(data_dir: Path, output_dir: Path, expected_control_recall: float | None = 0.254165, resume_index: Path | None = None, progress_path: Path | None = None) -> dict[str, object]:
    paths = DatasetPaths(data_dir)
    paths.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_path or output_dir / "shortlist-study-progress.json"
    workdir = resume_index.resolve() if resume_index else None
    db_dir: Path | None = None
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="tmp-study-", dir=output_dir)).resolve()
        db_dir = workdir
    db_path = workdir / "index.sqlite"
    progress_path = progress_path.resolve()
    data_path = paths.data_dir.resolve()
    saved_state: dict[str, object] = {}
    if progress_path.is_file():
        try:
            loaded = json.loads(progress_path.read_text(encoding="utf-8"))
            if loaded.get("status") in ("running", "failed") and loaded.get("workdir") == str(workdir) and loaded.get("data_dir") == str(data_path):
                saved_state = loaded
        except (OSError, json.JSONDecodeError):
            pass
    state: dict[str, object] = saved_state or {
        "status": "running",
        "workdir": str(workdir),
        "data_dir": str(data_path),
        "completed_methods": [],
        "bucket_results": {},
        "probe_offset": 0,
    }
    report_tmp = None
    conn = None
    start = time.perf_counter()

    def checkpoint(event: dict[str, object]) -> None:
        if event.get("stage") == "baseline_miss_diagnosis_complete":
            state.setdefault("bucket_results", {}).setdefault(str(event["bucket"]), {})["baseline_miss_causes"] = event["metrics"]
        elif event.get("stage") == "method_complete":
            bucket, method = str(event["bucket"]), str(event["method"])
            bucket_results = state.setdefault("bucket_results", {})
            bucket_results.setdefault(bucket, {})[method] = event["metrics"]
            completed = state.setdefault("completed_methods", [])
            name = f"{bucket}:{method}"
            if name not in completed:
                completed.append(name)
        state["status"] = "running"
        state["last_checkpoint"] = {key: value for key, value in event.items() if key != "metrics"}
        _save_progress(progress_path, state)

    published_backups: dict[Path, Path | None] | None = None
    try:
        report_fd, report_name = tempfile.mkstemp(prefix="shortlist-study-", suffix=".json.tmp", dir=output_dir)
        os.close(report_fd)
        report_tmp = Path(report_name)
        checkpoint({"stage": "loading_train_data"})
        free_disk_before = _disk_guard(output_dir)
        truths, s1_rows, gt_count, s1_count = _load_selected(paths)
        selected_truth_ids = set().union(*(set().union(*bucket.values()) if bucket else set() for bucket in truths.values()))
        target_raw, sources, source_counts = _target_truth_rows(paths, selected_truth_ids)
        checkpoint({"stage": "opening_resume_index" if resume_index else "building_target_index"})
        if resume_index:
            conn, target_rows, reused_preparation_seconds = _open_resume_index(workdir)
            target_index_seconds = reused_preparation_seconds
            name_fts_seconds = trigram_seconds = 0.0
        else:
            index_started = time.perf_counter()
            conn = baseline.build_index(paths.train_source2, paths.train_source3, db_path)
            target_index_seconds = time.perf_counter() - index_started
            target_rows = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            reused_preparation_seconds = None
        checkpoint({"stage": "auditing_targets", "target_rows": target_rows})
        indexed_truth_rows = _truth_rows(conn, selected_truth_ids)
        unindexed_valid_truths = sorted(entity_id for entity_id, row in target_raw.items() if row["country"] and row["name_norm"] and entity_id not in indexed_truth_rows)
        if unindexed_valid_truths:
            raise RuntimeError(f"Valid raw true target omitted from index: {unindexed_valid_truths[0]}")
        free_disk_after_index = _disk_guard(output_dir)
        checkpoint({"stage": "validating_control", "bucket": 1, "total_s1": len(truths[1])})
        baseline_control_recall = _control_recall(conn, truths[1], s1_rows[1], checkpoint)
        if expected_control_recall is not None and (baseline_control_recall is None or round(baseline_control_recall, 6) != expected_control_recall):
            raise RuntimeError(f"Bucket-1 control recall protocol mismatch before method comparisons: expected {expected_control_recall:.6f}, observed {baseline_control_recall}")
        if not resume_index:
            checkpoint({"stage": "building_name_token_index"})
            name_fts_started = time.perf_counter()
            _build_fts(conn, "target_name_fts", "name_norm, address_norm, tokenize='unicode61'")
            name_fts_seconds = time.perf_counter() - name_fts_started
        free_disk_after_name_fts = _disk_guard(output_dir)
        if not resume_index:
            checkpoint({"stage": "building_trigram_index"})
            trigram_started = time.perf_counter()
            _build_fts(conn, "target_name_trigram", "name_norm, tokenize='trigram'")
            trigram_seconds = time.perf_counter() - trigram_started
        free_disk_after_trigram = _disk_guard(output_dir)
        peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024)
        method_build_seconds = {
            "control": target_index_seconds,
            "exact": target_index_seconds,
            "name_tokens": None if target_index_seconds is None else target_index_seconds + name_fts_seconds,
            "name_address_tokens": None if target_index_seconds is None else target_index_seconds + name_fts_seconds,
            "tokens_trigram": None if target_index_seconds is None else target_index_seconds + name_fts_seconds + trigram_seconds,
        }
        summary: dict[str, object] = {
            "protocol": "CRC32(s1_id) % 200; bucket 0 development; bucket 1 hold-out; full train S2/S3 targets",
            "data_rows": {"ground_truth": gt_count, "train_source1": s1_count, **source_counts},
            "bucket_sizes": {str(bucket): {"s1_rows": len(truths[bucket]), "positive_rows": sum(bool(v) for v in truths[bucket].values()), "singleton_rows": sum(not v for v in truths[bucket].values()), "true_pairs": sum(map(len, truths[bucket].values()))} for bucket in (0, 1)},
            "target_row_audit": {
                "selected_true_target_ids": len(selected_truth_ids),
                "found_in_index": len(indexed_truth_rows),
                "omitted_for_blank_country_or_name": sum(not row["country"] or not row["name_norm"] for row in target_raw.values()),
                "absent_from_raw_targets": len(selected_truth_ids - target_raw.keys()),
            },
            "index_build_seconds": {"targets": target_index_seconds, "name_tokens_fts": name_fts_seconds, "name_trigram_fts": trigram_seconds, "reused_index_preparation_seconds": reused_preparation_seconds},
            "resume_index_target_rows": target_rows if resume_index else None,
            "methods": {},
            "constraints": {"max_candidates_per_s1": 150, "threshold": 0.5, "name_weight": 0.6, "address_weight": 0.4, "external_lookup": False},
            "baseline_control_validation": {"bucket": 1, "candidate_pair_recall_before_comparison": baseline_control_recall, "expected_recall": expected_control_recall},
            "disk_bytes_free": {"before_index": free_disk_before, "after_index": free_disk_after_index, "after_name_fts": free_disk_after_name_fts, "after_trigram_fts": free_disk_after_trigram},
            "progress_file": str(progress_path),
        }
        probe_tmp = workdir / "shortlist-probe.tsv.tmp"
        can_resume_probe = bool(saved_state) and probe_tmp.is_file()
        if can_resume_probe:
            probe_offset = int(state.get("probe_offset", 0))
            header_size = len(("\t".join(PROBE_COLUMNS) + "\n").encode("utf-8"))
            if probe_offset < header_size or probe_offset > probe_tmp.stat().st_size:
                can_resume_probe = False
        if saved_state and not can_resume_probe:
            for results in state.get("bucket_results", {}).values():
                results.pop("tokens_trigram", None)
            state["completed_methods"] = [name for name in state.get("completed_methods", []) if not name.endswith(":tokens_trigram")]
            state["probe_offset"] = 0
        with probe_tmp.open("r+" if can_resume_probe else "w+", encoding="utf-8", newline="") as probe_file:
            if can_resume_probe:
                probe_offset = int(state["probe_offset"])
                probe_file.truncate(probe_offset)
                probe_file.seek(probe_offset)
            else:
                writer = csv.writer(probe_file, delimiter="\t", lineterminator="\n")
                writer.writerow(PROBE_COLUMNS)
                probe_file.flush()
                os.fsync(probe_file.fileno())
                state["probe_offset"] = probe_file.tell()
                checkpoint({"stage": "probe_header_written"})
            writer = csv.writer(probe_file, delimiter="\t", lineterminator="\n")
            for bucket in (0, 1):
                checkpoint({"stage": "evaluating", "bucket": bucket, "total_s1": len(truths[bucket])})

                def report_method_progress(event: dict[str, object]) -> None:
                    if event.get("stage") == "method_complete":
                        probe_file.flush()
                        os.fsync(probe_file.fileno())
                        state["probe_offset"] = probe_file.tell()
                    checkpoint(event)

                prior = state.get("bucket_results", {}).get(str(bucket), {})
                bucket_results = _evaluate_bucket(
                    conn, bucket, truths[bucket], s1_rows[bucket], target_raw, sources, writer,
                    progress=report_method_progress,
                    saved_results=prior,
                )
                peak_rss_bytes = max(peak_rss_bytes, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024))
                control_recall = bucket_results["control"]["true_pair_candidate_recall"]
                for method, metrics in bucket_results.items():
                    if method == "baseline_miss_causes":
                        continue
                    metrics["marginal_vs_control"] = {
                        "true_pair_recall_delta": metrics["true_pair_candidate_recall"] - control_recall if control_recall is not None else None,
                        "true_link_gains_at_k150": metrics["marginal_true_link_gains"],
                        "true_link_losses_at_k150": metrics["marginal_true_link_losses"],
                        "net_true_link_gain_loss_at_k150": metrics["marginal_true_link_gains"] - metrics["marginal_true_link_losses"],
                    }
                state.setdefault("bucket_results", {})[str(bucket)] = bucket_results
                checkpoint({"stage": "bucket_complete", "bucket": bucket, "total_s1": len(truths[bucket])})
                summary["methods"][str(bucket)] = bucket_results
                if bucket == 0:
                    summary["recommendation"] = _choose_recommendation(bucket_results, method_build_seconds, peak_rss_bytes, free_disk_after_trigram, len(truths[bucket]))
        actual = summary["methods"]["1"]["control"]["true_pair_candidate_recall"]
        if actual is None or round(actual, 6) != round(baseline_control_recall, 6):
            raise RuntimeError(f"Bucket-1 control recall changed during method comparisons: checked {baseline_control_recall}, observed {actual}")
        peak_rss_bytes = max(peak_rss_bytes, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024))
        if peak_rss_bytes > 16 * 1024**3:
            raise RuntimeError(f"Full-corpus study peak RSS exceeds 16 GiB: {peak_rss_bytes} bytes")
        free_disk_final = _disk_guard(output_dir)
        summary["disk_bytes_free"]["after_comparison"] = free_disk_final
        summary["recommendation"]["observed_peak_rss_bytes"] = peak_rss_bytes
        summary["recommendation"]["observed_free_disk_bytes"] = free_disk_final
        summary["database_bytes"] = db_path.stat().st_size
        summary["peak_rss_bytes"] = peak_rss_bytes
        summary["elapsed_seconds"] = time.perf_counter() - start
        with report_tmp.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        conn.close()
        conn = None
        checkpoint({"stage": "publishing_outputs"})
        published_backups = _publish_outputs(report_tmp, probe_tmp, output_dir)
        report_tmp = None
        state["status"] = "complete"
        state["last_checkpoint"] = {"stage": "complete", "elapsed_seconds": summary["elapsed_seconds"]}
        _save_progress(progress_path, state)
        for backup in published_backups.values():
            if backup is not None:
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    pass
        published_backups = None
        try:
            if db_dir is not None:
                shutil.rmtree(db_dir)
                db_dir = None
            elif workdir.parent == output_dir.resolve() and workdir.name.startswith("tmp-study-"):
                shutil.rmtree(workdir)
        except OSError as exc:
            print(f"[shortlist-study] temporary-index cleanup deferred: {exc}", flush=True)
        return summary
    except BaseException as exc:
        if published_backups is not None:
            try:
                _restore_outputs(published_backups)
            except OSError as rollback_exc:
                state["output_rollback_error"] = str(rollback_exc)
            published_backups = None
        state["status"] = "failed"
        state["error"] = str(exc)
        try:
            _save_progress(progress_path, state)
        except OSError:
            pass
        if conn is not None:
            conn.close()
        if report_tmp is not None:
            report_tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose and compare entity-resolution shortlist retrieval methods")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--resume-index", type=Path, help="Reuse an existing tmp-study directory; completed method checkpoints are resumed")
    parser.add_argument("--progress-file", type=Path, help="Atomic progress/checkpoint JSON path")
    args = parser.parse_args()
    try:
        report = run_study(args.data_dir, args.output_dir, resume_index=args.resume_index, progress_path=args.progress_file)
    except Exception as exc:
        print(f"shortlist study failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1) from exc
    print(json.dumps({"bucket_sizes": report["bucket_sizes"], "database_bytes": report["database_bytes"], "peak_rss_bytes": report["peak_rss_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
