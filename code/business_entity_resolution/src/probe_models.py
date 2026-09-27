from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from model_probe_metrics import rank_lexical_rows, rank_probe_rows, ranking_metrics


MODELS = (
    {
        "id": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "license": "MIT",
        "license_url": "https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md",
        "prefixes": ("query: ", "passage: "),
    },
    {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/blob/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/README.md",
        "prefixes": ("", ""),
    },
)
MAX_PARAMETERS = 8_000_000_000


def _read_probe(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    required = {"bucket", "s1_id", "s1_name", "s1_address", "target_id", "target_name", "target_address", "label", "lexical_rank"}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"Probe TSV is empty or missing columns: {sorted(required - rows[0].keys()) if rows else sorted(required)}")
    if any(row["bucket"] not in {"0", "1"} or row["label"] not in {"0", "1"} for row in rows):
        raise ValueError("Probe rows contain an unknown bucket or label")
    return rows


def _encode(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return [vector.tolist() for vector in vectors]


def _run_model(spec: dict[str, object], snapshot: str, rows: list[dict[str, str]]) -> dict[str, object]:
    model = SentenceTransformer(snapshot, device="cpu")
    model.max_seq_length = 128
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > MAX_PARAMETERS:
        raise ValueError(f"{spec['id']} exceeds the {MAX_PARAMETERS:,}-parameter probe limit")

    query_prefix, target_prefix = spec["prefixes"]
    query_texts = {}
    target_texts = {}
    for row in rows:
        query_texts.setdefault(row["s1_id"], f"{query_prefix}{row['s1_name']} {row['s1_address']}".strip())
        target_texts.setdefault(row["target_id"], f"{target_prefix}{row['target_name']} {row['target_address']}".strip())
    query_vectors = dict(zip(query_texts, _encode(model, list(query_texts.values())), strict=True))
    target_vectors = dict(zip(target_texts, _encode(model, list(target_texts.values())), strict=True))

    bucket_metrics = {}
    for bucket in ("0", "1"):
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        ranked = rank_probe_rows(bucket_rows, query_vectors, target_vectors)
        bucket_metrics[bucket] = {
            "probe_s1_rows": len(ranked),
            "lexical_baseline": ranking_metrics(rank_lexical_rows(bucket_rows)),
            "model_ranking": ranking_metrics(ranked),
        }
    return {
        "revision": spec["revision"],
        "license": spec["license"],
        "license_source": spec["license_url"],
        "parameter_count": parameter_count,
        "device": "cpu",
        "max_sequence_length": model.max_seq_length,
        "buckets": bucket_metrics,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_probe(probe_path: Path, output_path: Path, cache_dir: Path) -> dict[str, object]:
    rows = _read_probe(probe_path)
    snapshots = {}
    for spec in MODELS:
        print(f"[model-probe] downloading {spec['id']}@{spec['revision']}", flush=True)
        snapshots[str(spec["id"])] = snapshot_download(repo_id=str(spec["id"]), revision=str(spec["revision"]), cache_dir=str(cache_dir))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    results = {}
    for spec in MODELS:
        print(f"[model-probe] encoding {spec['id']} on CPU", flush=True)
        results[str(spec["id"])] = _run_model(spec, snapshots[str(spec["id"])], rows)
    dev = results[MODELS[0]["id"]]["buckets"]["0"]
    selection = max(
        results,
        key=lambda model_id: (
            results[model_id]["buckets"]["0"]["model_ranking"]["recall_at_k"]["20"] or -1,
            results[model_id]["buckets"]["0"]["model_ranking"]["positive_row_coverage_at_k"]["20"] or -1,
            results[model_id]["buckets"]["0"]["model_ranking"]["mean_reciprocal_rank"] or -1,
        ),
    ) if dev["model_ranking"]["positive_pairs"] else None
    report: dict[str, object] = {
        "protocol": "Local CPU reranking only; probe rows are a sample from each raw lexical candidate pool, with all eligible true pairs and up to 20 hardest sampled negatives per S1; no business text is sent to a service.",
        "probe_rows": len(rows),
        "probe_s1_rows": {bucket: len({row["s1_id"] for row in rows if row["bucket"] == bucket}) for bucket in ("0", "1")},
        "models": results,
        "development_selection": {"bucket": 0, "selected_model": selection, "metric": "recall@20, then positive-row coverage@20, then MRR"},
        "holdout_bucket": 1,
    }
    _write_json_atomic(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pinned local sentence models on the restricted lexical probe pool")
    parser.add_argument("--probe-tsv", type=Path, default=Path("output/shortlist-probe.tsv"))
    parser.add_argument("--output", type=Path, default=Path("output/model-probe.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("output/model-cache"))
    args = parser.parse_args()
    report = run_probe(args.probe_tsv, args.output, args.cache_dir)
    print(json.dumps({"probe_rows": report["probe_rows"], "selected_model": report["development_selection"]["selected_model"]}, indent=2))


if __name__ == "__main__":
    main()
