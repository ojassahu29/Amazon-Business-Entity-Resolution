# Working in this repository

## Setup

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

The challenge data is supplied separately. Its directory must contain the `train/` and `test/` files listed in [challenge and data](problem-and-data.md).

## Run the baseline

From the repository root:

```sh
.venv/bin/python code/business_entity_resolution/src/baseline.py \
  --data-dir /path/to/dataset \
  --output-dir output
```

The baseline indexes the training targets to choose and evaluate a threshold. It then builds a separate index from test Source 2 and Source 3 and writes `candidate_pairs.tsv` and `matching_results.tsv` under the output directory. This can take substantial time and disk space on full challenge data.

## Run tests

From `code/business_entity_resolution/`:

```sh
PYTHONPATH=src ../../.venv/bin/python -m unittest discover -s tests -v
```

The tests use small temporary data. They do not require the full challenge dataset.

## Candidate-selector precision benchmark

From the repository root, run `bash autoresearch.sh`. It measures candidate-pair precision and recall on a fixed 5,000-query seed-42 sample against the full training S2/S3 targets. The first run builds and caches an active-key index in the system temporary directory; later runs reuse it. Progress is printed while indexing and evaluating. Set `DATA_DIR` to use another dataset directory or `AUTORESEARCH_CACHE_DIR` to relocate the cache.

The script prints `METRIC` lines for candidate precision, candidate recall, candidate-pair count, retrieved true-pair count, mean candidates per query, and elapsed seconds. Candidate recall is the guardrail when comparing precision changes; do not keep a higher-precision result that misses the agreed recall floor.

## Optional analysis tools

The profiler and match-analysis scripts accept `--data-dir` and sample/chunk options. The `study_candidates.py` tool builds additional SQLite full-text indexes and writes `shortlist-study.json` plus `shortlist-probe.tsv`. It requires adequate free disk space; do not run it on full data without checking available space first.

Use each script's `--help` output for its current options. Do not present an old README metric as a new result without rerunning the code on named data.
