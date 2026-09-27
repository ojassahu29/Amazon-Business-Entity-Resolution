# Code map

Core Python files live in `code/business_entity_resolution/src/`.

| File | Purpose |
| --- | --- |
| `baseline.py` | Main deterministic baseline: temporary SQLite index, candidate lookup, threshold selection, evaluation report, and output files. |
| `data_loader.py` | Dataset file paths, chunked TSV readers, and parsing of comma-separated true match IDs. |
| `preprocessing.py` | Unicode-aware text and country normalization, plus name-key helpers. |
| `evaluation.py` | Per-record and macro F0.5 calculations, including singleton behavior. |
| `blocking.py` | In-memory, multi-key inverted-index alternative with name and address token stopwords. Not used by the main baseline. |
| `study_candidates.py` | Compares candidate retrieval methods and writes a study report and probe pairs. It is for analysis, not the baseline output run. |
| `profile_data.py` | Streams through the data and reports data size and quality information. |
| `analyze_true_matches.py` | Studies labelled true matches and their properties. |
| `analyze_hard_negatives.py` | Studies false-match risks and common-token collisions. |

Baseline tests are in [`tests/test_baseline.py`](../code/business_entity_resolution/tests/test_baseline.py). Candidate-study tests are in [`tests/test_study_candidates.py`](../code/business_entity_resolution/tests/test_study_candidates.py). The repository-level overview is [`README.md`](../README.md); package setup and output formats are in the [pipeline README](../code/business_entity_resolution/README.md).
