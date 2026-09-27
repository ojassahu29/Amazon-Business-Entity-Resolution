# Matching pipeline

The runnable baseline is [`src/baseline.py`](../code/business_entity_resolution/src/baseline.py). It uses fixed rules. It does not train a classifier.

## How it works

1. Read the applicable S1 queries and derive their active retrieval keys.
2. Stream Source 2 and Source 3 in chunks. For each target, add only postings for active keys to `ProductionRetrievalIndex`; store its ID, normalized country, and normalized address in temporary SQLite.
3. Retrieve candidate IDs with Combo 3 + Secondary A + C2-A and the single rare-name rule (token length >= 4, DF <= 200). Preserve candidates from evidence sources weighted at least 4; cap lower-evidence candidates at 17,500 per query, ranked by aggregate evidence and entity ID. The index retains full posting lists for active keys, so document-frequency thresholds remain exact.
4. Look up candidate addresses in SQLite and score each pair with RapidFuzz character-level similarity. A blank address scores zero.
5. Keep candidates whose score reaches the threshold selected from the training sample. Write the candidate and match TSV files, including empty rows.

The query-key construction, index build, candidate lookup, and output are implemented in [`baseline.py`](../code/business_entity_resolution/src/baseline.py). Selector rules are in [`retrieval.py`](../code/business_entity_resolution/src/retrieval.py); normalization is in [`preprocessing.py`](../code/business_entity_resolution/src/preprocessing.py).

## Threshold selection

The program assigns labeled training rows to buckets with a repeatable CRC32 hash. Bucket 0 selects a threshold from a fixed list; bucket 1 evaluates it separately. The same candidate selector is used for both buckets and for test predictions. See [`_load_selected_training`, `active_keys_for_records`, `select_threshold`, and `run`](../code/business_entity_resolution/src/baseline.py).

## Selector validation

On fixed 5,000-query samples against the full S2+S3 training targets:
- Seed 42: 16,380 / 17,314 true pairs (94.61% recall), 7,464,219 candidates (0.2194% precision).
- Seed 123: 16,261 / 17,205 true pairs (94.51% recall), 8,056,698 candidates (0.2018% precision).

## What this does not do

- It does not search arbitrary fuzzy name variants beyond the current retrieval rules.
- It does not normalize legal company types into one shared form.
- It does not use address similarity to recover pairs missed by candidate retrieval.
- Query-key pruning avoids unrelated postings but is not a hard memory bound; measure peak memory on the full corpus.

The stopword-filtered blocker is in [`blocking.py`](../code/business_entity_resolution/src/blocking.py). The candidate-method study in [`study_candidates.py`](../code/business_entity_resolution/src/study_candidates.py) measures alternative retrieval methods; it is an analysis tool, not the baseline prediction command.
