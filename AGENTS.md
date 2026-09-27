# Agent Instructions

Read this file before changing the repository. Use the links below to read only the deeper pages needed for your task.

## Project in brief

This repository contains a deterministic baseline for linking business records across three sources. The baseline finds possible matches by country and normalized name, then filters them by address similarity. It is not a trained classifier. See the [pipeline guide](openwiki/pipeline.md).

## Rules for every agent

1. Treat source code and tests as the facts. Treat READMEs and the wiki as guides. If they disagree, trust the code and tests, then correct the documentation in the same change.
2. Keep the production baseline distinct from research tools. `code/business_entity_resolution/src/baseline.py` is the runnable baseline. `code/business_entity_resolution/src/blocking.py` is an in-memory alternative; do not describe it as the baseline's active path. `code/business_entity_resolution/src/study_candidates.py` compares candidate-search methods for analysis.
3. Preserve the data contract: each source row has an entity ID, business name, address, and country. The ground-truth file maps each Source 1 ID to zero or more matched IDs. Do not assume each Source 1 record has one match.
4. Candidate search sets the upper limit on matching recall. A true pair missing from the candidate list cannot be recovered by address scoring.
5. Do not load full challenge files into memory for convenience. Reuse the chunked readers in `code/business_entity_resolution/src/data_loader.py` and the disk-backed SQLite pattern in `code/business_entity_resolution/src/baseline.py` where they fit.
6. Preserve user data and existing outputs. Do not commit challenge input data or large generated outputs unless the task explicitly requires them.
7. Write clear English. Define specialist terms briefly. Link technical claims to the code, tests, or challenge statement that supports them.

## OpenWiki must stay current

The agent-maintained wiki is under `openwiki/`. For **every code change**, check whether it changes a fact that the wiki documents. If it changes behavior, structure, commands, data, tests, or known limits, update the relevant wiki page in the same change. Do not defer the update.

Before finishing:

- Check that changed wiki statements match current code and tests.
- Check links to other wiki pages and source files.
- Update the wiki home page if pages are added, removed, or renamed.
- In your final response, name the pages you updated, or state that no documented facts changed.

Do not make empty edits when the documented facts are unchanged. The wiki is hand-maintained Markdown; do not add OpenWiki tooling or generated metadata for it.

## Validation

From `code/business_entity_resolution/`, run the focused Python tests with:

```sh
PYTHONPATH=src ../../.venv/bin/python -m unittest discover -s tests -v
```

Use the test command for code changes. Run the full baseline only when the task needs an end-to-end check and the required dataset is available; it builds a database from the supplied records and writes prediction files. State exactly what you ran. See [working in this repository](openwiki/working-in-this-repo.md).

## Read next

- [Wiki home](openwiki/README.md)
- [Challenge data](openwiki/problem-and-data.md)
- [Matching pipeline](openwiki/pipeline.md)
- [Evaluation and limits](openwiki/evaluation.md)
- [Code map](openwiki/code-map.md)
- [Commands and tests](openwiki/working-in-this-repo.md)
