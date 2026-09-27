# Challenge and data

## The task

The task is to decide which records in Source 2 and Source 3 describe the same businesses as each record in Source 1. Source 1 is the reference set. Sources 2 and 3 are separate sets of records to search.

One Source 1 record can have no match, one match, or several matches. A record with no match is a **singleton**. The data reader defines the source columns and the ground-truth fields in [`data_loader.py`](../code/business_entity_resolution/src/data_loader.py).

## Files the code expects

Pass a data directory that contains this layout:

```text
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Each source TSV has these columns: `entity_id`, `business_name`, `business_address`, `country`. The training truth TSV has `source1_entity_id` and `matched_entity_ids`. The latter is a comma-separated list; an empty value means no true matches. See [`DatasetPaths`, the readers, and `parse_match_ids`](../code/business_entity_resolution/src/data_loader.py).

Training truth is used to choose and check the baseline's address threshold. Test truth is not part of the required test inputs.

## Important data properties

- Names and addresses can contain spelling errors, missing values, punctuation, and different legal forms.
- Names can use non-Latin scripts. Normalization must not assume English-only text.
- Country is used to restrict candidate searches. A wrong or missing country can prevent a true pair from being found.
- The links are many-to-many. Do not force one match per Source 1 record.

The [official challenge statement](../docs/Business%20Entity%20Resolution%20Challenge.pdf) defines the competition inputs and output requirements. The loader in the code defines the file names and column names this implementation accepts.
