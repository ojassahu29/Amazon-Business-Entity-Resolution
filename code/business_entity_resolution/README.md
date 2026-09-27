# Business Entity Resolution Engine

A deterministic production-retrieval pipeline for matching business records across Source 1, Source 2, and Source 3. It calibrates the frozen selector on training rows, reports held-out metrics, and writes the two required test TSV files.

For coding agents, read the repository [`AGENTS.md`](../../AGENTS.md) and use the [agent wiki](../../openwiki/README.md) for linked architecture and source notes.

---

## 📂 Directory Structure

```
code/business_entity_resolution/
├── README.md                      # End-to-end technical reproduction guide
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment definition
├── pyproject.toml                 # Package metadata
├── src/
│   ├── __init__.py                # Production retrieval API exports
│   ├── baseline.py                # Frozen-selector submission pipeline
│   ├── study_candidates.py        # Candidate-method analysis
│   ├── retrieval.py               # Frozen production retrieval engine
│   ├── blocking.py                # Legacy multi-index candidate blocker
│   ├── data_loader.py             # Chunked streaming loaders
│   ├── preprocessing.py           # Unicode-safe normalization and tokenization
│   ├── model_probe_metrics.py     # Local ranking probe metrics
│   ├── probe_models.py            # Optional local embedding-model probe
│   ├── profile_data.py            # Dataset profiler
│   ├── analyze_true_matches.py    # Ground-truth diagnostics
│   ├── analyze_hard_negatives.py  # False-match and collision analysis
│   └── evaluation.py              # Singleton-aware macro F0.5
└── tests/
    ├── test_baseline.py
    ├── test_study_candidates.py
    ├── test_study_progress.py
    ├── test_model_probe_metrics.py
    └── test_combo3_recovery.py
```

---

## ⚙️ Setup & Environment

### Prerequisites
- Python >= 3.10
- Git

### Installation Options

#### Option A: Pip & Virtual Environment
```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

#### Option B: Conda / Mamba
```bash
conda env create -f environment.yml
conda activate entity-resolution
pip install -e .
```

---

## 🔬 Pipeline Modules & CLI Usage

### 1. Data Profiling & Quality Audit (`profile_data.py`)
Memory-safe streaming profiler that inspects millions of records without exceeding RAM limits.

```bash
python src/profile_data.py --data-dir path/to/dataset --chunksize 250000
```

**Key Metrics Computed:**
- Total record counts and unique entity counts across S1, S2, and S3.
- Null value counts across `business_name`, `business_address`, and `country`.
- Ground-truth match cardinality (singletons vs 1-match vs multi-match entities).

---

### 2. Preprocessing & Normalization (`preprocessing.py`)
The baseline uses conservative Unicode NFKC normalization, case-folding, punctuation replacement, and whitespace normalization. It also uses sorted-token and compact name keys for candidate retrieval. It does not canonicalize legal company forms.

---

### 3. True Match Diagnostic Analysis (`analyze_true_matches.py`)
Performs an in-depth empirical diagnostic of true match relationships in training data to guide candidate generation and feature engineering.

```bash
python src/analyze_true_matches.py --data-dir path/to/dataset --sample-size 5000
```

**Key Empirical Insights:**
- **Exact Name Matches**: ~78.8% of true matches between S1 and S2/S3 share identical normalized names.
- **Country Consistency**: 99.4% of true matches share identical normalized country codes. Country is a reliable blocking key.
- **Cardinaity Breakdown**: Substantial fraction of S1 entities are singletons (0 matches in S2/S3), while others link to 1–5+ distinct records across S2 and S3.
- **Address Overlap**: S2/S3 addresses exhibit higher noise than names; token Jaccard and Levenshtein similarity provide strong discriminative signals for fuzzy pairs.

---

### 4. Hard Negative & Collision Analysis (`analyze_hard_negatives.py`)
Profiles token collisions and false positive hazards across high-frequency business tokens.

```bash
python src/analyze_hard_negatives.py --data-dir path/to/dataset --sample-size 5000
```

**Findings:**
- High-frequency legal and industry tokens (e.g., `private`, `limited`, `llc`, `inc`, `services`, `solutions`, `technologies`, `enterprises`) trigger severe combinatorial explosion if unindexed or unstopped.
- Informed the stopword filter design used in `blocking.py`.

---

### 5. Frozen Production Retrieval Engine (`retrieval.py`)
Canonical high-recall candidate retrieval architecture selected from conducted validation experiments. Combines three complementary retrieval layers:

1. **Combo 3 (Primary Frequency-Aware Multi-Index)**:
   - `(country, name_norm)` exact normalized match.
   - `(country, name_sorted)` token-sorted match.
   - `(country, name_compact)` space/punctuation-stripped compact match.
   - Informative name token overlap ($\ge 2$ shared tokens).
   - Informative address token overlap ($\ge 3$ shared tokens).
   - Rare informative name token ($DF \le 500$) co-occurring with name stopword token.
   - Rare address number ($DF \le 500, \text{len} \ge 3$) co-occurring with address token ($DF \le 1000$).
   - Rare address 2-token overlap ($DF \le 500$).
   - Single rare informative name token ($\text{len} \ge 4, DF \le 200$).
   - Rare compact prefix-5 ($\text{len} \ge 5, DF \le 50$).
2. **Secondary A (Address Overlap)**:
   - At least 2 shared informative address tokens with $DF \le 2000$.
3. **C2-A (Country + Building Number + Address Token)**:
   - Same country + shared building/address number + at least 1 address token with $DF \le 500$.

#### Evidence-ranked candidate selection
Candidates from evidence sources weighted $\ge 4$ are preserved. Up to 17,500 lower-evidence candidates per query are retained by aggregate evidence, with entity ID as a deterministic tie-break. This is a soft cap on lower-evidence candidates, not a hard total candidate limit.

#### Candidate-only validation (before address scoring)
- **Seed=42**: 16,380 / 17,314 true pairs (94.61% recall), 7,464,219 candidate pairs (0.2194% precision).
- **Seed=123**: 16,261 / 17,205 true pairs (94.51% recall), 8,056,698 candidate pairs (0.2018% precision).

`blocking.py` contains the early prototype blocker; production retrieval uses `retrieval.py`.

### 6. Production Submission Pipeline (`src/baseline.py`)
The submission uses the frozen `ProductionRetrievalIndex` selector for threshold calibration, hold-out reporting, and test prediction. It builds postings only for keys active in the corresponding S1 query set, while a temporary SQLite table stores target IDs, countries, and normalized addresses for scoring. This avoids a full unfiltered in-memory index; memory still depends on query-key overlap with the target corpus.

Run from the repository root:

```bash
.venv/bin/python code/business_entity_resolution/src/baseline.py \
  --data-dir ../6ab10eb3b23ba_student_resource/student_resource/dataset \
  --output-dir output
```

The selected production run calibrated threshold `0.85`; its labeled hold-out reported micro pair precision `0.357416`, recall `0.441166`, macro $F_{0.5}$ `0.465278`, and candidate pair recall `0.944430`. Test labels are unavailable, so these are hold-out—not test-set—scores.

### 7. Evaluation (`src/evaluation.py`)
`f_beta_per_s1(predicted, truth)` scores one S1 record, including correct empty predictions for singletons. `macro_f05(predictions, ground_truth)` averages that score across every labeled S1 record. The baseline reports candidate pair recall separately from held-out macro $F_{0.5}$.

---

### 8. Combo 3 Recall Comparison (`src/evaluate_targeted_retrieval.py`)
Run the deterministic 5,000-S1 validation comparison (seed 42) and write reports outside tracked outputs:

```bash
.venv/bin/python code/business_entity_resolution/src/evaluate_targeted_retrieval.py \
  --data-dir ../6ab10eb3b23ba_student_resource/student_resource/dataset \
  --output-dir /tmp/combo3-recall-study
```

The report's `combo3_b_recovery_variants` compares the verified Method B candidate set with relaxed name, address, and geographic recovery rules. It reports pair recall and candidate counts; this analysis does not change submission predictions. Candidate recall is not matching accuracy or macro $F_{0.5}$.


## Submission File Format

Both files contain one row per test S1 record, including rows with no candidates or matches. Column two contains a comma-separated, sorted list of IDs; an empty list is an empty second column.

```text
candidate_pairs.tsv:  source1_entity_id<TAB>candidate_entity_ids
matching_results.tsv: source1_entity_id<TAB>matched_entity_ids
```

Each matched ID comes from that row's candidate list. Candidate and match IDs come from test S2/S3 records, never from S1.
