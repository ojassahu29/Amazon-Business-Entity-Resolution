# Business Entity Resolution Engine

A scalable, memory-efficient machine learning pipeline for cross-source business entity resolution across multiple independent and noisy corporate datasets (Source 1 reference, Source 2, and Source 3).

---

## 📂 Directory Structure

```
code/business_entity_resolution/
├── README.md                      # End-to-end technical reproduction guide
├── requirements.txt               # Pinned pip dependencies
├── environment.yml                # Conda environment definition
├── pyproject.toml                 # PEP 517/621 packaging metadata
└── src/                           # Core source modules
    ├── __init__.py                # Package initialization (exports production retrieval API)
    ├── data_loader.py             # Chunked streaming loaders for multi-GB TSVs
    ├── preprocessing.py           # Unicode NFKC normalization, tokenization, & legal suffix cleaners
    ├── retrieval.py               # Canonical frozen production retrieval (Combo 3 + Sec A + C2-A)
    ├── blocking.py                # Legacy multi-index candidate blocker
    ├── profile_data.py            # Memory-safe dataset profiler & exploratory data analysis
    ├── analyze_true_matches.py     # Ground truth diagnostic: agreement rates, distributions, edit distances
    ├── analyze_hard_negatives.py   # Adversarial & hard negative mining, token collision profiling
    └── evaluation.py              # Micro, macro, and per-source F0.5 evaluation suite
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
Robust handling of multilingual text, noisy characters, and legal business structures:
- **Unicode NFKC Normalization**: Standardizes full-width/half-width characters and compatibility forms while preserving Indic scripts (Devanagari, Tamil, Bengali, Telugu, etc.).
- **Punctuation Stripping**: Custom regex cleaning preserving Unicode word characters and whitespace.
- **Order-Independent Sorting**: Converts names to sorted token representations (e.g. `"Alpha Tech Pvt Ltd"` → `"alpha ltd pvt tech"`).
- **Compact Forms**: Strips all non-alphanumeric characters for fuzzy boundary alignment (e.g. `"A.B.C. Corp"` → `"abccorp"`).
- **Legal Form Normalization**: Canonicalizes company types (`"private limited"` → `"pvt ltd"`, `"incorporated"` → `"inc"`, `"gesellschaft mit beschränkter haftung"` → `"gmbh"`).

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
   - Single rare informative name token ($\text{len} \ge 5, DF \le 50$).
   - Rare compact prefix-5 ($\text{len} \ge 5, DF \le 50$).
2. **Secondary A (Address Overlap)**:
   - $\ge 2$ shared informative address tokens with $DF \le 2000$.
3. **C2-A (Country + Building Number + Address Token)**:
   - Same country + shared building/address number + $\ge 1$ address token with $DF \le 500$.

#### Note on Secondary B Removal
Secondary B (single informative name token with length $\ge 4$ and $DF \le 100$) was rigorously evaluated across two independent validation splits:
- `seed=42`: Contributed only 10 unique true matches while adding 19,089 candidate pairs (~1,909 candidates per match).
- `seed=123`: Contributed only 9 unique true matches while adding 20,317 candidate pairs (~2,257 candidates per match).
Because the marginal recall gain was negligible relative to the substantial candidate volume explosion, Secondary B is excluded from production retrieval.

#### Validated Architecture Metrics (5,000 $S_1$ queries, full $S_2 + S_3$ universe):
- **Seed=42 Validation**:
  - Retrieved True Matches: 16,370 / 17,314 (94.55% recall)
  - Total Candidate Pairs: 8,555,167
  - Distribution: Mean 1,711.03 / $S_1$, Median 233.5, P90 3,792.4, P95 8,339.2, P99 26,163.2, Max 59,456
  - Canonical SHA-256 Digest: `873c791862d91ae0c91f26047c4787af50de93e32d06db878e0d5802956f2c5c`
- **Seed=123 Validation**:
  - Retrieved True Matches: 16,264 / 17,205 (94.53% recall)
  - Total Candidate Pairs: 9,595,099 (Mean 1,919.02 / $S_1$)

#### Legacy Blocker (`blocking.py`)
`blocking.py` contains the early prototype blocker (`InvertedIndexBlocker`) and is preserved for historical baseline comparisons. Production execution uses `retrieval.py`.

---

### 6. Evaluation Suite (`evaluation.py`)
Computes official competition metrics on candidate generators and downstream matchers:

```python
from evaluation import evaluate_matches, print_evaluation_report

# ground_truth: dict mapping s1_id -> set(matched_s2_s3_ids)
# predictions:  dict mapping s1_id -> set(predicted_s2_s3_ids)
metrics = evaluate_matches(ground_truth, predictions)
print_evaluation_report(metrics)
```

**Metrics Computed:**
- **Micro Precision, Recall, and $F_{0.5}$**: Evaluated globally over all predicted pairs.
- **Macro Precision, Recall, and $F_{0.5}$**: Evaluated per S1 entity and averaged (handles singletons gracefully).
- **Per-Source Breakdown**: Separate precision/recall tracking for Source 2 and Source 3.

$$\beta = 0.5 \implies F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

---

## 🧪 Submission File Constraints

Predictions must follow the competition specifications:
- **`candidate_pairs.tsv`**: Candidate pairs generated by the blocking stage (`s1_id\ts_candidate_id`).
- **`matching_results.tsv`**: Final matched pairs passing confidence thresholds (`s1_id\ts_match_id`).
- Empty predictions must be output for singletons.
