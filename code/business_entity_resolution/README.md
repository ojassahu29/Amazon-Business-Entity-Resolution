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
    ├── __init__.py                # Package initialization
    ├── data_loader.py             # Chunked streaming loaders for multi-GB TSVs
    ├── preprocessing.py           # Unicode NFKC normalization, tokenization, & legal suffix cleaners
    ├── profile_data.py            # Memory-safe dataset profiler & exploratory data analysis
    ├── analyze_true_matches.py     # Ground truth diagnostic: agreement rates, distributions, edit distances
    ├── analyze_hard_negatives.py   # Adversarial & hard negative mining, token collision profiling
    ├── blocking.py                # Multi-index candidate blocker with stopword filtering
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

### 5. Multi-Index Candidate Blocker (`blocking.py`)
High-recall inverted-index blocker designed to scale to 10M+ records without quadratic memory blowup.

```bash
python src/blocking.py --data-dir path/to/dataset --sample-size 5000
```

**Blocking Strategies Combined:**
1. **`country + name_norm`**: Catches exact normalized name matches within the same country (~78.8% of true pairs).
2. **`country + name_sorted`**: Catches token permutation differences (e.g., `"Amazon India"` vs `"India Amazon"`).
3. **`country + name_compact`**: Catches punctuation and spacing variations (e.g., `"Wal-Mart"` vs `"Walmart"`).
4. **`country + informative name tokens`**: Catches partial/sub-token matches using stopword-filtered tokens (length $\ge 3$, non-legal keywords).
5. **`country + address tokens`**: Catches name variations sharing distinctive address elements.

**Safety Mechanisms:**
- Enforces an empirical stopword list (`pvt`, `ltd`, `limited`, `private`, `llc`, `inc`, `corp`, `services`, `solutions`, etc.).
- Truncates oversized inverted-index buckets to protect against candidate explosion on adversarial queries.

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
