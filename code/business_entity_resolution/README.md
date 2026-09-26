# Business Entity Resolution Pipeline

A scalable, memory-efficient machine learning pipeline for cross-source business entity resolution across multiple independent and noisy data sources (Source 1 reference, Source 2, and Source 3).

## Directory Structure

```
code/business_entity_resolution/
├── README.md               # End-to-end reproduction guide
├── requirements.txt        # Pinned project dependencies
├── pyproject.toml          # PEP 517/621 packaging metadata
└── src/                    # Source code modules
    ├── __init__.py
    ├── data_loader.py      # Chunked streaming loaders for multi-GB TSVs
    ├── preprocessing.py    # Unicode NFKC normalization, tokenization, & cleaning
    └── profile_data.py     # Memory-safe dataset profiler & EDA
```

## Setup & Environment

### Prerequisites
- Python >= 3.10
- Git

### Installation
From the project root or inside `code/business_entity_resolution/`:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# (Optional) Install package in editable mode:
pip install -e .
```

## Pipeline Workflow

### 1. Data Profiling & Quality Audit
Profile raw training and test splits across all sources without loading full datasets into memory:

```bash
python src/profile_data.py --data-dir ../../dataset --chunksize 250000
```

Audits:
- Row counts, unique entity IDs, and duplicate detection
- Missing values across `business_name`, `business_address`, and `country`
- Distribution of ground-truth matches per Source 1 entity (including singletons)

### 2. Unicode-Safe Preprocessing
The dataset contains multilingual scripts (Latin, Devanagari, Tamil, etc.), noisy punctuation, and abbreviations.
- **NFKC Normalization**: Canonical and compatibility composition.
- **Unicode-aware Punctuation Stripping**: Preserves Indic combining characters while cleaning punctuation.
- **Token Sorting**: Creates order-independent representations (e.g., `"Alpha Tech Pvt Ltd"` → `"alpha ltd pvt tech"`).
- **Address & Country Normalization**: Handles open-set country labels (US, India, France).

### 3. Candidate Generation / Blocking
Generates plausible candidates from Source 2 and Source 3 for each Source 1 entity:
- Produces `output/candidate_pairs.tsv`
- Filters candidate space to maximize recall while keeping candidate set size manageable.

### 4. Matching & Classification Model
Scoring candidate pairs and selecting high-confidence matches:
- Produces `output/matching_results.tsv`
- Optimized for the precision-heavy $F_{0.5}$ metric:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Singletons (Source 1 entities with no true match) correctly output an empty match list.

### 5. Submission Validation
Validate final outputs against competition constraints:

```bash
python ../../utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```
