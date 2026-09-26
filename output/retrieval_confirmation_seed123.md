# Retrieval Architecture Confirmation Experiment: Seed=123 Validation

## Executive Summary

This experiment evaluated whether the retrieval pruning decision (specifically removing Secondary B) generalizes to an independent, unseen 5,000 $S_1$ validation sample using `seed=123` across the full 10.3M-record $S_2 + S_3$ universe.

- **Sample Size**: 5,000 $S_1$ entities (seed=123)
- **Total True Matches**: 17,205

### Measured Results for the 4 Candidate Configurations

| Configuration | Retrieved True Matches | Retrieval Recall | Total Candidates | Mean / $S_1$ | Median | p90 | p95 | p99 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C3** | 16,100 / 17,205 | 93.58% | 9,510,217 | 1902.04 | 214.0 | 4347.3 | 11280.3 | 29510.1 | 57,085 |
| **C3 + C2-A** | 16,206 / 17,205 | 94.19% | 9,514,202 | 1902.84 | 214.0 | 4355.4 | 11280.3 | 29510.1 | 57,085 |
| **C3 + A + C2-A** | 16,264 / 17,205 | 94.53% | 9,595,099 | 1919.02 | 222.0 | 4369.2 | 11344.6 | 29510.1 | 57,085 |
| **C3 + A + B + C2-A** | 16,273 / 17,205 | 94.58% | 9,615,416 | 1923.08 | 226.0 | 4369.2 | 11344.6 | 29510.1 | 57,085 |

---

## Incremental Marginal Contributions

| Step | Added Rule | Unique Matches Contributed | Added Candidates | Added Cands / $S_1$ | Incremental Candidates per Match |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C3 $\to$ C3 + C2-A** | C2-A ($DF \le 500$) | **+106** | **+3,985** | **+0.80** | **37.6** |
| **C3 + C2-A $\to$ C3 + A + C2-A** | Secondary A ($DF \le 2000$) | **+58** | **+80,897** | **+16.18** | **1394.8** |
| **C3 + A + C2-A $\to$ Full (+B)** | Secondary B ($DF \le 100$) | **+9** | **+20,317** | **+4.06** | **2257.4** |

---

## Cross-Seed Consistency Comparison: Seed=42 vs Seed=123

| Metric | Seed = 42 (17,314 matches) | Seed = 123 (17,205 matches) | Consistency Check |
| :--- | :---: | :---: | :---: |
| **C3 Recall** | 93.67% | 93.58% | Empirical measurement |
| **C3 + C2-A Recall** | 94.21% | 94.19% | Empirical measurement |
| **C3 + A + C2-A Recall** | 94.55% | 94.53% | Empirical measurement |
| **Full Stack Recall** | 94.61% | 94.58% | Empirical measurement |
| **Secondary B Unique Matches** | +10 matches (+0.058%) | +9 matches (+0.052%) | Confirms B is marginal |
| **Secondary B Added Candidates** | +19,089 (+3.82 / $S_1$) | +20,317 (+4.06 / $S_1$) | Measured scale |
| **B Candidates per Unique Match** | 1,908.9 cands / match | 2257.4 cands / match | Confirms low efficiency |

