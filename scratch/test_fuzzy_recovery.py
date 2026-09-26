import pickle
import re
from pathlib import Path
import random
import pandas as pd
from collections import defaultdict
import json
import sys

# pyrefly: ignore [missing-import]
from rapidfuzz import fuzz

sys.path.insert(0, "code/business_entity_resolution/src")
from preprocessing import compact, normalize_basic, sorted_tokens, tokenize
from blocking import ADDR_STOPWORDS, NAME_STOPWORDS

print("Loading cache...", flush=True)
with open("output/canonical_combo3_index_cache.pkl", "rb") as f:
    cache = pickle.load(f)

idx_name_norm = cache["idx_name_norm"]
idx_name_sorted = cache["idx_name_sorted"]
idx_name_compact = cache["idx_name_compact"]
idx_compact_prefix5 = cache["idx_compact_prefix5"]
idx_name_tokens = cache["idx_name_tokens"]
idx_name_stopwords = cache["idx_name_stopwords"]
idx_addr_tokens = cache["idx_addr_tokens"]
idx_addr_numbers = cache["idx_addr_numbers"]
true_match_records = cache["true_match_records"]

print("Loading GT...", flush=True)
gt = {}
for chunk in pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
    for row in chunk.itertuples(index=False):
        m = row.matched_entity_ids.strip()
        gt[row.source1_entity_id] = [x.strip() for x in m.split(",") if x.strip()] if m else []

val_s1_ids = random.Random(42).sample(sorted(gt.keys()), 5000)
val_s1_set = set(val_s1_ids)
val_gt = {s: gt[s] for s in val_s1_ids}

val_s1_records = {}
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
    mask = chunk["entity_id"].isin(val_s1_set)
    for eid, name, addr, country in zip(chunk.loc[mask, "entity_id"], chunk.loc[mask, "business_name"], chunk.loc[mask, "business_address"], chunk.loc[mask, "country"]):
        val_s1_records[eid] = {"name": name, "addr": addr, "country": normalize_basic(country)}

s1_parsed = {}
for s1_id, rec in val_s1_records.items():
    c = rec["country"]
    name = rec["name"]
    addr = rec["addr"]
    toks_n = tokenize(name)
    toks_a = tokenize(addr)
    info_n = [t for t in toks_n if len(t) >= 3 and t not in NAME_STOPWORDS]
    info_a = [t for t in toks_a if len(t) >= 3 and t not in ADDR_STOPWORDS]
    nums = re.findall(r"\b\d+[/\w-]*\b", addr)
    bldg = {n.strip(" ,.-/#").lower() for n in nums if len(n.strip(" ,.-/#")) >= 2 and n not in {"2019", "2020", "2021", "2022", "2023", "2024", "2025"} and not re.match(r"^\d{5,6}$", n.strip(" ,.-/#"))}
    nc = compact(name)
    p5 = nc[:5] if len(nc) >= 5 else ""
    s1_parsed[s1_id] = {"country": c, "name": name, "addr": addr, "name_norm": normalize_basic(name), "info_name": info_n, "info_addr": info_a, "bldg_nums": bldg, "prefix5": p5}

# Load current baseline retrieved pairs
current_baseline_pairs = set()
with open("output/cross_script_address_retrieval_results.json", "r", encoding="utf-8") as f:
    c2_res = json.load(f)

# Combo 3 + A + B + C2-A had 16,380 true matches
# Let's verify which true matches are already in current baseline:
# We know from analyze_missed_matches that 16,218 are in C3, and we have the exact missed matches.

# Let's check for each candidate true match if it is recovered by F1, F2, F3, F4:
# For true matches missed by current baseline:
print("Checking true matches recovery across F1, F2, F3, F4...", flush=True)

# For each validation true match:
# Check if it satisfies F1, F2, F3, F4 blocking conditions and what its fuzz.ratio is:
recovered_by_variant = defaultdict(lambda: defaultdict(int))

for s1_id in val_s1_ids:
    p = s1_parsed[s1_id]
    c = p["country"]
    s1_nn = p["name_norm"]
    
    for mid in val_gt[s1_id]:
        m_rec = true_match_records.get(mid)
        if not m_rec:
            continue
        cand_nn = normalize_basic(m_rec["business_name"])
        sim = fuzz.ratio(s1_nn, cand_nn)
        cand_toks_n = tokenize(cand_nn)
        cand_info_n = [t for t in cand_toks_n if len(t) >= 3 and t not in NAME_STOPWORDS]
        cand_toks_a = tokenize(m_rec["business_address"])
        cand_info_a = [t for t in cand_toks_a if len(t) >= 3 and t not in ADDR_STOPWORDS]
        cand_nums = re.findall(r"\b\d+[/\w-]*\b", m_rec["business_address"])
        cand_bldg = {n.strip(" ,.-/#").lower() for n in cand_nums if len(n.strip(" ,.-/#")) >= 2 and n not in {"2019", "2020", "2021", "2022", "2023", "2024", "2025"} and not re.match(r"^\d{5,6}$", n.strip(" ,.-/#"))}
        cand_nc = compact(cand_nn)
        cand_p5 = cand_nc[:5] if len(cand_nc) >= 5 else ""
        
        # Check F1: shared 1 info name token (where DF <= 500)
        shared_info_n = [t for t in (set(p["info_name"]) & set(cand_info_n)) if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
        f1_pass = len(shared_info_n) >= 1
        
        # Check F2: shared compact prefix5 (where DF <= 200)
        f2_pass = (p["prefix5"] and p["prefix5"] == cand_p5 and (c, p["prefix5"]) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p["prefix5"])]) <= 200)
        
        # Check F3: 1 shared name token (DF <= 1000) AND 1 shared addr token (DF <= 2000)
        shared_n_1000 = [t for t in (set(p["info_name"]) & set(cand_info_n)) if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 1000]
        shared_a_2000 = [t for t in (set(p["info_addr"]) & set(cand_info_a)) if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
        f3_pass = (len(shared_n_1000) >= 1 and len(shared_a_2000) >= 1)
        
        # Check F4: shared bldg num AND 1 shared addr token (DF <= 1000)
        shared_bldg = [n for n in (p["bldg_nums"] & cand_bldg) if (c, n) in idx_addr_numbers]
        shared_a_1000 = [t for t in (set(p["info_addr"]) & set(cand_info_a)) if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
        f4_pass = (len(shared_bldg) >= 1 and len(shared_a_1000) >= 1)
        
        for thresh in [80, 85, 90, 95]:
            if sim >= thresh:
                if f1_pass:
                    recovered_by_variant["F1"][thresh] += 1
                if f2_pass:
                    recovered_by_variant["F2"][thresh] += 1
                if f3_pass:
                    recovered_by_variant["F3"][thresh] += 1
                if f4_pass:
                    recovered_by_variant["F4"][thresh] += 1

print("\n--- True Matches Passing Blocking + Fuzzy Sim ---")
for var in ["F1", "F2", "F3", "F4"]:
    print(f"\n{var}:")
    for thresh in [80, 85, 90, 95]:
        print(f"  Thresh >={thresh}: {recovered_by_variant[var][thresh]} true matches pass")
