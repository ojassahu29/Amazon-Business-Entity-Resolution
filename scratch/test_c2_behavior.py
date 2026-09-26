import pickle
import re
from pathlib import Path
import random
import pandas as pd
from collections import defaultdict
import json
import sys

def normalize_basic(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list[str]:
    return [t for t in normalize_basic(text).split() if t]

def extract_address_numbers(addr: str) -> set[str]:
    raw_nums = re.findall(r"\b\d+[/\w-]*\b", addr)
    nums = set()
    for n in raw_nums:
        clean_n = n.strip(" ,.-/#")
        if len(clean_n) >= 2 and clean_n not in {"2019", "2020", "2021", "2022", "2023", "2024", "2025"}:
            nums.add(clean_n.lower())
    return nums

def extract_pincodes(addr: str) -> set[str]:
    return set(re.findall(r"\b\d{5,6}\b", addr))

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
val_gt = {s: gt[s] for s in val_s1_ids}

print("Loading S1...", flush=True)
val_s1_records = {}
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
    mask = chunk["entity_id"].isin(set(val_s1_ids))
    for eid, name, addr, country in zip(chunk.loc[mask, "entity_id"], chunk.loc[mask, "business_name"], chunk.loc[mask, "business_address"], chunk.loc[mask, "country"]):
        val_s1_records[eid] = {"name": name, "addr": addr, "country": normalize_basic(country)}

sys.path.insert(0, "code/business_entity_resolution/src")
# pyrefly: ignore [missing-import]
from blocking import ADDR_STOPWORDS, NAME_STOPWORDS

s1_parsed = {}
for s1_id, rec in val_s1_records.items():
    country = rec["country"]
    name = rec["name"]
    addr = rec["addr"]
    toks_name = tokenize(name)
    toks_addr = tokenize(addr)
    all_nums = extract_address_numbers(addr)
    pins = extract_pincodes(addr)
    bldg_nums = {n for n in all_nums if not re.match(r"^\d{5,6}$", n)}
    info_addr = [t for t in toks_addr if len(t) >= 3 and t not in ADDR_STOPWORDS]
    
    s1_parsed[s1_id] = {
        "country": country,
        "raw_name": name,
        "raw_addr": addr,
        "all_nums": all_nums,
        "bldg_nums": bldg_nums,
        "pincodes": pins,
        "info_addr": info_addr,
    }

# Test different configurations for C2
configs = [
    ("bldg_num + DF<=500", "bldg_nums", None, 500),
    ("bldg_num + DF<=1000", "bldg_nums", None, 1000),
    ("bldg_num + DF<=2000", "bldg_nums", None, 2000),
    ("pincode + DF<=2000", "pincodes", None, 2000),
    ("bldg_OR_pin + DF<=2000", "all_nums", None, 2000),
    # Also test with num_cap = 1000
    ("bldg_num(cap1000) + DF<=2000", "bldg_nums", 1000, 2000),
    ("pincode(cap2000) + DF<=2000", "pincodes", 2000, 2000),
    ("all_nums(cap1000) + DF<=2000", "all_nums", 1000, 2000),
]

for label, num_field, num_cap, addr_cap in configs:
    tot_cands = 0
    tms_found = 0
    for s1_id in val_s1_ids:
        p = s1_parsed[s1_id]
        c = p["country"]
        nums = [n for n in p[num_field] if (c, n) in idx_addr_numbers]
        if num_cap is not None:
            nums = [n for n in nums if len(idx_addr_numbers[(c, n)]) <= num_cap]
        addr_sets = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= addr_cap]
        if nums and addr_sets:
            a_union = set().union(*addr_sets)
            res = set()
            for n in nums:
                res |= (idx_addr_numbers[(c, n)] & a_union)
            tot_cands += len(res)
            tms_found += len(res & set(val_gt[s1_id]))
    print(f"{label:<30}: True Matches Found={tms_found:<4} Total Cands={tot_cands:<8} Mean Cands={tot_cands/5000:.1f}", flush=True)
