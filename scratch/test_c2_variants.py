import pickle
import re
from pathlib import Path
import random
import pandas as pd
from collections import defaultdict

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

print("Loading cache...")
with open("output/canonical_combo3_index_cache.pkl", "rb") as f:
    cache = pickle.load(f)

idx_addr_tokens = cache["idx_addr_tokens"]
idx_addr_numbers = cache["idx_addr_numbers"]
true_match_records = cache["true_match_records"]

print("Loading GT...")
gt = {}
for chunk in pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
    for row in chunk.itertuples(index=False):
        m = row.matched_entity_ids.strip()
        gt[row.source1_entity_id] = [x.strip() for x in m.split(",") if x.strip()] if m else []

val_s1_ids = random.Random(42).sample(sorted(gt.keys()), 5000)
val_gt = {s: gt[s] for s in val_s1_ids}

print("Loading S1...")
val_s1 = {}
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
    mask = chunk["entity_id"].isin(set(val_s1_ids))
    for eid, name, addr, country in zip(chunk.loc[mask, "entity_id"], chunk.loc[mask, "business_name"], chunk.loc[mask, "business_address"], chunk.loc[mask, "country"]):
        val_s1[eid] = {"name": name, "addr": addr, "country": country.strip().lower()}

# Load missed matches from previous report
with open("output/combo3_missed_matches_analysis.json", "r", encoding="utf-8") as f:
    missed_analysis = json.load(f)

cross_script_pairs = set()
for cat in missed_analysis["failure_mode_breakdown"]:
    if "Cross-Script" in cat["category"]:
        for ex in cat["examples"]:
            pass # wait, examples only has 3 items!
