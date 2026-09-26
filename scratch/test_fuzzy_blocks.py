import pickle
import re
from pathlib import Path
import random
import pandas as pd
from collections import defaultdict
import json
import sys

with open("output/canonical_combo3_index_cache.pkl", "rb") as f:
    cache = pickle.load(f)

idx_name_tokens = cache["idx_name_tokens"]
idx_compact_prefix5 = cache["idx_compact_prefix5"]
idx_addr_tokens = cache["idx_addr_tokens"]
idx_addr_numbers = cache["idx_addr_numbers"]

gt = {}
for chunk in pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype="string", keep_default_na=False, chunksize=500000):
    for row in chunk.itertuples(index=False):
        m = row.matched_entity_ids.strip()
        gt[row.source1_entity_id] = [x.strip() for x in m.split(",") if x.strip()] if m else []

val_s1_ids = random.Random(42).sample(sorted(gt.keys()), 5000)
val_s1_set = set(val_s1_ids)

sys.path.insert(0, "code/business_entity_resolution/src")
from preprocessing import compact, normalize_basic, sorted_tokens, tokenize
from blocking import ADDR_STOPWORDS, NAME_STOPWORDS

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
    s1_parsed[s1_id] = {"country": c, "name": name, "addr": addr, "info_name": info_n, "info_addr": info_a, "bldg_nums": bldg, "prefix5": p5}

# Check block sizes for F1, F2, F3, F4
all_block_eids = set()

# F1: 1 info name token (DF <= 200 or 500)
f1_eids = set()
for s1_id in val_s1_ids:
    p = s1_parsed[s1_id]
    c = p["country"]
    for t in p["info_name"]:
        if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500:
            f1_eids |= idx_name_tokens[(c, t)]

# F2: prefix5 (DF <= 200)
f2_eids = set()
for s1_id in val_s1_ids:
    p = s1_parsed[s1_id]
    c = p["country"]
    p5 = p["prefix5"]
    if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= 200:
        f2_eids |= idx_compact_prefix5[(c, p5)]

# F3: 1 info name token (DF <= 1000) & 1 info addr token (DF <= 2000)
f3_eids = set()
for s1_id in val_s1_ids:
    p = s1_parsed[s1_id]
    c = p["country"]
    n_sets = [idx_name_tokens[(c, t)] for t in p["info_name"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 1000]
    a_sets = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
    if n_sets and a_sets:
        n_u = set().union(*n_sets)
        a_u = set().union(*a_sets)
        f3_eids |= (n_u & a_u)

# F4: shared bldg num & 1 info addr token (DF <= 1000)
f4_eids = set()
for s1_id in val_s1_ids:
    p = s1_parsed[s1_id]
    c = p["country"]
    nums = [n for n in p["bldg_nums"] if (c, n) in idx_addr_numbers]
    a_sets = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
    if nums and a_sets:
        a_u = set().union(*a_sets)
        for n in nums:
            f4_eids |= (idx_addr_numbers[(c, n)] & a_u)

all_needed_eids = f1_eids | f2_eids | f3_eids | f4_eids
print(f"F1 unique EIDs: {len(f1_eids):,}")
print(f"F2 unique EIDs: {len(f2_eids):,}")
print(f"F3 unique EIDs: {len(f3_eids):,}")
print(f"F4 unique EIDs: {len(f4_eids):,}")
print(f"Total unique EIDs needed for names: {len(all_needed_eids):,}")
