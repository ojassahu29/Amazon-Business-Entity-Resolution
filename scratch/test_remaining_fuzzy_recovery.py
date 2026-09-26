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

with open("output/canonical_combo3_index_cache.pkl", "rb") as f:
    cache = pickle.load(f)

idx_name_tokens = cache["idx_name_tokens"]
idx_compact_prefix5 = cache["idx_compact_prefix5"]
idx_addr_tokens = cache["idx_addr_tokens"]
idx_addr_numbers = cache["idx_addr_numbers"]
true_match_records = cache["true_match_records"]

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

# Compute baseline retrieved true matches
from itertools import combinations

def get_token_overlap_candidates(token_sets, min_overlap):
    if len(token_sets) < min_overlap:
        return set()
    token_sets = sorted(token_sets, key=len)
    if min_overlap == 2 and len(token_sets) <= 6:
        res = set()
        for a, b in combinations(token_sets, 2):
            res |= (a & b)
        return res
    hits = defaultdict(int)
    for s in token_sets:
        for eid in s:
            hits[eid] += 1
    return {eid for eid, cnt in hits.items() if cnt >= min_overlap}

retrieved_base = set()
for s1_id in val_s1_ids:
    p = s1_parsed[s1_id]
    c = p["country"]
    c1 = cache["idx_name_norm"].get((c, p["name_norm"]), set())
    c2 = cache["idx_name_sorted"].get((c, sorted_tokens(p["name"])), set())
    c3 = cache["idx_name_compact"].get((c, compact(p["name"])), set())
    name_sets = [idx_name_tokens.get((c, t), set()) for t in p["info_name"] if (c, t) in idx_name_tokens]
    c4 = get_token_overlap_candidates(name_sets, min_overlap=2)
    addr_sets = [idx_addr_tokens.get((c, t), set()) for t in p["info_addr"] if (c, t) in idx_addr_tokens]
    c5 = get_token_overlap_candidates(addr_sets, min_overlap=3)
    c_set = c1 | c2 | c3 | c4 | c5
    
    stop_n = [t for t in tokenize(p["name"]) if t in NAME_STOPWORDS]
    rare_info = [idx_name_tokens[(c, t)] for t in p["info_name"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
    if rare_info and stop_n:
        s_union = set()
        for st in stop_n:
            s_union |= cache["idx_name_stopwords"].get((c, st), set())
        if s_union:
            for n_set in rare_info:
                c_set |= (n_set & s_union)
                
    nums = extract_address_numbers(p["addr"])
    valid_nums = [n for n in nums if len(n) >= 3 and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 500]
    valid_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 1000]
    if valid_nums and valid_addrs:
        a_union = set().union(*valid_addrs)
        for num in valid_nums:
            c_set |= (idx_addr_numbers[(c, num)] & a_union)
            
    rare_addrs = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
    if len(rare_addrs) >= 2:
        c_set |= get_token_overlap_candidates(rare_addrs, min_overlap=2)
        
    for t in p["info_name"]:
        if len(t) >= 5 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 50:
            c_set |= idx_name_tokens[(c, t)]
            
    p5 = p["prefix5"]
    if p5 and (c, p5) in idx_compact_prefix5 and len(idx_compact_prefix5[(c, p5)]) <= 50:
        c_set |= idx_compact_prefix5[(c, p5)]
        
    rare_addrs_2000 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
    if len(rare_addrs_2000) >= 2:
        c_set |= get_token_overlap_candidates(rare_addrs_2000, min_overlap=2)
        
    for t in p["info_name"]:
        if len(t) >= 4 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 100:
            c_set |= idx_name_tokens[(c, t)]
            
    b_nums = [n for n in p["bldg_nums"] if (c, n) in idx_addr_numbers]
    a_toks_500 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
    if b_nums and a_toks_500:
        a_u = set().union(*a_toks_500)
        for n in b_nums:
            c_set |= (idx_addr_numbers[(c, n)] & a_u)
            
    for m in set(val_gt[s1_id]) & c_set:
        retrieved_base.add((s1_id, m))

print(f"Base retrieved: {len(retrieved_base):,} / 17,314 ({len(retrieved_base)/17314*100:.2f}%)", flush=True)

# Now evaluate remaining misses:
missed_pairs = []
for s1_id in val_s1_ids:
    for mid in val_gt[s1_id]:
        if (s1_id, mid) not in retrieved_base:
            missed_pairs.append((s1_id, mid))

print(f"Total remaining missed true matches: {len(missed_pairs)}", flush=True)

# Profile newly recovered true matches by F1, F2, F3, F4:
new_rec = defaultdict(lambda: defaultdict(int))
fuzzy_rec = defaultdict(lambda: defaultdict(int))

for s1_id, mid in missed_pairs:
    p = s1_parsed[s1_id]
    c = p["country"]
    s1_nn = p["name_norm"]
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
    
    # Is it in fuzzy category?
    is_fuzzy_cat = (sim >= 70 or fuzz.token_sort_ratio(s1_nn, cand_nn) >= 75)
    
    # Check F1: shared 1 info name token (DF <= 500)
    shared_info_n = [t for t in (set(p["info_name"]) & set(cand_info_n)) if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
    f1_pass = len(shared_info_n) >= 1
    
    # Check F2: shared compact prefix5 (DF <= 200)
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
                new_rec["F1"][thresh] += 1
                if is_fuzzy_cat:
                    fuzzy_rec["F1"][thresh] += 1
            if f2_pass:
                new_rec["F2"][thresh] += 1
                if is_fuzzy_cat:
                    fuzzy_rec["F2"][thresh] += 1
            if f3_pass:
                new_rec["F3"][thresh] += 1
                if is_fuzzy_cat:
                    fuzzy_rec["F3"][thresh] += 1
            if f4_pass:
                new_rec["F4"][thresh] += 1
                if is_fuzzy_cat:
                    fuzzy_rec["F4"][thresh] += 1

print("\n--- NEWLY RECOVERED OVER CURRENT BASELINE (16,380) ---")
for var in ["F1", "F2", "F3", "F4"]:
    print(f"\n{var}:")
    for thresh in [80, 85, 90, 95]:
        print(f"  Thresh >={thresh}: +{new_rec[var][thresh]} newly recovered (Fuzzy cat: +{fuzzy_rec[var][thresh]})")
