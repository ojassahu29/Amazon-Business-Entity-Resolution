import pickle
import re
from pathlib import Path
import random
from itertools import combinations
import pandas as pd
from collections import Counter, defaultdict
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# pyrefly: ignore [missing-import]
from rapidfuzz import fuzz
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.insert(0, "code/business_entity_resolution/src")
# pyrefly: ignore [missing-import]
from blocking import ADDR_STOPWORDS, NAME_STOPWORDS
# pyrefly: ignore [missing-import]
from preprocessing import compact, normalize_basic, sorted_tokens, tokenize

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

def is_synthetic_or_cross_script(text: str) -> bool:
    if any(ord(c) > 127 for c in text):
        return True
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 6:
        vowels = sum(1 for c in cleaned if c in "aeiou")
        vowel_ratio = vowels / len(cleaned)
        if vowel_ratio < 0.15 or vowel_ratio > 0.70:
            return True
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", cleaned):
            return True
    return False

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
val_s1_set = set(val_s1_ids)

print("Loading S1 records...", flush=True)
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
    nn = normalize_basic(name)
    ns = sorted_tokens(name)
    nc = compact(name)
    p5 = nc[:5] if len(nc) >= 5 else ""
    toks_n = tokenize(name)
    toks_a = tokenize(addr)
    info_n = [t for t in toks_n if len(t) >= 3 and t not in NAME_STOPWORDS]
    stop_n = [t for t in toks_n if t in NAME_STOPWORDS]
    info_a = [t for t in toks_a if len(t) >= 3 and t not in ADDR_STOPWORDS]
    nums = extract_address_numbers(addr)
    bldg = {n for n in nums if not re.match(r"^\d{5,6}$", n)}
    pins = extract_pincodes(addr)
    
    s1_parsed[s1_id] = {
        "country": c,
        "raw_name": name,
        "raw_addr": addr,
        "name_norm": nn,
        "name_sorted": ns,
        "name_compact": nc,
        "prefix5": p5,
        "info_name": info_n,
        "stop_name": stop_n,
        "info_addr": info_a,
        "all_nums": nums,
        "bldg_nums": bldg,
        "pincodes": pins,
    }

print("Running current baseline retrieval (Combo 3 + Sec A + Sec B + C2-A)...", flush=True)
baseline_cands = {}
retrieved_pairs = set()

for s1_id in val_s1_ids:
    p = s1_parsed[s1_id]
    c = p["country"]
    
    # 1. Combo 3
    c1 = idx_name_norm.get((c, p["name_norm"]), set())
    c2 = idx_name_sorted.get((c, p["name_sorted"]), set())
    c3 = idx_name_compact.get((c, p["name_compact"]), set())
    name_sets = [idx_name_tokens.get((c, t), set()) for t in p["info_name"] if (c, t) in idx_name_tokens]
    c4 = get_token_overlap_candidates(name_sets, min_overlap=2)
    addr_sets = [idx_addr_tokens.get((c, t), set()) for t in p["info_addr"] if (c, t) in idx_addr_tokens]
    c5 = get_token_overlap_candidates(addr_sets, min_overlap=3)
    c_set = c1 | c2 | c3 | c4 | c5
    
    rare_info = [idx_name_tokens[(c, t)] for t in p["info_name"] if (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 500]
    if rare_info and p["stop_name"]:
        s_union = set()
        for st in p["stop_name"]:
            s_union |= idx_name_stopwords.get((c, st), set())
        if s_union:
            for n_set in rare_info:
                c_set |= (n_set & s_union)
                
    valid_nums = [n for n in p["all_nums"] if len(n) >= 3 and (c, n) in idx_addr_numbers and len(idx_addr_numbers[(c, n)]) <= 500]
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
        
    # 2. Sec A (2 rare addr tokens, DF <= 2000)
    rare_addrs_2000 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 2000]
    if len(rare_addrs_2000) >= 2:
        c_set |= get_token_overlap_candidates(rare_addrs_2000, min_overlap=2)
        
    # 3. Sec B (single name token len>=4, DF<=100)
    for t in p["info_name"]:
        if len(t) >= 4 and (c, t) in idx_name_tokens and len(idx_name_tokens[(c, t)]) <= 100:
            c_set |= idx_name_tokens[(c, t)]
            
    # 4. C2-A (shared bldg num + addr token DF<=500)
    b_nums = [n for n in p["bldg_nums"] if (c, n) in idx_addr_numbers]
    a_toks_500 = [idx_addr_tokens[(c, t)] for t in p["info_addr"] if (c, t) in idx_addr_tokens and len(idx_addr_tokens[(c, t)]) <= 500]
    if b_nums and a_toks_500:
        a_u = set().union(*a_toks_500)
        for n in b_nums:
            c_set |= (idx_addr_numbers[(c, n)] & a_u)
            
    baseline_cands[s1_id] = c_set
    for m in set(val_gt[s1_id]) & c_set:
        retrieved_pairs.add((s1_id, m))

print(f"Current Baseline Retrieved: {len(retrieved_pairs):,} / 17,314 ({len(retrieved_pairs)/17314*100:.2f}%)", flush=True)

# Identify remaining missed matches
remaining_misses = []
for s1_id in val_s1_ids:
    missed_for_s1 = set(val_gt[s1_id]) - baseline_cands[s1_id]
    for mid in missed_for_s1:
        s1_info = s1_parsed[s1_id]
        m_rec = true_match_records.get(mid)
        if not m_rec:
            continue
        s1_n = s1_info["name_norm"]
        cand_n = normalize_basic(m_rec["business_name"])
        s1_a = normalize_basic(s1_info["raw_addr"])
        cand_a = normalize_basic(m_rec["business_address"])
        
        # Similarities
        r_ratio = fuzz.ratio(s1_n, cand_n)
        r_tsort = fuzz.token_sort_ratio(s1_n, cand_n)
        r_tset = fuzz.token_set_ratio(s1_n, cand_n)
        r_part = fuzz.partial_ratio(s1_n, cand_n)
        
        # Token metrics
        toks_s1 = tokenize(s1_n)
        toks_cand = tokenize(cand_n)
        shared_exact_tokens = set(toks_s1) & set(toks_cand)
        shared_info_name = set(s1_info["info_name"]) & set([t for t in toks_cand if len(t) >= 3 and t not in NAME_STOPWORDS])
        
        # Address metrics
        shared_addr_tokens = set(s1_info["info_addr"]) & set([t for t in tokenize(cand_a) if len(t) >= 3 and t not in ADDR_STOPWORDS])
        m_nums = extract_address_numbers(m_rec["business_address"])
        shared_nums = set(s1_info["all_nums"]) & set(m_nums)
        
        # Character n-grams (3-grams and 4-grams)
        s1_c = compact(s1_n)
        cand_c = compact(cand_n)
        ng3_s1 = {s1_c[i:i+3] for i in range(len(s1_c)-2)} if len(s1_c) >= 3 else set()
        ng3_cand = {cand_c[i:i+3] for i in range(len(cand_c)-2)} if len(cand_c) >= 3 else set()
        shared_3grams = ng3_s1 & ng3_cand
        
        ng4_s1 = {s1_c[i:i+4] for i in range(len(s1_c)-3)} if len(s1_c) >= 4 else set()
        ng4_cand = {cand_c[i:i+4] for i in range(len(cand_c)-3)} if len(cand_c) >= 4 else set()
        shared_4grams = ng4_s1 & ng4_cand
        
        is_cross = is_synthetic_or_cross_script(s1_info["raw_name"]) or is_synthetic_or_cross_script(m_rec["business_name"])
        
        remaining_misses.append({
            "s1_id": s1_id,
            "mid": mid,
            "s1_name": s1_info["raw_name"],
            "cand_name": m_rec["business_name"],
            "s1_addr": s1_info["raw_addr"],
            "cand_addr": m_rec["business_address"],
            "country": s1_info["country"],
            "r_ratio": r_ratio,
            "r_tsort": r_tsort,
            "r_tset": r_tset,
            "r_part": r_part,
            "s1_len": len(s1_n),
            "cand_len": len(cand_n),
            "s1_tok_cnt": len(toks_s1),
            "cand_tok_cnt": len(toks_cand),
            "shared_exact_tokens": len(shared_exact_tokens),
            "shared_info_name": len(shared_info_name),
            "shared_addr_tokens": len(shared_addr_tokens),
            "shared_nums": len(shared_nums),
            "shared_3grams": len(shared_3grams),
            "shared_4grams": len(shared_4grams),
            "is_cross": is_cross,
        })

print(f"Total remaining missed matches: {len(remaining_misses)}", flush=True)

# Separate into Cross-Script vs Spelling/OCR/Fuzzy vs Other
cs_misses = [m for m in remaining_misses if m["is_cross"]]
non_cs = [m for m in remaining_misses if not m["is_cross"]]
fuzzy_misses = [m for m in non_cs if m["r_ratio"] >= 70 or m["r_tsort"] >= 75 or m["r_tset"] >= 75]
other_misses = [m for m in non_cs if m not in fuzzy_misses]

print(f"  Cross-script remaining: {len(cs_misses)}", flush=True)
print(f"  Spelling / OCR / Fuzzy remaining: {len(fuzzy_misses)}", flush=True)
print(f"  Other remaining: {len(other_misses)}", flush=True)

# Step 1: Characterize the remaining fuzzy misses
print("\n=== STEP 1: EMPIRICAL DISTRIBUTIONS OF REMAINING FUZZY MISSES ===", flush=True)
ratios = [m["r_ratio"] for m in fuzzy_misses]
tsorts = [m["r_tsort"] for m in fuzzy_misses]
tsets = [m["r_tset"] for m in fuzzy_misses]
parts = [m["r_part"] for m in fuzzy_misses]

for name, arr in [("RapidFuzz ratio", ratios), ("Token Sort Ratio", tsorts), ("Token Set Ratio", tsets), ("Partial Ratio", parts)]:
    print(f"{name:<20}: Mean={np.mean(arr):.1f} Median={np.median(arr):.1f} Min={np.min(arr):.1f} Max={np.max(arr):.1f}")
    print(f"  Counts >=95: {sum(1 for x in arr if x>=95)}, >=90: {sum(1 for x in arr if x>=90)}, >=85: {sum(1 for x in arr if x>=85)}, >=80: {sum(1 for x in arr if x>=80)}, >=75: {sum(1 for x in arr if x>=75)}")

print("\nShared exact tokens in fuzzy category:")
print(Counter(m["shared_exact_tokens"] for m in fuzzy_misses))

print("\nShared informative name tokens in fuzzy category:")
print(Counter(m["shared_info_name"] for m in fuzzy_misses))

print("\nShared address tokens in fuzzy category:")
print(Counter(m["shared_addr_tokens"] for m in fuzzy_misses))

print("\nShared address numbers in fuzzy category:")
print(Counter(m["shared_nums"] for m in fuzzy_misses))

print("\nShared 4-grams in fuzzy category:")
print(Counter(min(m["shared_4grams"], 10) for m in fuzzy_misses))
