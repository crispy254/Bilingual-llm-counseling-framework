"""Reduced human study: 30 messages per language, ours AND base rated for each (paired), + 6 duplicates.
60 + 6 = 66 items per rater per language (about 1.5-2 h). Uses the same key as the full design.
Run: python build_study_a_small.py --key ../private/study_a_key.csv --out ..
"""
import argparse, os, random
import pandas as pd
from build_study_a import write_workbook, order_for_rater

SEED = 2027
ALLOC = {"LISTEN": 8, "ASK": 7, "SUGGEST": 7, "NORMALIZE": 8}   # 30 messages per language
N_DUP = 6

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = random.Random(SEED)
    K = pd.read_csv(a.key); K["dup_of"] = K.dup_of.fillna(""); K = K[(K.set == "main") & (K.dup_of == "")]
    os.makedirs(f"{a.out}/rater_packets_small", exist_ok=True)
    keys = []
    for lang in ["en", "ko"]:
        L = K[(K.language == lang) & K.stratum.isin(["ours", "base"])]
        msgs = L[L.stratum == "ours"][["message_id", "intended"]]
        chosen = []
        for s, n in ALLOC.items():
            ids = sorted(msgs[msgs.intended == s].message_id.tolist())
            chosen += rng.sample(ids, n)
        it = L[L.message_id.isin(chosen)].copy().reset_index(drop=True)
        dups = it.sample(n=N_DUP, random_state=rng.randint(0, 10**6)).copy()
        dups["stratum"] = "dup"
        it["orig_item_id"] = it["item_id"]; dups["orig_item_id"] = dups["item_id"]
        it["dup_of"] = ""; allr = pd.concat([it, dups]).reset_index(drop=True)
        perm = list(range(len(allr))); rng.shuffle(perm)
        new = [None] * len(allr)
        for k, i in enumerate(perm, 1): new[i] = f"S-{lang.upper()}-{k:03d}"
        allr["item_id"] = new
        origmap = {o: n for o, n, s in zip(allr.orig_item_id, allr.item_id, allr.stratum) if s != "dup"}
        allr["dup_of"] = [origmap[o] if s == "dup" else "" for o, s in zip(allr.orig_item_id, allr.stratum)]
        allr["set"] = "main"
        keys.append(allr.drop(columns=["orig_item_id"]))
        for k in range(1, 4):
            rid = f"{lang.upper()}_R{k}"
            write_workbook(f"{a.out}/rater_packets_small/StudyA_main_{rid}.xlsx",
                           order_for_rater(allr, SEED + 10 * k + (lang == "ko")), lang, rid)
    key = pd.concat(keys)
    key.to_csv(f"{a.out}/private/study_a_key_small.csv", index=False, encoding="utf-8-sig")
    print(key.groupby(["language", "stratum"]).size().unstack(fill_value=0))
    print(key[key.stratum != "dup"].groupby(["language", "stratum"]).judge_correct.mean().round(3))

if __name__ == "__main__":
    main()
