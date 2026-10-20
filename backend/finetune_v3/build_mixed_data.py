"""
Builds the (smaller, faster) training data for BOTH models: a capped sample of the general data + the strategy data.
Use the SAME output files for the Gemma model and the Llama model so the comparison is fair.

Usage (from backend/finetune_v3):
  python build_mixed_data.py                       # 3,000 general examples, strategy data repeated 2 times
  python build_mixed_data.py --general-n 2000 --strategy-repeat 3
  python build_mixed_data.py --drop-sources ESConv MentalChat16K     (only if general_train.jsonl has source labels)
If the general file has source labels the cap is spread evenly over the sources; otherwise a random sample is taken.
Writes mixed_train.jsonl, mixed_dev.jsonl and prints the exact composition (report it in the paper).
"""
import argparse, json, random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--general-train", default="general_train.jsonl"); ap.add_argument("--general-dev", default="general_dev.jsonl")
ap.add_argument("--strategy-train", default="strategy_train.jsonl"); ap.add_argument("--strategy-dev", default="strategy_dev.jsonl")
ap.add_argument("--general-n", type=int, default=3000, help="number of general training examples to keep (0 = all)")
ap.add_argument("--strategy-repeat", type=int, default=2)
ap.add_argument("--general-dev-n", type=int, default=300)
ap.add_argument("--drop-sources", nargs="*", default=[], help="source names to remove (needs source labels)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rnd = random.Random(a.seed)

gen_tr, gen_dev = read(a.general_train), read(a.general_dev)
str_tr, str_dev = read(a.strategy_train), read(a.strategy_dev)
if a.drop_sources:
    drop = {d.lower() for d in a.drop_sources}
    gen_tr = [r for r in gen_tr if str(r.get("source", "")).lower() not in drop]
    gen_dev = [r for r in gen_dev if str(r.get("source", "")).lower() not in drop]
    print("After dropping", a.drop_sources, "->", len(gen_tr), "general training examples")

def cap(rows, n):
    if not n or n >= len(rows): return list(rows)
    groups = defaultdict(list)
    for r in rows: groups[r.get("source") or "_"].append(r)
    for g in groups.values(): rnd.shuffle(g)
    if len(groups) == 1:
        return next(iter(groups.values()))[:n]
    per = n // len(groups); chosen = []
    for g in groups.values(): chosen += g[:per]
    left = [r for g in groups.values() for r in g[per:]]; rnd.shuffle(left)
    return chosen + left[: n - len(chosen)]

gen_tr = cap(gen_tr, a.general_n)
rnd.shuffle(gen_dev); gen_dev = gen_dev[: a.general_dev_n]
train = gen_tr + str_tr * a.strategy_repeat
rnd.shuffle(train)
dev = gen_dev + str_dev
rnd.shuffle(dev)
for name, rows in (("mixed_train.jsonl", train), ("mixed_dev.jsonl", dev)):
    with open(name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"mixed_train.jsonl: {len(train)} = {len(gen_tr)} general + {len(str_tr)} strategy x {a.strategy_repeat} "
      f"({len(str_tr) * a.strategy_repeat / len(train):.1%} strategy)")
print(f"mixed_dev.jsonl: {len(dev)} = {len(gen_dev)} general + {len(str_dev)} strategy")
src = Counter(r.get("source") for r in gen_tr)
if any(src): print("General examples per source:", dict(src))
steps = -(-len(train) // 16)
print(f"Optimizer steps per epoch (batch 4 x accumulation 4): {steps}")