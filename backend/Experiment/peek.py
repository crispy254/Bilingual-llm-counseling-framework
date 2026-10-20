"""Prints a few replies of one model so you can sanity-check them.
Usage (from the Experiment folder):  python peek.py nova-gemma-v3 8     (model name, how many)"""
import sys, pandas as pd
import exp_config as C
model = sys.argv[1] if len(sys.argv) > 1 else "nova-gemma-v3"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
d = pd.read_csv(C.GENERATIONS_CSV)
d = d[d["model"] == model].sample(n=min(n, (d["model"] == model).sum()), random_state=1)
if d.empty: sys.exit(f"No rows for {model} in {C.GENERATIONS_CSV}")
for _, r in d.iterrows():
    print(f"\n[{r['language']}] requested: {r['strategy']}\nSTUDENT: {r['message']}\nREPLY:   {r['response']}")
print("\nLook for: normal 2-4 sentence replies; no '<start_of_turn>' or other odd tokens; no repetition; the right language.")