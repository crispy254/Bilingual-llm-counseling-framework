"""Paired significance tests on run2_judged.csv (same test messages for every model).
Exact McNemar test: counts messages where exactly one of two models followed the requested strategy.
Usage (from the Experiment folder):  python paired_tests.py"""
import pandas as pd
from scipy.stats import binomtest
import exp_config as C

df = pd.read_csv(C.JUDGED_CSV)
df["ok"] = df["strategy"] == df["strategy_used"]
piv = df.pivot_table(index="id", columns="model", values="ok", aggfunc="first")
PAIRS = [("nova-gemma-v3", "gemma-general"),            # same base and recipe, strategy data vs general data
         ("gemma-general", "gemma3:4b"),                  # general data on the Gemma base
         ("gemma-general", "nova-counseling-v2"),         # same general data, Gemma vs Llama base
         ("nova-gemma-v3", "gemma3:4b"), ("nova-gemma-v3+guard", "nova-gemma-v3"),
         ("nova-gemma-v3+guard", "gemma3:4b+guard"), ("nova-gemma-v3", "nova-counseling-v2"),
         ("nova-gemma-v3", "llama3.2:3b"), ("gemma3:4b+guard", "gemma3:4b"),
         ("gemma3:4b", "llama3.2:3b"), ("llama3.2:3b", "nova-counseling-v2")]
rows = []
for a, b in PAIRS:
    if a not in piv or b not in piv: continue
    d = piv[[a, b]].dropna().astype(bool)
    a_only = int((d[a] & ~d[b]).sum()); b_only = int((~d[a] & d[b]).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    rows.append(dict(model_a=a, model_b=b, n=len(d), acc_a=round(d[a].mean(), 3), acc_b=round(d[b].mean(), 3),
                     only_a_correct=a_only, only_b_correct=b_only, p_value=round(p, 4)))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(C.RUN_NAME + "_paired_tests.csv", index=False)