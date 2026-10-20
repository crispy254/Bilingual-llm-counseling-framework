"""Step 3: builds the three tables from the supervisor document.
Table 1 (strategy classification): table1_classification.csv
Table 2 (efficiency):               table2_efficiency.csv
Table 3 (pass/fail quality checks): table3_quality.csv
Extras: table1_per_strategy.csv, table1_by_language.csv, fig_*.png, confusion_*.png
Usage: python analyze.py"""
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import os, sys
import exp_config as C
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finetune_v3")))
from strategy_checks import rule_check
PFX = C.RUN_NAME + "_"

STRATS = ["validate_listen", "validate_ask", "validate_suggest", "validate_normalize"]
SHORT = {"validate_listen": "Listen", "validate_ask": "Ask",
         "validate_suggest": "Suggest", "validate_normalize": "Normalize"}

df = pd.read_csv(C.JUDGED_CSV).fillna({"context": ""})
tobool = lambda s: s.astype(str).str.lower().isin(["true", "1", "yes"])
df["made_up_facts"] = tobool(df["made_up_facts"]); df["unsafe"] = tobool(df["unsafe"])

# rule-based checks
hangul = lambda t: len(re.findall(r"[\uac00-\ud7a3]", str(t))) / max(1, len(re.findall(r"\S", str(t))))
df["hangul"] = df["response"].apply(hangul)
df["lang_ok"] = np.where(df["language"] == "ko", df["hangul"] > 0.3, df["hangul"] < 0.1)

LEAK = re.compile(r"^\s*(okay|ok,|alright|hmm|let me|first,|so,? the (user|student)|the (user|student)|i need to|i should)\b", re.I)
df["leak"] = df["response"].astype(str).apply(lambda t: bool(LEAK.match(t)))

def copied(resp, ctx, n=4):  # share of reply 4-grams copied from retrieved text
    w = str(resp).lower().split(); c = str(ctx).lower().split()
    if not str(ctx).strip() or len(w) < n: return 0.0
    cg = {tuple(c[i:i+n]) for i in range(len(c) - n + 1)}
    g = [tuple(w[i:i+n]) for i in range(len(w) - n + 1)]
    return sum(x in cg for x in g) / max(1, len(g))
df["copy_share"] = [copied(r, c) for r, c in zip(df["response"], df["context"])]
df["no_copy"] = df["copy_share"] < 0.5

df["rule_pass"] = [rule_check(s, str(r), l) for s, r, l in zip(df["strategy"], df["response"], df["language"])]

def order(t):  # Nova first
    t = t.copy(); t["_o"] = (t["model"] != C.NOVA_MODEL); return t.sort_values(["_o", "model"]).drop(columns="_o")

# ---- Table 1
t1, per = [], []
for m, g in df.groupby("model"):
    p, r, f, s = precision_recall_fscore_support(g["strategy"], g["strategy_used"], labels=STRATS, zero_division=0)
    t1.append(dict(model=m, n=len(g), accuracy=accuracy_score(g["strategy"], g["strategy_used"]),
                   rule_pass=g["rule_pass"].mean(),
                   precision=p.mean(), recall=r.mean(), f1=f.mean(),
                   **{f"f1_{SHORT[k]}": f[i] for i, k in enumerate(STRATS)}))
    for i, k in enumerate(STRATS):
        per.append(dict(model=m, strategy=k, precision=p[i], recall=r[i], f1=f[i], support=int(s[i])))
t1 = order(pd.DataFrame(t1)); t1.round(3).to_csv(PFX + "table1_classification.csv", index=False)
pd.DataFrame(per).round(3).to_csv(PFX + "table1_per_strategy.csv", index=False)

lang = (df.assign(ok=df["strategy"] == df["strategy_used"])
          .pivot_table(index="model", columns="language", values="ok", aggfunc="mean"))
lang.round(3).to_csv(PFX + "table1_by_language.csv")

# ---- Table 2
agg = dict(response_time_s=("latency_s", "mean"), tokens_per_s=("tokens_per_s", "mean"),
           vram_gb=("vram_gb", "max"), size=("model_size", "first"))
if "attempts" in df.columns: agg["mean_attempts"] = ("attempts", "mean")
t2 = order(df.groupby("model").agg(**agg).reset_index())
t2.round(2).to_csv(PFX + "table2_efficiency.csv", index=False)

# ---- Table 3 (% passing)
t3 = order(df.groupby("model").agg(language_correct=("lang_ok", "mean"),
        no_made_up_facts=("made_up_facts", lambda x: 1 - x.mean()),
        safe=("unsafe", lambda x: 1 - x.mean()), no_copying=("no_copy", "mean"),
        no_reasoning_leak=("leak", lambda x: 1 - x.mean())).reset_index())
t3.iloc[:, 1:] = (t3.iloc[:, 1:] * 100).round(1)
t3.to_csv(PFX + "table3_quality.csv", index=False)

for name, t in [("TABLE 1 - Strategy classification", t1.round(3)), ("TABLE 2 - Efficiency", t2.round(2)),
                ("TABLE 3 - Quality checks (% passing)", t3)]:
    print(f"\n{name}\n{t.to_string(index=False)}")
print("\nAccuracy by language\n", lang.round(3).to_string())

# ---- figures
ax = t1.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot.bar(figsize=(9, 4.5), ylim=(0, 1), rot=20)
ax.set_title("Strategy classification by model"); plt.tight_layout(); plt.savefig(PFX + "fig_classification.png", dpi=200); plt.close()
pv = pd.DataFrame(per).pivot(index="model", columns="strategy", values="f1")[STRATS].rename(columns=SHORT)
ax = pv.plot.bar(figsize=(9, 4.5), ylim=(0, 1), rot=20)
ax.set_title("F1 per strategy by model"); plt.tight_layout(); plt.savefig(PFX + "fig_f1_per_strategy.png", dpi=200); plt.close()
for m, g in df.groupby("model"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(g["strategy"], g["strategy_used"], labels=STRATS),
                           display_labels=[SHORT[s] for s in STRATS]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{m}\nrows = requested, columns = followed"); plt.tight_layout()
    plt.savefig(PFX + "confusion_" + re.sub(r"[^\w]+", "_", m) + ".png", dpi=200); plt.close()