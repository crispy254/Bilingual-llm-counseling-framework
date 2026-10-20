"""Human check of the judge. Two steps (run from the Experiment folder).

1) python human_check.py export  [--per-model 10]
   Writes human_check_sheet.csv (blind: no model name, no requested strategy) and human_check_key.csv.
   Give the sheet to 1-2 people. For each reply they write in the human_strategy column ONE of:
     listen    = validates feelings only; no advice and no question
     ask       = validates, then asks a follow-up question
     suggest   = validates, then gives a concrete suggestion or step
     normalize = validates, then says many people/students feel this way
   Save each person's copy as e.g. human_check_rater1.csv / human_check_rater2.csv.

2) python human_check.py score human_check_rater1.csv [human_check_rater2.csv]
   Reports agreement between humans and the judge (accuracy and Cohen's kappa), between the two raters,
   and per-model strategy adherence according to the humans."""
import sys, argparse, pandas as pd
from sklearn.metrics import cohen_kappa_score
import exp_config as C

NAMES = {"listen": "validate_listen", "ask": "validate_ask", "suggest": "validate_suggest",
         "normalize": "validate_normalize"}
def clean(x):
    x = str(x).strip().lower()
    return NAMES.get(x.replace("validate_", ""), None)

ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["export", "score"]); ap.add_argument("files", nargs="*")
ap.add_argument("--per-model", type=int, default=10); a = ap.parse_args()
J = pd.read_csv(C.JUDGED_CSV)

if a.cmd == "export":
    parts = [g.sample(n=min(a.per_model, len(g)), random_state=7) for _, g in J.groupby("model")]
    s = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)
    s["item_id"] = range(1, len(s) + 1)
    s[["item_id", "language", "message", "response"]].rename(columns={"response": "reply"}) \
        .assign(human_strategy="").to_csv("human_check_sheet.csv", index=False, encoding="utf-8-sig")
    s[["item_id", "model", "id", "strategy", "strategy_used"]].to_csv("human_check_key.csv", index=False)
    print(f"Wrote human_check_sheet.csv ({len(s)} replies, blind) and human_check_key.csv (keep the key private).")
else:
    key = pd.read_csv("human_check_key.csv")
    raters = []
    for f in a.files:
        h = pd.read_csv(f)[["item_id", "human_strategy"]]
        h["human_strategy"] = h["human_strategy"].map(clean)
        raters.append(h.rename(columns={"human_strategy": f}))
    d = key
    for r in raters: d = d.merge(r, on="item_id")
    cols = [r.columns[1] for r in raters]
    for c in cols:
        x = d.dropna(subset=[c])
        print(f"{c}: n={len(x)}  agreement with judge = {(x[c] == x['strategy_used']).mean():.3f}  "
              f"kappa = {cohen_kappa_score(x[c], x['strategy_used']):.3f}")
    if len(cols) >= 2:
        x = d.dropna(subset=cols[:2])
        print(f"rater vs rater: n={len(x)}  agreement = {(x[cols[0]] == x[cols[1]]).mean():.3f}  "
              f"kappa = {cohen_kappa_score(x[cols[0]], x[cols[1]]):.3f}")
    d["human"] = d[cols[0]]
    t = d.dropna(subset=["human"]).groupby("model").apply(lambda g: pd.Series({
        "n": len(g), "human_accuracy": (g["human"] == g["strategy"]).mean(),
        "judge_accuracy_same_rows": (g["strategy_used"] == g["strategy"]).mean()}), include_groups=False)
    print("\nPer model (rater 1):\n", t.round(3).to_string())
    t.round(3).to_csv(C.RUN_NAME + "_human_check_summary.csv")