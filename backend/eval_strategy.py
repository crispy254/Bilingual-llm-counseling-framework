"""
Per-strategy precision / recall / F1 for Nova's LinUCB strategy selector.

Classes: validate_listen, validate_ask, validate_suggest, validate_normalize

INPUTS
  1) rater_labels.csv  - human "best strategy" labels per context.
        emotion,stress_type,true_strategy
        anxious,academic,validate_ask
        anxious,academic,validate_suggest     <- a 2nd rater, same context
     Use 2-3 raters; every row counts as one sample. Cover all 24
     contexts (6 emotions x 4 stress types). Report rater agreement
     (Cohen's kappa) alongside the results.
  2) history.csv (optional) - exported StrategyLog rows with rewards:
        emotion,stress_type,strategy,reward
     Replayed via warm_start() so the bandit has learned something.
     Without it the bandit is untrained and picks the same class
     everywhere (useful only as a baseline).

USAGE
  python eval_strategy_prf1.py rater_labels.csv [history.csv]
Run from the folder containing strategy_selector.py.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score,
                             cohen_kappa_score, ConfusionMatrixDisplay)

import strategy_selector as ss


def exploit_only_choice(emotion, stress_type):
    """Greedy choice = highest predicted reward, NO exploration bonus,
    so results are deterministic. Ties -> first strategy in STRATEGIES."""
    old = ss.ALPHA
    ss.ALPHA = 0.0
    try:
        chosen, _ = ss.select_strategy(emotion, stress_type)
    finally:
        ss.ALPHA = old
    return chosen


def main(labels_path, history_path=None):
    if history_path:
        h = pd.read_csv(history_path)
        n = ss.warm_start(list(h[["emotion", "stress_type", "strategy", "reward"]]
                               .itertuples(index=False, name=None)))
        print(f"Warm-started bandit with {n} feedback rows.\n")
    else:
        print("No history given: evaluating the UNTRAINED bandit (baseline).\n")

    df = pd.read_csv(labels_path)
    df["pred"] = [exploit_only_choice(e, s)
                  for e, s in zip(df["emotion"], df["stress_type"])]

    labels = ss.STRATEGIES
    y_true, y_pred = df["true_strategy"], df["pred"]

    print(classification_report(y_true, y_pred, labels=labels,
                                zero_division=0, digits=3))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

    # Majority-class baseline for context
    base = y_true.value_counts(normalize=True).max()
    print(f"Majority-class baseline: {base:.3f}")

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    table = pd.DataFrame({"strategy": labels, "precision": p,
                          "recall": r, "f1": f, "support": s})
    table.loc[len(table)] = ["macro avg", p.mean(), r.mean(), f.mean(), s.sum()]
    print("\n", table.round(3).to_string(index=False))
    table.to_csv("strategy_prf1.csv", index=False)
    df.to_csv("predictions.csv", index=False)

    # Inter-rater agreement (if >=2 raters labelled the same contexts)
    groups = df.groupby(["emotion", "stress_type"])["true_strategy"].apply(list)
    pairs = [(g[0], g[1]) for g in groups if len(g) >= 2]
    if pairs:
        a, b = zip(*pairs)
        print(f"\nInter-rater Cohen's kappa (first two raters): "
              f"{cohen_kappa_score(a, b):.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title("Rows = human label, columns = bandit choice")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)

    ax = table.iloc[:-1].set_index("strategy")[["precision", "recall", "f1"]] \
        .plot.bar(figsize=(8, 4), ylim=(0, 1), rot=20)
    ax.set_title("Per-strategy precision / recall / F1")
    plt.tight_layout()
    plt.savefig("strategy_prf1.png", dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)