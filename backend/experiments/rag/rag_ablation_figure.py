#!/usr/bin/env python3
"""
rag_ablation_figure.py -- one figure answering the RAG-evaluation reviewer comment.

Reads  <csv>  (rag_ablation_judged_v2.csv; never modified)
Writes <output>/figures/fig_rag_ablation.png (300 dpi) + .pdf, figure_caption.txt
Prints the numbers plotted, so every value can be checked.

Panels (all computed from the CSV, nothing hard-coded)
 (a) Retrieval disabled vs enabled: judged response relevance, paired by message ID
 (b) Passage relevance: share of retrievals judged relevant / partial / poor (missing or
     misleading context), by language
 (c) Faithfulness label of rag_on responses by judged retrieval relevance
 (d) Effect of retrieval (rag_on - rag_off) by judged retrieval relevance

Notes: all scores come from an automated LLM judge (Phi-4 14B), not human raters.
Retrieval relevance is one score per retrieved set of passages. 'no_factual_claims' means
nothing checkable was found, not that the response is grounded.
Usage: python3 rag_ablation_figure.py [--csv rag_ablation_judged_v2.csv] [--output analysis_outputs]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})
C_OFF, C_ON = "#56B4E9", "#D55E00"
C_EN, C_KO = "#0072B2", "#E69F00"
C_REL, C_PART, C_POOR = "#009E73", "#E69F00", "#D55E00"
C_SUP, C_NOCL = "#009E73", "#BBBBBB"


def mean_ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def title(ax, t):
    ax.set_title(t, loc="left", fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="rag_ablation_judged_v2.csv")
    ap.add_argument("--output", default="analysis_outputs")
    a = ap.parse_args()
    if not Path(a.csv).exists():
        sys.exit(f"ERROR: {a.csv} not found (use --csv /path/to/rag_ablation_judged_v2.csv)")
    outdir = Path(a.output) / "figures"; outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.csv, encoding="utf-8-sig")
    on = df[df.condition == "rag_on"].set_index("id").sort_index()
    off = df[df.condition == "rag_off"].set_index("id").sort_index()
    if set(on.index) != set(off.index):
        sys.exit("ERROR: rag_on and rag_off IDs do not match.")
    on["rr_on"] = on.response_relevance
    on["rr_off"] = off.loc[on.index].response_relevance
    on["d"] = on.rr_on - on.rr_off
    ret = on.retrieval_relevance.astype(int)
    on["band"] = pd.cut(ret, [0, 2, 3, 5], labels=["poor\n(1–2)", "partial\n(3)", "relevant\n(4–5)"])
    bands = list(on.band.cat.categories)

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.2))
    fig.subplots_adjust(hspace=0.55, wspace=0.34)

    # (a) retrieval disabled vs enabled ------------------------------------------------
    A = ax[0, 0]
    print("\n(a) Response relevance, mean [95% CI]; paired diff on-off")
    for j, (lab, L) in enumerate([("All", None), ("English", "en"), ("Korean", "ko")]):
        g = on if L is None else on[on.language == L]
        for k, (col, nm, c) in enumerate([("rr_off", "Retrieval off", C_OFF), ("rr_on", "Retrieval on", C_ON)]):
            m, h = mean_ci(g[col])
            A.bar(j + (k - .5) * .38, m, .36, yerr=h, color=c, capsize=2, error_kw=dict(lw=.8),
                  label=nm if j == 0 else None)
            print(f"   {lab:8s}{nm:14s} n={len(g)} mean={m:.3f} CI=[{m-h:.3f}, {m+h:.3f}]")
        d = g.d
        p = stats.wilcoxon(d).pvalue
        dm, dh = mean_ci(d)
        print(f"   {lab:8s}diff={dm:+.3f} [{dm-dh:+.3f}, {dm+dh:+.3f}]  Wilcoxon p={p:.2g}")
        A.text(j, 4.72, f"$\\Delta$={dm + np.sign(dm) * 1e-9:+.2f}", ha="center", fontsize=7)
    A.set_xticks(range(3)); A.set_xticklabels(["All\n(n=144)", "English\n(n=72)", "Korean\n(n=72)"])
    A.set_ylim(1, 5); A.set_ylabel("Mean response relevance (1–5)")
    title(A, "(a) Retrieval disabled vs enabled")
    A.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=6.5)

    # (b) passage relevance by language (stacked horizontal) ---------------------------
    B = ax[0, 1]
    print("\n(b) Retrieval relevance bands (share of retrievals)")
    rows = [("All", on), ("English", on[on.language == "en"]), ("Korean", on[on.language == "ko"])]
    for i, (lab, g) in enumerate(rows):
        cnt = g.band.value_counts().reindex(bands)
        left = 0
        for bn, col, nm in zip(bands, [C_POOR, C_PART, C_REL], ["Poor (1–2)", "Partial (3)", "Relevant (4–5)"]):
            pct = 100 * cnt[bn] / len(g)
            B.barh(2 - i, pct, left=left, color=col, height=.6, label=nm if i == 0 else None)
            if pct >= 6:
                B.text(left + pct / 2, 2 - i, f"{pct:.0f}%", ha="center", va="center", fontsize=7, color="white")
            left += pct
        print(f"   {lab:8s} n={len(g)}  poor={cnt.iloc[0]}  partial={cnt.iloc[1]}  relevant={cnt.iloc[2]}")
    B.set_yticks([2, 1, 0]); B.set_yticklabels(["All\n(n=144)", "English\n(n=72)", "Korean\n(n=72)"])
    B.set_xlim(0, 100); B.set_xlabel("Share of retrievals (%)"); B.grid(axis="y", visible=False)
    title(B, "(b) Passage relevance (missing/misleading context)")
    B.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=6.5)

    # (c) faithfulness by retrieval band ------------------------------------------------
    C = ax[1, 0]
    print("\n(c) Faithfulness label by retrieval band (counts)")
    labs = ["supported", "partially_supported", "unsupported", "no_factual_claims"]
    ct = pd.crosstab(on.band, on.faithfulness).reindex(index=bands, columns=labs, fill_value=0)
    print(ct.to_string())
    x = np.arange(len(bands))
    C.bar(x, ct["no_factual_claims"], .6, color=C_NOCL, label="No checkable claims")
    C.bar(x, ct["supported"], .6, bottom=ct["no_factual_claims"], color=C_SUP, label="Supported")
    for i, bn in enumerate(bands):
        tot = ct.loc[bn].sum()
        C.text(i, tot + 2, f"{ct.loc[bn,'supported']}/{tot}\nwith claims", ha="center", fontsize=6.8)
    C.set_xticks(x); C.set_xticklabels([f"{b}\n(n={ct.loc[b].sum()})" for b in bands])
    C.set_ylim(0, ct.sum(axis=1).max() * 1.3)
    C.set_xlabel("Judged retrieval relevance"); C.set_ylabel("Retrieval-on responses")
    title(C, "(c) Faithfulness by passage relevance")
    C.legend(loc="upper left", frameon=False, fontsize=6.5)

    # (d) effect of retrieval by retrieval band -----------------------------------------
    D = ax[1, 1]
    print("\n(d) Paired difference (on-off) by retrieval band, mean [95% CI]")
    for k, (L, c, nm, mk) in enumerate([("en", C_EN, "English", "o"), ("ko", C_KO, "Korean", "s")]):
        for i, bn in enumerate(bands):
            v = on[(on.language == L) & (on.band == bn)].d
            if len(v) < 2:
                continue
            m, h = mean_ci(v)
            D.errorbar(i + (k - .5) * .2, m, yerr=h, fmt=mk, ms=4, capsize=2, lw=1, color=c,
                       label=nm if i == 0 else None)
            print(f"   {nm:8s}{bn.replace(chr(10),' '):14s} n={len(v):3d} mean={m:+.3f} [{m-h:+.3f}, {m+h:+.3f}]")
    D.axhline(0, color="k", lw=.7)
    D.set_xticks(range(3)); D.set_xticklabels([b for b in bands]); D.set_xlim(-.5, 2.5)
    D.set_xlabel("Judged retrieval relevance"); D.set_ylabel("Mean paired difference\n(on $-$ off)")
    title(D, "(d) Retrieval effect by passage relevance")
    D.legend(loc="lower right", frameon=False)

    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(outdir / f"fig_rag_ablation.{ext}", **kw)
    (outdir / "figure_caption.txt").write_text(
        "RAG ablation (144 student messages, 72 English / 72 Korean; same model with retrieval on and off). "
        "(a) Judged response relevance (1-5), mean and 95% CI; Delta = paired difference (on - off) per group. "
        "(b) Share of retrievals judged relevant, partially relevant or poorly relevant (missing or misleading "
        "context) by the automated judge, one score per retrieved set. (c) Faithfulness label of retrieval-on "
        "responses by retrieval relevance; 'no checkable claims' is not evidence of grounding. (d) Retrieval "
        "effect by retrieval relevance, mean paired difference and 95% CI. All scores are from an automated "
        "LLM judge (Phi-4 14B) without human validation.", encoding="utf-8")
    print(f"\nSaved: {outdir/'fig_rag_ablation.png'} (+ .pdf, figure_caption.txt)")


if __name__ == "__main__":
    main()
