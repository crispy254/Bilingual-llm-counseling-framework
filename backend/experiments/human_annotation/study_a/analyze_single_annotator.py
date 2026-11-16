"""Analyse ONE author annotator (not independent raters; no inter-rater statistics).
python analyze_single_annotator.py --en EN_file.xlsx --ko KO_file.xlsx --key study_a_key_small.csv \
       --bigkey study_a_key.csv --llm LLM_annotator.xlsx --out results_single
"""
import argparse, os, numpy as np, pandas as pd
from openpyxl import load_workbook
from scipy.stats import binomtest

LAB = ["LISTEN", "ASK", "SUGGEST", "NORMALIZE", "NONE"]


def load(f):
    ws = load_workbook(f, data_only=True)["Ratings"]; h = [c.value for c in ws[1]]
    d = pd.DataFrame([dict(zip(h, r)) for r in ws.iter_rows(min_row=2, values_only=True)])
    return d.dropna(subset=["item_id"])


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pe = sum((a == l).mean() * (b == l).mean() for l in set(a) | set(b))
    return ((a == b).mean() - pe) / (1 - pe)


def wil(k, n):
    z = 1.959964; ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** .5) / den
    lo, hi = c - h, c + h; return f"{k/n*100:.1f}% ({lo*100:.1f}-{hi*100:.1f})"


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    for x in ["en", "ko", "key", "bigkey", "llm", "out"]: ap.add_argument("--" + x, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    out = []
    def log(s=""): print(s); out.append(str(s))
    K = pd.read_csv(a.key); K["dup_of"] = K.dup_of.fillna("")
    H = pd.concat([load(a.en), load(a.ko)])[["item_id", "strategy", "mixed", "has_question", "has_advice", "has_normalizing", "safety", "language_ok"]]
    H = H.rename(columns={"strategy": "human"})
    D = K.merge(H, on="item_id")
    log(f"Items annotated: {len(D)} ({(D.dup_of == '').sum()} unique + {(D.dup_of != '').sum()} repeats)")
    # test-retest
    d = D[D.dup_of != ""].merge(D[["item_id", "human"]], left_on="dup_of", right_on="item_id", suffixes=("", "_orig"))
    log(f"Test-retest on repeated items: {(d.human == d.human_orig).sum()}/{len(d)} identical")
    U = D[D.dup_of == ""].copy()
    U["h_ok"] = U.human == U.intended; U["j_ok"] = U.judge == U.intended

    log("\n== Author-annotator adherence (label == intended), Wilson 95% CI; judge in last column ==")
    for lang in ["en", "ko", "all"]:
        for st in ["ours", "base"]:
            g = U[(U.stratum == st) & ((U.language == lang) | (lang == "all"))]
            log(f"{lang:>3} {st:<5} n={len(g):>3}  annotator {wil(int(g.h_ok.sum()), len(g))}   Phi-4 judge {g.j_ok.mean()*100:.1f}%")

    log("\n== Paired exact McNemar, ours vs base (same messages) ==")
    res = []
    for lang in ["en", "ko", "all"]:
        g = U if lang == "all" else U[U.language == lang]
        o = g[g.stratum == "ours"].set_index(["language", "message_id"]).h_ok
        b = g[g.stratum == "base"].set_index(["language", "message_id"]).h_ok.reindex(o.index)
        x, y = int((o & ~b).sum()), int((~o & b).sum())
        res.append((lang, x, y, 1.0 if x + y == 0 else binomtest(min(x, y), x + y, .5).pvalue))
    adj = holm(np.array([r[3] for r in res]))
    for r, p in zip(res, adj):
        log(f"  [{r[0]}] only-ours-correct {r[1]}, only-base-correct {r[2]}, p={r[3]:.4f}, Holm p={p:.4f}")

    log("\n== Annotator vs Phi-4 judge ==")
    for lang in ["en", "ko", "all"]:
        g = U if lang == "all" else U[U.language == lang]
        log(f"  [{lang}] agreement {wil(int((g.human == g.judge).sum()), len(g))}, Cohen kappa {kappa(g.human, g.judge):.3f}")
    log(pd.crosstab(U.human, U.judge, rownames=["annotator"], colnames=["judge"]).to_string())

    # AI annotator on the same items
    B = pd.read_csv(a.bigkey); B = B[(B.set == "main") & (B.dup_of.fillna("") == "")][["language", "message_id", "stratum", "item_id"]].rename(columns={"item_id": "big_id"})
    L = pd.concat([pd.read_excel(a.llm, sheet_name=s) for s in ["EN", "KO"]])[["item_id", "strategy"]].rename(columns={"item_id": "big_id", "strategy": "ai"})
    M = U.merge(B, on=["language", "message_id", "stratum"]).merge(L, on="big_id")
    log(f"\n== Claude AI annotator (separate automated annotator) on the same {len(M)} items ==")
    log(f"  AI vs author annotator: agreement {wil(int((M.ai == M.human).sum()), len(M))}, kappa {kappa(M.ai, M.human):.3f}")
    log(f"  AI vs Phi-4 judge:      agreement {wil(int((M.ai == M.judge).sum()), len(M))}, kappa {kappa(M.ai, M.judge):.3f}")

    log("\n== Flags (author annotator) ==")
    log(f"  mixed=Y {int((U.mixed=='Y').sum())}/{len(U)}; safety not OK {int((U.safety!='OK').sum())}; language_ok=N {int((U.language_ok=='N').sum())}")
    mm = U[(U.stratum == "ours") & (~U.j_ok)]
    log(f"\nOur-system items where Phi-4 judge != intended: {len(mm)}; annotator sides with intended in {int(mm.h_ok.sum())}")
    U.to_csv(os.path.join(a.out, "single_annotator_items.csv"), index=False, encoding="utf-8-sig")
    open(os.path.join(a.out, "single_annotator_report.txt"), "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
