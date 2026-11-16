"""Analyse completed Study A rater workbooks.

Usage: python analyze_study_a.py --packets <folder with completed StudyA_main_*.xlsx> \
                                 --key ../private/study_a_key.csv --out ../results
Only the MAIN files are analysed (pilot is for calibration and excluded).
Headline = human-majority adherence (label == intended) for ours vs base, paired by message.
"""
import argparse, glob, os, re, itertools
import numpy as np, pandas as pd
from openpyxl import load_workbook
from scipy.stats import binomtest

LABELS = ["LISTEN", "ASK", "SUGGEST", "NORMALIZE", "NONE"]
rng = np.random.default_rng(2026)


def load_ratings(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "StudyA_main_*_R*.xlsx"))):
        rater = re.search(r"(EN|KO)_R\d", f).group(0)
        ws = load_workbook(f, data_only=True)["Ratings"]
        hdr = [c.value for c in ws[1]]
        for r in ws.iter_rows(min_row=2, values_only=True):
            d = dict(zip(hdr, r)); d["rater"] = rater
            rows.append(d)
    df = pd.DataFrame(rows)
    missing = df.strategy.isna().sum()
    if missing:
        print(f"WARNING: {missing} rows without a strategy label were dropped")
    return df.dropna(subset=["strategy"])


# ---------- agreement statistics ----------
def cohen(a, b, labels=LABELS):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def fleiss(M, labels=LABELS):
    """M: items x raters array of labels (complete)."""
    n = M.shape[1]
    counts = np.array([[np.sum(row == l) for l in labels] for row in M])
    Pi = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    pj = counts.sum(0) / counts.sum()
    Pe = np.sum(pj ** 2)
    return (Pi.mean() - Pe) / (1 - Pe)


def gwet_ac1(M, labels=LABELS):
    n = M.shape[1]; q = len(labels)
    counts = np.array([[np.sum(row == l) for l in labels] for row in M])
    Pa = ((np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))).mean()
    pi = counts.sum(0) / counts.sum()
    Pe = np.sum(pi * (1 - pi)) / (q - 1)
    return (Pa - Pe) / (1 - Pe)


def krippendorff_nominal(M, labels=LABELS):
    n_items, m = M.shape
    idx = {l: i for i, l in enumerate(labels)}
    coin = np.zeros((len(labels), len(labels)))
    for row in M:
        for a, b in itertools.permutations(range(m), 2):
            coin[idx[row[a]], idx[row[b]]] += 1 / (m - 1)
    nc = coin.sum(1); N = nc.sum()
    Do = (coin.sum() - np.trace(coin))
    De = (N ** 2 - np.sum(nc ** 2)) / (N - 1)
    return 1 - Do / De if De else np.nan


def boot_ci(stat, groups, n=2000):
    """cluster bootstrap over message_id groups: groups = list of arrays of per-item values."""
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[i] for i in pick])))
    return np.percentile(vals, [2.5, 97.5])


def majority(labels):
    c = pd.Series(labels).value_counts()
    return c.index[0] if (c.iloc[0] >= 2) else "NO_MAJORITY"


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else binomtest(min(b, c), n, 0.5).pvalue


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj


def integrity_check(R):
    """Stop if raters look like copies of each other (independent raters never agree ~100% on every column)."""
    cols = ["strategy", "mixed", "has_question", "has_advice", "has_normalizing"]
    bad = []
    for lang in ["EN", "KO"]:
        rs = sorted(r for r in R.rater.unique() if r.startswith(lang))
        for r1, r2 in itertools.combinations(rs, 2):
            a = R[R.rater == r1].set_index("item_id")[cols]; b = R[R.rater == r2].set_index("item_id")[cols]
            i = a.index.intersection(b.index)
            same = (a.loc[i].values == b.loc[i].values).all(axis=1).mean()
            if same > 0.95:
                bad.append(f"{r1} and {r2} are identical on {same:.0%} of items")
    for r, g in R.groupby("rater"):
        for c in ["safety", "language_ok"]:
            if g[c].nunique() == 1:
                print(f"NOTE: {r} used a single value for '{c}' on every item; that column carries no information.")
    if bad:
        raise SystemExit("INTEGRITY CHECK FAILED - these files cannot be independent ratings:\n  " + "\n  ".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True); ap.add_argument("--key", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = load_ratings(a.packets)
    integrity_check(R)
    K = pd.read_csv(a.key); K["dup_of"] = K.dup_of.fillna(""); K = K[K.set == "main"]
    out = []
    def log(s=""):
        print(s); out.append(str(s))

    wide = R.pivot(index="item_id", columns="rater", values="strategy")
    D = K.set_index("item_id").join(wide, how="inner")
    summary = {}
    for lang in ["en", "ko"]:
        raters = [c for c in wide.columns if c.startswith(lang.upper())]
        d = D[D.language == lang].dropna(subset=raters).copy()
        d["human"] = d[raters].apply(lambda r: majority(r.values), axis=1)
        log(f"\n===== {lang.upper()}  ({len(d)} items, raters {raters}) =====")
        M = d[raters].values
        log(f"Fleiss kappa {fleiss(M):.3f} | Krippendorff alpha {krippendorff_nominal(M):.3f} | Gwet AC1 {gwet_ac1(M):.3f} | "
            f"mean exact agreement {np.mean([len(set(r)) == 1 for r in M]):.3f}")
        for r1, r2 in itertools.combinations(raters, 2):
            log(f"  Cohen {r1} vs {r2}: {cohen(d[r1], d[r2]):.3f}")
        nm = (d.human == "NO_MAJORITY").sum()
        log(f"No-majority items: {nm}")
        # test-retest on duplicates
        dup = d[d.dup_of != ""]
        orig = d.loc[dup.dup_of.values]
        for r in raters:
            same = np.mean(dup[r].values == orig[r].values)
            log(f"  test-retest {r}: {same:.2f} agreement on {len(dup)} duplicated items")
        # judge vs human majority (exclude dups and no-majority)
        v = d[(d.dup_of == "") & (d.human != "NO_MAJORITY")]
        acc = (v.judge == v.human).mean()
        k = cohen(v.judge, v.human)
        log(f"Judge vs human-majority: accuracy {acc:.3f}, Cohen kappa {k:.3f} (n={len(v)})")
        log("Confusion (rows=human, cols=judge):")
        log(pd.crosstab(v.human, v.judge).to_string())
        log("Per-strategy (human as truth): " + "; ".join(
            f"{s}: P={((v.judge==s)&(v.human==s)).sum()/max((v.judge==s).sum(),1):.2f} R={((v.judge==s)&(v.human==s)).sum()/max((v.human==s).sum(),1):.2f}"
            for s in LABELS[:4]))
        # features sanity
        summary[lang] = d

    # ---------- human-assessed adherence ----------
    log("\n===== Human-assessed adherence (human majority == intended) =====")
    allv = pd.concat(summary.values())
    allv = allv[(allv.dup_of == "") & (allv.human != "NO_MAJORITY")].copy()
    allv["human_ok"] = allv.human == allv.intended
    allv["judge_ok"] = allv.judge == allv.intended
    rows = []
    for (lang, st), g in list(allv.groupby(["language", "stratum"])) + [(("all", st), g) for st, g in allv.groupby("stratum")]:
        groups = [x.human_ok.values.astype(float) for _, x in g.groupby("message_id")]
        lo, hi = boot_ci(np.mean, groups, 1000)
        rows.append(dict(language=lang, stratum=st, n=len(g), human_acc=g.human_ok.mean(), ci_lo=lo, ci_hi=hi, judge_acc=g.judge_ok.mean()))
    tab = pd.DataFrame(rows).round(3); log(tab.to_string(index=False))

    # paired McNemar: ours vs base on the same message
    log("\nPaired exact McNemar on human labels (message-level):")
    tests = []
    for lang in ["en", "ko", "all"]:
        g = allv if lang == "all" else allv[allv.language == lang]
        o = g[g.stratum == "ours"].set_index(["language", "message_id"]).human_ok
        b = g[g.stratum == "base"].set_index(["language", "message_id"]).human_ok
        idx = o.index.intersection(b.index)
        o, b = o.loc[idx], b.loc[idx]
        only_o, only_b = int((o & ~b).sum()), int((~o & b).sum())
        tests.append((f"ours vs base [{lang}]", only_o, only_b, mcnemar_exact(only_o, only_b)))
    # guarded: ours_guard_only replaces unguarded reply for those messages
    gg = allv.copy()
    og = gg[gg.stratum == "ours_guard_only"].set_index(["language", "message_id"]).human_ok
    ou = gg[gg.stratum == "ours"].set_index(["language", "message_id"]).human_ok
    ou_g = ou.copy(); ou_g.loc[og.index.intersection(ou.index)] = og.loc[og.index.intersection(ou.index)]
    only_g, only_u = int((ou_g & ~ou).sum()), int((~ou_g & ou).sum())
    tests.append(("ours+guard vs ours [all]", only_g, only_u, mcnemar_exact(only_g, only_u)))
    adj = holm(np.array([t[3] for t in tests]))
    for t, p in zip(tests, adj):
        log(f"  {t[0]}: {t[1]} vs {t[2]} discordant, p={t[3]:.4f}, Holm p={p:.4f}")
    gm = ou_g.mean()
    log(f"  Human-assessed accuracy of guarded system: {gm:.3f}")

    # safety / language flags
    log("\nSafety/language flags (any rater):")
    R2 = R.merge(K[["item_id", "stratum", "language"]], on="item_id")
    log(R2.groupby("stratum").agg(unsafe_any=("safety", lambda s: (s != "OK").mean()), lang_bad=("language_ok", lambda s: (s == "N").mean())).round(3).to_string())

    # judge-mismatch review list (settles IDs 1001/1041-type cases)
    mm = allv[(allv.judge != allv.intended) & allv.stratum.isin(["ours", "ours_guard_only"])]
    mm[["language", "message_id", "stratum", "intended", "judge", "human", "reply"]].to_csv(f"{a.out}/judge_mismatch_review.csv", index=False, encoding="utf-8-sig")
    log(f"\nJudge-mismatch cases for our system: {len(mm)}; human agrees with intended strategy in "
        f"{(mm.human == mm.intended).sum()} ({(mm.human == mm.intended).mean():.0%}); these are judge errors, the rest are true failures")
    tab.to_csv(f"{a.out}/human_adherence_table.csv", index=False)
    open(f"{a.out}/study_a_report.txt", "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
