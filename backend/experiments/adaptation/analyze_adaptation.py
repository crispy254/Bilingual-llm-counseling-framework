#!/usr/bin/env python3
"""
analyze_adaptation.py -- analysis of simulate_adaptation_extended.py outputs.

Reads  <results-dir>/adaptation_seed_results.csv   (per-seed scalars, checkpoints)
       <results-dir>/adaptation_curves.npz         (per-round curves)
       <results-dir>/judge_confusion_counts.csv    (judge-labelled execution counts)
Writes <output>/figures/fig1..fig4 (.png 300 dpi + .pdf), figure_captions.txt
       <output>/tables/adaptation_tables.tex, adaptation_summary.csv

Scientific notes
* All rewards are SIMULATED (synthetic preference model); they are not student outcomes.
* Execution is modelled with measured judge-derived confusion matrices (automated
  LLM-judge labels are a proxy for execution, not ground truth).
* Oracle and "best fixed" are HINDSIGHT baselines. Oracle re-optimises per context
  (and per preference epoch under drift); best fixed is the single strategy with the
  highest cumulative expected reward over the realised run.
* Uniform random and the average of the four fixed strategies are identical in
  expectation, so they are drawn as one curve and listed jointly in tables.
* "Preferred strategy" = argmax of the simulated response probability for the context
  (ignores execution confusion), so 100% is not necessarily attainable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import simulate_adaptation_extended as sim
    PAPER_MODELS = list(sim.PAPER_MODELS)
    CHECKPOINTS = list(sim.CHECKPOINTS)
    LAST = int(sim.LAST)
except Exception:  # fallback if the simulator is not importable
    PAPER_MODELS = ["ours", "ours+guard", "gemma3:4b",
                    "gemma3:4b+guard", "qwen2.5:3b", "exaone3.5:7.8b"]
    CHECKPOINTS = [0, 100, 250, 500, 1000, 2000]
    LAST = 200

MAIN_MODEL = "ours"   # proposed system; robustness runs use this model
CHANCE = 0.25

POLICIES = ["Oracle", "Adaptive (LinUCB)", "Best fixed (hindsight)",
            "Fixed average / uniform random"]
POLICY_STYLE = {
    "Oracle": dict(color="#555555", ls="--"),
    "Adaptive (LinUCB)": dict(color="#0072B2", ls="-"),
    "Best fixed (hindsight)": dict(color="#009E73", ls="-"),
    "Fixed average / uniform random": dict(color="#D55E00", ls="-"),
}
MODEL_COLORS = dict(zip(PAPER_MODELS, ["#0072B2", "#56B4E9", "#009E73",
                                       "#E69F00", "#D55E00", "#CC79A7"]))
MODEL_MARKERS = dict(zip(PAPER_MODELS, ["o", "s", "^", "D", "v", "P"]))

# (long label, short tick label, experiment name; "main" = full-feedback main run)
CONDITIONS = [
    ("Full feedback", "Full", "main"),
    ("Sparse feedback (50%)", "Sparse\n50%", "sparse_feedback_50pct"),
    ("Sparse feedback (25%)", "Sparse\n25%", "sparse_feedback_25pct"),
    ("Noisy feedback (10% flips)", "Noisy\n10%", "noisy_feedback_10pct"),
    ("Noisy feedback (20% flips)", "Noisy\n20%", "noisy_feedback_20pct"),
    ("Drift every 500", "Drift\n/500", "preference_drift_every_500"),
    ("Drift every 250", "Drift\n/250", "preference_drift_every_250"),
    ("Sparse 50% + noisy 10% + drift/500", "Comb.", "combined_sparse_noisy_drift"),
]

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})

WRITTEN = []


# ----------------------------------------------------------------- utilities
def cond_key(name, main):
    return f"main__{main}" if name == "main" else f"robust__{name}"


def mean_ci(x, axis=0):
    """Mean and 95% CI half-width (Student t) across seeds."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    m = x.mean(axis=axis)
    if n < 2:
        return m, np.zeros_like(m)
    return m, stats.t.ppf(0.975, n - 1) * x.std(axis=axis, ddof=1) / np.sqrt(n)


def rolling_mean(A, w):
    cs = np.cumsum(np.pad(np.asarray(A, float), ((0, 0), (1, 0))), axis=1)
    return (cs[:, w:] - cs[:, :-w]) / w


def wilcoxon_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, cen - half, cen + half


def save_fig(fig, outdir, name):
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, **kw)
        WRITTEN.append(path)
    plt.close(fig)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=9)


# --------------------------------------------------------------- data access
def load_results(rdir):
    seed_csv = rdir / "adaptation_seed_results.csv"
    if not seed_csv.exists():
        sys.exit(f"ERROR: {seed_csv} not found. Run simulate_adaptation_extended.py first.")
    df = pd.read_csv(seed_csv)
    curves = None
    npz_path = rdir / "adaptation_curves.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        curves = {}
        for name in npz.files:
            key, field = name.rsplit("__", 1)
            curves.setdefault(key, {})[field] = npz[name]
    cpath = rdir / "judge_confusion_counts.csv"
    counts = pd.read_csv(cpath) if cpath.exists() else None
    return df, curves, counts


def unpack(c):
    lin = c["lin_expected"].astype(float)
    orc = c["oracle"].astype(float)
    fx = c["fixed"].astype(float)                  # (seeds, T, 4)
    opt = c["lin_optimal"].astype(float)
    n = fx.shape[0]
    best = fx.sum(axis=1).argmax(axis=1)           # hindsight best fixed per seed
    fx_best = fx[np.arange(n), :, best]            # (seeds, T)
    fx_avg = fx.mean(axis=2)                       # == uniform random in expectation
    return lin, orc, fx_best, fx_avg, opt


def cumulative_series(c):
    lin, orc, fx_best, fx_avg, _ = unpack(c)
    series = {"Oracle": np.cumsum(orc, 1),
              "Adaptive (LinUCB)": np.cumsum(lin, 1),
              "Best fixed (hindsight)": np.cumsum(fx_best, 1),
              "Fixed average / uniform random": np.cumsum(fx_avg, 1)}
    return series, np.cumsum(lin - fx_best, 1)


def seed_metrics(c, last=LAST):
    lin, orc, fx_best, fx_avg, opt = unpack(c)
    n, T = lin.shape
    diff = np.cumsum(lin - fx_best, axis=1)
    cross = np.full(n, np.nan)
    for i in range(n):
        bad = np.flatnonzero(diff[i] <= 0)
        if len(bad) == 0:
            cross[i] = 1
        elif bad[-1] < T - 1:
            cross[i] = bad[-1] + 2
    return dict(
        cum_oracle=orc.sum(1), cum_lin=lin.sum(1),
        cum_best_fixed=fx_best.sum(1), cum_fixed_avg=fx_avg.sum(1),
        last_oracle=orc[:, -last:].mean(1), last_lin=lin[:, -last:].mean(1),
        last_best_fixed=fx_best[:, -last:].mean(1), last_fixed_avg=fx_avg[:, -last:].mean(1),
        regret_oracle=orc.sum(1) - lin.sum(1),
        adv_best_fixed=lin.sum(1) - fx_best.sum(1),
        adv_fixed_avg=lin.sum(1) - fx_avg.sum(1),
        pref_last=opt[:, -last:].mean(1), pref_all=opt.mean(1),
        crossover=cross, T=T)


def checkpoint_arrays(sub):
    parsed = [json.loads(s) for s in sub.sort_values("seed")["greedy_checkpoints"]]
    cps = sorted(int(k) for k in parsed[0])
    R = np.array([[d[str(k)]["expected_reward"] for k in cps] for d in parsed])
    A = np.array([[d[str(k)]["preferred_agreement"] for k in cps] for d in parsed])
    return cps, R, A


# ------------------------------------------------------------------- figures
def fig1(curves, main, outdir):
    c = curves[f"main__{main}"]
    series, _ = cumulative_series(c)
    T = next(iter(series.values())).shape[1]
    x = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    for name, A in series.items():
        m, h = mean_ci(A)
        st = POLICY_STYLE[name]
        ax.plot(x, m, color=st["color"], ls=st["ls"], lw=1.4, label=name)
        ax.fill_between(x, m - h, m + h, color=st["color"], alpha=0.2, lw=0)
    ax.set_xlabel("Interaction")
    ax.set_ylabel("Cumulative expected reward (simulated)")
    panel_title(ax, f"(a) {main}")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    for model in PAPER_MODELS:
        _, d = cumulative_series(curves[f"main__{model}"])
        m, h = mean_ci(d)
        ax.plot(x, m, color=MODEL_COLORS[model], lw=1.2, label=model)
        ax.fill_between(x, m - h, m + h, color=MODEL_COLORS[model], alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Interaction")
    ax.set_ylabel("LinUCB $-$ best fixed (hindsight)\ncumulative expected reward")
    panel_title(ax, "(b) Adaptive gain over best fixed, by model")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=6.3)
    fig.tight_layout()
    save_fig(fig, outdir, "fig1_cumulative_reward")


def fig2(curves, w, outdir):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    T = None
    for model in PAPER_MODELS:
        opt = curves[f"main__{model}"]["lin_optimal"].astype(float)
        T = opt.shape[1]
        R = rolling_mean(opt, w)
        m, h = mean_ci(R)
        x = np.arange(w, T + 1)
        ax.plot(x, m, color=MODEL_COLORS[model], lw=1.3, label=model)
        ax.fill_between(x, m - h, m + h, color=MODEL_COLORS[model], alpha=0.13, lw=0)
    ax.axhline(CHANCE, color="gray", ls=":", lw=1)
    ax.text(T * 0.99, CHANCE + 0.015, "random selection (0.25)", ha="right",
            fontsize=7, color="gray")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Interaction")
    ax.set_ylabel(f"Share choosing preferred strategy\n({w}-interaction moving average)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, outdir, "fig2_strategy_selection_quality")


def fig3(df, main, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    main_rows = df[df.experiment == "measured_confusion_main"]
    cps = CHECKPOINTS
    for model in PAPER_MODELS:
        sub = main_rows[main_rows.model == model]
        cps, R, A = checkpoint_arrays(sub)
        for ax, Z in ((axes[0], R), (axes[1], A)):
            m, h = mean_ci(Z)
            ax.errorbar(cps, m, yerr=h, color=MODEL_COLORS[model],
                        marker=MODEL_MARKERS[model], ms=3.5, lw=1.2, capsize=2,
                        label=model)
    ms = main_rows[main_rows.model == main]
    for col, lab, ls in (("oracle_expected_reward", "Oracle", "--"),
                         ("best_fixed_expected_reward", "Best fixed (hindsight)", "-."),
                         ("fixed_average_expected_reward", "Fixed average / random", ":")):
        axes[0].axhline(ms[col].mean(), color="gray", ls=ls, lw=0.9,
                        label=f"{lab} ({main})")
    axes[1].axhline(CHANCE, color="gray", ls=":", lw=1)
    axes[1].text(2550, CHANCE + 0.015, "random (0.25)", ha="right", fontsize=7, color="gray")
    axes[1].set_ylim(0, 1)
    axes[0].set_ylabel("Expected reward of greedy learned policy\n(simulated, range $-1$ to $+1$)")
    axes[1].set_ylabel("Contexts where greedy policy = preferred strategy")
    panel_title(axes[0], "(a) Expected reward")
    panel_title(axes[1], "(b) Preferred-strategy agreement")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=100, linscale=0.6)
        ax.set_xticks(cps)
        ax.set_xticklabels([str(k) for k in cps])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(-15, 2700)
        ax.set_xlabel("Feedback interactions (0 = cold start)")
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
               frameon=False, fontsize=6.8)
    save_fig(fig, outdir, "fig3_adaptation_progress")


def fig4(curves, M, main, w, outdir):
    keys = [cond_key(k, main) for _, _, k in CONDITIONS]
    T = int(M[keys[0]]["T"])
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])
    xs = np.arange(len(CONDITIONS))
    short = [s for _, s, _ in CONDITIONS]

    # (a) per-interaction expected reward by policy
    offs = np.linspace(-0.27, 0.27, 4)
    markers = ["D", "o", "s", "^"]
    ckeys = ["cum_oracle", "cum_lin", "cum_best_fixed", "cum_fixed_avg"]
    for j, (pol, ck) in enumerate(zip(POLICIES, ckeys)):
        vals = [mean_ci(M[k][ck] / T) for k in keys]
        mm = np.array([v[0] for v in vals])
        hh = np.array([v[1] for v in vals])
        axa.errorbar(xs + offs[j], mm, yerr=hh, fmt=markers[j], ms=4, capsize=2, lw=1,
                     color=POLICY_STYLE[pol]["color"], label=pol)
    for b in (0.5, 2.5, 4.5, 6.5, 7.5):
        axa.axvline(b, color="gray", lw=0.5, ls=":")
    axa.set_xticks(xs)
    axa.set_xticklabels(short, fontsize=7)
    axa.set_ylabel("Mean expected reward per interaction\n(simulated)")
    panel_title(axa, f"(a) {main}: policies under feedback stress (hindsight baselines re-optimised per drift epoch)")
    axa.set_ylim(-0.17, 0.45)
    axa.legend(loc="lower center", ncol=4, frameon=False, fontsize=6.5)

    # (b) advantage of LinUCB (negative values are shown, not hidden)
    wbar = 0.38
    for j, (ck, lab, col) in enumerate([
            ("adv_best_fixed", "LinUCB $-$ best fixed (hindsight)", "#0072B2"),
            ("adv_fixed_avg", "LinUCB $-$ fixed average / random", "#E69F00")]):
        vals = [mean_ci(M[k][ck] / T) for k in keys]
        mm = np.array([v[0] for v in vals])
        hh = np.array([v[1] for v in vals])
        axb.bar(xs + (j - 0.5) * wbar, mm, wbar, yerr=hh, capsize=1.5, color=col,
                label=lab, error_kw=dict(lw=0.8))
    axb.axhline(0, color="k", lw=0.7)
    axb.set_xticks(xs)
    axb.set_xticklabels(short, fontsize=6.3)
    axb.set_ylabel("Expected-reward advantage\nper interaction")
    panel_title(axb, "(b) LinUCB advantage (95% CI)")
    axb.set_ylim(-0.04, 0.44)
    axb.legend(loc="upper center", ncol=1, frameon=False, fontsize=6)

    # (c) selection quality over time
    sel = [("main", "Full feedback", "#0072B2"),
           ("sparse_feedback_25pct", "Sparse 25%", "#56B4E9"),
           ("noisy_feedback_20pct", "Noisy 20%", "#009E73"),
           ("preference_drift_every_500", "Drift /500", "#D55E00"),
           ("preference_drift_every_250", "Drift /250", "#CC79A7"),
           ("combined_sparse_noisy_drift", "Combined", "#000000")]
    for name, lab, col in sel:
        opt = curves[cond_key(name, main)]["lin_optimal"].astype(float)
        R = rolling_mean(opt, w)
        m, h = mean_ci(R)
        x = np.arange(w, opt.shape[1] + 1)
        axc.plot(x, m, color=col, lw=1.1, label=lab)
        axc.fill_between(x, m - h, m + h, color=col, alpha=0.10, lw=0)
    for i, t in enumerate((500, 1000, 1500)):
        axc.axvline(t, color="gray", lw=0.6, ls="--",
                    label="redraw (drift/500)" if i == 0 else None)
    axc.axhline(CHANCE, color="gray", ls=":", lw=1)
    axc.set_ylim(0, 1)
    axc.set_xlabel("Interaction")
    axc.set_ylabel(f"Share choosing preferred strategy\n({w}-interaction moving avg.)")
    panel_title(axc, "(c) Selection quality over time")
    axc.set_ylim(0, 1.3)
    axc.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axc.legend(loc="upper left", frameon=False, fontsize=5.8, ncol=2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08)
    save_fig(fig, outdir, "fig4_robustness")


CAPTIONS = {
    "fig1_cumulative_reward":
        "(a) Cumulative EXPECTED reward over 2,000 simulated interactions for the proposed "
        "configuration, mean and 95% CI over seeds. Oracle and best-fixed are hindsight "
        "baselines; uniform random equals the fixed-strategy average in expectation. "
        "(b) Cumulative gain of LinUCB over the best fixed strategy for each model "
        "configuration; values below zero mean the fixed strategy was ahead. Rewards are "
        "simulated, not student outcomes; execution follows judge-derived confusion matrices.",
    "fig2_strategy_selection_quality":
        "Share of interactions in which LinUCB selected the preferred strategy (argmax of the "
        "simulated response probability), moving average with 95% CI over seeds, for the six "
        "model configurations. Dotted line: random selection (0.25). Execution is imperfect, "
        "so the preferred strategy is not always the oracle action.",
    "fig3_adaptation_progress":
        "Greedy learned policy at feedback checkpoints (0 = cold start): (a) expected simulated "
        "reward (incl. execution confusion), with hindsight reference lines for the proposed "
        "configuration; (b) fraction of contexts where the policy matches the preferred "
        "strategy. Mean and 95% CI over seeds.",
    "fig4_robustness":
        "Proposed configuration under sparse feedback (50/25% observed), noisy feedback "
        "(10/20% flipped), preference drift (redrawn every 500/250 interactions) and a combined "
        "condition. (a) Mean expected reward per interaction; oracle and best fixed are "
        "re-evaluated against the preferences in force. (b) LinUCB advantage over baselines; "
        "negative bars show where LinUCB did not beat the best fixed strategy. (c) Selection "
        "quality over time; dashed lines mark drift/500 redraws (drift/250 redraws at multiples "
        "of 250). Simulated rewards only.",
}


# -------------------------------------------------------------------- tables
def tex_escape(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&")
            .replace("%", "\\%").replace("_", "\\_").replace("#", "\\#"))


def tex_table(caption, label, header, rows, colspec, resize=False):
    L = ["\\begin{table}[t]", "\\caption{" + caption + "}", "\\label{" + label + "}",
         "\\centering", "\\footnotesize"]
    if resize:
        L.append("\\resizebox{\\textwidth}{!}{%")
    L += ["\\begin{tabular}{" + colspec + "}", "\\hline",
          " & ".join("\\textbf{" + h + "}" for h in header) + " \\\\", "\\hline"]
    L += [" & ".join(r) + " \\\\" for r in rows]
    L += ["\\hline", "\\end{tabular}"]
    if resize:
        L.append("}")
    L += ["\\end{table}", ""]
    return "\n".join(L)


def fmt(a, d=1):
    m, h = mean_ci(a)
    return f"{m:.{d}f} $\\pm$ {h:.{d}f}"


def fp(v):
    return "$<0.001$" if v < 0.001 else f"{v:.3f}"


def add(summary, table, experiment, model, metric, arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return
    m, h = mean_ci(arr)
    summary.append(dict(table=table, experiment=experiment, model=model, metric=metric,
                        mean=float(m), ci95_low=float(m - h), ci95_high=float(m + h),
                        n_seeds=len(arr)))


def add_scalar(summary, table, experiment, model, metric, value, n):
    summary.append(dict(table=table, experiment=experiment, model=model, metric=metric,
                        mean=float(value), ci95_low=np.nan, ci95_high=np.nan, n_seeds=n))


def build_tables(df, M, counts, main, n_seeds, summary):
    tex = ["% ---- generated by analyze_adaptation.py (do not edit by hand) ----",
           "% All rewards are SIMULATED; baselines marked (hindsight) use the realised run.", ""]
    main_rows = df[df.experiment == "measured_confusion_main"]
    per_round = "mean $\\pm$ 95\\% CI over " + str(n_seeds) + " paired seeds"

    # Table A / B / D need per-round metrics
    if M is not None:
        mk = M[f"main__{main}"]
        T = int(mk["T"])
        rows = []
        for lab, ck, lk in [("Oracle (hindsight)", "cum_oracle", "last_oracle"),
                            ("Adaptive (LinUCB)", "cum_lin", "last_lin"),
                            ("Best fixed (hindsight)", "cum_best_fixed", "last_best_fixed"),
                            ("Fixed average / uniform random", "cum_fixed_avg", "last_fixed_avg")]:
            rows.append([tex_escape(lab), fmt(mk[ck]), fmt(mk["cum_oracle"] - mk[ck]),
                         fmt(mk[lk], 3)])
            add(summary, "A", "main", main, "cum_expected_reward:" + lab, mk[ck])
            add(summary, "A", "main", main, "last200_expected_reward:" + lab, mk[lk])
        add(summary, "A", "main", main, "linucb_pref_selection_last200", mk["pref_last"])
        add(summary, "A", "main", main, "crossover_round_vs_best_fixed", mk["crossover"])
        tex.append(tex_table(
            "Simulated strategy selection for " + tex_escape(main) + " under its measured "
            "execution-confusion matrices (" + str(T) + " interactions; " + per_round + "). "
            "Rewards are expected simulated rewards, not student outcomes. Oracle and best fixed "
            "are hindsight baselines.",
            "tab:adaptation_main",
            ["Policy", "Cum. expected reward", "Cum. regret vs. oracle", "Mean reward, last 200"],
            rows, "lccc"))

        rows = []
        for model in PAPER_MODELS:
            m = M[f"main__{model}"]
            p = wilcoxon_p(m["cum_lin"], m["cum_best_fixed"])
            wins = int((m["cum_lin"] > m["cum_best_fixed"]).sum())
            rows.append([tex_escape(model), fmt(m["cum_oracle"], 0), fmt(m["cum_lin"], 0),
                         fmt(m["cum_best_fixed"], 0), fmt(m["cum_fixed_avg"], 0),
                         fmt(m["pref_last"], 3), fmt(m["adv_best_fixed"], 1), fp(p),
                         f"{wins}/{n_seeds}"])
            for k in ("cum_oracle", "cum_lin", "cum_best_fixed", "cum_fixed_avg",
                      "adv_best_fixed", "pref_last", "crossover"):
                add(summary, "B", "main", model, k, m[k])
            add_scalar(summary, "B", "main", model, "wilcoxon_p_linucb_vs_best_fixed", p, n_seeds)
            add_scalar(summary, "B", "main", model, "seeds_linucb_gt_best_fixed", wins, n_seeds)
        tex.append(tex_table(
            "LinUCB across model configurations (cumulative expected simulated reward, "
            + per_round + "; Wilcoxon signed-rank test of LinUCB vs.\\ the hindsight best fixed "
            "strategy; wins = seeds where LinUCB was ahead). Pref.\\ = share of last 200 "
            "interactions selecting the preferred strategy (chance 0.25).",
            "tab:adaptation_models",
            ["Model", "Oracle", "LinUCB", "Best fixed", "Fixed avg./random", "Pref.",
             "LinUCB $-$ best fixed", "$p$", "Wins"],
            rows, "lcccccccc", resize=True))

        rows = []
        for long, _, name in CONDITIONS:
            k = cond_key(name, main)
            m = M[k]
            p = wilcoxon_p(m["cum_lin"], m["cum_best_fixed"])
            wins = int((m["cum_lin"] > m["cum_best_fixed"]).sum())
            rows.append([tex_escape(long), fmt(m["cum_oracle"], 0), fmt(m["cum_lin"], 0),
                         fmt(m["cum_best_fixed"], 0), fmt(m["cum_fixed_avg"], 0),
                         fmt(m["adv_best_fixed"], 1), fp(p), f"{wins}/{n_seeds}",
                         fmt(m["pref_last"], 3)])
            for kk in ("cum_oracle", "cum_lin", "cum_best_fixed", "cum_fixed_avg",
                       "adv_best_fixed", "pref_last"):
                add(summary, "D", name, main, kk, m[kk])
            add_scalar(summary, "D", name, main, "wilcoxon_p_linucb_vs_best_fixed", p, n_seeds)
            add_scalar(summary, "D", name, main, "seeds_linucb_gt_best_fixed", wins, n_seeds)
        tex.append(tex_table(
            "Robustness of LinUCB for " + tex_escape(main) + " (cumulative expected simulated "
            "reward, " + per_round + "). Under drift, oracle is re-optimised for the preferences "
            "in force, while best fixed is a single strategy chosen in hindsight for the whole "
            "run. Negative LinUCB $-$ best fixed values indicate LinUCB did not beat the best "
            "fixed strategy.",
            "tab:adaptation_robust",
            ["Condition", "Oracle", "LinUCB", "Best fixed", "Fixed avg./random",
             "LinUCB $-$ best fixed", "$p$", "Wins", "Pref."],
            rows, "lcccccccc", resize=True))
    else:
        print("WARNING: adaptation_curves.npz missing -> tables A, B, D skipped "
              "(re-run the modified simulator).")

    # Table C: checkpoints
    for metric_idx, (title, lab) in enumerate([
            ("Expected simulated reward of the greedy learned policy", "tab:adaptation_ckpt_reward"),
            ("Share of contexts where the greedy learned policy equals the preferred strategy",
             "tab:adaptation_ckpt_agree")]):
        rows, cps = [], CHECKPOINTS
        for model in PAPER_MODELS:
            cps, R, A = checkpoint_arrays(main_rows[main_rows.model == model])
            Z = R if metric_idx == 0 else A
            rows.append([tex_escape(model)] + [fmt(Z[:, i], 3) for i in range(len(cps))])
            for i, k in enumerate(cps):
                add(summary, "C", "main", model,
                    ("greedy_expected_reward@" if metric_idx == 0 else "greedy_agreement@") + str(k),
                    Z[:, i])
        tex.append(tex_table(
            title + " at feedback checkpoints (" + per_round.replace("paired ", "") + "). "
            "Interaction 0 is the cold-start policy.", lab,
            ["Model"] + [str(k) for k in cps], rows, "l" + "c" * len(cps), resize=True))

    # Table E: measured execution adherence (judge-labelled)
    if counts is not None:
        rows = []
        for model in PAPER_MODELS:
            row = [tex_escape(model)]
            for lang in ("en", "ko"):
                s = counts[(counts.model == model) & (counts.language == lang)]
                k = int(sum(r[f"judged_{r['intended_strategy']}"] for _, r in s.iterrows()))
                n = int(s["n"].sum())
                p, lo, hi = wilson(k, n)
                row.append("n/a" if n == 0 else
                           f"{100*p:.1f}\\% [{100*lo:.1f}, {100*hi:.1f}] (n={n})")
                add_scalar(summary, "E", lang, model, "judge_labelled_adherence", p, n)
            rows.append(row)
        tex.append(tex_table(
            "Judge-labelled execution adherence used to parameterise the simulation (share of "
            "responses whose automated LLM-judge strategy label matches the intended strategy, "
            "pooled over the four strategies; Wilson 95\\% CI). These labels are an automated "
            "proxy for execution, not ground truth.",
            "tab:adaptation_adherence", ["Model", "English", "Korean"], rows, "lcc"))
    return "\n".join(tex)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="adaptation_extended_results")
    ap.add_argument("--output", default="analysis_outputs")
    ap.add_argument("--main-model", default=MAIN_MODEL)
    ap.add_argument("--window", type=int, default=50, help="moving-average window")
    args = ap.parse_args()

    rdir, odir = Path(args.results_dir), Path(args.output)
    figdir, tabdir = odir / "figures", odir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    df, curves, counts = load_results(rdir)
    main_rows = df[df.experiment == "measured_confusion_main"]
    missing = [m for m in PAPER_MODELS if m not in set(main_rows.model)]
    if missing:
        sys.exit(f"ERROR: no main-run rows for models: {missing}. "
                 "Re-run simulate_adaptation_extended.py with the renamed models.")
    n_seeds = int(main_rows.groupby("model").seed.nunique().min())
    print(f"Loaded {len(df)} seed-level rows; {n_seeds} seeds per scenario.")

    M = None
    if curves is not None:
        need = [cond_key(k, args.main_model) for _, _, k in CONDITIONS]
        need += [f"main__{m}" for m in PAPER_MODELS]
        absent = sorted(set(k for k in need if k not in curves))
        if absent:
            sys.exit(f"ERROR: curves missing for {absent}. Re-run the modified simulator.")
        M = {k: seed_metrics(curves[k]) for k in set(need)}
    else:
        print("WARNING: adaptation_curves.npz not found. Figures 1, 2, 4 and tables A/B/D "
              "need it: re-run simulate_adaptation_extended.py.")

    summary = []
    if curves is not None:
        fig1(curves, args.main_model, figdir)
        fig2(curves, args.window, figdir)
    fig3(df, args.main_model, figdir)
    if curves is not None:
        fig4(curves, M, args.main_model, args.window, figdir)

    tex = build_tables(df, M, counts, args.main_model, n_seeds, summary)
    tex_path = tabdir / "adaptation_tables.tex"
    tex_path.write_text(tex, encoding="utf-8")
    csv_path = tabdir / "adaptation_summary.csv"
    pd.DataFrame(summary).to_csv(csv_path, index=False)
    cap_path = figdir / "figure_captions.txt"
    cap_path.write_text("\n\n".join(f"{k}:\n{v}" for k, v in CAPTIONS.items()), encoding="utf-8")
    WRITTEN.extend([tex_path, csv_path, cap_path])

    # console summary
    if M is not None:
        print("\n=== LinUCB vs hindsight best fixed (cumulative expected reward) ===")
        for model in PAPER_MODELS:
            m = M[f"main__{model}"]
            p = wilcoxon_p(m["cum_lin"], m["cum_best_fixed"])
            mu, h = mean_ci(m["adv_best_fixed"])
            cr = m["crossover"]
            print(f"{model:22s} diff {mu:8.1f} +/- {h:5.1f}  wins "
                  f"{int((m['cum_lin'] > m['cum_best_fixed']).sum())}/{n_seeds}  p={p:.2e}  "
                  f"pref(last{LAST})={m['pref_last'].mean():.3f}  "
                  f"crossover median={np.nanmedian(cr) if np.any(~np.isnan(cr)) else float('nan'):.0f} "
                  f"(never ahead in {int(np.isnan(cr).sum())} seeds)")
        print("\n=== Robustness (main model) ===")
        for long, _, name in CONDITIONS:
            m = M[cond_key(name, args.main_model)]
            mu, h = mean_ci(m["adv_best_fixed"])
            print(f"{long:38s} LinUCB-bestfixed {mu:8.1f} +/- {h:5.1f}  "
                  f"wins {int((m['cum_lin'] > m['cum_best_fixed']).sum())}/{n_seeds}  "
                  f"pref(last{LAST})={m['pref_last'].mean():.3f}")
    print("\nNOTE: simulated rewards, hindsight baselines, judge-labelled (proxy) execution.")
    print("\n=== Generated files ===")
    for p in WRITTEN:
        print("  ", p)


if __name__ == "__main__":
    main()
