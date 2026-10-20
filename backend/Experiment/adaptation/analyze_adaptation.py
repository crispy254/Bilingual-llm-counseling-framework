
"""Builds the tables and figures from sim_results.pkl (made by simulate_adaptation.py)."""
import pickle, sys
import numpy as np
from scipy import stats
import simulate_adaptation as sim

path = sys.argv[1] if len(sys.argv) > 1 else "sim_results.pkl"
R = pickle.load(open(path, "rb")); store, robust = R["store"], R["robust"]
T, NS = R["rounds"], R["seeds"]
POL = ["Oracle", "Adaptive (LinUCB)", "Best fixed (hindsight)", "Fixed (average of four)", "Random (no learning)"]
main_label = "Our system (q = 0.868)"
MAIN_Q = 0.868

def col(runs, pol, key): return np.array([r[pol][key] for r in runs])
def fmt(x, d=1): m, h = sim.ci95(x); return f"{m:.{d}f} $\\pm$ {h:.{d}f}"
def wil(a, b):
    try: return stats.wilcoxon(a, b).pvalue
    except ValueError: return 1.0

# ---- Table A: policies at our system's adherence
print(f"\n=== TABLE A: policies, q={MAIN_Q:.3f}, {NS} seeds, T={T} ===")
runs = store[main_label]
tabA = []
for pol in POL:
    row = [pol, fmt(col(runs, pol, "final"), 0), fmt(col(runs, pol, "regret"), 0),
           fmt(col(runs, pol, "first"), 3), fmt(col(runs, pol, "last"), 3),
           (fmt(col(runs, pol, "optimal_last"), 3) if pol != "Oracle" else "-")]
    tabA.append(row); print(row)
ad = col(runs, "Adaptive (LinUCB)", "final")
for ref in ["Best fixed (hindsight)", "Fixed (average of four)", "Random (no learning)"]:
    b = col(runs, ref, "final")
    print(f"LinUCB vs {ref}: mean diff {np.mean(ad-b):.1f}, wins {int((ad>b).sum())}/{NS}, Wilcoxon p={wil(ad,b):.2e}")

# ---- Table B: LinUCB across adherence levels
print("\n=== TABLE B: adherence levels ===")
tabB = []
for label, rr in store.items():
    ad = col(rr, "Adaptive (LinUCB)", "final"); rnd = col(rr, "Random (no learning)", "final")
    bf = col(rr, "Best fixed (hindsight)", "final")
    row = [label, fmt(ad, 0), fmt(rnd, 0), fmt(bf, 0), fmt(col(rr, "Adaptive (LinUCB)", "last"), 3),
           fmt(col(rr, "Adaptive (LinUCB)", "optimal_last"), 3), wil(ad, rnd), wil(ad, bf), int((ad > bf).sum())]
    tabB.append(row); print(row)

# ---- Table C: before / after feedback (value of the greedy learned policy)
print("\n=== TABLE C: expected reward of the learned (greedy) policy ===")
tabC = []
for t in sim.CHECKPOINTS:
    row = [t]
    for label, rr in store.items():
        v = np.array([r["_greedy_val"][t][0] for r in rr]); row.append(fmt(v, 3))
    tabC.append(row); print(row)

# ---- Table D: robustness
print(f"\n=== TABLE D: robustness (q={MAIN_Q:.3f}) ===")
tabD = []
for label, rr in robust.items():
    row = [label] + [fmt(col(rr, p, "last"), 3) for p in ["Oracle", "Adaptive (LinUCB)", "Best fixed (hindsight)", "Random (no learning)"]]
    ad = col(rr, "Adaptive (LinUCB)", "final"); bf = col(rr, "Best fixed (hindsight)", "final")
    row += [wil(ad, bf), int((ad > bf).sum())]
    tabD.append(row); print(row)

# ---- strategy-selection behaviour
print(f"\n=== strategy selection distribution (LinUCB, q={MAIN_Q:.3f}) ===")
df = np.mean([r["Adaptive (LinUCB)"]["dist_first"] for r in runs], 0)
dl = np.mean([r["Adaptive (LinUCB)"]["dist_last"] for r in runs], 0)
do = np.mean([r["_oracle_dist"] for r in runs], 0)
print("first200", df.round(3), "last200", dl.round(3), "truly best share", do.round(3))
print("optimal-selection first200 %.3f last200 %.3f" % (col(runs,"Adaptive (LinUCB)","optimal_first").mean(), col(runs,"Adaptive (LinUCB)","optimal_last").mean()))

# ---- how quickly does adaptation pay off?
cross = []
for r in runs:
    diff = r["Best fixed (hindsight)"]["cum_regret"] - r["Adaptive (LinUCB)"]["cum_regret"]
    bad = np.where(diff <= 0)[0]
    cross.append(int(bad[-1]) + 2 if len(bad) else 1)
print("\ncrossover round (LinUCB permanently ahead of best fixed in expected regret): median %d, IQR %d-%d, never(>T) %d" % (
    np.median(cross), np.percentile(cross, 25), np.percentile(cross, 75), sum(c > T for c in cross)))

print("\n=== agreement of learned greedy policy with preferred strategy (24 contexts) ===")
for label, rr in store.items():
    print(label, [round(float(np.mean([r["_greedy_val"][t][1] for r in rr])), 3) for t in sim.CHECKPOINTS])

# ---- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.arange(1, T + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
colors = {"Oracle": "#7F7F7F", "Adaptive (LinUCB)": "#1F3864", "Best fixed (hindsight)": "#2F75B5",
          "Fixed (average of four)": "#ED7D31", "Random (no learning)": "#C00000"}

for pol in POL:
    A = np.array([r[pol]["cum_reward"] for r in runs]); m = A.mean(0); h = 1.96 * A.std(0, ddof=1) / np.sqrt(NS)
    ax[0].plot(x, m, label=pol, color=colors[pol], lw=1.8, ls="--" if pol == "Oracle" else "-")
    ax[0].fill_between(x, m - h, m + h, color=colors[pol], alpha=0.15)

ax[0].set_xlabel("Interaction")
ax[0].set_ylabel("Cumulative reward")
ax[0].set_title(f"(a) Cumulative reward ($q$ = {MAIN_Q:.3f})", fontsize=10)
ax[0].legend(fontsize=7, loc="upper left")
ax[0].grid(alpha=0.3)

k = 50; kern = np.ones(k) / k
for label, rr in store.items():
    c = np.mean([r["Adaptive (LinUCB)"]["optimal_curve"] for r in rr], 0)
    ax[1].plot(x[k - 1:], np.convolve(c, kern, "valid"), label=label, lw=1.6)

ax[1].axhline(0.25, color="gray", ls=":", lw=1)
ax[1].text(T * 0.98, 0.265, "chance", ha="right", fontsize=7, color="gray")
ax[1].set_xlabel("Interaction")
ax[1].set_ylabel("Share of rounds choosing the preferred strategy")
ax[1].set_title("(b) LinUCB selection quality by execution reliability", fontsize=10)
ax[1].legend(fontsize=6.5, loc="center right")
ax[1].grid(alpha=0.3)
ax[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("fig_adaptation_curves.png", dpi=220)
plt.close()

fig, ax = plt.subplots(figsize=(6.2, 3.6))
for label, rr in store.items():
    A = np.array([[r["_greedy_val"][t][1] for t in sim.CHECKPOINTS] for r in rr])
    m = A.mean(0); h = 1.96 * A.std(0, ddof=1) / np.sqrt(NS)
    ax.errorbar(sim.CHECKPOINTS, m, yerr=h, marker="o", ms=4, capsize=2, lw=1.5, label=label)

ax.axhline(0.25, color="gray", ls=":", lw=1)
ax.text(T * 0.98, 0.265, "chance", ha="right", fontsize=7, color="gray")
ax.set_xlabel("Number of feedback interactions")
ax.set_ylabel("Contexts with preferred strategy chosen")
ax.set_title("Learned strategy policy versus the preferred strategy", fontsize=10)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(fontsize=6.5, loc="center right")

plt.tight_layout()
plt.savefig("fig_adaptation_policy.png", dpi=220)
plt.close()

# ---- LaTeX tables written to adaptation_tables.tex
def p(v): return "$<0.001$" if v < 0.001 else f"{v:.3f}"

L = []
L.append("% ---- generated by analyze_adaptation.py ----")

L.append(
    "\\begin{table}[t]\n"
    "\\caption{Adaptive versus fixed and random strategy selection "
    f"(simulation, $q={MAIN_Q:.3f}$, {NS} seeds, {T} interactions). "
    "Mean $\\pm$ 95\\% CI.}\n"
    "\\label{tab:adaptation_main}\n"
    "\\centering\n\\small\n"
    "\\begin{tabular}{p{0.30\\columnwidth}ccc}\n"
    "\\hline\n"
    "\\textbf{Policy} & \\textbf{Cum. reward} & \\textbf{Cum. regret} & "
    "\\textbf{Avg. reward, last 200} \\\\\n\\hline"
)
for r in tabA:
    L.append(f"{r[0]} & {r[1]} & {r[2]} & {r[4]} \\\\")
L.append("\\hline\n\\end{tabular}\n\\end{table}\n")

L.append(
    "\\begin{table}[t]\n"
    "\\caption{Adaptive (LinUCB) performance by execution reliability $q$ "
    f"of the language model. Cumulative reward after {T} interactions "
    "(mean $\\pm$ 95\\% CI); $p$: Wilcoxon signed-rank test against "
    f"the comparison policy over {NS} paired seeds.}}\n"
    "\\label{tab:adaptation_q}\n"
    "\\centering\n\\footnotesize\n"
    "\\begin{tabular}{p{0.30\\columnwidth}cccc}\n"
    "\\hline\n"
    "\\textbf{Generator ($q$)} & \\textbf{LinUCB} & \\textbf{Random} & "
    "\\textbf{Best fixed} & \\textbf{Optimal choice, last 200} \\\\\n\\hline"
)
for r in tabB:
    L.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[5]} \\\\")
L.append("\\hline\n\\end{tabular}\n\\end{table}\n")

# Table C: dynamically supports all seven configurations
labels = list(store.keys())
cols = "r" + "c" * len(labels)

L.append(
    "\\begin{table}[t]\n"
    "\\caption{Expected reward of the learned (greedy) strategy policy "
    f"before and after feedback, by execution reliability $q$ "
    f"(mean $\\pm$ 95\\% CI over {NS} seeds). "
    "Interaction 0 is the cold-start policy.}\n"
    "\\label{tab:adaptation_before_after}\n"
    "\\centering\n\\footnotesize\n"
    "\\resizebox{\\textwidth}{!}{%\n"
    f"\\begin{{tabular}}{{{cols}}}\n"
    "\\hline\n"
    "\\textbf{Feedback} & "
    + " & ".join(f"\\textbf{{{label}}}" for label in labels)
    + " \\\\\n\\hline"
)
for r in tabC:
    L.append(f"{r[0]} & " + " & ".join(r[1:]) + " \\\\")
L.append("\\hline\n\\end{tabular}%\n}\n\\end{table}\n")

L.append(
    "\\begin{table}[t]\n"
    f"\\caption{{Robustness at $q={MAIN_Q:.3f}$: average reward over "
    f"the last 200 interactions (mean $\\pm$ 95\\% CI over {NS} seeds), "
    "and the number of seeds in which LinUCB exceeded the best fixed "
    "strategy in cumulative reward.}\n"
    "\\label{tab:adaptation_robust}\n"
    "\\centering\n\\footnotesize\n"
    "\\begin{tabular}{p{0.27\\columnwidth}ccccc}\n"
    "\\hline\n"
    "\\textbf{Student model} & \\textbf{Oracle} & \\textbf{LinUCB} & "
    "\\textbf{Best fixed} & \\textbf{Random} & \\textbf{LinUCB wins} \\\\\n\\hline"
)
for r in tabD:
    L.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[6]}/{NS} \\\\")
L.append("\\hline\n\\end{tabular}\n\\end{table}\n")

open("adaptation_tables.tex", "w").write("\n".join(L))

print("\nwrote adaptation_tables.tex, fig_adaptation_curves.png, fig_adaptation_policy.png")
for r in tabB:
    print("p-values", r[0], p(r[6]), p(r[7]), "wins vs best fixed", r[8])
for r in tabD:
    print("robust p", r[0], p(r[5]))
