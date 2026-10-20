"""
Adaptive (LinUCB) versus fixed / random strategy selection: a simulation study.

What this is: a controlled simulation of the feedback loop (context -> strategy -> response -> thumbs up/down ->
update). Simulated students have assumed, randomly drawn strategy preferences per context. It shows the learning
behaviour of the bandit and how it depends on how reliably the language model executes the selected strategy.
What this is NOT: evidence that real students benefit. State this in the paper.

Design
  * Contexts: 6 emotions x 4 stress types (24), sampled uniformly each round; features = 10-d one-hot (as in the system).
  * Student model: P(thumbs up | context, delivered strategy) = sigmoid(beta_s + u[e,s] + v[t,s] (+ g[e,t,s])), drawn
    afresh for every seed. g adds context-specific interactions that the linear LinUCB model cannot represent.
  * Execution model: the generator delivers the SELECTED strategy with probability q (its measured strategy adherence);
    otherwise a uniformly random OTHER strategy is delivered. Feedback is attributed to the SELECTED strategy
    (exactly as in the system), so low q corrupts the learning signal. At q = 0.25 the delivered strategy is independent
    of the selection and nothing can be learned.
  * Reward: +1 (thumbs up) / -1 (thumbs down), as in the system. LinUCB: alpha = 0.8, A = I, b = 0 (as in the system).
  * Policies: oracle (knows the true preferences), adaptive LinUCB, best fixed strategy chosen in hindsight,
    fixed strategies (average of the four), random (no learning).
  * All policies of one seed see the same environment, context sequence and random numbers (paired comparison).

Usage:  python simulate_adaptation.py [--seeds 50] [--rounds 2000]
"""
import argparse, json, os
import numpy as np
from scipy import stats

EM, ST, S, D = 6, 4, 4, 10
STRATS = ["Listen", "Ask", "Suggest", "Normalize"]
ALPHA = 0.8
CHECKPOINTS = [0, 100, 250, 500, 1000, 2000]
LAST = 200                                  # window for "after learning" statistics
ADHERENCE = {
    "Perfect execution (q = 1.000)": 1.000,
    "Our system (q = 0.868)": 0.868,
    "Our system + guard (q = 0.917)": 0.917,
    "Gemma 3 4B base (q = 0.722)": 0.722,
    "Gemma 3 4B base + guard (q = 0.750)": 0.750,
    "Qwen2.5 3B (q = 0.396)": 0.396,
    "EXAONE 3.5 7.8B (q = 0.361)": 0.361,
}

MAIN_Q = 0.868

def make_env(rng, strength=1.0, interaction=0.0):
    beta = rng.normal(0, 0.3, S)
    u = rng.normal(0, 1.0 * strength, (EM, S))
    v = rng.normal(0, 0.7 * strength, (ST, S))
    g = rng.normal(0, interaction * strength, (EM, ST, S)) if interaction > 0 else np.zeros((EM, ST, S))
    logits = beta[None, None, :] + u[:, None, :] + v[None, :, :] + g
    return (1 / (1 + np.exp(-logits))).reshape(EM * ST, S)          # p[c, s], c = e * ST + t


def eff_reward(p, q):
    """Expected P(thumbs up) when selecting s and delivering s with prob q (else a uniform other strategy)."""
    return q * p + (1 - q) / (S - 1) * (p.sum(1, keepdims=True) - p)


def deliver(sel, q, u_exec, u_alt):
    """Vectorised delivered strategy for selected strategies `sel` (array)."""
    alt = (sel + 1 + np.minimum((u_alt * (S - 1)).astype(int), S - 2)) % S      # uniform among the other S-1
    return np.where(u_exec < q, sel, alt)


class LinUCB:
    """Same maths as strategy_selector.py: per-strategy A (init I) and b; score = theta.x + alpha*sqrt(x A^-1 x).
    A^-1 is kept up to date with the Sherman-Morrison identity (numerically equivalent to inverting A)."""
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.Ainv = np.stack([np.eye(D)] * S)
        self.b = np.zeros((S, D))

    def scores(self, i, j):                         # x = e_i + e_j (emotion index i, stress index 6 + t)
        th = np.einsum("sab,sb->sa", self.Ainv, self.b)
        mean = th[:, i] + th[:, j]
        quad = self.Ainv[:, i, i] + self.Ainv[:, j, j] + 2 * self.Ainv[:, i, j]
        return mean + self.alpha * np.sqrt(np.maximum(quad, 0))

    def select(self, i, j):
        return int(np.argmax(self.scores(i, j)))     # first maximum, like max() over STRATEGIES

    def update(self, s, i, j, r):
        x = np.zeros(D); x[i] = 1; x[j] = 1
        Ax = self.Ainv[s] @ x
        self.Ainv[s] -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b[s] += r * x

    def greedy_policy(self):
        """argmax of the learned reward estimate (no exploration bonus) for each of the 24 contexts."""
        th = np.einsum("sab,sb->sa", self.Ainv, self.b)
        out = np.zeros(EM * ST, int)
        for c in range(EM * ST):
            i, j = c // ST, EM + c % ST
            out[c] = int(np.argmax(th[:, i] + th[:, j]))
        return out


def run_seed(seed, q, rounds, strength=1.0, interaction=0.0, alpha=ALPHA):
    rng = np.random.default_rng(seed)
    p = make_env(rng, strength, interaction)
    c_seq = rng.integers(0, EM * ST, rounds)
    u_exec, u_alt, u_rew, u_sel = (rng.random(rounds) for _ in range(4))
    eff = eff_reward(p, q)
    best_true = p.argmax(1)                                         # strategy students actually prefer per context
    rows = np.arange(rounds)

    def realised(sel):
        d = deliver(sel, q, u_exec, u_alt)
        return np.where(u_rew < p[c_seq, d], 1.0, -1.0)

    out = {}
    oracle_sel = eff.argmax(1)[c_seq]
    out["Oracle"] = (realised(oracle_sel), 2 * eff[c_seq, oracle_sel] - 1, oracle_sel)
    fixed = []
    for s0 in range(S):
        sel = np.full(rounds, s0)
        fixed.append((realised(sel), 2 * eff[c_seq, sel] - 1, sel))
    best_s0 = int(np.argmax(eff.mean(0)))
    out["Best fixed (hindsight)"] = fixed[best_s0]
    out["Fixed (average of four)"] = tuple(np.mean([f[k] for f in fixed], 0) for k in range(2)) + (fixed[0][2],)
    sel = np.minimum((u_sel * S).astype(int), S - 1)
    out["Random (no learning)"] = (realised(sel), 2 * eff[c_seq, sel] - 1, sel)

    bandit = LinUCB(alpha)
    sel_l = np.zeros(rounds, int); r_l = np.zeros(rounds)
    greedy_val = {}
    def policy_value():
        pol = bandit.greedy_policy()
        return (float((2 * eff[np.arange(EM * ST), pol] - 1).mean()),     # expected reward of the learned policy
                float((pol == best_true).mean()))                          # share of the 24 contexts where it picks the preferred strategy
    greedy_val[0] = policy_value()
    for t in range(rounds):
        c = c_seq[t]; i, j = c // ST, EM + c % ST
        s = bandit.select(i, j)
        d = s if u_exec[t] < q else (s + 1 + min(int(u_alt[t] * (S - 1)), S - 2)) % S
        r = 1.0 if u_rew[t] < p[c, d] else -1.0
        bandit.update(s, i, j, r)                     # feedback attributed to the SELECTED strategy
        sel_l[t], r_l[t] = s, r
        if (t + 1) in CHECKPOINTS: greedy_val[t + 1] = policy_value()
    out["Adaptive (LinUCB)"] = (r_l, 2 * eff[c_seq, sel_l] - 1, sel_l)

    res = {}
    oracle_exp = out["Oracle"][1]
    for name, (r, er, sel) in out.items():
        res[name] = dict(
            cum_reward=np.cumsum(r), cum_regret=np.cumsum(oracle_exp - er),
            final=float(r.sum()), last=float(r[-LAST:].mean()), first=float(r[:LAST].mean()),
            regret=float((oracle_exp - er).sum()),
            optimal_last=float((sel[-LAST:] == best_true[c_seq][-LAST:]).mean()),
            optimal_first=float((sel[:LAST] == best_true[c_seq][:LAST]).mean()),
            optimal_curve=(sel == best_true[c_seq]).astype(float),
            dist_first=np.bincount(sel[:LAST], minlength=S) / LAST,
            dist_last=np.bincount(sel[-LAST:], minlength=S) / LAST)
    res["_greedy_val"] = greedy_val
    res["_oracle_dist"] = np.bincount(best_true, minlength=S) / (EM * ST)
    return res


def ci95(x):
    x = np.asarray(x, float); m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, h


def run_all(seeds, rounds, **kw):
    return [run_seed(s, rounds=rounds, **kw) for s in range(seeds)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50); ap.add_argument("--rounds", type=int, default=2000)
    a = ap.parse_args()
    import pickle
    store = {}
    for label, q in ADHERENCE.items():
        store[label] = run_all(a.seeds, a.rounds, q=q)
        print("done", label, flush=True)
    robust = {}
    for label, (strength, inter) in {"Additive preferences, weaker (strength 0.5)": (0.5, 0.0),
                                     "Additive preferences (main setting)": (1.0, 0.0),
                                     "With context interactions (0.7)": (1.0, 0.7),
                                     "With context interactions (1.0)": (1.0, 1.0)}.items():
        robust[label] = run_all(a.seeds, a.rounds, q=MAIN_Q, strength=strength, interaction=inter)
        print("done", label, flush=True)
    pickle.dump(dict(store=store, robust=robust, seeds=a.seeds, rounds=a.rounds), open("sim_results.pkl", "wb"))
    print("saved sim_results.pkl")