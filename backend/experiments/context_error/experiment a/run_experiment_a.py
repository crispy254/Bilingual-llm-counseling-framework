"""
run_experiment_a.py  -  Experiment A (READ-ONLY, separate from the app)
=======================================================================
Compares LinUCB strategy selection when given
  Run 1: the REFERENCE (annotated) emotion/stress labels from the CSV
  Run 2: the PREDICTED labels from inference.infer_emotional_state()
against the SAME frozen bandit state, restored from historical feedback
exactly as app.py::_warm_start_bandit() does it.

Safety properties
-----------------
* Opens the SQLite DB in read-only mode (use a COPY anyway).
* Never imports app.py, models.py, database.py, llm.py or rag.py.
* Never writes to the application folder.
* Calls strategy_selector.warm_start() ONCE, then replaces
  record_feedback() with a function that raises, so any accidental
  update during the runs crashes loudly.
* Snapshots _A/_b after warm start and asserts they are unchanged at
  the end.
* detect_intent()/COUNSELING_INTENTS are taken from the REAL
  chatbot_engine.py source via ast (no import, so no LLM/RAG loading,
  and no copy that could drift from the app).

Usage
-----
python run_experiment_a.py --app-dir /path/to/app --history-csv strategy_logs_export.csv \
       --csv /path/to/prompts_v2.csv --out-dir ./exp_a_out
(SQLite users can pass --db copy.db instead of --history-csv.)
"""
import argparse
import ast
import copy
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

EXPECTED = {"emotion": 44, "stress": 73, "joint": 22, "routed": 77, "n": 144}  # from the paper's earlier results
TIE_EPS = 1e-9


def load_routing(engine_path: Path):
    """Extract detect_intent + its constants from the real source text."""
    tree = ast.parse(engine_path.read_text(encoding="utf-8"))
    keep = {"INTENT_PATTERNS", "COUNSELING_INTENTS", "GENERIC_DISTRESS_PATTERNS"}
    ns = {"re": re}
    for node in tree.body:
        is_const = isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in keep
        is_fn = isinstance(node, ast.FunctionDef) and node.name == "detect_intent"
        if is_const or is_fn:
            exec(compile(ast.Module([node], []), str(engine_path), "exec"), ns)
    for k in ("INTENT_PATTERNS", "COUNSELING_INTENTS", "GENERIC_DISTRESS_PATTERNS", "detect_intent"):
        if k not in ns:
            sys.exit(f"Could not extract {k} from {engine_path}")
    return ns["detect_intent"], ns["COUNSELING_INTENTS"]


def read_history_rows(db_path: Path):
    """Same query as app.py::_warm_start_bandit (reward NOT NULL, ORDER BY created_at)."""
    uri = db_path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT emotion, stress_type, strategy, reward FROM strategy_logs "
            "WHERE reward IS NOT NULL ORDER BY created_at"
        ).fetchall()
        total = con.execute("SELECT COUNT(*) FROM strategy_logs").fetchone()[0]
    finally:
        con.close()
    return [(e, s, st, float(r)) for e, s, st, r in rows], total


def read_history_csv(csv_path: Path):
    """Reads an export of strategy_logs (already filtered to reward NOT NULL and
    ordered by created_at - see the export command in the instructions)."""
    text = csv_path.read_text(encoding="utf-8-sig")
    rdr = csv.DictReader(io.StringIO(text))
    need = {"emotion", "stress_type", "strategy", "reward"}
    if not rdr.fieldnames or not need.issubset(set(rdr.fieldnames)):
        sys.exit(f"History CSV must have columns {sorted(need)}; found {rdr.fieldnames}")
    rows = []
    for r in rdr:
        if r["reward"] in ("", None):
            continue  # same rule as app.py: only rows with a reward are replayed
        rows.append((r["emotion"], r["stress_type"], r["strategy"], float(r["reward"])))
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(ss, emotion, stress):
    """Returns (strategy, margin_top_minus_second, is_tie). Read-only."""
    chosen, scores = ss.select_strategy(emotion, stress)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    return chosen, margin, margin < TIE_EPS


def summarize(records, strategies):
    n = len(records)
    changed = [r for r in records if r["changed"]]
    return {
        "n_messages": n,
        "agree": n - len(changed),
        "agreement_pct": round(100 * (n - len(changed)) / n, 1) if n else None,
        "changed": len(changed),
        "changed_pct": round(100 * len(changed) / n, 1) if n else None,
        "context_identical_so_strategy_cannot_differ": sum(
            1 for r in records if (r["ref_emotion"], r["ref_stress"]) == (r["pred_emotion"], r["pred_stress"])
        ),
        "dist_reference": {s: sum(1 for r in records if r["ref_strategy"] == s) for s in strategies},
        "dist_predicted": {s: sum(1 for r in records if r["pred_strategy"] == s) for s in strategies},
        "transitions_ref_to_pred": dict(Counter(f'{r["ref_strategy"]} -> {r["pred_strategy"]}' for r in changed)),
        "ties_reference_run": sum(1 for r in records if r["ref_tie"]),
        "ties_predicted_run": sum(1 for r in records if r["pred_tie"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--db", help="path to a COPY of a SQLite DB (use this OR --history-csv)")
    ap.add_argument("--history-csv", help="CSV export of strategy_logs (PostgreSQL users)")
    ap.add_argument("--total-logs", type=int, default=None,
                    help="optional: total rows in strategy_logs incl. unrated, for the report")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    if bool(a.db) == bool(a.history_csv):
        sys.exit("Give exactly one of --db or --history-csv")
    app_dir, csv_path, out_dir = (Path(x).resolve() for x in (a.app_dir, a.csv, a.out_dir))
    hist_src = Path(a.db or a.history_csv).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    detect_intent, counseling_intents = load_routing(app_dir / "chatbot_engine.py")

    # Import only the two light modules, with the app dir on the path.
    cwd = os.getcwd()
    sys.path.insert(0, str(app_dir))
    os.chdir(app_dir)
    try:
        import strategy_selector as ss
        import inference
    finally:
        os.chdir(cwd)

    # ---- restore the application bandit state from history ----
    if a.db:
        hist_rows, total_logs = read_history_rows(hist_src)
    else:
        hist_rows, total_logs = read_history_csv(hist_src), a.total_logs
    n_replayed = ss.warm_start(hist_rows)
    snapshot_A = copy.deepcopy(ss._A)
    snapshot_b = copy.deepcopy(ss._b)

    def _forbidden(*args, **kwargs):
        raise RuntimeError("record_feedback() called during the experiment - bandit must stay frozen")

    ss.record_feedback = _forbidden  # warm_start already finished; any later call is a bug

    reward_counts = Counter(r[3] for r in hist_rows)
    strategy_counts = Counter(r[2] for r in hist_rows)

    # ---- load CSV ----
    text = csv_path.read_text(encoding="utf-8-sig")
    data = list(csv.DictReader(io.StringIO(text)))

    records = []
    for row in data:
        msg = row["message"]
        ref_e, ref_s = row["emotion"], row["stress_type"]
        pred = inference.infer_emotional_state(msg)  # history is ignored by this function
        pred_e, pred_s = pred["emotion"], pred["stress_type"]
        crisis = bool(inference.check_crisis_risk(msg))
        intent = detect_intent(msg)
        routed = (not crisis) and intent in counseling_intents

        rs, rm, rt = pick(ss, ref_e, ref_s)
        ps, pm, pt = pick(ss, pred_e, pred_s)
        records.append({
            "id": row["id"], "language": row["language"], "message": msg,
            "ref_emotion": ref_e, "ref_stress": ref_s,
            "pred_emotion": pred_e, "pred_stress": pred_s,
            "emotion_correct": ref_e == pred_e, "stress_correct": ref_s == pred_s,
            "crisis_flag": crisis, "intent": intent, "reaches_linucb": routed,
            "ref_strategy": rs, "ref_margin": round(rm, 6), "ref_tie": rt,
            "pred_strategy": ps, "pred_margin": round(pm, 6), "pred_tie": pt,
            "changed": rs != ps,
        })

    # ---- integrity: bandit untouched ----
    for s in ss.STRATEGIES:
        assert np.array_equal(snapshot_A[s], ss._A[s]) and np.array_equal(snapshot_b[s], ss._b[s]), \
            f"Bandit state changed for {s}!"

    # ---- sanity checks against previously reported numbers ----
    n = len(records)
    emo = sum(r["emotion_correct"] for r in records)
    stress = sum(r["stress_correct"] for r in records)
    joint = sum(r["emotion_correct"] and r["stress_correct"] for r in records)
    routed_n = sum(r["reaches_linucb"] for r in records)
    checks = {
        "n": (n, EXPECTED["n"]), "emotion_correct": (emo, EXPECTED["emotion"]),
        "stress_correct": (stress, EXPECTED["stress"]), "joint_correct": (joint, EXPECTED["joint"]),
        "reaches_linucb": (routed_n, EXPECTED["routed"]),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "history_source": str(hist_src.name),
        "history_source_sha256": sha256(hist_src),
        "csv_sha256": sha256(csv_path),
        "numpy_version": np.__version__,
        "bandit_state": {
            "strategy_logs_total_rows": total_logs,
            "rows_with_reward": len(hist_rows),
            "rows_replayed_by_warm_start": n_replayed,
            "reward_counts": {str(k): v for k, v in reward_counts.items()},
            "replayed_per_strategy": dict(strategy_counts),
            "looks_untrained": len(hist_rows) == 0,
        },
        "sanity_vs_previously_reported": {k: {"observed": v[0], "expected": v[1]} for k, v in checks.items()},
        "sanity_mismatches": {k: {"observed": v[0], "expected": v[1]} for k, v in mismatches.items()},
        "all_144": summarize(records, ss.STRATEGIES),
        "reaching_linucb_only": summarize([r for r in records if r["reaches_linucb"]], ss.STRATEGIES),
        "crisis_flagged_in_csv": sum(r["crisis_flag"] for r in records),
        "bandit_state_unchanged_assertion": "passed",
    }

    with open(out_dir / "per_message.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "changed_ids.txt").write_text(
        "\n".join(r["id"] for r in records if r["changed"]), encoding="utf-8")

    # ---- console report ----
    print(f"Bandit restored: {n_replayed} rows replayed ({total_logs} total logs, {len(hist_rows)} with reward)")
    if not hist_rows:
        print("WARNING: no feedback rows -> bandit is in its untrained state. Do NOT describe it as trained.")
    if mismatches:
        print("WARNING: sanity mismatches vs previously reported numbers:", mismatches)
    else:
        print("Sanity: counts match previously reported (44/73/22 and 77 routed).")
    for name in ("all_144", "reaching_linucb_only"):
        s = summary[name]
        print(f"\n[{name}] n={s['n_messages']}  agree={s['agree']} ({s['agreement_pct']}%)  "
              f"changed={s['changed']} ({s['changed_pct']}%)  ties(ref/pred)={s['ties_reference_run']}/{s['ties_predicted_run']}")
        print("  reference dist:", s["dist_reference"])
        print("  predicted dist:", s["dist_predicted"])
    print(f"\nWrote: {out_dir/'per_message.csv'}, summary.json, changed_ids.txt")


if __name__ == "__main__":
    main()