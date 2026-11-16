"""
Crisis-routing and context-detection evaluation (response to reviewer comment 6)
================================================================================

CPU-only: it calls your RULE-BASED detectors, so it runs in seconds and needs no GPU/LLM.
Place it in backend/Experiment/ next to crisis_eval_set.csv and rag_ablation_generations.csv.

STEP 1 (once): edit predict_crisis() and predict_context() below so they call your real
               detectors. Check the wiring first with:  python crisis_context_eval.py --demo
STEP 2:        python crisis_context_eval.py            (both parts)
               python crisis_context_eval.py --crisis   (crisis set only)
               python crisis_context_eval.py --context  (144 evaluation messages only)

Part A  Crisis routing on an independent labelled set (crisis_eval_set.csv):
        TP / FN / FP / TN, sensitivity, specificity, precision with Wilson 95% CIs,
        overall, per language, per category and per dev/test half; every error is listed.
Part B  Context detection on the 144 evaluation messages: detector PREDICTIONS compared
        with the REFERENCE labels used to generate the messages (accuracy per language,
        per-class recall, confusion matrices, number of wrong context-vector coordinates),
        plus the crisis false-positive rate on those 144 non-crisis messages.
        It writes context_detection_predictions.csv, which you can feed back into the
        generation pipeline to measure how detection errors affect generation.

Independence note: crisis_eval_set.csv was written without access to the detector rules.
Do not tune the rules on the "test" half. If you revise rules after inspecting "dev"
errors, say so in the paper and report the final numbers on "test" only.
"""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CRISIS_SET = HERE / "crisis_eval_set.csv"
ABLATION = HERE / "rag_ablation_generations.csv"
OUT_CRISIS_SUMMARY = HERE / "crisis_eval_summary.txt"
OUT_CRISIS_PRED = HERE / "crisis_eval_predictions.csv"
OUT_CRISIS_ERRORS = HERE / "crisis_eval_errors.csv"
OUT_CONTEXT_SUMMARY = HERE / "context_detection_summary.txt"
OUT_CONTEXT_PRED = HERE / "context_detection_predictions.csv"
OUT_CONF_EMO = HERE / "context_confusion_emotion.csv"
OUT_CONF_STRESS = HERE / "context_confusion_stress.csv"


# =============================================================================
# EDIT THESE TWO FUNCTIONS TO CALL YOUR DETECTORS
# =============================================================================
# Example (adapt module/function names to your backend):
#     import sys; sys.path.insert(0, str(HERE.parent))
#     from services.context_detector import detect_crisis, detect_emotion, detect_stress
#
# lang is "en", "ko" or "mixed" (code-switched). If your detector auto-detects the
# language, ignore it; if it needs "en"/"ko", map "mixed" the same way production does.

def predict_crisis(text: str, lang: str) -> bool:
    """Return True if the production crisis detector would route this message to the
    fixed crisis response."""
    raise NotImplementedError("Connect predict_crisis() to your crisis detector.")


def predict_context(text: str, lang: str) -> tuple[str, str]:
    """Return (emotion, stress_type) exactly as the production classifier outputs them."""
    raise NotImplementedError("Connect predict_context() to your emotion/stress classifier.")

# =============================================================================


def _demo_crisis(text: str, lang: str) -> bool:      # toy detector, pipeline test only
    t = text.lower()
    return any(k in t for k in ("kill myself", "end my life", "want to die", "죽고 싶", "자살"))


def _demo_context(text: str, lang: str) -> tuple[str, str]:  # toy classifier, pipeline test only
    t = text.lower()
    return ("anxious" if ("anxi" in t or "불안" in t) else "sad",
            "academic" if ("exam" in t or "시험" in t) else "career")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _fmt(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}] ({k}/{n})"


def crisis_line(name: str, g: pd.DataFrame) -> str:
    tp = int(((g.label == 1) & (g.pred == 1)).sum())
    fn = int(((g.label == 1) & (g.pred == 0)).sum())
    fp = int(((g.label == 0) & (g.pred == 1)).sum())
    tn = int(((g.label == 0) & (g.pred == 0)).sum())
    return (f"{name}: n={len(g)} (crisis={tp + fn}, non-crisis={fp + tn}) | TP={tp} FN={fn} FP={fp} TN={tn} | "
            f"sensitivity={_fmt(tp, tp + fn)} | specificity={_fmt(tn, tn + fp)} | "
            f"precision={_fmt(tp, tp + fp)}")


def run_crisis(crisis_fn) -> None:
    if not CRISIS_SET.exists():
        raise FileNotFoundError(f"Missing {CRISIS_SET.name}")
    df = pd.read_csv(CRISIS_SET, dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    df["split"] = df["id"].map(lambda s: "dev" if zlib.crc32(s.encode("utf-8")) % 2 == 0 else "test")
    df["pred"] = [int(bool(crisis_fn(t, l))) for t, l in zip(df["text"], df["language"])]
    df["outcome"] = df.apply(lambda r: ("TP" if r.pred else "FN") if r.label else ("FP" if r.pred else "TN"), axis=1)
    df.to_csv(OUT_CRISIS_PRED, index=False, encoding="utf-8-sig")
    df[df["outcome"].isin(["FN", "FP"])].sort_values(["outcome", "language", "category"]).to_csv(
        OUT_CRISIS_ERRORS, index=False, encoding="utf-8-sig")

    lines = ["CRISIS ROUTING EVALUATION (independent labelled set)",
             f"Messages: {len(df)}; crisis={int((df.label == 1).sum())}, non-crisis={int((df.label == 0).sum())}",
             "Counts by language and label:"]
    for lang, g in df.groupby("language"):
        lines.append(f"  {lang}: crisis={int((g.label == 1).sum())}, non-crisis={int((g.label == 0).sum())}")
    lines += ["", crisis_line("ALL", df), "", "Per language:"]
    lines += [crisis_line(f"  {lang}", g) for lang, g in df.groupby("language")]
    lines += ["", "Per category (crisis categories -> sensitivity; non-crisis categories -> specificity):"]
    lines += [crisis_line(f"  {cat}", g) for cat, g in df.groupby("category")]
    lines += ["", "Per language x category:"]
    lines += [crisis_line(f"  {lang}/{cat}", g) for (lang, cat), g in df.groupby(["language", "category"])]
    lines += ["", "Per split (report 'test' as the final numbers if rules were revised after 'dev'):"]
    lines += [crisis_line(f"  {s}", g) for s, g in df.groupby("split")]

    errs = df[df["outcome"].isin(["FN", "FP"])]
    lines += ["", f"ERRORS ({len(errs)}): false negatives are the safety-critical failure."]
    for _, r in errs.sort_values(["outcome", "language"]).iterrows():
        lines.append(f"  [{r.outcome}] {r.id} ({r.language}/{r.category}, {r.split}): {r.text}")
    lines += ["", "Interpretation: a missed crisis message (FN) bypasses the crisis pathway entirely; "
                  "report sensitivity per category, not only overall. Confidence intervals are wide "
                  "for small n; state the sample size next to every rate."]
    OUT_CRISIS_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def run_context(crisis_fn, context_fn) -> None:
    if not ABLATION.exists():
        raise FileNotFoundError(f"Missing {ABLATION.name}")
    data = pd.read_csv(ABLATION, dtype=str, keep_default_na=False)
    df = data[data["condition"].str.lower() == "rag_on"].copy().reset_index(drop=True)
    need = {"id", "language", "message", "emotion", "stress_type", "strategy"}
    if need - set(df.columns):
        raise ValueError(f"Ablation file missing columns: {sorted(need - set(df.columns))}")
    df["language"] = df["language"].str.lower()

    preds = [context_fn(m, l) for m, l in zip(df["message"], df["language"])]
    df["pred_emotion"] = [str(p[0]).strip().lower() for p in preds]
    df["pred_stress"] = [str(p[1]).strip().lower() for p in preds]
    df["ref_emotion"] = df["emotion"].str.strip().str.lower()
    df["ref_stress"] = df["stress_type"].str.strip().str.lower()
    df["emotion_match"] = df["pred_emotion"] == df["ref_emotion"]
    df["stress_match"] = df["pred_stress"] == df["ref_stress"]
    df["joint_match"] = df["emotion_match"] & df["stress_match"]
    df["wrong_coordinates"] = (~df["emotion_match"]).astype(int) + (~df["stress_match"]).astype(int)
    df["crisis_false_positive"] = [int(bool(crisis_fn(m, l))) for m, l in zip(df["message"], df["language"])]
    df[["id", "language", "message", "ref_emotion", "ref_stress", "pred_emotion", "pred_stress",
        "strategy", "emotion_match", "stress_match", "joint_match", "wrong_coordinates",
        "crisis_false_positive"]].to_csv(OUT_CONTEXT_PRED, index=False, encoding="utf-8-sig")
    pd.crosstab(df["ref_emotion"], df["pred_emotion"]).to_csv(OUT_CONF_EMO, encoding="utf-8-sig")
    pd.crosstab(df["ref_stress"], df["pred_stress"]).to_csv(OUT_CONF_STRESS, encoding="utf-8-sig")

    lines = ["CONTEXT DETECTION: PREDICTIONS vs REFERENCE LABELS (144 evaluation messages)",
             "Reference labels = the emotion/stress combination each message was generated for; they are "
             "NOT independently human-verified unless you add that check.", ""]
    for name, g in [("ALL", df)] + [(l, x) for l, x in df.groupby("language")]:
        n = len(g)
        lines.append(f"{name}: n={n} | emotion acc={_fmt(int(g.emotion_match.sum()), n)} | "
                     f"stress acc={_fmt(int(g.stress_match.sum()), n)} | "
                     f"joint acc={_fmt(int(g.joint_match.sum()), n)}")
    lines += ["", "Wrong context-vector coordinates per message (0 = identical vector; 2 = both wrong):"]
    for name, g in [("ALL", df)] + [(l, x) for l, x in df.groupby("language")]:
        vc = g["wrong_coordinates"].value_counts().reindex([0, 1, 2], fill_value=0)
        lines.append(f"  {name}: 0 wrong={vc[0]}, 1 wrong={vc[1]}, 2 wrong={vc[2]}")
    lines += ["", "Per-class recall (emotion):"]
    for cls, g in df.groupby("ref_emotion"):
        lines.append(f"  {cls}: {_fmt(int(g.emotion_match.sum()), len(g))}")
    lines += ["Per-class recall (stress):"]
    for cls, g in df.groupby("ref_stress"):
        lines.append(f"  {cls}: {_fmt(int(g.stress_match.sum()), len(g))}")
    fp = int(df["crisis_false_positive"].sum())
    lines += ["", f"Crisis detector false positives on these {len(df)} non-crisis messages: {fp} "
                  f"(by language: {df.groupby('language')['crisis_false_positive'].sum().to_dict()})",
              "", "NEXT (error propagation, not computed here):",
              " 1. Generation: regenerate 'Our system' for the same 144 messages using pred_emotion/pred_stress "
              "(from context_detection_predictions.csv) instead of the reference labels, keep the intended strategy, "
              "re-judge adherence, and compare with the reference-label result (McNemar).",
              " 2. Bandit: in the simulation, let LinUCB observe the context sampled from the measured confusion "
              "matrices (context_confusion_*.csv) while the simulated preferences depend on the TRUE context; "
              "report the advantage over the best fixed strategy as in the main simulation."]
    OUT_CONTEXT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crisis", action="store_true", help="Run Part A only.")
    ap.add_argument("--context", action="store_true", help="Run Part B only.")
    ap.add_argument("--demo", action="store_true", help="Use toy detectors to test the pipeline.")
    a = ap.parse_args()
    crisis_fn, context_fn = (_demo_crisis, _demo_context) if a.demo else (predict_crisis, predict_context)
    both = not (a.crisis or a.context)
    if a.demo:
        print("*** DEMO MODE: toy detectors; numbers are meaningless. ***\n")
    if a.crisis or both:
        run_crisis(crisis_fn)
        print()
    if a.context or both:
        run_context(crisis_fn, context_fn)


if __name__ == "__main__":
    main()
