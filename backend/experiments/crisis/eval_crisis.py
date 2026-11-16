"""
eval_crisis.py - R1.6 evaluation: crisis detection, routing, context detection.

Run from the backend folder (next to inference.py, chatbot_engine.py, crisis_detector.py):
    python crisis\\eval_crisis.py crisis_eval_set_v1.csv crisis_eval_set_v2_heldout.csv --eval144 Experiment\\prompts_v2.csv

v1 set = development set (v2 patterns were designed after seeing its results).
v2 held-out set = the number to report. llm/rag are stubbed; nothing is written to the DB.
"""
import sys, re, types, argparse, math, os
import pandas as pd

# Works from backend\ or from backend\crisis\ : modules are imported from backend,
# CSVs and outputs live next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "crisis" else HERE
sys.path.insert(0, BACKEND)
def resolve(p): return p if os.path.exists(p) else os.path.join(HERE, p)
def out(name): return os.path.join(HERE, name)

llm_stub = types.ModuleType("llm")
class LLMConnectionError(Exception): pass
llm_stub.LLMConnectionError = LLMConnectionError
llm_stub.generate_response = lambda messages, max_tokens=200: "[LLM_STUB]"
rag_stub = types.ModuleType("rag"); rag_stub.retrieve = lambda q, k=3, category=None: []
sys.modules["llm"], sys.modules["rag"] = llm_stub, rag_stub

import inference, chatbot_engine, strategy_selector, prompt_builder, crisis_detector

calls = {"select": 0}
_orig = strategy_selector.select_strategy
def _counted(*a, **k):
    calls["select"] += 1
    return _orig(*a, **k)
strategy_selector.select_strategy = _counted

OUT = []
def say(s=""): print(s); OUT.append(s)

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0.0, c-h), min(1.0, c+h)
def fmt(k, n):
    if not n: return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n}={100*k/n:.1f}% (CI {100*lo:.1f}-{100*hi:.1f})"
def has_hangul(t): return bool(re.search(r"[\uac00-\ud7a3]", t))

def conf(sub, col):
    tp = int(((sub.label=="crisis") & sub[col]).sum()); fn = int(((sub.label=="crisis") & ~sub[col]).sum())
    fp = int(((sub.label=="non_crisis") & sub[col]).sum()); tn = int(((sub.label=="non_crisis") & ~sub[col]).sum())
    return tp, fn, fp, tn
def block(name, sub, col):
    tp, fn, fp, tn = conf(sub, col)
    say(f"  {name:<9} TP={tp:<2} FN={fn:<2} FP={fp:<2} TN={tn:<2} | sens {fmt(tp,tp+fn)} | spec {fmt(tn,tn+fp)}")

def detectors():
    v1 = getattr(inference, "check_crisis_risk_v1", inference.check_crisis_risk)
    def v2_noguard(m):
        old = crisis_detector.USE_GUARD; crisis_detector.USE_GUARD = False
        try: return crisis_detector.check_crisis_risk(m)
        finally: crisis_detector.USE_GUARD = old
    def v2(m):
        old = crisis_detector.USE_GUARD; crisis_detector.USE_GUARD = True
        try: return crisis_detector.check_crisis_risk(m)
        finally: crisis_detector.USE_GUARD = old
    return {"v1_original": v1, "v2_no_guard": v2_noguard, "v2_final": v2}

def part_a(path, tag):
    df = pd.read_csv(resolve(path))
    say("="*74); say(f"PART A [{tag}] {os.path.basename(path)}  (crisis={int((df.label=='crisis').sum())}, non-crisis={int((df.label=='non_crisis').sum())})"); say("="*74)
    for name, fn in detectors().items():
        df[name] = df.text.map(fn)
        say(f"-- detector: {name}")
        block("OVERALL", df, name)
        for l in ["en","ko","cs"]: block(f"lang={l}", df[df.language==l], name)
    final = "v2_final"
    say("\nBy category, v1_original -> v2_final (crisis cats: recall; non-crisis cats: false-positive rate):")
    for cat, sub in df.groupby("category"):
        a = conf(sub, "v1_original"); b = conf(sub, final)
        if sub.label.iloc[0]=="crisis":
            say(f"  {cat:<24} recall  {fmt(a[0],a[0]+a[1])}  ->  {fmt(b[0],b[0]+b[1])}")
        else:
            say(f"  {cat:<24} FP rate {fmt(a[2],a[2]+a[3])}  ->  {fmt(b[2],b[2]+b[3])}")
    say("\nv2_final FALSE NEGATIVES:")
    for _, r in df[(df.label=="crisis") & ~df[final]].iterrows(): say(f"  {r.id} [{r.language}/{r.category}] {r.text}")
    say("v2_final FALSE POSITIVES:")
    for _, r in df[(df.label=="non_crisis") & df[final]].iterrows(): say(f"  {r.id} [{r.language}/{r.category}] {r.text}")
    return df

def part_b(df, tag):
    say("\n"+"-"*74); say(f"PART B [{tag}] end-to-end routing via chatbot_engine.handle_message (installed detector)"); say("-"*74)
    rows = []
    for _, r in df.iterrows():
        calls["select"] = 0
        res = chatbot_engine.handle_message(r.text, language="en")      # app default language
        expect = prompt_builder.get_crisis_response(crisis_detector.crisis_response_language(r.text))
        rows.append(dict(id=r.id, language=r.language, label=r.label, category=r.category, text=r.text,
                         intent=res["intent"], strategy=res["strategy"], bandit_calls=calls["select"],
                         response_ok=(res["response"]==expect)))
    R = pd.DataFrame(rows)
    cr, nc = R[R.label=="crisis"], R[R.label=="non_crisis"]
    routed = cr[cr.intent=="crisis"]
    ok = ((routed.bandit_calls==0) & routed.strategy.isna() & routed.response_ok).sum()
    say(f"Crisis messages sent to the fixed-response pathway: {fmt(len(routed), len(cr))}")
    say(f"  of those: no bandit call, no strategy, correct-language fixed response: {fmt(int(ok), len(routed))}")
    say(f"Non-crisis messages wrongly sent to the crisis pathway: {fmt(int((nc.intent=='crisis').sum()), len(nc))}")
    miss = cr[cr.intent!="crisis"]
    if len(miss):
        say(f"Missed crisis messages ({len(miss)}) went to: {miss.intent.value_counts().to_dict()}; reached bandit: {int((miss.bandit_calls>0).sum())}")
    say(f"Non-crisis messages reaching the bandit: {fmt(int((nc.bandit_calls>0).sum()), len(nc))}; by intent: {nc.intent.value_counts().to_dict()}")
    R.to_csv(out(f"routing_{tag}.csv"), index=False, encoding="utf-8-sig")

def part_c(path, tcol, ecol, scol, lcol):
    say("\n"+"="*74); say("PART C - emotion/stress detection on the evaluation messages"); say("="*74)
    d = pd.read_csv(resolve(path))
    for c in (tcol, ecol, scol):
        if c not in d.columns:
            say(f"Column '{c}' not found; columns are {list(d.columns)}. Use --text/--emotion/--stress/--lang."); return
    if lcol not in d.columns: d[lcol] = d[tcol].map(lambda t: "ko" if has_hangul(str(t)) else "en")
    inf = d[tcol].map(lambda t: inference.infer_emotional_state(str(t)))
    d["pred_emotion"] = inf.map(lambda x: x["emotion"]); d["pred_stress"] = inf.map(lambda x: x["stress_type"])
    d["emo_ok"] = d.pred_emotion==d[ecol]; d["str_ok"] = d.pred_stress==d[scol]; d["both_ok"] = d.emo_ok & d.str_ok
    d["crisis_flag"] = d[tcol].map(lambda t: crisis_detector.check_crisis_risk(str(t)))
    d["intent"] = d[tcol].map(lambda t: chatbot_engine.detect_intent(str(t)))
    d["reaches_bandit"] = d.intent.isin(chatbot_engine.COUNSELING_INTENTS) & ~d.crisis_flag
    for name, sub in [("OVERALL", d)] + [(f"lang={l}", s) for l, s in d.groupby(lcol)]:
        n = len(sub); say(f"{name:<10} n={n} emotion {fmt(int(sub.emo_ok.sum()),n)} | stress {fmt(int(sub.str_ok.sum()),n)} | joint {fmt(int(sub.both_ok.sum()),n)}")
    say(f"Flagged as crisis (all are non-crisis messages): {fmt(int(d.crisis_flag.sum()), len(d))}")
    say(f"Reach the bandit through intent routing: {fmt(int(d.reaches_bandit.sum()), len(d))}; intents: {d.intent.value_counts().to_dict()}")
    say("\nEmotion confusion (rows=reference, cols=detected):"); say(pd.crosstab(d[ecol], d.pred_emotion).to_string())
    say("\nStress confusion (rows=reference, cols=detected):"); say(pd.crosstab(d[scol], d.pred_stress).to_string())
    d.to_csv(out("context_detection_144.csv"), index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csvs", nargs="+")
    ap.add_argument("--eval144"); ap.add_argument("--text", default="message")
    ap.add_argument("--emotion", default="emotion"); ap.add_argument("--stress", default="stress_type"); ap.add_argument("--lang", default="language")
    a = ap.parse_args()
    for i, p in enumerate(a.csvs):
        tag = "dev" if i == 0 else "heldout"
        part_b(part_a(p, tag), tag)
    if a.eval144: part_c(a.eval144, a.text, a.emotion, a.stress, a.lang)
    open(out("crisis_eval_summary.txt"), "w", encoding="utf-8").write("\n".join(OUT))
