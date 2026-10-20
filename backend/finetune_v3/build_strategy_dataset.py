"""
Builds strategy-conditioned training data for nova-counseling-v3.

Every example is built with Nova's REAL prompt_builder.build_counseling_prompt(), so the
training prompt is identical to what the model sees at inference. Run from backend/finetune_v3/.

Order of use:
  1) python build_strategy_dataset.py testset  --per-context 3     (new held-out test set, FIRST)
  2) python build_strategy_dataset.py esconv   --esconv ESConv.json
  3) python build_strategy_dataset.py synth    --msgs-per-context 15
  4) python build_strategy_dataset.py merge
Outputs: strategy_train.jsonl, strategy_dev.jsonl  (chat format: system/user/assistant)

Test messages (prompts.csv, prompts_v2.csv, ...) are excluded from all training data.
Teacher/judge models are listed below (pull qwen2.5:7b and gemma3:4b first); the EVALUATION judge (phi4) is deliberately not used
for filtering, so the model is not trained toward the evaluation judge's quirks.
"""
import argparse, glob, hashlib, itertools, json, os, random, re, sys, time
import requests, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # backend/ (prompt_builder, rag)
sys.path.insert(0, HERE)
import prompt_builder, rag
from strategy_checks import (STRATEGY_MAP, rule_check, map_emotion, map_stress, language_ok,
                             is_near_duplicate, norm)

OLLAMA = "http://localhost:11434"
MESSAGE_WRITER = "gemma3:4b"
TEACHER = {"en": "qwen2.5:7b", "ko": "qwen2.5:7b"}   # non-thinking instruct model; run: ollama pull qwen2.5:7b
DATA_JUDGE = {"en": "gemma3:4b", "ko": "gemma3:4b"}  # different model family from the teacher

EMOTIONS = ["anxious", "overwhelmed", "sad", "angry", "hopeful", "neutral"]
STRESS = ["academic", "career", "interpersonal", "general"]
STRATS = ["validate_listen", "validate_ask", "validate_suggest", "validate_normalize"]
TOPICS = {
    "academic": ["an exam", "a group project", "a low grade", "a thesis or paper deadline", "a difficult course",
                 "falling behind in lectures", "feedback from a professor", "study habits", "choosing classes"],
    "career": ["job applications", "an internship", "not knowing what to do after graduation", "a resume",
               "an interview", "comparing myself with classmates' offers", "choosing a major or direction"],
    "interpersonal": ["a roommate", "a close friend", "a partner", "a club or team", "family expectations",
                      "feeling lonely on campus", "a conflict in a group"],
    "general": ["sleep", "motivation", "homesickness", "money worries", "daily routine", "feeling stuck",
                "self-confidence", "free time"],
}
EMOTION_HINT = {
    "anxious": "worried and nervous", "overwhelmed": "overwhelmed, with too much to handle",
    "sad": "sad and low", "angry": "angry or frustrated", "hopeful": "hopeful and looking forward",
    "neutral": "calm; asking a practical question rather than venting",
}

RUBRIC = """You are a strict evaluator of counseling-style chatbot replies for university students.
You see the student message and ONE reply. Answer:
1) strategy_used: which single strategy does the reply actually follow?
 - validate_listen: validates feelings only; no advice and no question
 - validate_ask: validates, then asks a follow-up question
 - validate_suggest: validates, then gives a concrete suggestion or step
 - validate_normalize: validates, then says many people/students feel this way
2) made_up_facts (true/false): true if it states specific facts/statistics/names not common knowledge.
3) unsafe (true/false): true if it gives harmful advice, diagnoses the student, or dismisses serious distress.
Return ONLY JSON: {"strategy_used":"...","made_up_facts":false,"unsafe":false}"""

def chat(model, messages, temperature=0.7, num_predict=250, fmt=None):
    if model.startswith("qwen3"):   # qwen3 reasons out loud unless told not to
        messages = [dict(m) for m in messages]; messages[-1]["content"] += " /no_think"
    body = {"model": model, "stream": False, "think": False, "messages": messages,
            "options": {"temperature": temperature, "num_predict": num_predict}}
    if fmt: body["format"] = fmt
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

JUDGE_ERRORS = {"n": 0}
def judge_ok(lang, message, reply, target):
    try:
        txt = chat(DATA_JUDGE[lang], [{"role": "system", "content": RUBRIC},
                   {"role": "user", "content": f"Student message:\n{message}\n\nReply:\n{reply}"}],
                   temperature=0, num_predict=120, fmt="json")
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        return d.get("strategy_used") == target and not d.get("made_up_facts") and not d.get("unsafe")
    except Exception as e:
        JUDGE_ERRORS["n"] += 1
        if JUDGE_ERRORS["n"] <= 3:
            print(f"  JUDGE ERROR ({DATA_JUDGE[lang]}): {type(e).__name__}: {str(e)[:200]}")
        return False

def write_message(emotion, stress, lang):
    topic = random.choice(TOPICS[stress])
    language = "Korean" if lang == "ko" else "English"
    prompt = (f"Write ONE realistic message (1-3 sentences, casual, first person) that a university student "
              f"might send to a supportive chatbot. Feeling: {EMOTION_HINT[emotion]}. Context: {stress} concerns, "
              f"about {topic}. Language: {language}. Vary the wording; do not name the feeling directly. "
              f"Output only the message, no quotes.")
    out = chat(MESSAGE_WRITER, [{"role": "user", "content": prompt}], temperature=1.0, num_predict=120)
    out = out.strip().strip('"“”').split("\n")[0].strip()
    return out if 6 <= len(out) <= 300 and language_ok(out, lang) else None

def load_exclusions(paths):
    toks = []
    for p in paths:
        if os.path.exists(p):
            toks += [set(norm(m)) for m in pd.read_csv(p)["message"].dropna()]
        else:
            print(f"  (exclusion file not found: {p})")
    print(f"Excluding {len(toks)} test messages from training data.")
    return toks

def split_of(key):
    return "dev" if int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 20 == 0 else "train"

def build_example(msg, emotion, stress, lang, strategy, rag_prob):
    chunks = rag.retrieve(msg, k=3) if random.random() < rag_prob else []
    messages = prompt_builder.build_counseling_prompt(
        user_message=msg, history_summary=None, retrieved_chunks=chunks,
        emotion_info={"emotion": emotion, "stress_type": stress}, trend_info=None,
        language=lang, strategy=strategy)
    return messages

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ------------------------------------------------------------------ testset
def cmd_testset(a):
    excl = load_exclusions(a.exclude)
    rows, seen = [], list(excl)
    for lang in ("en", "ko"):
        for emotion, stress in itertools.product(EMOTIONS, STRESS):
            got, tries = 0, 0
            while got < a.per_context and tries < 12:
                tries += 1
                try: m = write_message(emotion, stress, lang)
                except Exception as e: print("error:", e); continue
                if not m or is_near_duplicate(m, seen): continue
                seen.append(set(norm(m))); got += 1
                rows.append(dict(language=lang, emotion=emotion, stress_type=stress, message=m))
            print(f"{lang} {emotion}/{stress}: {got}")
    random.Random(42).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = 1000 + i; r["strategy"] = STRATS[i % 4]
    df = pd.DataFrame(rows)[["id", "language", "emotion", "stress_type", "strategy", "message"]]
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {a.out}: {len(df)} messages. REVIEW the messages by hand before using them.")

# ------------------------------------------------------------------ esconv
def cmd_esconv(a):
    data = json.load(open(a.esconv, encoding="utf-8"))
    excl = load_exclusions(a.exclude)
    random.seed(42)
    for p in ("esconv_train.jsonl", "esconv_dev.jsonl"):
        if os.path.exists(p): os.remove(p)
    kept, seen = {s: 0 for s in STRATS}, set()
    for ci, conv in enumerate(data):
        emotion, stress = map_emotion(conv["emotion_type"]), map_stress(conv["problem_type"])
        d = conv["dialog"]
        for j in range(2, len(d)):
            t, prev = d[j], d[j - 1]
            if t["speaker"] != "supporter" or prev["speaker"] != "seeker": continue
            strat = STRATEGY_MAP.get(t.get("annotation", {}).get("strategy"))
            if not strat: continue
            msg = " ".join(prev["content"].split()); reply = " ".join(t["content"].split())
            if not (6 <= len(msg.split()) <= 120): continue
            if not rule_check(strat, reply, "en"): continue
            if (msg, strat) in seen or is_near_duplicate(msg, excl): continue
            seen.add((msg, strat))
            messages = build_example(msg, emotion, stress, "en", strat, a.rag_prob)
            messages.append({"role": "assistant", "content": reply})
            append_jsonl(f"esconv_{split_of(ci)}.jsonl", dict(messages=messages, strategy=strat,
                         language="en", source="esconv"))
            kept[strat] += 1
    print("ESConv examples kept per strategy:", kept)

# ------------------------------------------------------------------ synth
TEACHER_HINT = {
    "validate_listen": "Reply with 2-3 sentences that only reflect and validate their feelings in specific words. Do NOT give any advice, suggestion or tip, and do NOT ask any question.",
    "validate_ask": "Reply with one sentence validating their feelings, then ask exactly ONE gentle question as the final sentence. Do NOT give any advice, suggestion or tip.",
    "validate_suggest": "Reply with one sentence validating their feelings, then give exactly ONE small, concrete next step they could try. Do NOT ask any question.",
    "validate_normalize": "Reply with one sentence validating their feelings, then say this is a common, understandable reaction that many students experience. Do NOT give any advice or suggestion and do NOT ask any question.",
}

def cmd_synth(a):
    """The teacher sees the real inference prompt PLUS a stronger one-line instruction (TEACHER_HINT).
    The saved training example keeps only the real inference prompt, so the student learns to follow
    the strategy from the normal prompt alone."""
    excl = load_exclusions(a.exclude)
    random.seed(a.seed)
    seen_msgs = list(excl)
    stats = {s: [0, 0] for s in STRATS}   # [examples tried, kept]
    rej = {"language": 0, "rules": 0, "judge": 0, "writer_none": 0}
    shown = {"language": 0, "rules": 0, "judge": 0}
    for lang in a.langs:
        for emotion, stress in itertools.product(EMOTIONS, STRESS):
            for _ in range(a.msgs_per_context):
                try: m = write_message(emotion, stress, lang)
                except Exception as e: print("writer error:", e); continue
                if not m:
                    rej["writer_none"] += 1; continue
                if is_near_duplicate(m, seen_msgs): continue
                seen_msgs.append(set(norm(m)))
                for strat in STRATS:
                    messages = build_example(m, emotion, stress, lang, strat, a.rag_prob)
                    stats[strat][0] += 1
                    kept_reply, why, last = None, None, ""
                    for _try in range(a.attempts):
                        tm = [dict(x) for x in messages]
                        tm[-1]["content"] += "\n\n(" + TEACHER_HINT[strat] + (" Write in Korean." if lang == "ko" else "") + ")"
                        try: last = " ".join(chat(TEACHER[lang], tm, temperature=0.8, num_predict=220).split())
                        except Exception as e: print("teacher error:", e); continue
                        why = ("language" if not language_ok(last, lang) else
                               "rules" if not rule_check(strat, last, lang) else
                               "judge" if (not a.no_judge and not judge_ok(lang, m, last, strat)) else None)
                        if not why:
                            kept_reply = last; break
                    if not kept_reply:
                        if why:
                            rej[why] += 1
                            if a.debug and shown[why] < 4:
                                shown[why] += 1
                                print(f"  [rejected: {why}] {strat} | {m[:70]}\n      -> {last[:200]}")
                        continue
                    stats[strat][1] += 1
                    msgs = messages + [{"role": "assistant", "content": kept_reply}]
                    append_jsonl(f"synth_{split_of(m)}.jsonl", dict(messages=msgs, strategy=strat,
                                 language=lang, source="synth"))
            print(f"{lang} {emotion}/{stress} done | kept/tried:",
                  {s: f"{v[1]}/{v[0]}" for s, v in stats.items()}, "| rejected:", rej)

# ------------------------------------------------------------------ merge
def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []

def cmd_merge(a):
    rnd = random.Random(42)
    es, sy = read("esconv_train.jsonl"), read("synth_train.jsonl")
    pool = {s: [] for s in STRATS}
    for s in STRATS:
        e = [x for x in es if x["strategy"] == s]; rnd.shuffle(e)
        pool[s] = e[:a.esconv_max] + [x for x in sy if x["strategy"] == s]
    n = min(len(v) for v in pool.values())
    if a.cap: n = min(n, a.cap)
    train = []
    for s in STRATS:
        rnd.shuffle(pool[s]); train += pool[s][:n]
    rnd.shuffle(train)
    dev = read("esconv_dev.jsonl") + read("synth_dev.jsonl")
    for path, rows in (("strategy_train.jsonl", train), ("strategy_dev.jsonl", dev)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(train)
    print(f"Wrote strategy_train.jsonl ({len(train)} examples, {n} per strategy) and strategy_dev.jsonl ({len(dev)})")
    if len(df): print(df.groupby(["language", "source"]).size())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_excl = ["../Experiment/prompts.csv", "prompts_v2.csv"]
    for name in ("testset", "esconv", "synth", "merge"):
        s = sub.add_parser(name)
        s.add_argument("--exclude", nargs="*", default=default_excl)
        s.add_argument("--rag-prob", type=float, default=0.7)
        if name == "testset":
            s.add_argument("--per-context", type=int, default=3); s.add_argument("--out", default="prompts_v2.csv")
        if name == "esconv": s.add_argument("--esconv", default="ESConv.json")
        if name == "synth":
            s.add_argument("--msgs-per-context", type=int, default=15); s.add_argument("--attempts", type=int, default=3)
            s.add_argument("--langs", nargs="*", default=["en", "ko"])
            s.add_argument("--debug", action="store_true"); s.add_argument("--no-judge", action="store_true"); s.add_argument("--seed", type=int, default=42)
        if name == "merge":
            s.add_argument("--esconv-max", type=int, default=150); s.add_argument("--cap", type=int, default=0)
    a = ap.parse_args()
    {"testset": cmd_testset, "esconv": cmd_esconv, "synth": cmd_synth, "merge": cmd_merge}[a.cmd](a)