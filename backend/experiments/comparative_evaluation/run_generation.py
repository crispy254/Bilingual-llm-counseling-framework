"""Step 1: every model answers every prompt through Nova's real prompt + RAG pipeline.
Resumable: re-run to continue after an interruption.
Usage (from backend/experiment):  python run_generation.py"""
import os, re, sys, time, requests, pandas as pd
import exp_config as C
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finetune_v3")))
from strategy_checks import rule_check   # same rules used to build the training data
import pipeline_hooks as H

THINK = re.compile(r"<think>.*?</think>", re.S)
LEAK = re.compile(r"^\s*(okay|ok,|alright|hmm|let me|first,|so,? the (user|student)|the (user|student)|i need to|i should)\b", re.I)

def chat(model, messages, seed=None, temperature=None):
    messages = [dict(m) for m in messages]
    if model.startswith("qwen3"):   # qwen3 reasons out loud unless told not to
        messages[-1]["content"] += " /no_think"
    t0 = time.time()
    r = requests.post(f"{C.OLLAMA_URL}/api/chat", json={
        "model": model, "stream": False, "think": False, "messages": messages,
        "options": {"temperature": C.TEMPERATURE if temperature is None else temperature,
                    "num_predict": C.MAX_TOKENS, "seed": C.SEED if seed is None else seed},
    }, timeout=600)
    r.raise_for_status()
    d = r.json()
    wall = time.time() - t0
    out_tok = d.get("eval_count", 0)
    tps = out_tok / (d["eval_duration"] / 1e9) if d.get("eval_duration") else 0
    return THINK.sub("", d["message"]["content"]).strip(), wall, out_tok, tps

def generate(variant, messages, row):
    """variant = 'model' or 'model+guard'. With the guard, up to GUARD_ATTEMPTS replies are generated and the
    first that passes the strategy rule check is kept; if none passes, the LAST reply is kept and the failure
    is recorded (it is never hidden). Latency and tokens are summed over attempts."""
    base, _, flag = variant.partition("+")
    attempts = C.GUARD_ATTEMPTS if flag == "guard" else 1
    total_wall = total_tok = 0
    for k in range(attempts):
        resp, wall, ntok, tps = chat(base, messages, seed=C.SEED + k,
                                     temperature=C.TEMPERATURE if k == 0 else 0.7)
        total_wall += wall; total_tok += ntok
        ok = rule_check(row["strategy"], resp, row["language"])
        if ok: break
    return resp, total_wall, total_tok, tps, k + 1, ok

def append_row(path, row):
    """Append one row, matching the existing file's columns (extra keys dropped, missing ones blank)."""
    df = pd.DataFrame([row])
    if os.path.exists(path):
        cols = list(pd.read_csv(path, nrows=0).columns)
        df = df.reindex(columns=cols)
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")

def model_info(model):
    """VRAM currently used by this model (GB) and its parameter size."""
    vram, size = None, None
    try:
        for m in requests.get(f"{C.OLLAMA_URL}/api/ps", timeout=30).json().get("models", []):
            if m["name"] == model or m["name"].split(":")[0] == model.split(":")[0]:
                vram = round(m.get("size_vram", 0) / 1e9, 2)
        d = requests.post(f"{C.OLLAMA_URL}/api/show", json={"model": model}, timeout=30).json()
        size = d.get("details", {}).get("parameter_size")
    except Exception:
        pass
    return vram, size

def main():
    if not H.rag_is_working():
        print("WARNING: RAG returned nothing. Run `python rag.py` in backend/ first "
              "(needs `ollama pull nomic-embed-text`). Continuing WITHOUT retrieval would "
              "not match Nova's real pipeline.")
        if input("Continue anyway? (y/N) ").strip().lower() != "y":
            return
    prompts = pd.read_csv(C.PROMPTS_CSV).dropna(subset=["message"])
    prompts = prompts[prompts["message"].astype(str).str.strip() != ""]
    done = set()
    if os.path.exists(C.GENERATIONS_CSV):
        g = pd.read_csv(C.GENERATIONS_CSV)
        done = set(zip(g["model"], g["id"]))
    header = not os.path.exists(C.GENERATIONS_CSV)
    for model in C.MODELS:
        print(f"== {model}")
        for _, p in prompts.iterrows():
            if (model, p["id"]) in done:
                continue
            messages, ctx = H.build_messages(p)
            try:
                resp, wall, ntok, tps, attempts, guard_ok = generate(model, messages, p)
            except Exception as e:
                print("  error on id", p["id"], e); continue
            vram, size = model_info(model.split("+")[0])
            row = dict(model=model, id=p["id"], language=p["language"], emotion=p["emotion"],
                       stress_type=p["stress_type"], strategy=p["strategy"], message=p["message"],
                       context=ctx, response=resp, latency_s=round(wall, 2),
                       out_tokens=ntok, tokens_per_s=round(tps, 1), vram_gb=vram, model_size=size,
                       reasoning_leak=bool(LEAK.match(resp)), attempts=attempts, guard_pass=guard_ok)
            append_row(C.GENERATIONS_CSV, row)
            print(f"  id {p['id']} ok ({wall:.1f}s)")

if __name__ == "__main__":
    main()