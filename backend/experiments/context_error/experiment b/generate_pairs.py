"""
generate_pairs.py  -  Experiment B, step 1: paired generation (no judging)
==========================================================================
For every message that (a) reaches LinUCB through intent routing and
(b) got a DIFFERENT strategy under reference vs predicted context in
Experiment A (read from Experiment A's per_message.csv, so the bandit
is NOT re-run and cannot be modified), generate responses with:

  * the same model            (llm.MODEL_NAME)
  * the same RAG chunks       (retrieved ONCE per message, reused in both arms)
  * the same generation opts  (llm default temperature, same num_predict
                               cap per intent as chatbot_engine.py, same seeds)
  * history=None, trend_info=None in both arms
  * prompt built by the REAL prompt_builder.build_counseling_prompt()

The two arms differ ONLY in emotion_info (emotion/stress labels) and the
strategy chosen. This script does NOT judge quality; it records the raw
responses plus simple objective checks.

Safety
------
* Does not import app.py, models.py, database.py, chatbot_engine.py or
  strategy_selector's state; never touches the DB; never writes to the
  app folder.
* Only deliberate deviation from production: an explicit "seed" option
  (llm.generate_response() has no seed parameter), needed for
  reproducibility. Same seeds are used in both arms.

Usage
-----
python generate_pairs.py --app-dir <backend> --per-message exp_a_out/per_message.csv \
       --out-dir exp_b_out [--seeds 3]
"""
import argparse
import ast
import csv
import difflib
import hashlib
import inspect
import io
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import requests

ENGINE_SNIPPETS = [  # copied-by-hand logic from handle_message(); fail loudly if the app changes
    'rag_category = "career_employment" if intent == "career_anxiety" else None',
    "rag.retrieve(user_message, k=3, category=rag_category)",
]
NGRAM_N = 6


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_token_caps(engine_path: Path) -> dict:
    src = engine_path.read_text(encoding="utf-8")
    for s in ENGINE_SNIPPETS:
        if s not in src:
            sys.exit(f"chatbot_engine.py no longer contains expected line:\n  {s}\n"
                     "Experiment B copies this logic; re-check it before running.")
    caps = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "MAX_TOKENS_BY_INTENT":
            caps = ast.literal_eval(node.value)
    if caps is None:
        sys.exit("MAX_TOKENS_BY_INTENT not found in chatbot_engine.py")
    return caps


LETTER = re.compile(r"[A-Za-z\uac00-\ud7a3]")
HANGUL = re.compile(r"[\uac00-\ud7a3]")
WORD = re.compile(r"\w+")


def ngrams(text: str, n: int = NGRAM_N):
    toks = WORD.findall(text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def objective_metrics(text: str, chunks, language: str, strategy: str) -> dict:
    letters = LETTER.findall(text)
    hangul_ratio = (len(HANGUL.findall(text)) / len(letters)) if letters else 0.0
    q = text.count("?") + text.count("\uff1f")
    resp_ng = ngrams(text)
    chunk_ng = set()
    for c in chunks:
        chunk_ng |= ngrams(c)
    overlap = (len(resp_ng & chunk_ng) / len(resp_ng)) if resp_ng else 0.0
    rule_ok = {"validate_listen": q == 0, "validate_ask": q == 1}.get(strategy)
    return {
        "word_count": len(text.split()),
        "question_marks": q,
        "strategy_rule_ok": rule_ok,
        "language_ok": (hangul_ratio > 0.5) if language == "ko" else (hangul_ratio < 0.1),
        "chunk_6gram_overlap": round(overlap, 4),
    }


def ollama_chat(llm, messages, temperature, num_predict, seed, retries=2):
    payload = {
        "model": llm.MODEL_NAME, "messages": messages, "stream": False,
        "keep_alive": llm.KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": num_predict, "seed": seed},
    }
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(llm.OLLAMA_URL, json=payload, timeout=llm.DEFAULT_TIMEOUT)
            r.raise_for_status()
            d = r.json()
            return d.get("message", {}).get("content", "").strip(), d.get("done_reason")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise RuntimeError(f"Ollama call failed after retries: {last}")


def model_digest(llm):
    try:
        base = llm.OLLAMA_URL.split("/api/")[0]
        tags = requests.get(base + "/api/tags", timeout=10).json().get("models", [])
        for m in tags:
            if m.get("name", "").split(":")[0] == llm.MODEL_NAME.split(":")[0]:
                return {"name": m.get("name"), "digest": m.get("digest")}
    except Exception:  # noqa: BLE001
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--per-message", required=True, help="Experiment A per_message.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3, help="samples per arm per message (seeds 1..N)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: whatever llm.generate_response() uses")
    ap.add_argument("--allow-empty-rag", action="store_true")
    a = ap.parse_args()

    app_dir, pm_path, out_dir = Path(a.app_dir).resolve(), Path(a.per_message).resolve(), Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token_caps = load_token_caps(app_dir / "chatbot_engine.py")

    cwd = os.getcwd()
    sys.path.insert(0, str(app_dir))
    os.chdir(app_dir)
    try:
        import llm
        import prompt_builder
        import rag
    finally:
        os.chdir(cwd)

    temperature = a.temperature
    if temperature is None:
        temperature = inspect.signature(llm.generate_response).parameters["temperature"].default
    if not llm.check_ollama_status():
        sys.exit("Ollama is not reachable at localhost:11434 - start it (`ollama serve`) and retry.")

    # ---- select the pairs ----
    rows = list(csv.DictReader(io.StringIO(pm_path.read_text(encoding="utf-8-sig"))))
    pairs = [r for r in rows if r["reaches_linucb"] == "True" and r["changed"] == "True"]
    print(f"{len(pairs)} routed messages with changed strategy -> {len(pairs) * 2 * a.seeds} generations")

    # ---- retrieve ONCE per message (same chunks for both arms), exactly as handle_message does ----
    chunks_by_id, empty = {}, []
    for r in pairs:
        category = "career_employment" if r["intent"] == "career_anxiety" else None
        chunks = rag.retrieve(r["message"], k=3, category=category)
        chunks_by_id[r["id"]] = chunks
        if not chunks:
            empty.append(r["id"])
    if empty and not a.allow_empty_rag:
        sys.exit(f"RAG returned NO chunks for {len(empty)} message(s) {empty[:10]}... "
                 "(rag.retrieve swallows errors). Check chroma_store and `ollama pull nomic-embed-text`, "
                 "or pass --allow-empty-rag if this matches how the app really behaves.")
    (out_dir / "chunks.json").write_text(json.dumps(chunks_by_id, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- resume support ----
    gen_path = out_dir / "generations.jsonl"
    done = set()
    if gen_path.exists():
        for line in gen_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                done.add((j["id"], j["arm"], j["seed"]))

    prompt_audit = {}
    with open(gen_path, "a", encoding="utf-8") as out:
        for i, r in enumerate(pairs, 1):
            chunks = chunks_by_id[r["id"]]
            lang, intent = r["language"], r["intent"]
            arms = {
                "reference": ({"emotion": r["ref_emotion"], "stress_type": r["ref_stress"]}, r["ref_strategy"]),
                "predicted": ({"emotion": r["pred_emotion"], "stress_type": r["pred_stress"]}, r["pred_strategy"]),
            }
            msgs = {}
            for arm, (emo, strat) in arms.items():
                msgs[arm] = prompt_builder.build_counseling_prompt(
                    r["message"], None, chunks, emo, None, lang, strat)
            assert msgs["reference"][1] == msgs["predicted"][1], "user turns differ between arms"
            diff = [l for l in difflib.unified_diff(
                msgs["reference"][0]["content"].splitlines(), msgs["predicted"][0]["content"].splitlines(),
                lineterm="", n=0) if l and l[0] in "+-" and not l.startswith(("+++", "---"))]
            prompt_audit[r["id"]] = {"n_diff_lines": len(diff), "diff": diff}

            cap = token_caps.get(intent, 200)
            for arm, (emo, strat) in arms.items():
                for seed in range(1, a.seeds + 1):
                    if (r["id"], arm, seed) in done:
                        continue
                    text, done_reason = ollama_chat(llm, msgs[arm], temperature, cap, seed)
                    rec = {
                        "id": r["id"], "arm": arm, "seed": seed, "language": lang, "intent": intent,
                        "emotion": emo["emotion"], "stress_type": emo["stress_type"], "strategy": strat,
                        "num_predict": cap, "done_reason": done_reason, "truncated": done_reason == "length",
                        "response": text,
                    }
                    rec.update(objective_metrics(text, chunks, lang, strat))
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
            print(f"  [{i}/{len(pairs)}] id={r['id']} {arms['reference'][1]} vs {arms['predicted'][1]}")

    (out_dir / "prompt_audit.json").write_text(json.dumps(prompt_audit, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- flat CSV + run info ----
    recs = [json.loads(l) for l in gen_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if recs:
        with open(out_dir / "generations.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
    info = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": llm.MODEL_NAME, "model_digest": model_digest(llm),
        "temperature": temperature, "seeds": list(range(1, a.seeds + 1)),
        "token_caps_by_intent_used": {k: token_caps.get(k, 200) for k in sorted({r["intent"] for r in pairs})},
        "n_pairs": len(pairs), "n_generations_in_file": len(recs),
        "rag": {"k": 3, "messages_with_empty_retrieval": empty},
        "per_message_sha256": sha256_file(pm_path),
        "prompt_diff_lines_per_pair": sorted({v["n_diff_lines"] for v in prompt_audit.values()}),
        "python": platform.python_version(),
        "deviation_from_production": "explicit options.seed added; history=None and trend_info=None in both arms",
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Wrote generations.jsonl/.csv, chunks.json, prompt_audit.json, run_info.json to {out_dir}")


if __name__ == "__main__":
    main()
