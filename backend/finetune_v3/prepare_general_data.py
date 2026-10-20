"""
Converts your processed GENERAL counseling dataset (the 23,037 examples used to fine-tune the Llama model) into the
chat-format JSONL that finetune_v3.py trains on, split 80/10/10 with a fixed seed.

Understands: JSONL / JSON / CSV / Parquet with
  - a "messages" list (role/content)                     - ShareGPT "conversations" (from/value)
  - question/answer style columns (instruction+output, question+answer, prompt+response, Context+Response,
    questionText+answerText, input+output, ...)
  - a single "text" column already written in the Llama 3 or Gemma chat template (it is parsed back into messages)

Usage (from backend/finetune_v3):
  python prepare_general_data.py --input PATH --peek       # look at the structure and the first records, writes nothing
  python prepare_general_data.py --input PATH              # writes general_train.jsonl / general_dev.jsonl / general_test.jsonl
Options: --user-col NAME --assistant-col NAME (override auto-detection), --system "text" (add a system message
if your records have none), --seed 42
"""
import argparse, json, os, random, re, sys

LLAMA = re.compile(r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\s*(.*?)<\|eot_id\|>", re.S)
GEMMA = re.compile(r"<start_of_turn>(user|model)\s*(.*?)<end_of_turn>", re.S)
PAIRS = [("instruction", "output"), ("question", "answer"), ("prompt", "response"), ("context", "response"),
         ("input", "output"), ("questiontext", "answertext"), ("user", "assistant"), ("human", "assistant"),
         ("query", "response"), ("question", "response"), ("text_input", "output")]


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if ext == ".json":
        d = json.load(open(path, encoding="utf-8"))
        return d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
    import pandas as pd
    if ext == ".csv": return pd.read_csv(path).to_dict("records")
    if ext == ".parquet": return pd.read_parquet(path).to_dict("records")
    sys.exit(f"Unsupported file type {ext}. Use jsonl, json, csv or parquet.")


def clean(x): return re.sub(r"\n{3,}", "\n\n", str(x).strip())


def to_messages(row, user_col=None, asst_col=None, system=None):
    msgs = None
    low = {str(k).lower(): k for k in row}
    if "messages" in low and isinstance(row[low["messages"]], list):
        msgs = [{"role": m["role"], "content": clean(m["content"])} for m in row[low["messages"]]]
    elif "conversations" in low and isinstance(row[low["conversations"]], list):
        role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
        msgs = [{"role": role.get(m.get("from", m.get("role")), "user"), "content": clean(m.get("value", m.get("content")))}
                for m in row[low["conversations"]]]
    elif user_col and asst_col:
        msgs = [{"role": "user", "content": clean(row[user_col])}, {"role": "assistant", "content": clean(row[asst_col])}]
    elif "text" in low and isinstance(row[low["text"]], str) and ("<|start_header_id|>" in row[low["text"]] or "<start_of_turn>" in row[low["text"]]):
        t = row[low["text"]]
        if "<|start_header_id|>" in t:
            msgs = [{"role": r, "content": clean(c)} for r, c in LLAMA.findall(t)]
        else:
            msgs = [{"role": "user" if r == "user" else "assistant", "content": clean(c)} for r, c in GEMMA.findall(t)]
    else:
        for u, a in PAIRS:
            if u in low and a in low and row[low[u]] is not None and row[low[a]] is not None:
                user = clean(row[low[u]])
                if u == "instruction" and "input" in low and str(row[low["input"]]).strip() not in ("", "nan", "None"):
                    user += "\n\n" + clean(row[low["input"]])
                msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": clean(row[low[a]])}]
                break
    if not msgs or msgs[-1]["role"] != "assistant" or not msgs[-1]["content"] or not any(m["role"] == "user" and m["content"] for m in msgs):
        return None
    if system and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": system}] + msgs
    return msgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--peek", action="store_true")
    ap.add_argument("--user-col"); ap.add_argument("--assistant-col"); ap.add_argument("--system")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = load(a.input)
    print(f"Loaded {len(rows)} records. Fields of the first record: {list(rows[0].keys())}")
    def source_of(r):
        for k in r:
            if str(k).lower() in ("source", "dataset", "origin", "data_source"):
                return str(r[k])
        return None
    conv = [(to_messages(r, a.user_col, a.assistant_col, a.system), source_of(r)) for r in rows]
    ok = [(m, src) for m, src in conv if m]
    if any(src for _, src in ok):
        from collections import Counter
        print("Source labels found:", dict(Counter(src for _, src in ok)))
    print(f"Converted {len(ok)} of {len(rows)} records ({len(rows) - len(ok)} skipped as empty or unrecognised).")
    if a.peek or not ok:
        for m, _ in ok[:2]: print(json.dumps(m, ensure_ascii=False, indent=1)[:900])
        if not ok: sys.exit("Nothing converted: tell me the field names (or use --user-col / --assistant-col).")
        if a.peek: sys.exit(0)
    random.Random(a.seed).shuffle(ok)
    n = len(ok); n_tr, n_dev = int(n * 0.8), int(n * 0.1)
    for name, part in (("train", ok[:n_tr]), ("dev", ok[n_tr:n_tr + n_dev]), ("test", ok[n_tr + n_dev:])):
        with open(f"general_{name}.jsonl", "w", encoding="utf-8") as f:
            for m, src in part:
                rec = {"messages": m}
                if src: rec["source"] = src
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"general_{name}.jsonl: {len(part)} examples")