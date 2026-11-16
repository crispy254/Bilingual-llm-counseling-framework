#!/usr/bin/env python3
"""
RAG retrieval experiment for the bilingual (EN/KO) counseling-framework paper.

This script evaluates retrieval quality only. It does not generate chatbot
responses or re-evaluate response faithfulness.

Workflow (note: --work is a TOP-LEVEL option and goes BEFORE the command):

  python rag_multilingual_experiment.py --work rag_exp prepare \
      --kb-zip knowledge_base.zip --chroma-zip chroma_store.zip \
      --generations run2_generations.csv
  python rag_multilingual_experiment.py --work rag_exp translate-draft
  #   -> human reviews rag_exp/kb_ko_draft.json, saves as rag_exp/kb_ko.json
  python rag_multilingual_experiment.py --work rag_exp check-kb
  python rag_multilingual_experiment.py --work rag_exp tune-export --tuning tuning.jsonl
  #   -> label tune_pairs.csv, save as tune_pairs_labeled.csv
  python rag_multilingual_experiment.py --work rag_exp tune-select
  python rag_multilingual_experiment.py --work rag_exp retrieve
  python rag_multilingual_experiment.py --work rag_exp annotate-export
  #   -> label annotate_pairs.csv -> annotate_pairs_labeled.csv
  #   -> double-label human_sample.csv -> human_sample_labeled.csv
  python rag_multilingual_experiment.py --work rag_exp metrics
  python rag_multilingual_experiment.py --work rag_exp agreement
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

TOPK = 3
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CAREER_CATEGORY = "career_employment"

HANGUL = re.compile(r"[\uac00-\ud7a3]")

CRISIS_RE = re.compile(
    r"suicid|self[\s-]?harm|kill myself|end my life|overdose|"
    r"자살|자해|죽고\s*싶|죽어버|극단적\s*선택|"
    r"\b109\b|1577-0199|\b112\b|\b119\b",
    re.I,
)

CONFIGS = {
    "A_original_chroma": {"kind": "chroma", "embedder": "nomic-embed-text"},
    "B_bge3_same_index": {"kind": "archive_dense", "embedder": "bge-m3"},
    "C_bilingual": {
        "kind": "bilingual", "embedder": "bge-m3",
        "drop_crisis": False, "threshold": False,
    },
    "D_no_crisis": {
        "kind": "bilingual", "embedder": "bge-m3",
        "drop_crisis": True, "threshold": False,
    },
    "E_threshold": {
        "kind": "bilingual", "embedder": "bge-m3",
        "drop_crisis": True, "threshold": True,
    },
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def language_of(text):
    chars = [c for c in str(text) if not c.isspace()]
    if not chars:
        return "en"
    ratio = len(HANGUL.findall(str(text))) / len(chars)
    return "ko" if ratio > 0.30 else "en"


def norm_lang(value, text=""):
    v = str(value or "").strip().lower()
    if v in ("ko", "korean"):
        return "ko"
    if v in ("en", "english"):
        return "en"
    return language_of(text)


def chunk_text(text, size=500, overlap=80):
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [x for x in chunks if x]


def is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def read_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_label(raw, where):
    try:
        v = int(str(raw).strip())
    except ValueError:
        sys.exit(f"Non-integer label {raw!r} in {where}")
    if v not in (0, 1, 2, 3):
        sys.exit(f"Label {v} out of range 0-3 in {where}")
    return v


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def prepare(args):
    import pandas as pd

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    # ---- English knowledge base --------------------------------------
    kb_dir = work / "knowledge_base"
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
    kb_dir.mkdir()

    with zipfile.ZipFile(args.kb_zip) as z:
        for member in z.namelist():
            p = Path(member)
            if p.suffix.lower() != ".txt":
                continue
            parts = p.parts
            if "knowledge_base" not in parts:
                continue
            rel = Path(*parts[parts.index("knowledge_base") + 1:])
            if not rel.parts or ".." in rel.parts:
                continue
            target = kb_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))

    chunks = []
    for fp in sorted(kb_dir.rglob("*.txt")):
        category = fp.parent.relative_to(kb_dir).as_posix()
        if category == ".":
            category = "general"
        text = fp.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "id": f"{category}-{fp.name}-{i}",
                "doc": fp.name,
                "category": category,
                "text": chunk,
                "is_crisis": bool(CRISIS_RE.search(chunk)),
            })

    if not chunks:
        sys.exit("No .txt chunks found in knowledge_base zip.")

    (work / "kb_en.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Original ChromaDB archive -----------------------------------
    chroma_dir = work / "original_chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir()

    with zipfile.ZipFile(args.chroma_zip) as z:
        for member in z.namelist():
            p = Path(member)
            if "chroma_store" not in p.parts:
                continue
            rel = Path(*p.parts[p.parts.index("chroma_store") + 1:])
            if not rel.parts or ".." in rel.parts:
                continue
            target = chroma_dir / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))

    # ---- Generation dataset -> unique messages -----------------------
    df = pd.read_csv(args.generations)

    required = {"id", "language", "message", "model"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Generation CSV missing required columns: {sorted(missing)}")

    consistency = df.groupby("id").agg(
        n_text=("message", "nunique"), n_lang=("language", "nunique"))
    bad = consistency[(consistency.n_text != 1) | (consistency.n_lang != 1)]
    if len(bad):
        sys.exit("Message text/language differs across models for IDs: "
                 + str(bad.index.tolist()[:10]))

    cols = ["id", "language", "message"]
    if "stress_type" in df.columns:
        cols.append("stress_type")

    msgs = df[cols].drop_duplicates("id").copy()
    msgs["language"] = msgs["language"].astype(str).str.lower().map(
        lambda x: "ko" if x in ("ko", "korean")
        else "en" if x in ("en", "english") else x)

    if set(msgs.language) - {"en", "ko"}:
        sys.exit("Unexpected language values: "
                 + str(sorted(set(msgs.language) - {"en", "ko"})))

    # Keyword screen only. Review manually against the production detector.
    msgs["crisis_bypass"] = msgs.message.astype(str).map(
        lambda x: bool(CRISIS_RE.search(x)))

    msgs.to_csv(work / "messages_144.csv", index=False, encoding="utf-8-sig")

    summary = (
        f"English chunks created: {len(chunks)}\n"
        f"Unique messages: {len(msgs)}\n"
        f"Languages: {msgs.language.value_counts().to_dict()}\n"
        f"Keyword-screened crisis messages: {int(msgs.crisis_bypass.sum())}\n"
        "NOTE: Review messages_144.csv and manually mark any additional "
        "crisis messages before running retrieval.\n"
    )
    (work / "prepare_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Created {work / 'kb_en.json'}")
    print(f"Created {work / 'messages_144.csv'}")
    print(f"Extracted ChromaDB to {chroma_dir}")


# --------------------------------------------------------------------------
# Embeddings (Ollama) with on-disk cache
# --------------------------------------------------------------------------

class Embedder:
    def __init__(self, model, work, legacy=False):
        self.model = model
        self.work = Path(work)
        self.legacy = legacy
        self.cache_path = self.work / "embedding_cache.pkl"
        self.cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self.cache = pickle.load(f)

    def _request(self, texts):
        import requests

        result = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            if self.legacy:
                for text in batch:
                    r = requests.post(
                        f"{OLLAMA_URL}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=600)
                    r.raise_for_status()
                    result.append(r.json()["embedding"])
            else:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=600)
                r.raise_for_status()
                result.extend(r.json()["embeddings"])
        return result

    def __call__(self, texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [f"{self.model}|{hashlib.sha256(t.encode('utf-8')).hexdigest()}"
                for t in texts]

        # De-duplicate missing items so identical texts are embedded once.
        missing = {}
        for key, text in zip(keys, texts):
            if key not in self.cache and key not in missing:
                missing[key] = text

        if missing:
            miss_keys = list(missing)
            vectors = self._request([missing[k] for k in miss_keys])
            if len(vectors) != len(miss_keys):
                sys.exit("Embedding server returned an unexpected number "
                         "of vectors.")
            for key, vector in zip(miss_keys, vectors):
                arr = np.asarray(vector, dtype=np.float32)
                # Legacy (nomic) vectors stay unnormalised to preserve the
                # original Chroma L2 behaviour.
                if not self.legacy:
                    arr = arr / (np.linalg.norm(arr) + 1e-12)
                self.cache[key] = arr.tolist()

            tmp = self.cache_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.cache, f)
            os.replace(tmp, self.cache_path)

        return np.asarray([self.cache[k] for k in keys], dtype=np.float32)


# --------------------------------------------------------------------------
# Knowledge-base loaders
# --------------------------------------------------------------------------

def load_clean_kb(work):
    en_path = Path(work) / "kb_en.json"
    ko_path = Path(work) / "kb_ko.json"

    if not en_path.exists():
        sys.exit("kb_en.json missing. Run prepare first.")
    en = json.loads(en_path.read_text(encoding="utf-8"))

    if not ko_path.exists():
        sys.exit("kb_ko.json missing. Create and review Korean "
                 "translations first.")
    ko = json.loads(ko_path.read_text(encoding="utf-8"))

    en_by_id = {x["id"]: x for x in en}
    if len(en_by_id) != len(en) or len(ko) != len(en):
        sys.exit("KB IDs must be unique and English/Korean chunk counts "
                 "must match.")

    ko_by_id = {x["id"]: x for x in ko}
    if set(ko_by_id) != set(en_by_id):
        sys.exit("Korean and English KB IDs do not match.")

    out = []
    for x in en:
        out.append({
            "cid": x["id"], "lang": "en", "doc": x["doc"],
            "category": x["category"], "text": x["text"],
            "crisis": bool(x["is_crisis"]),
        })

    for x in en:
        y = ko_by_id[x["id"]]
        if language_of(y["text"]) != "ko":
            sys.exit(f"Korean translation appears invalid: {x['id']}")
        out.append({
            "cid": x["id"] + "_ko", "lang": "ko", "doc": x["doc"],
            "category": x["category"], "text": y["text"],
            "crisis": bool(x["is_crisis"]),
        })
    return out


def load_archive(work):
    try:
        import chromadb
    except ImportError:
        sys.exit("Install chromadb: pip install chromadb")

    client = chromadb.PersistentClient(
        path=str(Path(work) / "original_chroma"))
    col = client.get_collection("counseling_kb")
    raw = col.get(include=["documents", "metadatas"])

    records = []
    for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        meta = meta or {}
        records.append({
            "cid": cid,
            "doc": meta.get("source", ""),
            "category": meta.get("category", "general"),
            "text": doc or "",
            "crisis": bool(CRISIS_RE.search(doc or "")),
        })

    if not records:
        sys.exit("Archived Chroma collection is empty.")
    return client, col, records


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def candidate_indices(cfg, kb, lang, stress_type):
    """Same career-category filter for EVERY configuration, so the
    comparison between A-E is not confounded by filtering."""
    idx = list(range(len(kb)))

    if cfg.get("kind") == "bilingual":
        idx = [i for i in idx if kb[i]["lang"] == lang]
        if cfg.get("drop_crisis"):
            idx = [i for i in idx if not kb[i]["crisis"]]

    if stress_type == "career":
        idx = [i for i in idx if kb[i]["category"] == CAREER_CATEGORY]
    return idx


def retrieve_dense(text, lang, stress_type, cfg, kb, vectors, embedder,
                   threshold=None):
    candidates = candidate_indices(cfg, kb, lang, stress_type)
    if not candidates:
        return []

    q = embedder([text])[0]
    scores = vectors[candidates] @ q
    order = np.argsort(-scores)[:TOPK]

    out = []
    for j in order:
        c = kb[candidates[j]]
        score = float(scores[j])
        if threshold is not None and score < threshold:
            continue
        out.append({
            "cid": c["cid"], "doc": c["doc"], "category": c["category"],
            "crisis": c["crisis"], "score": score, "text": c["text"],
        })
    return out


def retrieve_chroma(text, stress_type, collection, embedder):
    q = embedder([text])[0].tolist()

    kwargs = {
        "query_embeddings": [q],
        "n_results": TOPK,
        "include": ["documents", "metadatas", "distances"],
    }
    if stress_type == "career":
        kwargs["where"] = {"category": CAREER_CATEGORY}

    r = collection.query(**kwargs)

    out = []
    for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0],
                                    r["metadatas"][0], r["distances"][0]):
        meta = meta or {}
        out.append({
            "cid": cid,
            "doc": meta.get("source", ""),
            "category": meta.get("category", "general"),
            "crisis": bool(CRISIS_RE.search(doc or "")),
            "distance": float(dist),
            "score": -float(dist),
            "text": doc or "",
        })
    return out


# --------------------------------------------------------------------------
# translate-draft / check-kb
# --------------------------------------------------------------------------

def cmd_translate(args):
    import requests

    en = json.loads((Path(args.work) / "kb_en.json").read_text(encoding="utf-8"))
    out_path = Path(args.work) / "kb_ko_draft.json"

    # Resume support: keep translations already drafted.
    done = {}
    if out_path.exists():
        done = {x["id"]: x["text"]
                for x in json.loads(out_path.read_text(encoding="utf-8"))}

    out = []
    for i, row in enumerate(en, 1):
        if row["id"] in done:
            out.append({"id": row["id"], "text": done[row["id"]]})
            continue

        prompt = (
            "Translate this university-student counseling guidance passage "
            "into natural, accurate Korean. Preserve meaning; do not add or "
            "remove information. Output only the translation.\n\n"
            + row["text"])

        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": args.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0}},
            timeout=600)
        r.raise_for_status()

        out.append({"id": row["id"], "text": r.json()["response"].strip()})
        print(f"Drafted {i}/{len(en)}: {row['id']}")

        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("Draft saved. A Korean-speaking author must review it and save "
          "the corrected file as kb_ko.json.")


def cmd_check(args):
    kb = load_clean_kb(args.work)
    print(f"English chunks: {sum(x['lang'] == 'en' for x in kb)}")
    print(f"Korean chunks: {sum(x['lang'] == 'ko' for x in kb)}")
    print("Crisis-flagged English chunks: "
          f"{sum(x['crisis'] for x in kb if x['lang'] == 'en')}")
    print("KB checks passed. This checks structure and language only, "
          "not translation quality.")


# --------------------------------------------------------------------------
# Threshold tuning (independent message set)
# --------------------------------------------------------------------------

def cmd_tune_export(args):
    work = Path(args.work)
    msgs = read_jsonl(args.tuning)

    eval_msgs = {str(x["id"]) for x in read_csv_rows(work / "messages_144.csv")}
    if any(str(x["id"]) in eval_msgs for x in msgs):
        sys.exit("Tuning set overlaps with the 144-message evaluation set. "
                 "Use independent messages.")

    kb = load_clean_kb(work)
    emb = Embedder("bge-m3", work)
    vec = emb([x["text"] for x in kb])

    rows = []
    mapping = {}
    cfg = {"kind": "bilingual", "drop_crisis": True}

    for m in msgs:
        if m.get("crisis_bypass", False) or CRISIS_RE.search(m["text"]):
            continue
        lang = language_of(m["text"])
        passages = retrieve_dense(m["text"], lang, m.get("stress_type", ""),
                                  cfg, kb, vec, emb)
        for p in passages:
            pid = f"T{len(rows):05d}"
            mapping[pid] = {"message_id": str(m["id"]), "cid": p["cid"],
                            "score": p["score"]}
            rows.append([pid, m["text"], p["text"], ""])

    with open(work / "tune_pairs.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "message", "passage", "relevance_0_3"])
        w.writerows(rows)

    (work / "tune_pair_map.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8")

    print(f"Exported {len(rows)} tuning pairs. Have annotators label "
          "relevance 0-3. Do not use evaluation messages.")


def cmd_tune_select(args):
    work = Path(args.work)
    mp = json.loads((work / "tune_pair_map.json").read_text(encoding="utf-8"))
    rows = read_csv_rows(work / "tune_pairs_labeled.csv")

    if not rows or any(not r["relevance_0_3"].strip() for r in rows):
        sys.exit("Label every row in tune_pairs_labeled.csv first.")

    pairs = [(mp[r["pair_id"]]["score"],
              parse_label(r["relevance_0_3"], "tune_pairs_labeled.csv") >= 2)
             for r in rows]

    best = None
    for threshold in sorted(set(x[0] for x in pairs)):
        tp = sum(s >= threshold and rel for s, rel in pairs)
        fp = sum(s >= threshold and not rel for s, rel in pairs)
        fn = sum(s < threshold and rel for s, rel in pairs)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)

        if best is None or (f1, threshold) > (best[0], best[1]):
            best = (f1, threshold, precision, recall)

    result = {"threshold": best[1], "f1": best[0], "precision": best[2],
              "recall": best[3], "n_pairs": len(pairs)}
    (work / "threshold.json").write_text(json.dumps(result, indent=2),
                                         encoding="utf-8")
    print(result)


# --------------------------------------------------------------------------
# retrieve
# --------------------------------------------------------------------------

def cmd_retrieve(args):
    work = Path(args.work)
    msgs = read_csv_rows(args.messages)

    if len(msgs) != 144:
        print(f"WARNING: expected 144 unique messages; found {len(msgs)}")
    if len({r["id"] for r in msgs}) != len(msgs):
        sys.exit("Message file contains duplicate IDs.")

    _client, archive_col, archive_kb = load_archive(work)
    clean_kb = load_clean_kb(work)

    emb_nomic = Embedder("nomic-embed-text", work, legacy=True)
    emb_bge = Embedder("bge-m3", work)

    archive_vec = emb_bge([x["text"] for x in archive_kb])
    clean_vec = emb_bge([x["text"] for x in clean_kb])

    configs = dict(CONFIGS)
    threshold = None
    threshold_file = work / "threshold.json"

    if args.skip_threshold:
        configs.pop("E_threshold")
    elif threshold_file.exists():
        threshold = json.loads(
            threshold_file.read_text(encoding="utf-8"))["threshold"]
    else:
        sys.exit("threshold.json missing. Complete the independent tuning "
                 "workflow, or explicitly pass --skip-threshold.")

    for name, cfg in configs.items():
        records = []

        for m in msgs:
            text = m["message"]
            lang = norm_lang(m.get("language"), text)
            crisis = is_true(m.get("crisis_bypass", ""))
            stress = m.get("stress_type", "") or ""

            if crisis:
                passages = []   # crisis messages bypass retrieval
            elif name == "A_original_chroma":
                passages = retrieve_chroma(text, stress, archive_col,
                                           emb_nomic)
            elif name == "B_bge3_same_index":
                passages = retrieve_dense(text, lang, stress, cfg,
                                          archive_kb, archive_vec, emb_bge)
            else:
                passages = retrieve_dense(
                    text, lang, stress, cfg, clean_kb, clean_vec, emb_bge,
                    threshold if name == "E_threshold" else None)

            records.append({
                "id": str(m["id"]), "language": lang, "stress_type": stress,
                "crisis_bypass": crisis, "text": text, "passages": passages,
            })

        write_jsonl(work / f"retrieval_{name}.jsonl", records)

        for lang in ("en", "ko"):
            rr = [r for r in records if r["language"] == lang]
            if not rr:
                continue
            abstain = np.mean([not r["passages"] for r in rr])
            crisis_rate = np.mean(
                [any(p["crisis"] for p in r["passages"]) for r in rr])
            print(f"{name:22s} {lang}: n={len(rr)}, "
                  f"crisis_bypass={sum(r['crisis_bypass'] for r in rr)}, "
                  f"abstain={abstain:.1%}, "
                  f"crisis_in_top3={crisis_rate:.1%}")

    print("Retrieval files written. A uses archived Chroma; "
          "B re-embeds its stored passages.")


# --------------------------------------------------------------------------
# annotate-export
# --------------------------------------------------------------------------

def passage_key(message_id, passage):
    text_hash = hashlib.sha256(passage["text"].encode("utf-8")).hexdigest()
    return (str(message_id), str(passage["cid"]), text_hash)


def cmd_annotate(args):
    work = Path(args.work)

    names = [n for n in CONFIGS if (work / f"retrieval_{n}.jsonl").exists()]
    if not names:
        sys.exit("No retrieval_*.jsonl files found. Run retrieve first.")

    pair_data = {}
    for name in names:
        for r in read_jsonl(work / f"retrieval_{name}.jsonl"):
            for p in r["passages"]:
                pair_data[passage_key(r["id"], p)] = {
                    "message": r["text"], "passage": p["text"],
                    "language": r["language"],
                }

    pairs = sorted(pair_data)
    random.Random(17).shuffle(pairs)

    pair_map = {}
    allrows = []
    pair_languages = {}

    for i, key in enumerate(pairs):
        pid = f"P{i:05d}"
        pair_map[pid] = {"message_id": key[0], "cid": key[1],
                         "text_hash": key[2]}
        item = pair_data[key]
        allrows.append([pid, item["message"], item["passage"], ""])
        pair_languages[key] = item["language"]

    with open(work / "annotate_pairs.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "message", "passage", "relevance_0_3"])
        writer.writerows(allrows)

    (work / "pair_map.json").write_text(json.dumps(pair_map, indent=2),
                                        encoding="utf-8")

    ids_by_lang = {
        lang: [i for i, key in enumerate(pairs) if pair_languages[key] == lang]
        for lang in ("en", "ko")
    }

    rng = random.Random(2026)
    selected = []
    for lang in ("en", "ko"):
        selected.extend(rng.sample(ids_by_lang[lang],
                                   min(30, len(ids_by_lang[lang]))))

    with open(work / "human_sample.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "message", "passage",
                         "annotator_1_0_3", "annotator_2_0_3"])
        for i in selected:
            row = allrows[i]
            writer.writerow([row[0], row[1], row[2], "", ""])

    print(f"{len(allrows)} unique message-passage pairs exported.")
    print(f"{len(selected)} pairs exported for double annotation.")
    print("Label annotate_pairs.csv and save it as annotate_pairs_labeled.csv.")
    print("Double-label human_sample.csv and save it as "
          "human_sample_labeled.csv.")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def weighted_kappa(a, b, k=4):
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[x, y] += 1
    if observed.sum() == 0:
        return float("nan")

    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    weights = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)

    denominator = (weights * expected).sum()
    if not denominator:
        return float("nan")
    return 1 - (weights * observed).sum() / denominator


def cmd_metrics(args):
    from scipy.stats import binomtest, wilcoxon

    work = Path(args.work)
    mp = json.loads((work / "pair_map.json").read_text(encoding="utf-8"))

    labels = {}
    for row in read_csv_rows(work / "annotate_pairs_labeled.csv"):
        if not row["relevance_0_3"].strip():
            sys.exit("Unlabelled rows in annotate_pairs_labeled.csv")
        pair = mp[row["pair_id"]]
        key = (str(pair["message_id"]), str(pair["cid"]),
               str(pair["text_hash"]))
        labels[key] = parse_label(row["relevance_0_3"],
                                  "annotate_pairs_labeled.csv")

    names = [n for n in CONFIGS if (work / f"retrieval_{n}.jsonl").exists()]
    if not names:
        sys.exit("No retrieval_*.jsonl files found. Run retrieve first.")

    data = {n: read_jsonl(work / f"retrieval_{n}.jsonl") for n in names}

    def get_label(message_id, passage):
        key = passage_key(message_id, passage)
        if key not in labels:
            sys.exit("Missing annotation for a retrieved passage: " + str(key))
        return labels[key]

    # Pool of judged passages per message (across all configurations).
    pooled = {}
    for records in data.values():
        for record in records:
            mid = str(record["id"])
            for passage in record["passages"]:
                pooled.setdefault(mid, {})[passage_key(mid, passage)] = \
                    get_label(mid, passage)

    per = {}
    for name, records in data.items():
        per[name] = {}
        for record in records:
            mid = str(record["id"])
            relevance = [get_label(mid, p) for p in record["passages"]]
            relevant_count = sum(s >= 2 for s in relevance)

            ideal_pooled = dcg(sorted(pooled.get(mid, {}).values(),
                                      reverse=True)[:TOPK])
            ideal_returned = dcg(sorted(relevance, reverse=True))

            per[name][mid] = {
                "lang": record["language"],
                "crisis_bypass": bool(record.get("crisis_bypass", False)),
                "no_rel": int(relevant_count == 0),
                "p3": relevant_count / TOPK,
                "p_returned": (relevant_count / len(relevance)
                               if relevance else float("nan")),
                "ndcg": (dcg(relevance) / ideal_returned
                         if ideal_returned else 0.0),
                "ndcg_pooled": (dcg(relevance) / ideal_pooled
                                if ideal_pooled else 0.0),
                "abstain": int(not relevance),
                "irrelevant_injected": int(bool(relevance)
                                           and relevant_count == 0),
                "crisis": int(any(p["crisis"] for p in record["passages"])),
            }

    def select(name, lang):
        return [v for v in per[name].values()
                if (lang == "all" or v["lang"] == lang)
                and (args.include_crisis_bypass or not v["crisis_bypass"])]

    if not args.include_crisis_bypass:
        n_excl = sum(v["crisis_bypass"] for v in per[names[0]].values())
        print(f"Excluding {n_excl} crisis-bypass messages from metrics "
              "(retrieval never runs for them). "
              "Use --include-crisis-bypass to keep them.")

    output_rows = []
    print("\nConfiguration | Language | N | No relevant passage % [95% CI] | "
          "P@3 | P(returned) | nDCG@3 | Pooled nDCG@3 | Abstain % | Crisis %")

    for name in names:
        for lang in ("all", "en", "ko"):
            rows = select(name, lang)
            if not rows:
                continue

            k = sum(x["no_rel"] for x in rows)
            low, high = wilson(k, len(rows))

            returned_precision = [x["p_returned"] for x in rows
                                  if not math.isnan(x["p_returned"])]

            result = [
                name, lang, len(rows),
                100 * k / len(rows), 100 * low, 100 * high,
                float(np.mean([x["p3"] for x in rows])),
                (float(np.mean(returned_precision))
                 if returned_precision else float("nan")),
                float(np.mean([x["ndcg"] for x in rows])),
                float(np.mean([x["ndcg_pooled"] for x in rows])),
                100 * float(np.mean([x["abstain"] for x in rows])),
                100 * float(np.mean([x["crisis"] for x in rows])),
            ]
            output_rows.append(result)

            print(f"{name:22s} {lang:3s} | {len(rows):3d} | "
                  f"{result[3]:5.1f} [{result[4]:.1f}-{result[5]:.1f}] | "
                  f"{result[6]:.3f} | {result[7]:.3f} | {result[8]:.3f} | "
                  f"{result[9]:.3f} | {result[10]:5.1f} | {result[11]:5.1f}")

    with open(work / "metrics_summary.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "language", "n", "no_relevant_pct",
                         "ci_low", "ci_high", "P_at_3", "P_returned",
                         "nDCG_at_3", "nDCG_pooled", "abstain_pct",
                         "crisis_in_top3_pct"])
        writer.writerows(output_rows)

    # ---- Paired tests against the archived Chroma baseline -----------
    if "A_original_chroma" in per:
        tests = []
        base = per["A_original_chroma"]

        for name in names:
            if name == "A_original_chroma":
                continue

            for lang in ("all", "en", "ko"):
                ids = [i for i in base
                       if i in per[name]
                       and (lang == "all" or base[i]["lang"] == lang)
                       and (args.include_crisis_bypass
                            or not base[i]["crisis_bypass"])]
                if not ids:
                    continue

                b = sum(base[i]["no_rel"] and not per[name][i]["no_rel"]
                        for i in ids)
                c = sum((not base[i]["no_rel"]) and per[name][i]["no_rel"]
                        for i in ids)

                p_mcnemar = (binomtest(min(b, c), b + c, 0.5).pvalue
                             if b + c else 1.0)
                tests.append([name, lang, "no_relevant_McNemar", b, c,
                              float(p_mcnemar)])

                for metric in ("p3", "ndcg"):
                    diffs = [per[name][i][metric] - base[i][metric]
                             for i in ids]
                    p_w = (wilcoxon(diffs).pvalue
                           if any(x != 0 for x in diffs) else 1.0)
                    tests.append([name, lang, metric + "_Wilcoxon",
                                  float(np.mean(diffs)), "", float(p_w)])

        adjusted = holm([row[5] for row in tests])

        with open(work / "paired_tests.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["configuration", "language", "test",
                             "statistic_1", "statistic_2", "p_raw", "p_holm"])
            for row, adj in zip(tests, adjusted):
                writer.writerow(row + [adj])

        print("\nPaired tests saved to paired_tests.csv.")

    print("\nP@3 uses 3 as the denominator, so abstentions count as zero. "
          "Interpret P@3 alongside abstention rate.")


def cmd_agreement(args):
    work = Path(args.work)
    rows = read_csv_rows(work / "human_sample_labeled.csv")

    if not rows or any(not r["annotator_1_0_3"].strip()
                       or not r["annotator_2_0_3"].strip() for r in rows):
        sys.exit("Complete human_sample_labeled.csv first.")

    a = [parse_label(r["annotator_1_0_3"], "annotator_1") for r in rows]
    b = [parse_label(r["annotator_2_0_3"], "annotator_2") for r in rows]

    exact = np.mean(np.array(a) == np.array(b))
    binary = np.mean((np.array(a) >= 2) == (np.array(b) >= 2))

    result = (f"n={len(rows)}\n"
              f"Exact agreement={exact:.3f}\n"
              f"Linear-weighted Cohen kappa={weighted_kappa(a, b):.3f}\n"
              f"Binary agreement (relevance>=2)={binary:.3f}\n")

    (work / "human_agreement.txt").write_text(result, encoding="utf-8")
    print(result)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", default="rag_exp")

    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare")
    s.add_argument("--kb-zip", required=True)
    s.add_argument("--chroma-zip", required=True)
    s.add_argument("--generations", required=True)
    s.set_defaults(func=prepare)

    s = sub.add_parser("translate-draft")
    s.add_argument("--model", default="qwen2.5:7b-instruct")
    s.set_defaults(func=cmd_translate)

    s = sub.add_parser("check-kb")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("tune-export")
    s.add_argument("--tuning", required=True,
                   help="Independent JSONL messages: id, text, "
                        "optional stress_type")
    s.set_defaults(func=cmd_tune_export)

    s = sub.add_parser("tune-select")
    s.set_defaults(func=cmd_tune_select)

    s = sub.add_parser("retrieve")
    s.add_argument("--messages", default=None,
                   help="Defaults to <work>/messages_144.csv")
    s.add_argument("--skip-threshold", action="store_true",
                   help="Omit E_threshold; do not report threshold results.")
    s.set_defaults(func=cmd_retrieve)

    s = sub.add_parser("annotate-export")
    s.set_defaults(func=cmd_annotate)

    s = sub.add_parser("metrics")
    s.add_argument("--include-crisis-bypass", action="store_true",
                   help="Keep crisis-bypass messages in the metrics "
                        "(excluded by default).")
    s.set_defaults(func=cmd_metrics)

    s = sub.add_parser("agreement")
    s.set_defaults(func=cmd_agreement)

    args = parser.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)

    if args.cmd == "retrieve" and not args.messages:
        args.messages = str(Path(args.work) / "messages_144.csv")

    args.func(args)


if __name__ == "__main__":
    main()