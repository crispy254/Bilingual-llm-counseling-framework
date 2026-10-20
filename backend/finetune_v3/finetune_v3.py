"""
Fine-tunes a generator on chat-format JSONL (strategy-conditioned data, or the combined general + strategy data).

Default: Gemma 3 4B  -> output name nova-gemma-v3   (python finetune_v3.py)
Option : Llama 3.2 3B -> output name nova-counseling-v3 (python finetune_v3.py --family llama)
Always pass --out to choose the output folder name so earlier models are never overwritten.

FAIR COMPARISON: to compare the two base models, train both with the SAME files and the SAME settings:
  python finetune_v3.py --family gemma --train mixed_train.jsonl --dev mixed_dev.jsonl --out proposed-mixed
  python finetune_v3.py --family llama --train mixed_train.jsonl --dev mixed_dev.jsonl --out llama-mixed

Recipe (identical for both families): LoRA r=16, alpha=16, bf16, batch 4 x grad-accum 4, 3 epochs, lr 2e-4 cosine with 5%
warmup, loss on the reply only, best checkpoint by dev loss.

This version = your normalize_messages / --attn / --no-compile fixes + --no-gguf, the GGUF-failure fallback, and a
run summary file (<out>-run_summary.json) holding the data counts and losses you need for the paper.
Smoke test first:  --max-steps 20 --no-gguf
"""
import argparse, inspect, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["gemma", "llama"], default="gemma")
ap.add_argument("--train", default="strategy_train.jsonl")
ap.add_argument("--dev", default="strategy_dev.jsonl")
ap.add_argument("--replay", default=None, help="optional jsonl with {'messages': [...]} from earlier training data")
ap.add_argument("--replay-frac", type=float, default=0.25)
ap.add_argument("--base", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max-steps", type=int, default=-1, help="use e.g. 20 for a smoke test")
ap.add_argument("--max-len", type=int, default=2048, help="maximum tokens per example; longer examples are dropped (never truncated)")
ap.add_argument("--group-by-length", action="store_true", help="batch similar lengths together (less padding, faster); try it, drop it if it errors")
ap.add_argument("--batch-size", type=int, default=4, choices=[1, 2, 4], help="per-device batch size; gradient accumulation is adjusted so the effective batch stays 16 (use 2 if you run out of memory)")
ap.add_argument("--no-gguf", action="store_true", help="skip the GGUF export (use for the smoke test)")
ap.add_argument("--attn", default=None, choices=["eager", "sdpa", "flash_attention_2", "flex_attention"],
                help="attention implementation; use 'eager' if Triton flex-attention kernels fail on your GPU")
ap.add_argument("--no-compile", action="store_true",
                help="disable Unsloth's compiled kernels (slower, but avoids Triton/Inductor kernel failures)")
a = ap.parse_args()
if a.no_compile:
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"   # must be set before unsloth is imported

from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

CFG = {
    "gemma": dict(base="unsloth/gemma-3-4b-it", out="nova-gemma-v3", template="gemma-3",
                  inst="<start_of_turn>user\n", resp="<start_of_turn>model\n"),
    "llama": dict(base="unsloth/Llama-3.2-3B-Instruct", out="nova-counseling-v3", template="llama-3.2",
                  inst="<|start_header_id|>user<|end_header_id|>\n\n",
                  resp="<|start_header_id|>assistant<|end_header_id|>\n\n"),
}[a.family]
base, out = a.base or CFG["base"], a.out or CFG["out"]
summary = dict(family=a.family, base=base, out=out, train_file=a.train, dev_file=a.dev, replay=a.replay,
               epochs=a.epochs, lr=a.lr, max_steps=a.max_steps, batch_size=a.batch_size, attn=a.attn)


def normalize_messages(msgs):
    """One system message first; merge consecutive same-role turns; start with user; end with assistant."""
    if not msgs:
        return None
    system = [str(m["content"]).strip() for m in msgs if m.get("role") == "system" and str(m.get("content", "")).strip()]
    turns = []
    for m in msgs:
        role, content = m.get("role"), str(m.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    if not turns:
        return None
    return ([{"role": "system", "content": "\n\n".join(system)}] if system else []) + turns


def read(p, label):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    kept, changed, dropped = [], 0, 0
    for r in rows:
        fixed = normalize_messages(r["messages"])
        if not fixed:
            dropped += 1
            continue
        if fixed != r["messages"]:
            changed += 1
        r = dict(r); r["messages"] = fixed
        kept.append(r)
    print(f"{label}: {len(rows)} read, {changed} normalized, {dropped} dropped -> {len(kept)}")
    summary[f"{label}_read"], summary[f"{label}_normalized"] = len(rows), changed
    summary[f"{label}_dropped"], summary[f"{label}_kept"] = dropped, len(kept)
    return kept


train, dev = read(a.train, "train"), read(a.dev, "dev")
n_main = len(train)
if a.replay:
    rep = read(a.replay, "replay"); random.Random(42).shuffle(rep)
    train += rep[: int(len(train) * a.replay_frac)]
random.Random(42).shuffle(train)
print(f"family={a.family} base={base}  train={len(train)} (from --train: {n_main}, from --replay: {len(train) - n_main})  dev={len(dev)}")

if a.family == "gemma":
    from unsloth import FastModel
    kw = dict(model_name=base, max_seq_length=a.max_len, load_in_4bit=False, load_in_8bit=False, full_finetuning=False)
    if a.attn:
        kw["attn_implementation"] = a.attn
    try:
        model, tok = FastModel.from_pretrained(**kw)
    except TypeError as e:
        print("attn_implementation not accepted by this Unsloth version, loading without it:", e)
        kw.pop("attn_implementation", None)
        model, tok = FastModel.from_pretrained(**kw)
    print("attention implementation:", getattr(model.config, "_attn_implementation", "unknown"))
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True,
                                     r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=42)
else:
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(base, max_seq_length=a.max_len, dtype=None, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=42,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")
tok = get_chat_template(tok, chat_template=CFG["template"])


def to_text(rows):
    texts, skipped = [], 0
    for r in rows:
        try:
            t = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        except Exception as e:           # never let one bad record kill a multi-hour run
            skipped += 1
            if skipped <= 3: print("skipping record:", str(e)[:120])
            continue
        texts.append(t.removeprefix("<bos>"))
    return Dataset.from_dict({"text": texts}), skipped


def drop_too_long(ds, label):
    """Examples longer than --max-len would be truncated, cutting off the reply. Drop them instead and report."""
    try:
        tk = getattr(tok, "tokenizer", tok)
        lens = [len(tk(t, add_special_tokens=False)["input_ids"]) for t in ds["text"]]
    except Exception as e:
        print(f"{label}: could not measure token lengths ({str(e)[:80]}); keeping all")
        return ds, 0
    srt = sorted(lens)
    keep = [t for t, n in zip(ds["text"], lens) if n <= a.max_len]
    print(f"{label}: token length median {srt[len(srt)//2]}, 95th percentile {srt[int(len(srt)*0.95)]}, "
          f"max {srt[-1]}; dropped {len(lens) - len(keep)} longer than {a.max_len}")
    return Dataset.from_dict({"text": keep}), len(lens) - len(keep)


train_ds, sk_tr = to_text(train)
dev_ds, sk_dev = to_text(dev)
train_ds, too_long_tr = drop_too_long(train_ds, "train")
dev_ds, too_long_dev = drop_too_long(dev_ds, "dev")
summary.update(too_long_train=too_long_tr, too_long_dev=too_long_dev, max_len=a.max_len)
steps_epoch = -(-len(train_ds) // 16)
print(f"Optimizer steps: {steps_epoch} per epoch x {a.epochs} epochs = {steps_epoch * a.epochs} "
      f"(multiply by the seconds per step shown once training starts to estimate the time)")
summary.update(templated_train=len(train_ds), templated_dev=len(dev_ds), skipped_train=sk_tr, skipped_dev=sk_dev)
print(f"After templating: train={len(train_ds)} (skipped {sk_tr})  dev={len(dev_ds)} (skipped {sk_dev})")
print("Example training text:\n", train_ds[0]["text"][:600], "\n...")

cfg_params = inspect.signature(SFTConfig.__init__).parameters
cfg_kwargs = dict(
    output_dir=out + "-ckpt", dataset_text_field="text",
    per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=16 // a.batch_size,
    num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
    lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=True, logging_steps=10,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
    seed=42, report_to="none")
cfg_kwargs["max_length" if "max_length" in cfg_params else "max_seq_length"] = a.max_len
if a.group_by_length:
    cfg_kwargs["group_by_length"] = True
tr_params = inspect.signature(SFTTrainer.__init__).parameters
tok_kw = {"processing_class" if "processing_class" in tr_params else "tokenizer": tok}
trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=dev_ds,
                     args=SFTConfig(**cfg_kwargs), **tok_kw)
trainer = train_on_responses_only(trainer, instruction_part=CFG["inst"], response_part=CFG["resp"])
trainer.train()

try:   # losses for the paper; never allowed to break the run
    hist = trainer.state.log_history
    tr_losses = [h["loss"] for h in hist if "loss" in h]
    ev_losses = [h["eval_loss"] for h in hist if "eval_loss" in h]
    summary.update(final_train_loss=tr_losses[-1] if tr_losses else None, eval_losses=ev_losses,
                   best_eval_loss=min(ev_losses) if ev_losses else None, steps=trainer.state.global_step)
    json.dump(summary, open(out + "-run_summary.json", "w"), indent=1)
    print("Run summary saved to", out + "-run_summary.json", {k: summary.get(k) for k in ("final_train_loss", "best_eval_loss")})
except Exception as e:
    print("could not write run summary:", e)

model.save_pretrained(out + "-lora"); tok.save_pretrained(out + "-lora")
print("Saved LoRA adapter to", out + "-lora")
if a.no_gguf:
    raise SystemExit("--no-gguf given: skipping GGUF export. Smoke test finished OK.")
# GGUF for Ollama. Needs llama.cpp tooling that Unsloth builds on first use (can fail on Windows).
try:
    model.save_pretrained_gguf(out + "-gguf", tok, quantization_method="q4_k_m")
except Exception as e:
    print("\nGGUF export failed:", type(e).__name__, str(e)[:300])
    print("Saving merged 16-bit weights as a fallback to", out + "-merged")
    model.save_pretrained_merged(out + "-merged", tok, save_method="merged_16bit")
    raise SystemExit("Fallback saved. Convert it with llama.cpp (convert_hf_to_gguf.py) or paste the error above "
                     "in the chat; the LoRA adapter is safe in " + out + "-lora")
print(f"Done. To use in Ollama:\n"
      f"  python import_to_ollama.py --name <tag> --gguf-dir {out}-gguf_gguf --base {'gemma3:4b' if a.family == 'gemma' else 'llama3.2:3b'}")