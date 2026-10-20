"""
Runs the whole nova-gemma-v3 pipeline in order. Run from backend/finetune_v3/:

    python run_all.py                       # everything: data -> train -> import -> eval
    python run_all.py --stages data         # only some stages: data | train | import | eval
    python run_all.py --dry-run             # print the commands without running them
    python run_all.py --train-python C:\\path\\to\\env\\python.exe   # if training needs another Python environment

Stages
  data   : synthetic English + Korean strategy data, then merge into strategy_train.jsonl (slow: hours)
  train  : fine-tune Gemma 3 4B (finetune_v3.py)  -> nova-gemma-v3-lora, nova-gemma-v3-gguf
  import : create the Ollama model nova-gemma-v3 from the GGUF
  eval   : ../Experiment: run_generation, run_judging, analyze, paired_tests (nova-gemma-v3 is added automatically)
Every step is resumable; if a step fails, fix the cause and re-run the same command.
"""
import argparse, json, os, subprocess, sys
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "Experiment"))
OLLAMA = "http://localhost:11434"

ap = argparse.ArgumentParser()
ap.add_argument("--stages", nargs="*", default=["data", "train", "import", "eval"])
ap.add_argument("--msgs-per-context", type=int, default=12)
ap.add_argument("--min-per-strategy", type=int, default=100,
                help="stop before training if the merged data has fewer examples per strategy than this")
ap.add_argument("--train-python", default=sys.executable)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

def run(cmd, cwd):
    print(f"\n>>> {' '.join(cmd)}   (in {cwd})", flush=True)
    if a.dry_run: return
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit("\nStep failed (see the message above). Fix it and re-run the same command.")

def unload_ollama_models():
    """Free GPU memory before training: unload every model Ollama currently holds."""
    if a.dry_run: print("(would unload Ollama models)"); return
    try:
        for m in requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", []):
            requests.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
            print("unloaded", m["name"])
    except Exception as e:
        print("could not unload models automatically:", e)

py = sys.executable
if "data" in a.stages:
    for lang in ("en", "ko"):
        run([py, "build_strategy_dataset.py", "synth", "--msgs-per-context", str(a.msgs_per_context), "--langs", lang], HERE)
    run([py, "build_strategy_dataset.py", "merge"], HERE)
    if not a.dry_run:
        n = {}
        for line in open(os.path.join(HERE, "strategy_train.jsonl"), encoding="utf-8"):
            s = json.loads(line)["strategy"]; n[s] = n.get(s, 0) + 1
        print("Examples per strategy:", n)
        if not n or min(n.values()) < a.min_per_strategy:
            sys.exit(f"Fewer than {a.min_per_strategy} examples for some strategy. Run the data stage again to add "
                     f"more (each run adds data), then re-run. Or lower --min-per-strategy if you accept less data.")

if "train" in a.stages:
    unload_ollama_models()
    run([a.train_python, "finetune_v3.py"], HERE)

if "import" in a.stages:
    run([py, "import_to_ollama.py"], HERE)

if "eval" in a.stages:
    for script in ("run_generation.py", "run_judging.py", "analyze.py", "paired_tests.py"):
        run([py, script], EXP)
print("\nAll requested stages finished.")