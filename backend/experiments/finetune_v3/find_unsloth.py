"""Finds which of your conda environments has Unsloth installed (the one used to train v2).
Run with any Python, from anywhere:  python find_unsloth.py"""
import json, os, subprocess, sys

def conda_envs():
    try:
        out = subprocess.run("conda env list --json", shell=True, capture_output=True, text=True, timeout=60).stdout
        return json.loads(out).get("envs", [])
    except Exception as e:
        print("Could not run `conda env list` (is conda on PATH? try the Anaconda/Miniconda PowerShell):", e)
        return []

def python_of(env):
    for p in (os.path.join(env, "python.exe"), os.path.join(env, "bin", "python")):
        if os.path.exists(p): return p
    return None

CHECK = ("import unsloth, torch, trl; "
         "print('OK torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())")
found = []
for env in conda_envs():
    py = python_of(env)
    if not py: continue
    print(f"checking {env} ...", flush=True)
    r = subprocess.run([py, "-c", CHECK], capture_output=True, text=True, timeout=600)
    if r.returncode == 0:
        line = [l for l in r.stdout.splitlines() if l.startswith("OK")][-1]
        print(f"  FOUND: {py}\n  {line}")
        found.append(py)
    else:
        print("  no unsloth here")
print()
if found:
    print("Use this Python for training:\n  " + found[0])
    print("\nSmoke test (from backend\\finetune_v3):\n  " + found[0] + " finetune_v3.py --max-steps 20")
    print("\nThen the rest:\n  python run_all.py --stages train import eval --train-python " + found[0])
else:
    print("No environment with Unsloth found. Either you trained v2 in WSL, or you need to install it (see the steps in the chat).")