"""Imports the fine-tuned GGUF into Ollama using Gemma 3's own Modelfile (template and parameters),
changing ONLY the FROM line. Usage (from finetune_v3):
    python import_to_ollama.py                       # defaults: nova-gemma-v3, nova-gemma-v3-gguf, gemma3:4b
    python import_to_ollama.py --name nova-gemma-v3 --gguf-dir nova-gemma-v3-gguf --base gemma3:4b"""
import argparse, glob, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="nova-gemma-v3")
ap.add_argument("--gguf-dir", default=None)
ap.add_argument("--base", default="gemma3:4b")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
gguf_dir = a.gguf_dir or (a.name + "-gguf")

cands = []
for d in (gguf_dir, gguf_dir + "_gguf"):          # Unsloth writes the GGUF files to "<name>-gguf_gguf"
    cands = [f for f in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True)
             if "mmproj" not in os.path.basename(f).lower()]          # skip the vision projector file
    if cands:
        gguf_dir = d
        break
if not cands:
    sys.exit(f"No .gguf model file found in {gguf_dir} or {gguf_dir}_gguf. Did finetune_v3.py finish the GGUF export?")
pref = [f for f in cands if "q4_k_m" in os.path.basename(f).lower()]
gguf = os.path.abspath((pref or cands)[0]).replace("\\", "/")
print("Using GGUF:", gguf)

text = subprocess.run(["ollama", "show", a.base, "--modelfile"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
lines, replaced = [], False
for line in text.splitlines():
    if line.strip().upper().startswith("FROM ") and not replaced:
        lines.append(f"FROM {gguf}"); replaced = True
    else:
        lines.append(line)
if not replaced: sys.exit("Could not find a FROM line in the base Modelfile.")
mf = f"Modelfile_{a.name}"
open(mf, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"Wrote {mf}")
cmd = ["ollama", "create", a.name, "-f", mf]
print(">>>", " ".join(cmd))
if not a.dry_run:
    sys.exit(subprocess.run(cmd).returncode)