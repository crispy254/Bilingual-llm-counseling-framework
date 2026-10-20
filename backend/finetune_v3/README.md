# nova-gemma-v3: strategy-conditioned fine-tuning (Gemma 3 4B base)

Place this folder at `backend/finetune_v3/`. Run commands from inside it.

1. Download ESConv: https://github.com/thu-coai/Emotional-Support-Conversation (file `ESConv.json`) into this folder.
2. Pull helper models (already used in the experiment): `ollama pull gemma3:4b` and `ollama pull qwen3:4b`.
3. Build the retrieval index if you have not: `python ../rag.py` (from backend/).
4. Data:
   - `python build_strategy_dataset.py testset --per-context 3`   -> prompts_v2.csv (144 NEW test messages; review by hand)
   - `python build_strategy_dataset.py esconv --esconv ESConv.json`
   - `python build_strategy_dataset.py synth --msgs-per-context 15`   (slow; re-run adds more data)
   - `python build_strategy_dataset.py merge`
   Inspect 30 random lines of strategy_train.jsonl yourself before training.
5. Train: `python finetune_v3.py --max-steps 20` (smoke test), then `python finetune_v3.py` (Gemma 3 4B -> nova-gemma-v3; use `--family llama` for Llama 3.2 3B).
   Then import it into Ollama as printed at the end of the script (base the Modelfile on `ollama show gemma3:4b --modelfile`, change only FROM).
6. Evaluate: in Experiment/exp_config.py, uncomment `nova-gemma-v3` and `nova-gemma-v3+guard` in MODELS, then re-run run_generation.py, run_judging.py, analyze.py (finished models are skipped).
Do not change the data or prompts after you have looked at the test results.