"""Removes one or more models' rows from the generations and judged files of the current run (exp_config.RUN_NAME),
after backing both up, so run_generation.py / run_judging.py redo only those models.
Usage (from the Experiment folder):
    python drop_model.py nova-gemma-v3 "nova-gemma-v3+guard\""""
import sys, shutil, pandas as pd
import exp_config as C
models = sys.argv[1:]
if not models: sys.exit("Give at least one model name, for example: python drop_model.py nova-gemma-v3 \"nova-gemma-v3+guard\"")
for f in [C.GENERATIONS_CSV, C.JUDGED_CSV]:
    try:
        shutil.copy(f, f.replace(".csv", "_backup.csv"))
        d = pd.read_csv(f)
        kept = d[~d["model"].isin(models)]
        kept.to_csv(f, index=False, encoding="utf-8-sig")
        print(f"{f}: removed {len(d) - len(kept)} rows for {', '.join(models)} (backup saved)")
    except FileNotFoundError:
        print(f"{f}: not found, skipped")