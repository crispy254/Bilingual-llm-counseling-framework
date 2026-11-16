"""Repairs generations.csv / judged.csv when some rows have extra trailing columns
(the old header has fewer columns than rows added by a newer script version).
Makes a backup first. Usage (from the Experiment folder):  python fix_csv.py"""
import csv, shutil, sys
csv.field_size_limit(10**9)
for f in ["generations.csv", "judged.csv"]:
    try:
        with open(f, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    except FileNotFoundError:
        print(f"{f}: not found, skipped"); continue
    header, body = rows[0], rows[1:]
    extra = sum(1 for r in body if len(r) > len(header))
    short = sum(1 for r in body if len(r) < len(header))
    if not extra and not short:
        print(f"{f}: OK ({len(body)} rows)"); continue
    shutil.copy(f, f.replace(".csv", "_before_fix.csv"))
    fixed = [r[:len(header)] + [""] * (len(header) - len(r)) for r in body]
    with open(f, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(fixed)
    print(f"{f}: fixed {extra} rows with extra columns and {short} short rows ({len(fixed)} rows total)")