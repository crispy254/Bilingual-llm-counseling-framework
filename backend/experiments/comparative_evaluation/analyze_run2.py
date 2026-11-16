
import pandas as pd
from pathlib import Path

# Run this script from backend/experiment
GEN_FILE = Path("run2_generations.csv")
JUDGE_FILE = Path("run2_judged.csv")

if not GEN_FILE.exists():
    raise FileNotFoundError("Cannot find run2_generations.csv")
if not JUDGE_FILE.exists():
    raise FileNotFoundError("Cannot find run2_judged.csv")

gen = pd.read_csv(GEN_FILE)
judge = pd.read_csv(JUDGE_FILE)

def bool_col(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    )

def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

# --------------------------------------------------
# 1. Model and language counts
# --------------------------------------------------
section("1. GENERATION COUNTS")

print("Total generated rows:", len(gen))
print("\nRows per model:")
print(gen.groupby("model").size().to_string())

print("\nRows per model and language:")
print(gen.groupby(["model", "language"]).size().to_string())

print("\nUnique input messages:")
print(gen["id"].nunique())

print("\nInput language distribution:")
print(
    gen.drop_duplicates("id")["language"]
    .value_counts().to_string()
)

# --------------------------------------------------
# 2. Missing and duplicate model-input pairs
# --------------------------------------------------
section("2. DATA COMPLETENESS")

duplicates = gen[gen.duplicated(["model", "id"], keep=False)]
print("Duplicate model-input rows:", len(duplicates))

print("\nUnique IDs per model:")
print(gen.groupby("model")["id"].nunique().to_string())

print("\nExpected rows per model if all 144 inputs were processed: 144")

# --------------------------------------------------
# 3. Guard performance
# --------------------------------------------------
section("3. GUARD RESULTS")

if "guard_pass" in gen.columns:
    gen["guard_bool"] = bool_col(gen["guard_pass"])

    print("\nGuard pass rate by model:")
    print(
        (gen.groupby("model")["guard_bool"].mean() * 100)
        .round(2).to_string()
    )

    print("\nGuard pass rate by model and language:")
    print(
        (gen.groupby(["model", "language"])["guard_bool"].mean() * 100)
        .round(2).to_string()
    )
else:
    print("guard_pass column not found.")

if "attempts" in gen.columns:
    print("\nAverage attempts by model:")
    print(gen.groupby("model")["attempts"].mean().round(2).to_string())

    print("\nAttempt distribution:")
    print(pd.crosstab(gen["model"], gen["attempts"]).to_string())

# --------------------------------------------------
# 4. Join generation and judge results
# --------------------------------------------------
section("4. JUDGING DATA COMPLETENESS")

keys = ["model", "id"]

gen_keys = gen[keys].drop_duplicates()
judge_keys = judge[keys].drop_duplicates()

missing_judgments = gen_keys.merge(
    judge_keys, on=keys, how="left", indicator=True
)
missing_judgments = missing_judgments[
    missing_judgments["_merge"] == "left_only"
]

print("Generation rows without a corresponding judgment:",
      len(missing_judgments))

print("\nJudged rows per model:")
print(judge.groupby("model").size().to_string())

# --------------------------------------------------
# 5. Phi-4 results
# --------------------------------------------------
section("5. PHI-4 JUDGE RESULTS")

if "strategy_used" in judge.columns:
    judge["strategy_adherent"] = (
        judge["strategy"].astype(str).str.strip()
        == judge["strategy_used"].astype(str).str.strip()
    )

    print("\nStrategy adherence by model:")
    print(
        (judge.groupby("model")["strategy_adherent"].mean() * 100)
        .round(2).to_string()
    )

    print("\nStrategy adherence by model and language:")
    print(
        (judge.groupby(["model", "language"])["strategy_adherent"].mean() * 100)
        .round(2).to_string()
    )

    print("\nStrategy adherence by intended strategy:")
    print(
        (judge.groupby(["model", "strategy"])["strategy_adherent"].mean() * 100)
        .round(2).to_string()
    )

if "made_up_facts" in judge.columns:
    judge["made_up_bool"] = bool_col(judge["made_up_facts"])
    print("\nMade-up fact rate by model (%):")
    print(
        (judge.groupby("model")["made_up_bool"].mean() * 100)
        .round(2).to_string()
    )

if "unsafe" in judge.columns:
    judge["unsafe_bool"] = bool_col(judge["unsafe"])
    print("\nUnsafe response rate by model (%):")
    print(
        (judge.groupby("model")["unsafe_bool"].mean() * 100)
        .round(2).to_string()
    )

# --------------------------------------------------
# 6. Failure examples for manual inspection
# --------------------------------------------------
section("6. FAILURE EXAMPLES")

conditions = []

if "strategy_used" in judge.columns:
    conditions.append(judge["strategy_adherent"] == False)

if "made_up_bool" in judge.columns:
    conditions.append(judge["made_up_bool"])

if "unsafe_bool" in judge.columns:
    conditions.append(judge["unsafe_bool"])

if conditions:
    failure_mask = conditions[0]
    for condition in conditions[1:]:
        failure_mask = failure_mask | condition

    failures = judge[failure_mask].copy()

    columns = [
        "model", "id", "language", "strategy",
        "strategy_used", "made_up_facts", "unsafe",
        "message", "response"
    ]

    columns = [c for c in columns if c in failures.columns]

    failures[columns].head(40).to_csv(
        "run2_failure_samples.csv", index=False, encoding="utf-8-sig"
    )

    print("Total rows with at least one identified issue:", len(failures))
    print("Saved up to 40 examples in run2_failure_samples.csv")
else:
    print("No usable judge columns found for failure analysis.")

# --------------------------------------------------
# 7. Save complete audit
# --------------------------------------------------
print("\nAudit completed.")
print("Original experiment files were not modified.")
