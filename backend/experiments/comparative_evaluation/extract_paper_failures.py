
import pandas as pd

# The six configurations included in the paper
paper_models = [
    "nova-gemma-v3",
    "nova-gemma-v3+guard",
    "gemma3:4b",
    "gemma3:4b+guard",
    "qwen2.5:3b",
    "exaone3.5:7.8b"
]

# Load the original generation and judging results
generations = pd.read_csv("run2_generations.csv")
judgments = pd.read_csv("run2_judged.csv")

# Keep only the required judgment columns
judgment_columns = [
    "model", "id", "strategy_used",
    "made_up_facts", "unsafe"
]

# Combine generated responses with their Phi-4 judgments
merged = generations.merge(
    judgments[judgment_columns],
    on=["model", "id"],
    how="inner"
)

# Keep only the six paper configurations
merged = merged[merged["model"].isin(paper_models)].copy()

# Convert judgment flags into reliable Boolean values
def to_bool(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    )

merged["made_up_facts"] = to_bool(merged["made_up_facts"])
merged["unsafe"] = to_bool(merged["unsafe"])

# Identify failures:
# 1. The strategy used differs from the intended strategy.
# 2. Phi-4 flags unsupported facts.
# 3. Phi-4 flags unsafe content.
failures = merged[
    (merged["strategy"] != merged["strategy_used"])
    | merged["made_up_facts"]
    | merged["unsafe"]
].copy()

# Select useful columns for manual inspection
columns = [
    "model", "id", "language", "strategy",
    "strategy_used", "made_up_facts", "unsafe",
    "message", "response"
]

failures = failures[columns]

# Save the results
failures.to_csv(
    "paper_failure_samples.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFailure examples by model:")
print(failures["model"].value_counts().reindex(paper_models, fill_value=0))

print("\nFailure examples by language:")
print(failures["language"].value_counts())

print("\nTotal failure examples:", len(failures))
print("\nSaved paper_failure_samples.csv")
