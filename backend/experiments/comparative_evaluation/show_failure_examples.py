
import pandas as pd

df = pd.read_csv("paper_failure_samples.csv")

models = [
    "nova-gemma-v3",
    "nova-gemma-v3+guard",
    "gemma3:4b",
    "gemma3:4b+guard",
    "qwen2.5:3b",
    "exaone3.5:7.8b"
]

output = []

for model in models:
    for language in ["en", "ko"]:
        subset = df[
            (df["model"] == model) &
            (df["language"] == language)
        ]

        if subset.empty:
            continue

        # Select one representative example
        example = subset.sample(n=1, random_state=42).iloc[0]

        output.append({
            "model": model,
            "language": language,
            "id": example["id"],
            "intended_strategy": example["strategy"],
            "judge_strategy": example["strategy_used"],
            "made_up_facts": example["made_up_facts"],
            "unsafe": example["unsafe"],
            "message": str(example["message"])[:250],
            "response": str(example["response"])[:700]
        })

result = pd.DataFrame(output)

result.to_csv(
    "compact_failure_examples.csv",
    index=False,
    encoding="utf-8-sig"
)

with open("compact_failure_examples.txt", "w", encoding="utf-8") as f:
    for _, row in result.iterrows():
        f.write("\n" + "=" * 65 + "\n")
        f.write(f"MODEL: {row['model']}\n")
        f.write(f"LANGUAGE: {row['language']}\n")
        f.write(f"ID: {row['id']}\n")
        f.write(f"INTENDED STRATEGY: {row['intended_strategy']}\n")
        f.write(f"JUDGE STRATEGY: {row['judge_strategy']}\n")
        f.write(f"MADE-UP FACTS: {row['made_up_facts']}\n")
        f.write(f"UNSAFE: {row['unsafe']}\n")
        f.write(f"MESSAGE: {row['message']}\n")
        f.write(f"RESPONSE: {row['response']}\n")

print("Created 12 compact examples (where available).")
print("Open compact_failure_examples.txt and copy its contents.")
