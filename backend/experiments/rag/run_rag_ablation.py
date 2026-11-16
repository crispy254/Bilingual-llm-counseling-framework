
"""
RAG Ablation Experiment
=======================

Compares Nova with retrieval enabled and disabled.

Both conditions use:
- The same Nova model
- The same 144 student messages
- The same strategy instructions
- The same generation settings
- The same random seed

Only retrieved reference passages are changed.

Run from backend/experiment:

    python run_rag_ablation.py

Output:
    rag_ablation_generations.csv

The experiment is resumable.
"""

import os
import pandas as pd

import exp_config as C
import pipeline_hooks as H
import run_generation as G


OUTPUT_CSV = "rag_ablation_generations.csv"

CONDITIONS = [
    ("rag_on", True),
    ("rag_off", False),
]


def append_row(path, row):
    df = pd.DataFrame([row])

    if os.path.exists(path):
        columns = list(pd.read_csv(path, nrows=0).columns)
        df = df.reindex(columns=columns)
        df.to_csv(
            path,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig"
        )
    else:
        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig"
        )


def main():

    # Ensure the actual fine-tuned Nova model is being evaluated.
    if C.NOVA_MODEL != C.MAIN_NOVA:
        raise RuntimeError(
            f"Expected {C.MAIN_NOVA}, but configuration selected "
            f"{C.NOVA_MODEL}. Check that the fine-tuned Nova model "
            "is installed in Ollama before running this experiment."
        )

    model = C.NOVA_MODEL

    # Ensure retrieval is available for the RAG-enabled condition.
    if not H.rag_is_working():
        raise RuntimeError(
            "RAG returned no results. Rebuild the knowledge base "
            "using rag.py before running this experiment."
        )

    prompts = pd.read_csv(C.PROMPTS_CSV)

    prompts = prompts.dropna(subset=["message"])
    prompts = prompts[
        prompts["message"].astype(str).str.strip() != ""
    ]

    # Resume support.
    done = set()

    if os.path.exists(OUTPUT_CSV):
        previous = pd.read_csv(OUTPUT_CSV)

        done = set(
            zip(
                previous["condition"].astype(str),
                previous["id"].astype(str)
            )
        )

    print("=" * 60)
    print("RAG ABLATION EXPERIMENT")
    print("=" * 60)
    print("Model:", model)
    print("Prompts:", len(prompts))
    print("Temperature:", C.TEMPERATURE)
    print("Seed:", C.SEED)
    print("Output:", OUTPUT_CSV)
    print("=" * 60)

    for _, p in prompts.iterrows():

        for condition, use_rag in CONDITIONS:

            key = (condition, str(p["id"]))

            if key in done:
                print(
                    f"Skipping {condition}, "
                    f"id {p['id']} (already completed)"
                )
                continue

            print(
                f"\nGenerating: {condition} | "
                f"id {p['id']} | {p['language']}"
            )

            try:

                # Build the same Nova prompt with RAG on or off.
                messages, context = H.build_messages(
                    p,
                    use_rag=use_rag
                )

                # No guard: isolate the effect of retrieval.
                response, wall, tokens, tps = G.chat(
                    model,
                    messages,
                    seed=C.SEED,
                    temperature=C.TEMPERATURE
                )

                row = {
                    "model": model,
                    "condition": condition,
                    "id": p["id"],
                    "language": p["language"],
                    "emotion": p["emotion"],
                    "stress_type": p["stress_type"],
                    "strategy": p["strategy"],
                    "message": p["message"],
                    "context": context,
                    "response": response,
                    "latency_s": round(wall, 2),
                    "out_tokens": tokens,
                    "tokens_per_s": round(tps, 1),
                    "reasoning_leak": bool(
                        G.LEAK.match(response)
                    )
                }

                append_row(OUTPUT_CSV, row)

                print(
                    f"Completed {condition} | "
                    f"{wall:.1f}s | {tokens} tokens"
                )

            except Exception as e:

                print(
                    f"ERROR: {condition}, "
                    f"id {p['id']}: {e}"
                )

    print("\nRAG ablation generation completed.")
    print("Results saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
