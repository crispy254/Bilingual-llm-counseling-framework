"""Shared settings for the Nova multi-model experiment. EDIT THIS FIRST.

RUN 2: new 144-message test set (prompts_v2.csv), Gemma-based Nova variants.
Run 1 results (prompts.csv, 48 messages) stay untouched in generations.csv / judged.csv.
"""
RUN_NAME = "run2"

OLLAMA_URL = "http://localhost:11434"

# Candidate models (Ollama tags). Check yours with:  ollama list
# "<model>+guard" = same model with the check-and-retry guard (up to 3 attempts, rule-based strategy check).
MODELS = [
    "nova-counseling-v2",   # previous Nova (Llama 3.2 3B + LoRA)
    "llama3.2:3b",          # base of v2 (no fine-tuning)
    "gemma3:4b",            # new base model for Nova (no fine-tuning)
    "gemma3:4b+guard",      # new base + check-and-retry guard
    "qwen2.5:3b",
    "exaone3.5:7.8b",
    # nova-gemma-v3 and nova-gemma-v3+guard are added automatically below once the model exists in Ollama
]

def _installed():
    try:
        import requests
        return [m["name"] for m in requests.get(OLLAMA_URL + "/api/tags", timeout=3).json().get("models", [])]
    except Exception:
        return []

MAIN_NOVA = "nova-gemma-v3"   # the main Nova model: fine-tuned Gemma 3 4B
if any(n == MAIN_NOVA or n.startswith(MAIN_NOVA + ":") for n in _installed()):
    MODELS += [MAIN_NOVA, MAIN_NOVA + "+guard"]
    NOVA_MODEL = MAIN_NOVA
else:
    NOVA_MODEL = "gemma3:4b"   # until the fine-tuned model exists, Gemma 3 4B stands in as the main base

# Judge must NOT be one of the candidates (avoid self-preference).
# Gemma 3 4B fine-tuned on the same general counseling data as the Llama model (added automatically once it exists)
GENERAL_TAG = "gemma-general"
if any(n == GENERAL_TAG or n.startswith(GENERAL_TAG + ":") for n in _installed()):
    MODELS.append(GENERAL_TAG)

JUDGE_MODEL = "phi4:14b"

TEMPERATURE = 0.3
MAX_TOKENS = 300
SEED = 42
GUARD_ATTEMPTS = 3

PROMPTS_CSV = "../finetune_v3/prompts_v2.csv"   # the 144 NEW test messages (review them by hand first)
GENERATIONS_CSV = f"{RUN_NAME}_generations.csv"
JUDGED_CSV = f"{RUN_NAME}_judged.csv"