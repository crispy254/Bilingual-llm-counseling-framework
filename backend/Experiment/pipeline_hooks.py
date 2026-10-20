"""
Connects the experiment to Nova's REAL pipeline (backend/prompt_builder.py and backend/rag.py).
Every model receives exactly the same messages: the same strategy rule block, persona,
retrieved chunks and language reminder that Nova itself would build.
Place this file in backend/experiment/ and run scripts from inside that folder.
"""
import os, sys
from functools import lru_cache

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

import prompt_builder   # backend/prompt_builder.py
import rag              # backend/rag.py

TOP_K = 3

@lru_cache(maxsize=None)
def _retrieve(message: str):
    """Retrieved once per message and cached, so all models see identical context."""
    return tuple(rag.retrieve(message, k=TOP_K))

def build_messages(row):
    """row: dict/Series with message, emotion, stress_type, language, strategy.
    Returns (messages_for_the_llm, context_text_for_logging)."""
    chunks = list(_retrieve(row["message"]))
    messages = prompt_builder.build_counseling_prompt(
        user_message=row["message"],
        history_summary=None,
        retrieved_chunks=chunks,
        emotion_info={"emotion": row["emotion"], "stress_type": row["stress_type"]},
        trend_info=None,
        language=row["language"],
        strategy=row["strategy"],
    )
    return messages, "\n".join(chunks)

def rag_is_working() -> bool:
    return len(rag.retrieve("exam anxiety and stress", k=1)) > 0