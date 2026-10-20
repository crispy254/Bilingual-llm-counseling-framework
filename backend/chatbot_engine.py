"""
============================================================
AI Academic Success Companion - Chatbot Engine
============================================================

Purpose
-------
Main controller for the chatbot: crisis check, intent detection, RAG +
emotion inference for counseling intents, RL strategy selection for
counseling intents, prompt construction, and LLM generation.

RL integration (strategy_selector.py)
--------------------------------------
For counseling-intent messages only, after inferring emotion/stress
type, the bandit in strategy_selector.py picks ONE of four pre-written
response strategies (see prompt_builder.STRATEGIES / _STRATEGY_RULE_BLOCKS)
based on that context. The chosen strategy is returned in the result
dict as "strategy", so the caller (chatbot_service.py) can log it -
that log is both how the bandit learns (once feedback arrives) and the
raw evaluation data for the research writeup.

The bandit NEVER runs for crisis-risk messages (those short-circuit
before intent detection even happens) and never runs for academic
intents (gpa/study_plan/quiz/explanation) - it is scoped exclusively
to the counseling-support response style, by design.

Intent coverage (why this matters for RL)
-------------------------------------------
The RL bandit only ever sees a (context, action, reward) sample when a
message is classified into a COUNSELING_INTENTS category. If a
student's real phrasing doesn't match any INTENT_PATTERNS regex, the
message falls into "general" - which never touches RAG, emotion
inference, or the bandit at all, and never gets a feedback_id, so no
thumbs-up/down is even possible on that turn.

Two things address this:
  1. INTENT_PATTERNS below is broadened to catch common bare phrasings
     ("am stressed", "i'm stressed") in addition to cause-specific ones.
  2. A generic fallback in detect_intent(): if nothing else matches but
     the message contains clear emotional-distress language, it routes
     to "emotional_checkin" (a COUNSELING_INTENTS member) rather than
     "general".

Multi-turn history workaround (nova-counseling-v2)
-----------------------------------------------------
nova-counseling-v2 was fine-tuned almost entirely on single-turn
examples (CounselChat, Amod, Career Guidance QA, and the context-free
Dolly subset are all one-shot Q&A). A standalone test confirmed it
answers a single message coherently, but when the app fed it 5-6 raw
prior turns (see chatbot_service._recent_history), it started losing
track of the actual current message - repeating an old assistant
answer verbatim, inventing unstated details ("...about your exams
this week" when exams were never mentioned), or replying with
non-sequitur filler ("You are correct, I am sorry about that").

Rather than retraining, summarize_history() below compresses prior
turns into a single deterministic topic-summary LINE (e.g. "Earlier in
this conversation, the student touched on career worries, then
academic stress") instead of handing the model the raw exchange. Only
the CURRENT user message is ever sent as an actual conversational
turn to the LLM for counseling intents - see build_counseling_prompt
in prompt_builder.py, which embeds this summary in the system prompt
instead of extending the messages list with old turns. This keeps the
model in the single-turn regime it was actually trained on and
observed to handle correctly, while still giving it *some* awareness
of what's already been discussed.

This module is stateless. Sessions, persistence, and the database are
handled by other layers (see services/chatbot_service.py).
"""

import re
from typing import Optional, List, Dict

import llm
import prompt_builder
import inference
import rag
import strategy_selector


INTENT_PATTERNS = {
    "career_anxiety": [
        r"job", r"career", r"internship", r"resume", r"\bcv\b",
        r"interview", r"진로", r"취업", r"면접", r"what should i do after (i )?graduat"
    ],
    "stress_academic": [
        r"stressed.*(exam|class|grade|assignment|deadline)",
        r"(exam|class|grade|assignment|deadline).*stress",
        r"burnt? out", r"burned out", r"overwhelm", r"can'?t keep up",
        r"\bam stressed\b", r"\bi'?m stressed\b", r"\bso stressed\b",
        r"\breally stressed\b", r"\bfeeling stressed\b", r"\bstressed out\b",
        r"학업\s*스트레스", r"시험\s*스트레스"
    ],
    "stress_interpersonal": [
        r"lonely", r"isolated", r"fight with (my )?(friend|roommate|family)",
        r"relationship trouble", r"인간관계", r"외로", r"고립"
    ],
    "emotional_checkin": [
        r"feeling (down|low|anxious|sad|overwhelmed)",
        r"not (feeling|doing) (ok|okay|well|good)",
        r"having a hard time", r"i'?m struggling",
        r"기분이\s*안좋", r"힘들어"
    ],
    "gpa": [
        r"\bgpa\b", r"grade point", r"improve.*grade", r"raise my grade",
        r"academic performance", r"성적", r"학점", r"평점"
    ],
    "study_plan": [
        r"study plan", r"revision plan", r"study schedule", r"prepare for",
        r"plan.*exam", r"how should i study", r"공부\s*계획", r"학습\s*계획"
    ],
    "quiz": [
        r"\bquiz\b", r"practice question", r"mock exam", r"mock test",
        r"test me", r"give me questions", r"퀴즈", r"문제\s*내", r"연습\s*문제"
    ],
    "explanation": [
        r"^explain", r"define", r"what is", r"what are", r"how does",
        r"can you explain", r"설명해", r"이란\s*무엇", r"뭐야\??$"
    ]
}

COUNSELING_INTENTS = {
    "career_anxiety",
    "stress_academic",
    "stress_interpersonal",
    "emotional_checkin",
}

GENERIC_DISTRESS_PATTERNS = [
    r"\bstress(ed)?\b", r"\banxious\b", r"\banxiety\b", r"\bworried\b",
    r"\bexhausted\b", r"\btired of\b", r"\bstruggling\b", r"\boverwhelmed\b",
    r"\bsad\b", r"\bdown\b",
    r"스트레스", r"불안", r"걱정", r"힘들",
]

# Human-readable labels used by summarize_history() below. "general" is
# intentionally mapped to None - casual chit-chat isn't worth
# summarizing, and academic-intent turns (gpa/quiz/etc.) are separate,
# single-turn flows that don't currently need history context anyway.
_INTENT_TOPIC_LABELS = {
    "career_anxiety": "career worries",
    "stress_academic": "academic stress",
    "stress_interpersonal": "a relationship or social concern",
    "emotional_checkin": "how they've been feeling",
    "gpa": "GPA questions",
    "study_plan": "a study plan",
    "quiz": "practice questions",
    "explanation": "a concept explanation",
    "general": None,
}


def detect_intent(message: str) -> str:
    text = message.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent

    for pattern in GENERIC_DISTRESS_PATTERNS:
        if re.search(pattern, text):
            return "emotional_checkin"

    return "general"


def summarize_history(history: Optional[List[Dict]]) -> Optional[str]:
    """
    Builds a short ONE-LINE summary of topics touched on earlier in
    the conversation, instead of the raw verbatim turns - see module
    docstring for why (nova-counseling-v2 loses coherence on raw
    multi-turn input).

    Walks prior USER turns only (assistant turns are ignored - we only
    care what topics were raised, not exact prior wording, since exact
    wording is precisely what we don't want to hand back to the model
    to potentially copy), classifies each with the same detect_intent()
    used for routing, and joins consecutive-unique topic labels into
    one sentence. Returns None if there's no history or nothing worth
    summarizing (e.g. it was all casual chit-chat).
    """
    if not history:
        return None

    topics: List[str] = []
    for turn in history:
        if turn.get("role") != "user":
            continue
        intent = detect_intent(turn.get("content", ""))
        label = _INTENT_TOPIC_LABELS.get(intent)
        if label and (not topics or topics[-1] != label):
            topics.append(label)

    if not topics:
        return None
    if len(topics) == 1:
        return f"Earlier in this conversation, the student asked about {topics[0]}."
    return f"Earlier in this conversation, the student touched on {', then '.join(topics)}."


MAX_TOKENS_BY_INTENT = {
    "general": 180, "gpa": 160, "explanation": 200, "study_plan": 300,
    "quiz": 380, "career_anxiety": 180, "stress_academic": 180,
    "stress_interpersonal": 180, "emotional_checkin": 160,
}

PROMPT_BUILDERS = {
    "gpa": prompt_builder.build_gpa_prompt,
    "study_plan": prompt_builder.build_study_plan_prompt,
    "quiz": prompt_builder.build_quiz_prompt,
    "explanation": prompt_builder.build_explanation_prompt,
}


def handle_message(
    user_message: str,
    history: Optional[List[Dict]] = None,
    trend_info: Optional[Dict] = None,
    language: str = "en",
) -> Dict:
    """
    Returns:
        {
            "intent": "...",
            "response": "...",
            "emotion": {...} | None,
            "strategy": "validate_ask" | ... | None,   # only set for counseling intents
            "error": None
        }
    """
    if not user_message.strip():
        return {"intent": None, "response": "", "emotion": None, "strategy": None,
                "error": "Message cannot be empty."}

    if inference.check_crisis_risk(user_message):
        return {
            "intent": "crisis",
            "response": prompt_builder.get_crisis_response(language),
            "emotion": {"emotion": "crisis", "stress_type": "crisis", "risk": True},
            "strategy": None,  # RL never touches crisis handling
            "error": None,
        }

    intent = detect_intent(user_message)
    emotion_info = None
    strategy = None

    if intent in COUNSELING_INTENTS:
        emotion_info = inference.infer_emotional_state(user_message, history)

        # --- RL strategy selection happens here, and only here ---
        strategy, _scores = strategy_selector.select_strategy(
            emotion_info["emotion"], emotion_info["stress_type"]
        )

        rag_category = "career_employment" if intent == "career_anxiety" else None
        retrieved_chunks = rag.retrieve(user_message, k=3, category=rag_category)

        # NOTE: raw `history` is deliberately NOT passed to the prompt
        # builder here - only a one-line summary is. See
        # summarize_history() docstring for why.
        history_summary = summarize_history(history)
        messages = prompt_builder.build_counseling_prompt(
            user_message, history_summary, retrieved_chunks, emotion_info,
            trend_info, language, strategy,
        )
    elif intent == "general":
        messages = prompt_builder.build_messages(user_message, history, language)
    else:
        messages = PROMPT_BUILDERS[intent](user_message, history, language)

    try:
        token_cap = MAX_TOKENS_BY_INTENT.get(intent, 200)
        response = llm.generate_response(messages, max_tokens=token_cap)
        return {
            "intent": intent, "response": response, "emotion": emotion_info,
            "strategy": strategy, "error": None,
        }
    except llm.LLMConnectionError as e:
        return {
            "intent": intent, "response": "", "emotion": emotion_info,
            "strategy": strategy, "error": str(e),
        }