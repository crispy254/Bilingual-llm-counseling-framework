r"""
inference.py - FIXED: plural/boundary bug across multiple keywords.

Root cause found: patterns like r"\bexam\b" require a word boundary
immediately after "exam" - so "exams" (plural) never matches, since
there's no boundary between "exam" and the trailing "s". This silently
affected every plural-form keyword: exam/exams, course/courses,
friend/friends, job/jobs, grade/grades.

Fix: use r"exams?\b" style patterns (optional trailing "s") instead of
bare r"\bword\b" wherever a natural plural exists.

Also added: "stress\w*" to the overwhelmed emotion category - "I'm
stressed" / "I'm so stressed out" previously matched NO emotion
category at all (fell through to "neutral"), despite being one of the
single most common ways students actually phrase this.
"""
import re
from typing import Dict, List, Optional

CRISIS_PATTERNS = [
    r"\bkill myself\b", r"\bend my life\b", r"\bend it all\b",
    r"\bsuicid\w*", r"\bwant to die\b", r"\bdon'?t want to (be alive|live)\b",
    r"\bno reason to live\b", r"\bhurt myself\b", r"\bself[- ]?harm\b",
    r"죽고\s*싶", r"자살", r"자해", r"살기\s*싫", r"사라지고\s*싶",
    r"극단적\s*선택",
]

EMOTION_PATTERNS = {
    "anxious": [r"\banxious\b", r"\banxiety\b", r"\bworried\b", r"\bnervous\b",
                r"\bpanick\w*", r"불안", r"걱정"],
    "overwhelmed": [r"overwhelm\w*", r"too much", r"can'?t cope", r"burnt out",
                     r"burned out", r"\bstress\w*", r"지쳤", r"버겁"],  # ADDED stress\w*
    "sad": [r"\bsad\b", r"depress\w*", r"down lately", r"crying", r"우울", r"슬프"],
    "angry": [r"\bangry\b", r"frustrat\w*", r"irritat\w*", r"\bfight\w*", r"화가", r"짜증"],  # ADDED fight\w*
    "hopeful": [r"\bexcited\b", r"looking forward", r"\bhopeful\b", r"기대", r"희망"],
}

STRESS_TYPE_PATTERNS = {
    "academic": [r"\bexams?\b", r"\bgrades?\b", r"\bgpa\b", r"assignment", r"deadline",
                 r"\bclass(es)?\b", r"\bcourses?\b", r"\bfinals?\b", r"\bsemester\b",
                 r"\bproject\b", r"\bschool\b", r"\bhomework\b",
                 r"학점", r"성적", r"시험", r"과제", r"수업"],
    "career": [r"\bjobs?\b", r"\bcareer\b", r"internship", r"resume", r"\bcv\b",
               r"interview", r"scholarship", r"funding", r"financial aid",
               r"진로", r"취업", r"면접"],
    "interpersonal": [r"\bfriends?\b", r"relationship", r"\bfamily\b", r"roommates?\b",
                       r"\blonely\b", r"인간관계", r"외로"],
}


def check_crisis_risk(message: str) -> bool:
    text = message.lower()
    return any(re.search(p, text) for p in CRISIS_PATTERNS)


def _match_category(text: str, patterns: Dict[str, List[str]]) -> Optional[str]:
    for label, pats in patterns.items():
        for p in pats:
            if re.search(p, text):
                return label
    return None


def infer_emotional_state(message: str, history: Optional[List[Dict]] = None) -> Dict:
    text = message.lower()
    emotion = _match_category(text, EMOTION_PATTERNS) or "neutral"
    stress_type = _match_category(text, STRESS_TYPE_PATTERNS) or "general"
    return {
        "emotion": emotion,
        "stress_type": stress_type,
        "risk": check_crisis_risk(message),
    }