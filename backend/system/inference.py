r"""
inference.py - rule-based intent/emotion/stress inference + crisis screening.

Crisis detection now lives in crisis_detector.py (same folder). This file
re-exports it, so chatbot_engine.py keeps calling inference.check_crisis_risk(...).
The previous crisis patterns are kept as check_crisis_risk_v1 (baseline only).
"""
import re
from typing import Dict, List, Optional

from crisis_detector import check_crisis_risk   # noqa: F401  (v2 detector, used everywhere)

# ---------------------------------------------------------------------
# Previous detector (baseline for the paper; NOT used at runtime)
# ---------------------------------------------------------------------
_V1_PATTERNS = [
    r"\bkill\s+myself\b", r"\bkilling\s+myself\b", r"\bend\s+my\s+life\b", r"\bend\s+it\s+all\b", r"\bsuicid\w*",
    r"\bwant\s+to\s+die\b", r"\bwish\s+i\s+were\s+dead\b", r"\bi'?d\s+rather\s+be\s+dead\b",
    r"\bdon'?t\s+want\s+to\s+(?:be\s+alive|live)\b", r"\bno\s+reason\s+to\s+live\b", r"\bhurt\w*\s+myself\b",
    r"\bself[- ]?harm\b", r"\bself[- ]?injur\w*", r"살고\s*싶지\s*않", r"살아가고\s*싶지\s*않", r"살아야\s*할\s*이유가\s*없",
    r"내가\s*없(?:는|어지는)\s*(?:게|것이)\s*(?:모두에게\s*)?나을",
    r"\bbetter\s+off\s+without\s+me\b", r"\beveryone\s+would\s+be\s+better\s+off\s+without\s+me\b",
    r"\bcan'?t\s+go\s+on\s+(?:anymore|like\s+this)\b", r"\bcannot\s+go\s+on\s+(?:anymore|like\s+this)\b",
    r"\bno\s+way\s+out\b", r"\bno\s+way\s+to\s+keep\s+going\b", r"\bhope\s+i\s+don'?t\s+wake\s+up\b",
    r"\bwish\s+i\s+wouldn'?t\s+wake\s+up\b", r"\bi\s+want\s+to\s+disappear\s+forever\b", r"\bi\s+wish\s+i\s+could\s+disappear\b",
    r"죽고\s*싶", r"죽어\s*버리고\s*싶", r"자살", r"자해", r"살기\s*싫", r"살\s*이유가\s*없", r"살아갈\s*이유가\s*없",
    r"삶을\s*끝내", r"목숨을\s*끊", r"극단적\s*선택", r"더\s*이상\s*못\s*버티", r"더\s*이상\s*견딜\s*수\s*없",
    r"사라지고\s*싶", r"눈을\s*뜨지\s*않았으면", r"태어나지\s*않았으면", r"내가\s*없어지는\s*게\s*나을", r"내가\s*없는\s*게\s*나을",
]
_V1_NEGATIONS = [
    r"\b(?:i\s+)?am\s+not\s+suicidal\b", r"\b(?:i\s+)?am\s+not\s+thinking\s+about\s+suicide\b", r"\b(?:i\s+)?am\s+not\s+considering\s+suicide\b",
    r"\b(?:i\s+)?have\s+no\s+suicidal\s+thoughts?\b", r"\bno\s+suicide\s+ideation\b", r"\b(?:i\s+)?do\s+not\s+have\s+suicidal\s+thoughts?\b",
    r"\b(?:i\s+)?don'?t\s+want\s+to\s+die\b", r"\b(?:i\s+)?do\s+not\s+want\s+to\s+die\b", r"\b(?:i\s+)?am\s+not\s+thinking\s+of\s+hurting\s+myself\b",
    r"죽고\s*싶지\s*않", r"죽을\s*생각\s*(?:은|이)?\s*없", r"자살(?:을)?\s*(?:할\s*)?생각\s*(?:은|이)?\s*없", r"자해(?:를)?\s*(?:할\s*)?생각\s*(?:은|이)?\s*없",
    r"자살하고\s*싶지\s*않", r"자해하고\s*싶지\s*않", r"죽고\s*싶은\s*마음\s*(?:은\s*)?없", r"살고\s*싶지\s*않은\s*게\s*아니",
]
def check_crisis_risk_v1(message: str) -> bool:
    if not message or not message.strip():
        return False
    text = message.lower()
    for p in _V1_NEGATIONS:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _V1_PATTERNS)

# ---------------------------------------------------------------------
# Emotion / stress inference (unchanged)
# ---------------------------------------------------------------------
EMOTION_PATTERNS = {
    "anxious": [r"\banxious\b", r"\banxiety\b", r"\bworried\b", r"\bnervous\b",
                r"\bpanick\w*", r"불안", r"걱정"],
    "overwhelmed": [r"overwhelm\w*", r"too much", r"can'?t cope", r"burnt out",
                     r"burned out", r"\bstress\w*", r"지쳤", r"버겁"],
    "sad": [r"\bsad\b", r"depress\w*", r"down lately", r"crying", r"우울", r"슬프"],
    "angry": [r"\bangry\b", r"frustrat\w*", r"irritat\w*", r"\bfight\w*", r"화가", r"짜증"],
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