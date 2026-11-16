"""Rule-based checks, label mappings and de-duplication helpers shared by the dataset builder.
These rules are INDEPENDENT of the judge model used to evaluate Nova, which keeps the
training data from being tuned to the evaluation judge's quirks."""
import re

# ESConv supporter strategy -> Nova strategy (others are skipped)
STRATEGY_MAP = {
    "Question": "validate_ask",
    "Reflection of feelings": "validate_listen",
    "Restatement or Paraphrasing": "validate_listen",
    "Providing Suggestions": "validate_suggest",
    "Affirmation and Reassurance": "validate_normalize",
}

def map_emotion(esconv_emotion: str) -> str:
    e = (esconv_emotion or "").lower()
    if e in ("anxiety", "fear", "nervousness"): return "anxious"
    if e in ("depression", "sadness", "pain", "guilt", "shame"): return "sad"
    if e in ("anger", "disgust", "jealousy"): return "angry"
    return "neutral"

def map_stress(problem_type: str) -> str:
    p = (problem_type or "").lower()
    if p in ("academic pressure", "procrastination"): return "academic"
    if p == "job crisis": return "career"
    if p in ("breakup with partner", "problems with friends", "conflict with parents",
             "issues with parents", "school bullying"): return "interpersonal"
    return "general"

HANGUL = re.compile(r"[\uac00-\ud7a3]")
def hangul_share(t):
    return len(HANGUL.findall(t)) / max(1, len(re.findall(r"\S", t)))

def language_ok(text, lang):
    h = hangul_share(text)
    return h > 0.3 if lang == "ko" else h < 0.1

# ---- cues
ADVICE_EN = re.compile(r"\b(try|maybe|could|should|consider|might want|suggest|recommend|how about|why not|make sure|you can|it may help|it might help|one thing)\b", re.I)
SUGGEST_EN = re.compile(r"\b(try|maybe|could|consider|might want|suggest|recommend|how about|you can|it may help|it might help|one thing|take a|set aside|write down|reach out|talk to|break)\b", re.I)
NORMAL_EN = re.compile(r"\b(common|normal|understandable|natural|many (people|students)|lots of (people|students)|a lot of (people|students)|others|not alone|anyone|everyone|so many|often)\b", re.I)
ADVICE_KO = re.compile(r"(해\s?보|시도|추천|좋을 것 같|보세요|해\s?봐|권해|써\s?보|정리해)")
NORMAL_KO = re.compile(r"(흔|누구나|많은 학생|많은 사람|자연스러|당연|혼자가 아니|다들|많은 분)")
SELF_DISCLOSURE = re.compile(r"\bwhen I\b|\bI (once|went through|have been through|had to|used to|remember when)\b|\bmy (own|mom|mother|dad|father|wife|husband|boyfriend|girlfriend|sister|brother|job|boss|friend|son|daughter)\b|\bme too\b", re.I)
BAD_ANY = re.compile(r"(https?://|www\.|as an ai\b|language model|take care|good luck|have a (great|good|nice)|goodbye|\bbye\b|\bhello\b|\bhi there\b|thank you for (sharing|your time))", re.I)

VALIDATION_EN = re.compile(r"\b(sounds?|hear|feel(ing|s)?|understand|must (be|feel|have been)|that'?s|that is|sorry|difficult|hard|tough|stress(ful|ed)?|overwhelm\w*|frustrat\w*|worr\w*|anxi\w*|scary|lonely|upset|painful|exhaust\w*|heavy|a lot)\b", re.I)
INTERROG_START = re.compile(r"(^|[.!]\s+)(do you|did you|are you|is it|have you|would you|could you|can you|will you|what|how|why|when|where|who)\b", re.I)

REASONING_LEAK = re.compile(r"^\s*(okay|ok,|alright|hmm|let me|first,|so,? the (user|student)|the (user|student)|i need to|i should|<think>)\b", re.I)

def sentences(text):
    return [s for s in re.split(r"(?<=[.!?。？！])\s+", text.strip()) if s]

def rule_check(strategy, text, lang="en", max_words=80):
    """Returns True if the reply plausibly follows the strategy's rule block."""
    t = " ".join(text.split())
    if not t or BAD_ANY.search(t) or SELF_DISCLOSURE.search(t) or REASONING_LEAK.match(t): return False
    if lang == "en":
        n = len(t.split())
        if n < 10 or n > max_words: return False
    else:
        if len(t) < 20 or len(t) > 260: return False
    ss = sentences(t)
    if len(ss) < 2 or len(ss) > 5: return False
    q = t.count("?") + t.count("？")
    if lang == "en" and not VALIDATION_EN.search(" ".join(ss[:2])): return False
    if strategy != "validate_ask" and lang == "en" and INTERROG_START.search(t): return False
    adv = ADVICE_EN if lang == "en" else ADVICE_KO
    if strategy == "validate_ask":
        return q == 1 and t.rstrip().endswith(("?", "？")) and not adv.search(t)
    if q != 0: return False
    if strategy == "validate_listen":
        return not adv.search(t)
    if strategy == "validate_suggest":
        return bool((SUGGEST_EN if lang == "en" else ADVICE_KO).search(t))
    if strategy == "validate_normalize":
        return bool((NORMAL_EN if lang == "en" else NORMAL_KO).search(t)) and not adv.search(t)
    return False

# ---- near-duplicate detection (protects the test set from leaking into training)
def norm(t):
    return re.sub(r"[^\w\s]", " ", str(t).lower()).split()

def is_near_duplicate(msg, pool_token_sets, thr=0.7):
    a = set(norm(msg))
    if not a: return True
    for b in pool_token_sets:
        if not b: continue
        inter = len(a & b)
        if inter / len(a | b) >= thr or inter / min(len(a), len(b)) >= 0.9:
            return True
    return False