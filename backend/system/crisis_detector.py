"""
crisis_detector.py  -  bilingual (EN/KO/code-switched) crisis detector, v2

Design (replaces the 14-pattern list in inference.py):
  1. Normalise: lowercase, curly quotes -> ASCII, plus a whitespace-free view
     (Korean spacing is inconsistent, so Korean patterns run on that view).
  2. Scoped span removal (USE_GUARD): explicit first-person DENIALS
     ("I'm not suicidal", "자살 생각은 없어요") and well-known FIGURATIVE idioms
     ("this exam will kill me", "죽겠어요", "dying to know") are blanked out.
     Only the matched span is removed; anything else in the message is still
     scanned, so "I'm not suicidal... but I want to die" is still flagged.
  3. Pattern families organised by suicide-risk category: ideation / wish to
     be dead, method and means, plan and intent, preparatory behaviour,
     self-harm, burden and hopelessness. Plus a few co-occurrence rules.
Priority is recall: a false alarm shows a helpline message, a miss does not.

Public API
    check_crisis_risk(message)           -> bool
    explain(message)                     -> list of matched pattern ids (debug)
    detect_message_language(message)     -> "en" | "ko" | "mixed"
    crisis_response_language(message, ui_language="en") -> "en" | "ko" | "both"
"""
import re

USE_GUARD = True   # set False to switch off denial/idiom span removal (ablation)

# ---------------------------------------------------------------- denial / idiom
DENIAL_EN = [
    r"\bnot (depressed or )?(suicidal|in crisis|in danger)\b",
    r"\b(i would|i will|i could|i'?d) never (hurt|harm|kill|end)\b[^.,;!?]{0,12}\b(myself|my life)\b",
    r"\b(i )?(don'?t|do not) want to die\b",
    r"\b(i )?(don'?t|do not) have (any )?(plans?|thoughts?|intentions?|urges?) (to|of|about) (hurt|harm|kill|end|suicide|self)[^.,;!?]*",
    r"\bnot (thinking|planning) (about|of|on) (killing myself|suicide|ending my life|hurting myself|harming myself)\b",
    r"\bnever (had|have had) (any )?(thoughts?|urges?) (of|about) (self[- ]?harm|suicide|hurting myself)\b",
    r"\bno suicidal (thinking|thoughts|ideation)\b",
    r"\bi'?m not going to (hurt|kill) myself\b",
]
DENIAL_KO = [
    r"자살(할)?(생각|계획)(은|이)?(전혀|하나도)?(없|안)",
    r"자(살|해)하고싶지않",
    r"자해(를|할)?(생각|계획)?(해본적)?(은|이|는)?(한번도|전혀)?없",
    r"자해를생각해본적(은|이)?(한번도|전혀)?없",
    r"스스로를해칠(생각|계획)(은|이)?(전혀)?없",
    r"죽고싶(은건|은게|지는|지)(절대)?(아니|않|없)",
    r"죽을생각(은|이)?(전혀)?없",
    r"목숨을끊을생각(은|이)?없",
    r"위기상황(은|이)?아니",
    r"극단적인?생각(은|을)?(하지않|없)",
]
FIGURATIVE_EN = [
    r"\b(kill|killing|murder|murdering)\s+me\b",
    r"\bdying to\b", r"\bi'?m dying\b(?! (inside|alone))",
    r"\b(dying|died|die) (of|from|laughing)\b",
    r"\b(nearly|almost) died\b",
    r"\bi could (just )?die\b",
    r"\bdeath of me\b", r"\bmurder on\b", r"\bkill for\b",
]
FIGURATIVE_KO = [
    r"죽겠", r"죽을것같", r"죽을맛", r"죽을지경", r"죽는줄", r"죽을만큼",
    r"(쪽팔|민망|창피|부끄)[가-힣]{0,3}죽고싶",
]

# ---------------------------------------------------------------- crisis patterns
EN = {
 "kill_self":   r"\bkill(ing)? myself\b",
 "end_life":    r"\bend(ing)? (my|my own) life\b|\bend it all\b|\b(going to|gonna|plan to|want to) end it\b",
 "suicide":     r"\bsuicid\w*",
 "want_die":    r"\b(can\'?t|cannot|can not) go on( like this| anymore)\b|\bi\'?d rather be dead\b|\bno way out\b|\bwant(ed)? to die\b|\bwish i (was|were|had been) (dead|gone)\b|\bwish i (had )?never been born\b|\bwish i could (just )?(die|disappear|not wake)\b",
 "not_live":    r"\bdon'?t want to (be alive|live|be here|exist)\b|\b(no|nothing).{0,12}(reason|point) (to|in) (live|living|going on|being alive|be here)\b|\bnothing left to live for\b|\bno reason to be here\b|\b(can\'?t|cannot) see any reason to (be here|go on|live)\b",
 "self_harm":   r"\b(hurt|harm|cut|burn)(ing|ting)? myself\b|\bcutting\b|\bself[- ]?(harm|injur)\w*",
 "method":      r"\bhang myself\b|\boverdos\w*|\btoo many pills\b|\bpills?\b.{0,40}\b(all of them|tonight|take them all)\b|\bjump (off|from)\b.{0,40}\b(roof|bridge|building|window|dorm)\b|\b(almost|nearly) jump\w*",
 "means_place": r"\b(bridge|roof|ledge|window|tracks?)\b.{0,70}\b(fall|jump\w*|step\w* in front)\b|\bstep\w* in front of (a |an |the |one )?(train|car|bus|subway)\b",
 "not_wake":    r"\b(never|not) (wake|waking) up\b|\bhope i don\'?t wake up\b|\bwish i wouldn\'?t wake up\b|\bdon\'?t wake up tomorrow\b|\bnot come back\b|\bnever wake\b|\bstop existing\b|\bwant to disappear\b|\bdisappear(ed)? for good\b",
 "better_off":  r"\bbetter off (if i|without me|dead)\b|\b(at peace|relieved) if i (were|was) gone\b|\bbetter off if i\b",
 "burden":      r"\b(a )?burden\b.{0,60}\b(stop existing|disappear|die|gone|end)\b",
 "passive_dead":r"\bhope something happens to me\b|\bdon'?t care if i (die|get hit|get killed)\b|\bi'?m already gone\b|\bnobody would notice if i\b|\bno point in tomorrow\b|\bwant everything to stop\b",
 "preparing":   r"\bgoodbye (messages?|letters?|notes?)\b|\bsuicide note\b|\b(picked|chosen|set) (the|a) (day|date)\b.{0,60}(not telling|nobody|no one|won'?t tell)|\bwon'?t be around\b.{0,20}$|\bgiv(e|ing) away my (things|stuff|belongings)\b|\bgoogling\b.{0,20}\b(ways to die|how to die)\b|\blooking up ways to (die|stop existing|kill)\b",
}
KO = {   # run on the whitespace-free view
 "죽고싶":       r"죽고싶|죽고만싶|죽어버리|죽을래|죽을거|죽으려|죽는방법|죽는법",
 "자살해":       r"자살|자해|극단적선택|목숨을?끊|목을?매|손목을?(그었|긋)|몸에상처를?내|스스로.{0,6}상처",
 "약":           r"수면제.{0,10}(먹|모아|한꺼번)|(약|알약)을?(다|전부|한꺼번에).{0,6}먹",
 "살기싫":       r"살이유가없|삶을끝내|더이상(못버티|견딜수없|못견디)|살고싶지않|살기싫|살고싶은이유(가|는)?(하나도)?(없|안)|살의미(가)?없|살아갈이유|살아야할이유",
 "사라지":       r"사라지고싶|사라지는방법|사라지면.{0,12}(편|나을|좋)|없어지고싶|없어져도|없어지는게.{0,6}나|이미없는사람|(제가|내가)없(으면|어지면).{0,10}(편|나)|(제가|내가|저)가?없는게.{0,8}나|없는게모두에게나",
 "투신":         r"뛰어내리|뛰어들(면|까|고)|(옥상|다리|창문|철길|난간|한강).{0,15}(뛰|떨어)",
 "깨어나지":     r"눈을?뜨지않았으면|태어나지않았으면|깨어나지않|다시는.{0,6}(깨|일어나)|돌아오지않았으면|태어나지말걸|내일이.{0,3}(오지않|와야할이유)|(더이상)?버틸이유(를|가)?(모르|없)|(더|안)버텨도됐으면",
 "준비":         r"마지막인사|유서|날짜(는|를)?(이미)?정했|곧없을|(물건|노트북|책|옷).{0,12}(다|전부)?(나눠|줬|주었|정리했).{0,16}(필요없|곧|없을)|필요없으니까",
 "차":           r"차에치여도(상관|괜찮)|모든게(영원히)?멈췄으면|다멈췄으면",
 "계획":         r"(자살|죽).{0,10}(계획이있|방법도정)|거짓말이었어요.{0,12}계획",
}
# co-occurrence rules: both regexes must match (run on the spaced, lowercased view)
RULES = [
 ("better_off_ko", r"better off|편할|편해질", r"제가\s*없|내가\s*없|저\s*없|제가\s*사라|without me|if i('?m| am| were| was)? gone"),
 ("end_cant",      r"끝내|\bend\b", r"더\s*이상|못\s*버티|못\s*하겠|can'?t (do|take) this|anymore"),
 ("pills_now",     r"\bpills?\b|약", r"tonight|오늘\s*밤|다\s*먹|all of them|한꺼번"),
 ("gave_leaving",  r"\b(gave|giving|give away|given away)\b", r"won'?t (be|need)|will not (be|need)|since i('?m| am) (leaving|gone)"),
]

def _norm(message: str):
    t = message.lower().replace("\u2019", "'").replace("\u2018", "'")
    return t, re.sub(r"\s+", "", t)

def _strip(pats, text):
    for p in pats:
        text = re.sub(p, " ", text)
    return text

def explain(message: str):
    t, k = _norm(message)
    if USE_GUARD:
        t = _strip(DENIAL_EN + FIGURATIVE_EN, t)
        k = _strip(DENIAL_KO + FIGURATIVE_KO, k)
    hits = [n for n, p in EN.items() if re.search(p, t)]
    hits += ["ko:" + n for n, p in KO.items() if re.search(p, k)]
    hits += [n for n, a, b in RULES if re.search(a, t) and re.search(b, t)]
    return hits

def check_crisis_risk(message: str) -> bool:
    return len(explain(message)) > 0

def detect_message_language(message: str) -> str:
    has_ko = bool(re.search(r"[\uac00-\ud7a3]", message))
    latin_words = len(re.findall(r"[A-Za-z]{2,}", message))
    if has_ko and latin_words >= 2:
        return "mixed"
    return "ko" if has_ko else "en"

def crisis_response_language(message: str, ui_language: str = "en") -> str:
    """Language of the fixed crisis response, chosen from what the student wrote.
    Mixed messages receive both versions."""
    lang = detect_message_language(message)
    return "both" if lang == "mixed" else lang