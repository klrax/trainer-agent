"""Pain is handled in code, before the model is called.

Why: the safety rule was in the prompt and the local models still answered "start light" to sharp knee pain (0 of 3).
A fixed reply is instant and can't be talked round. It doesn't diagnose: it stops, refers, and offers a swap.

Two levels, because a list of complaint words is never complete (one that passed all its own tests missed 15 of 18
new phrasings):
- clearly pain ("my knee hurts") -> stop and refer
- the user's own joint, and nothing clearly neutral about it ("my knee is being weird") -> ask, no weights meanwhile

Unsure -> stop. Being told to rest by mistake costs one message; being told to squat on a bad knee costs more.
"""
from __future__ import annotations

import re
from typing import Iterable

from .rag.text import normalize

_JOINT = (r"knees?|kneecaps?|shoulders?|(?:lower |upper )?back|elbows?|wrists?|hips?|ankles?|necks?|spine|lumbar|groin|achilles|tendons?|joints?")
_PART = _JOINT + r"|legs?|arms?|foot|feet|hands?|hamstrings?|quads?|calf|calves|chest|ribs?|biceps?|triceps?|glutes?|shins?|thighs?|fingers?|toes?"
_LIFT = (r"squat\w*|bench\w*|deadlift\w*|press\w*|row(?:s|ing)?|lunge\w*|pull-?ups?|pulldowns?|curl\w*|plank\w*|run(?:ning)?|"
         r"jump\w*|lift\w*|train\w*|exercis\w*|workout|walk\w*|bend\w*|rais\w*")
_SHARP = (r"hurt(?:s|ing)?|pain(?:s|ful)?|stabbing|throbbing|pinch(?:es|ing)|swollen|swelling|sprain\w*|injur(?:y|ed|ies)|"
          r"tweaked|torn|tore|popped|snapped|cracked|gave out|giving out|give out|numb(?:ness)?|tingl\w+|dislocat\w+|fractur\w+|"
          r"burning|shooting|radiat\w+|pinch|twinge\w*|discomfort|uncomfortable|bother\w*|tender|bruis\w+|inflam\w+|"
          r"killing me|acting up|feels? (?:\w+ )?(?:off|weird|wrong|bad|unstable|loose|stuck|funny|weak)|gives? me (?:trouble|issues|grief)|"
          r"ouch|aggravat\w+|flar\w+ up|clicking|clicks|grinding|locks up|locking|unstable|buckl\w+|strained|sprain")
# Named conditions and accidents need no body part.
_CONDITION = (r"tend[io]n[io]tis|bursitis|arthritis|sciatica|hernia\w*|(?:slipped|bulging|herniated) dis[ck]|meniscus|acl|rotator cuff|"
              r"impingement|plantar fasciitis|shin splints|concussion|tennis elbow|golfer'?s elbow|runner'?s knee|frozen shoulder|stress fracture|carpal tunnel|whiplash")

# Not pain now. Removed before matching.
_NOT_NOW = [
    r"\b(?:no|without|zero|any|isn'?t any)(?: \w+)? (?:pain|injur(?:y|ies))\b", r"\bpain[- ]?free\b",
    r"\b(?:doesn'?t|don'?t|does not|do not|didn'?t|isn'?t|is not|not|never|no longer|stopped|wouldn'?t|won'?t) (?:\w+ )?(?:hurt\w*|painful|injured)\b",
    r"\b(?:used to|would it|will it|could it|can it) (?:hurt|be painful)\b", r"\bhurt(?:s|ing)? (?:my|the|our) (?:gains|progress|feelings|results|goals?|wallet)\b",
    r"\bpain (?:au|perdu)\b", r"\b(?:it'?s|what|such|is) a (?:real |huge )?pain\b(?! in (?:my|the))", r"\bpainkillers?\b",
    r"\bdoms\b", r"\b(?:burning|burn) (?:fat|calories|kcal)\b", r"\bhurt(?:ing)? for\b", r"\b(?:avoid|prevent)\w* (?:\w+ )?(?:injur\w+|pain)\b", r"\binjury prevention\b",
    r"\b(?:old|past|previous|former|history of)(?: \w+){0,2} (?:pain|injur(?:y|ies))\b", r"\binjuries?:? (?:none|no)\b",
    r"筋肉痛", r"肌肉酸痛|酸痛", r"痛み?(?:は|も)?(?:ない|ありません|なし|なく)", r"痛くない", r"不(?:疼|痛)", r"没有?(?:疼|痛|受伤)", r"もう痛く",
]
_PAIN = [
    rf"\b(?:{_SHARP})\b.{{0,60}}\b(?:{_PART}|{_LIFT})\b", rf"\b(?:{_PART}|{_LIFT})\b.{{0,60}}\b(?:{_SHARP})\b",
    rf"\b(?:{_JOINT})\b.{{0,20}}\b(?:ach(?:e|es|ing)|sore(?:ness)?|locks?|locking|clicks? and)\b", rf"\b(?:sore|aching|achy|bad) (?:{_JOINT})\b",
    rf"\bsharp(?:ly)?\b.{{0,40}}\b(?:{_PART})\b", rf"\b(?:{_PART})\b.{{0,40}}\bsharp(?:ly)?\b",
    rf"\b(?:{_CONDITION})\b", rf"\b(?:rolled|twisted|jammed|banged|landed on|wrenched|threw out|put out) (?:my|an?) (?:{_PART})\b",
    r"\b(?:it|this|that)(?: still| really| also)? hurts\b", r"\bi(?:'m| am) (?:injured|hurt|in pain)\b",
    r"\bi(?:'ve)? (?:hurt|injured|tweaked|strained|pulled) (?:my|a|something)\b", r"\b(?:back|muscle) spasms?\b",
    r"違和感|痛く|痛い|痛み|痛む|痛めた|ひねった|捻った|つった|ズキズキ|しびれ|腫れ|捻挫|怪我|ケガ|けが", r"疼|痛|受伤|肿|扭|拉伤|刺痛|发麻|(?:膝|肩|腰|背|肘|腕|踝|髋|颈|脖)\S{0,4}(?:不舒服|难受|不适)",
]
_URGENT = [
    r"\bchest\b.{0,25}\b(?:pain|hurt\w*|tight\w*|pressure|crush\w*)", r"\b(?:pain|tightness|pressure)\b.{0,15}\bchest\b",
    r"\b(?:can'?t|cannot|hard to|trouble|struggling to) breath\w*", r"\bshort(?:ness)? of breath\b.{0,30}\b(?:rest|sitting|nothing)\b",
    r"\b(?:faint(?:ed|ing)?|pass(?:ed|ing)? out|blacked out|collapsed)\b", r"\b(?:dizzy|dizziness|light-?headed)\b",
    r"\b(?:can'?t|cannot) (?:feel|move) my\b", r"胸.{0,4}(?:痛|苦し|圧迫)|息が(?:できない|苦しい)|失神|気を失|めまい", r"胸(?:口)?(?:疼|痛|闷)|呼吸困难|晕倒|头晕",
]
# The user's own joint, or a joint as the subject. Neutral: training it, describing it as good, or a lift's name.
_WATCH = _JOINT + r"|chest|hamstrings?|hammy|calf|calves|quad|shin|thumb|si joint|rib|biceps? tendon"
_OWN = re.compile(rf"\b(?:my|mine|the) (?:\w+ ){{0,2}}?(?:{_WATCH})\b|(?:^|[.,!?;] ?|\band |\bbut )(?:left |right |lower |upper )*(?:{_WATCH}) "
                  r"(?:is|was|are|has|have|had|feels?|felt|went|seized|gave|clunks?|grinds?|clicks?|keeps?|won'?t|can'?t|still|again)\b")
_NEUTRAL = re.compile(
    rf"\b(?:{_WATCH}) (?:day|days|workouts?|sessions?|exercises?|training|muscles?|squats?|press(?:es)?|thrusts?|raises?|curls?|extensions?|rows?|gains|pr|volume|routine|program|fly|flyes)\b"
    rf"|\b(?:train|grow|build|strengthen|stretch(?: for)?|exercises? for|target|hit|work|work on|working|focus on|bulk|develop|activate|warm up) (?:my|the) (?:\w+ )?(?:{_WATCH})\b"
    rf"|\b(?:{_WATCH}) (?:is|are|feels?|felt) (?:\w+ )?(?:my fav\w+|fine|good|great|ok|okay|strong\w*|better|healthy|normal|bigger|growing|lagging|all good)\b")
_OWN_CJK = re.compile(r"膝|肘|手首|足首|股関節|首が|腰|肩が|肩も|肩に|背中が|肩膀|膝盖|手腕|脚踝|脖子|颈|髋|背(?=怪|不|很)")
_NEUTRAL_CJK = re.compile(r"トレ|鍛え|の日|種目|メニュー|ストレッチ|肩こりなし|大丈夫|问题不大|没事|没问题|练|训练|动作|拉伸|好了")
_BETTER = re.compile(r"\b(?:better now|all better|cleared|healed|recovered|went away|(?:doctor|physio)\w* said (?:it'?s )?(?:ok|okay|fine))\b|治った|治りました|好了")

PAIN_REPLY = ("Please stop that exercise today, and skip anything else that hurts. I won't suggest a weight for it: "
              "pain like this needs checking, so see a doctor or physiotherapist before you train it again.\n"
              "I can suggest exercises that stay away from the sore area. Tell me which area to leave out.")
URGENT_REPLY = ("Stop exercising now and sit or lie down. Chest pain, trouble breathing, dizziness or fainting can be serious: "
                "get medical help now. In Japan call 119 (or your local emergency number), especially if it doesn't pass "
                "within a few minutes, spreads, or comes back.\nNo training today until a doctor has checked you.")
CHECK_REPLY = ("Before I suggest anything: does it hurt, feel unstable or not move normally? If so, stop that exercise today, "
               "skip anything that bothers it, and see a doctor or physiotherapist. I won't suggest weights until I know.\n"
               "If it's fine, tell me (\"my knee is fine\") and I'll carry on.")
_REST = "\nIf you asked for something else too, send it as a separate message: I didn't act on this one."


_KCAL_ASK = re.compile(r"(\d[\d,]{2,4})\s*(?:k?cals?|kilocalories|calories)\b", re.IGNORECASE)
_DIET = re.compile(r"\b(diet|meal plan|plan|menu|a day|per day|daily|eat|eating|only)\b", re.IGNORECASE)
DIET_REPLY = ("I can't plan a diet under about 1200 kcal a day: it's too little to train on and isn't safe without a "
              "doctor. For fat loss I can switch your goal to cut, which gives a moderate deficit you can keep up "
              "while lifting. Want me to do that?")


def very_low_calorie(user: str) -> bool:
    """"Give me an 800 kcal a day meal plan" -> True. The prompt forbade it; the model planned lunches anyway (0/3)."""
    asked = [int(n.replace(",", "")) for n in _KCAL_ASK.findall(user)]
    return bool(asked) and min(asked) < 1200 and bool(_DIET.search(user))


def safety_reply(user: str, earlier: Iterable[str] = ()) -> str | None:
    """The fixed reply for a message about pain, or None to carry on as normal.

    earlier: the user's previous messages in this conversation. After "my knee hurts when I squat", "so what weight
    for squats?" gets the same answer, unless they say it is better.
    """
    if very_low_calorie(user):
        return DIET_REPLY
    text = _clean(user)
    if _any(_URGENT, text):
        return URGENT_REPLY + _REST
    if _any(_PAIN, text):
        return PAIN_REPLY + _REST
    if _unclear(user, text):
        return CHECK_REPLY + _REST
    sore: set[str] = set()
    for message in earlier:
        before = _clean(message)
        if _BETTER.search(before) or _NEUTRAL.search(before):
            sore = set()
        elif _any(_PAIN, before) or _any(_URGENT, before) or _unclear(message, before):
            sore |= _topics(before)
    if sore & _topics(text) and not _BETTER.search(text) and not _NEUTRAL.search(text):
        return PAIN_REPLY + _REST
    return None


def _unclear(original: str, cleaned: str) -> bool:
    if cleaned.strip() != normalize(original).replace("’", "'").strip() or _BETTER.search(cleaned):
        return False                                        # they said it does NOT hurt, or it was long ago
    if _OWN_CJK.search(cleaned) and not _NEUTRAL_CJK.search(cleaned):
        return True
    return bool(_OWN.search(cleaned)) and not _NEUTRAL.search(cleaned)


def _clean(text: str) -> str:
    text = normalize(text).replace("’", "'")
    for p in _NOT_NOW:
        text = re.sub(p, " ", text)
    return text


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.DOTALL) for p in patterns)


def _topics(text: str) -> set[str]:
    """Body parts and lifts named, roughly singular: "squatting" and "squats" -> "squat"."""
    words = re.findall(rf"\b(?:{_PART}|{_LIFT})\b", text)
    skip = {"lift", "train", "exercis", "workout", "walk", "rais", "bend"}          # too general to carry over
    return {w for w in (re.sub(r"(?:ting|ing|ed|s)$", "", x) for x in words) if w not in skip}
