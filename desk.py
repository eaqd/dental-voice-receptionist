"""Plain code decides. The AI reads each caller turn; these rules choose what the receptionist does next:
send an emergency to a human, ask the next missing question, offer a free slot, or book it."""
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import yaml

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_WORDS = {w: i for i, name in enumerate(WEEKDAYS) for w in (name, name + "s")} | {
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
PREFERENCES = ("preferred_day", "preferred_time")  # a new wish replaces the old one; everything else fills gaps
WISH = r"\b(want|wanted|prefer|rather|need it|can you do|could you do|how about|what about|anything|earliest|instead|better)\b"
NEGATED_WISH = re.compile(r"\b(?:no|not|never|cannot|\w+n't|avoid|bad for me)\b")
YES = r"\s*(yes|yeah|yep|ok|okay|sure|fine|great|perfect|lovely|that's fine|that works|that's great)\b"


def load(path="clinic.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Call:
    caller_id: str
    started: datetime
    fields: dict = field(default_factory=dict)
    offered: datetime | None = None
    declined: list = field(default_factory=list)
    booked: datetime | None = None
    outcome: str = ""  # booked | urgent | message ("" while the call goes on)
    said: str = ""     # the receptionist's last line, which the AI needs to read "yes" or "no"

    @property
    def done(self) -> bool:
        return bool(self.outcome)


# ---------------------------------------------------------------- reading days and times (no AI)

def parse_day(text: str, now: datetime) -> date | None:
    """'tomorrow', 'Thursday', 'thurs', 'next week', '2026-11-19', 'the 19th' -> a date; anything else -> None."""
    t = (text or "").lower()
    today = now.date()
    if not t:
        return None
    if "day after tomorrow" in t:
        return today + timedelta(days=2)
    if "tomorrow" in t:  # before "today": "not today, tomorrow" means tomorrow
        return today + timedelta(days=1)
    if "today" in t or "tonight" in t:
        return today
    if m := re.search(r"\d{4}-\d{2}-\d{2}", t):
        try:
            day = date.fromisoformat(m.group())
        except ValueError:  # "2026-13-45"
            return None
        return day if day >= today else None  # a date in the past is no wish we can meet
    # Full names or the usual short forms; never "sat" or "sun" ("I sat too long" is no Saturday). If several are
    # named, the last one counts: "Tuesday is no good, how about Thursday" means Thursday.
    named = [WEEKDAY_WORDS[w] for w in re.findall(r"[a-z]+", t) if w in WEEKDAY_WORDS]
    if named:
        i = named[-1]
        return today + timedelta(days=(i - today.weekday()) % 7 or 7)  # "Monday" on a Monday = next week
    if "next week" in t:
        return today + timedelta(days=7 - today.weekday())
    if m := re.search(r"\b(\d{1,2})(st|nd|rd|th)\b", t):
        day = int(m.group(1))
        month = today.replace(day=1)
        if day < today.day:  # already passed this month: next month
            month = (month + timedelta(days=32)).replace(day=1)
        try:
            return month.replace(day=day)
        except ValueError:
            return None
    return None


def _hour(h: str, minutes: str | None, ampm: str | None) -> float:
    hour = int(h) % 24
    if ampm == "am" and hour == 12:  # 12 am is midnight
        hour = 0
    elif ampm == "pm" and hour < 12:
        hour += 12
    elif ampm is None and 1 <= hour <= 7:  # "after 3" at a dental practice means 3 pm
        hour += 12
    return hour + (int(minutes) / 60 if minutes else 0)


def parse_time(text: str) -> tuple[float, float] | None:
    """'morning', 'after 3', 'before 11am', '2:30', "three o'clock" -> (from_hour, to_hour); else None."""
    t = (text or "").lower()
    t = t.replace("’", "'").replace("a.m.", "am").replace("p.m.", "pm")  # how speech-to-text often writes them
    t = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", lambda m: NUMBER_WORDS[m.group()], t)
    clock = r"(\d{1,2})(?::(\d\d))?\s*(am|pm)?"
    if m := re.search(r"\b(after|from)\s+" + clock, t):
        return _hour(m.group(2), m.group(3), m.group(4)), 24.0
    if m := re.search(r"\b(before|by)\s+" + clock, t):
        return 0.0, _hour(m.group(2), m.group(3), m.group(4))
    if m := (re.search(r"\b(\d{1,2})(?::(\d\d))?\s*(am|pm|o'?clock)", t) or re.search(r"\b(\d{1,2}):(\d\d)()", t)
             or re.search(r"\bat\s+(\d{1,2})()()\b", t)):  # "10am", "2:30", "3 o'clock", "at 8"
        h = _hour(m.group(1), m.group(2), m.group(3) if m.group(3) in ("am", "pm") else None)
        return h, h + 0.5
    for words, window in ((("after school", "school run", "school pick"), (15, 24)),  # found by the caller simulator
                          (("morning",), (0, 12)), (("lunch",), (12, 14)), (("afternoon",), (12, 17)),
                          (("evening", "after work"), (17, 24))):
        if any(w in t for w in words):
            return float(window[0]), float(window[1])
    return None


def spoken(slot: datetime) -> str:
    """Thursday 19 November at 2 pm: the way a receptionist says it (fixed English, whatever the PC's language)."""
    h, m = slot.hour, slot.minute
    clock = f"{(h - 1) % 12 + 1}{f':{m:02d}' if m else ''} {'am' if h < 12 else 'pm'}"
    return f"{WEEKDAYS[slot.weekday()].title()} {slot.day} {MONTHS[slot.month - 1]} at {clock}"


# ---------------------------------------------------------------- the calendar

def open_slots(rules: dict, after: datetime, taken=(), days=14):
    """Free slots from the opening hours in clinic.yaml, at least an hour from now, minus the taken ones."""
    step = timedelta(minutes=rules.get("slot_minutes", 30))
    for d in range(days):
        day = after.date() + timedelta(days=d)
        hours = rules["hours"].get(WEEKDAYS[day.weekday()][:3])
        if not hours:
            continue
        start, end = (datetime.combine(day, time.fromisoformat(x.strip())) for x in hours.split("-"))
        slot = start
        while slot + step <= end:
            if slot >= after + timedelta(hours=1) and slot not in taken:
                yield slot
            slot += step


def pick_slot(rules: dict, now: datetime, day: date | None, window, taken) -> tuple[datetime | None, str]:
    """The earliest free slot matching the wish; if none, relax the time, then the day. Returns (slot, relaxed)."""
    slots = list(open_slots(rules, now, taken))
    fits = lambda s, d, w: (d is None or s.date() == d) and (w is None or w[0] <= s.hour + s.minute / 60 < w[1])  # noqa: E731
    for d, w, relaxed in ((day, window, ""), (day, None, "time"), (None, window, "day"), (None, None, "both")):
        hit = next((s for s in slots if fits(s, d, w)), None)
        if hit:
            return hit, relaxed
    return None, "none"


# ---------------------------------------------------------------- one caller turn

def is_emergency(caller_text: str, rules: dict, intent: str) -> bool:
    """Either the AI or the patterns in clinic.yaml can raise the alarm; neither can lower it."""
    said = (caller_text or "").lower().replace("’", "'")
    return intent == "emergency" or any(re.search(p, said) for p in rules["emergency_patterns"])


def clean(got: dict) -> dict:
    return {k: "" if v is None else v.strip() if isinstance(v, str) else v for k, v in got.items()}


FILLER = {"the", "and", "for", "you", "your", "are", "any", "can", "with", "that", "this", "have", "like", "would",
          "what", "not", "but", "just", "please", "its", "it's", "is", "be", "to", "an", "in", "on", "at", "of"}


def _words(text: str) -> set:
    """The words that carry meaning ("three" counts as "3"), for checking an answer against what was said."""
    text = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", lambda m: NUMBER_WORDS[m.group()], (text or "").lower())
    return {w for w in re.findall(r"[a-z0-9']+", text) if (w.isdigit() or len(w) >= 3) and w not in FILLER}


def positive_wishes(text: str) -> str:
    lower = text.lower().replace("’", "'").replace("a.m.", "am").replace("p.m.", "pm")
    if not NEGATED_WISH.search(lower):
        return text
    clauses = re.split(r"[.,?!;]|\bbut\b|\band\s+(?=i\b|would\b)", lower)
    return " ".join(clause for clause in clauses if not NEGATED_WISH.search(clause))


def drop_echoes(got: dict, rules: dict, said: str, heard: str = "", wishes: str | None = None) -> dict:
    """Check the AI's reading against what was actually said. A small model sometimes:
    - copies OUR question from clinic.yaml into a field as if it were the answer (seen on the first real run);
    - writes "none" instead of leaving a field empty;
    - invents a detail or a wish nobody said ("plan: private", or the time we just offered copied back as the
      caller's new wish). So optional details and day/time wishes only count if the caller said a word of them."""
    questions = {q.strip().lower() for q in {**rules["required"], **rules.get("optional", {})}.values()}
    grounded = set(PREFERENCES) | set(rules.get("optional", {}))
    rejected = _words(heard) - _words(wishes) if wishes is not None and wishes != heard else set()
    out = {}
    for k, v in got.items():
        source = wishes if k in PREFERENCES and wishes is not None else heard
        if isinstance(v, str) and (
                v.lower() in questions or v.lower() in ("none", "n/a", "unknown", "not mentioned")
                or (k in grounded and v and not _words(v) & _words(source))
                or (k in PREFERENCES and _words(v) & rejected)):
            v = ""
        out[k] = v
    return out


def merge(known: dict, new: dict) -> dict:
    """Details fill gaps (a later mishearing never overwrites a name); a new day or time wish replaces the old."""
    out = dict(known)
    for k, v in new.items():
        if v and (k in PREFERENCES or not out.get(k)):
            out[k] = v
    return out


def missing(fields: dict, rules: dict) -> list[str]:
    return [name for name in rules["required"] if not fields.get(name)]


def say(call: Call, line: str) -> str:
    call.said = line
    return line


def turn(call: Call, caller_text: str, got: dict, rules: dict, taken=(), reserve=lambda slot: True) -> str:
    """What the receptionist says after the caller's turn. `got` is the AI's reading; every choice is made here.
    `reserve(slot)` writes the booking and returns False if someone else got the slot first."""
    if not (caller_text or "").strip():  # silence or a failed hearing: never act on it, just ask again
        return say(call, "Sorry, I didn't catch that. Could you say it again?")
    wishes = positive_wishes(caller_text)
    got = drop_echoes(clean(got), rules, call.said, caller_text, wishes)
    intent, answer = got.pop("intent", "other"), got.pop("answer_to_offer", "none")
    # 1. Safety first, before anything else.
    if is_emergency(caller_text, rules, intent):
        call.outcome = "urgent"
        return say(call, rules["emergency_reply"])
    if intent in ("change_or_cancel", "question") and not call.fields.get("reason") and not call.offered:
        call.outcome = "message"  # not a new booking: a person handles it tomorrow
        call.fields = merge(call.fields, got)
        return say(call, rules["message_reply"])
    # The AI sometimes misses a day or time wish ("a slot after school on any weekday": found by the caller
    # simulator, tutorial 3). Plain code reads the caller's own words too, unless they only repeat our offer.
    # Only for a clear wish ("I want...", "how about...", "anything after...") and never when the caller says yes.
    if re.search(WISH, wishes, re.I) and not re.match(YES, caller_text.lower()):
        window, day = parse_time(wishes), parse_day(wishes, call.started)
        at = call.offered.hour + call.offered.minute / 60 if call.offered else None
        if not got.get("preferred_time") and window and not (at is not None and window[0] <= at < window[1]):
            got["preferred_time"] = wishes
        if not got.get("preferred_day") and day and not (call.offered and day == call.offered.date()):
            got["preferred_day"] = wishes
    call.fields = merge(call.fields, got)
    taken = set(taken) | set(call.declined)
    # 2. An answer to the time we offered.
    if call.offered:
        if answer == "yes" and call.offered not in taken and reserve(call.offered):  # booked at this very moment
            call.booked, call.outcome = call.offered, "booked"
            first = (call.fields.get("patient_name") or "").split()
            return say(call, rules["booked_reply"].format(name=f", {first[0]}" if first else "",
                                                          slot=spoken(call.booked)))
        if answer == "yes":  # someone else booked it in the meantime
            taken.add(call.offered)
        if answer == "no" or any(got.get(p) for p in PREFERENCES) or call.offered in taken:
            call.declined.append(call.offered)  # a slot someone else just took counts as gone too
            taken.add(call.offered)
            call.offered = None
    # 3. The next missing detail, one question at a time (easier to answer on the phone).
    gaps = missing(call.fields, rules)
    if gaps:
        return say(call, rules["required"][gaps[0]])
    if call.offered:  # we offered a time and didn't hear a yes or a no
        return say(call, f"Sorry, I didn't catch that. Would {spoken(call.offered)} work, or is there a better "
                         f"day or time?")
    # 4. Offer the best free slot. The calendar decides, never the AI.
    day = parse_day(call.fields.get("preferred_day", ""), call.started)
    window = parse_time(call.fields.get("preferred_time", ""))
    slot, relaxed = pick_slot(rules, call.started, day, window, taken)
    if slot is None:
        call.outcome = "message"
        return say(call, "I can't see a free slot in the next two weeks, so I've passed your details to the team "
                         "to call you back.")
    call.offered = slot
    lead = {"": "I can do", "time": "That time is taken that day, but I can do",
            "day": "That day is full, but I can do", "both": "The nearest I can do is"}[relaxed]
    if relaxed in ("day", "both") and day and not any(s.date() == day for s in open_slots(rules, call.started)):
        lead = "We're closed then, but I can do"  # a closed day or the rest of today, not a full one
    return say(call, f"{lead} {spoken(slot)}. Would that work?")
