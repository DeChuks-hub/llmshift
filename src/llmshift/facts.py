"""Fact extraction and matching.

Facts are the checkable details of an answer: numbers (with sign, currency,
scale, unit and percent), dates, times, IDs and phone numbers, URLs, e-mail
addresses, inline code and named entities. `extract` pulls them out of a
baseline answer; `match` reports which ones a new answer is missing and which
numbers changed value ("2.1 million -> 2 million").
"""

from __future__ import annotations

import bisect
import re

from .textutil import normalize

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

_DATE_RES = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}})\b)?", re.I), "mdy"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b\.?(?:,?\s+(\d{{4}})\b)?", re.I), "dmy"),
    (re.compile(rf"\b({_MONTH})\.?,?\s+(\d{{4}})\b", re.I), "my"),
    (re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])"), "numeric"),
]
# Month/day written with digits only ("3/14") - accepted when matching, never extracted.
_SHORT_NUMERIC_DATE_RE = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?![\d/])")
_TIME_RES = [
    re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(?:([ap])\.?\s?m\b\.?)?(?![\w:])", re.I),
    re.compile(r"\b(\d{1,2})()\s?([ap])\.?\s?m\b\.?", re.I),  # 9am, 5 p.m.
]
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"'`)\]]+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]{1,64}@[\w-]+(?:\.[\w-]+)+\b")  # bounded: a long a-a-a... run stays linear
_CODE_RE = re.compile(r"`([^`\n]{1,60})`")
_FENCED_RE = re.compile(r"```.*?(```|$)", re.S)
_LIST_MARKER_RE = re.compile(r"(?m)^([^\S\n]*)(\d{1,3}[.)])(\s+)")  # indentation never spans lines: no quadratic blank runs
_ID_RES = [
    re.compile(r"(?<![\w-])(?:\+?1[ .-])?\(?\d{3}\)?[ .-]?\d{3}[ .-]\d{4}(?![\w-])"),  # phone (NANP)
    re.compile(r"(?<![\w-])\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?![\w-])"),  # phone (international)
    re.compile(r"#\d{2,}\b"),  # order / ticket numbers
    re.compile(r"\b[A-Z]{2,}[-_]?\d{2,}(?:[-_][A-Z0-9]+)*\b"),  # ORD-12345, INV2024-7
    re.compile(r"\bv?\d+\.\d+\.\d+(?:[-+][\w.]+)?\b"),  # versions
]
_UNITS = {
    "kg": "kg", "g": "g", "mg": "mg", "mcg": "mcg", "ug": "mcg", "lb": "lb", "lbs": "lb", "oz": "oz", "t": "t",
    "km": "km", "m": "m", "cm": "cm", "mm": "mm", "nm": "nm", "mi": "mi", "ft": "ft", "in": "in", "yd": "yd",
    "ml": "ml", "l": "l", "gal": "gal", "qt": "qt", "cc": "cc",
    "ms": "ms", "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "min": "min", "mins": "min", "minute": "min",
    "minutes": "min", "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "day": "day", "days": "day", "week": "week",
    "weeks": "week", "wk": "week", "wks": "week", "month": "month", "months": "month", "mo": "month", "year": "year",
    "years": "year", "yr": "year", "yrs": "year",
    "gb": "gb", "mb": "mb", "kb": "kb", "tb": "tb", "px": "px", "dpi": "dpi", "fps": "fps", "rpm": "rpm",
    "mph": "mph", "kph": "kph", "kmh": "kph", "hz": "hz", "ghz": "ghz", "mhz": "mhz", "w": "w", "kw": "kw", "kwh": "kwh", "v": "v",
    "kcal": "kcal", "cal": "cal", "ppm": "ppm", "db": "db", "x": "x", "pt": "pt", "pts": "pt",
}
_SPACED_UNITS = sorted((u for u in _UNITS if len(u) > 1 or u in ("g", "m", "l", "h", "s")), key=len, reverse=True)
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_./#:,])(?<![A-Za-z]-)"
    r"(?:(?P<sign>[-−])(?=[$€£¥₹]?\d))?"
    r"(?P<cur>[$€£¥₹])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)"
    r"(?:\s?(?P<scalew>(?i:thousand|million|billion|trillion|bn))(?![A-Za-z])|(?P<scalel>[kKMB])(?![A-Za-z]))?"
    r"(?P<pct>\s?(?:%|(?i:percent|per\s?cent)\b))?"
    r"(?:\s(?P<sunit>(?i:" + "|".join(_SPACED_UNITS) + r"))(?![A-Za-z]))?"
    r"(?P<tail>[A-Za-z]{0,4})"
)
_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}
_ORDINALS = {"st", "nd", "rd", "th"}
_CURRENCIES = set("$€£¥₹")
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12,
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.I)

_STOP_CAPS = set(
    """a an the this that these those it its in on at for to of and or but if when while as however yes no sure here there
    please thank thanks hi hello dear we you i our your my me us they their them he she his her him also first second third
    finally then next let based unfortunately absolutely great certainly course note overall additionally moreover furthermore
    therefore thus so because although though with without by from after before during about above below over under between
    into through not all any some each every both either neither one two three four five what which who whom whose where why
    how is are was were be been being have has had do does did can could will would shall should may might must okay ok hey
    well now today tomorrow yesterday just only even still yet again step example summary answer question result results
    conclusion introduction important tip tips warning true false null none hope best regards kind sincerely good
    order customer support team user sir madam welcome congratulations unless since until upon via per like
    monday tuesday wednesday thursday friday saturday sunday january february march april june july august september
    october november december
    check click open use make try consider contact visit call select enter add remove keep ensure remember start begin
    follow see read find go get set update install run review verify confirm choose create delete save send share ask tell
    give take wait avoid include provide apply change enable disable configure download upload sign log reset restart
    request submit return pay cancel book reply respond schedule
    currently generally typically usually basically essentially ultimately alternatively otherwise meanwhile instead
    similarly likewise specifically especially actually honestly obviously clearly simply really perhaps maybe probably
    hopefully luckily fortunately sadly interestingly notably importantly regarding according most many several other
    another such more less few much very both either great happy glad sorry""".split()
)
_CONNECTORS = {"of", "de", "du", "da", "di", "van", "von", "der", "la", "le", "del", "the", "y", "al", "bin"}
_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z0-9À-ɏ][\w'’&.\-]*\w|[A-Za-z0-9À-ɏ]")

MAX_HARD = 15
MAX_ENTITIES = 10


class _Taken:
    """Character spans already claimed, kept merged into disjoint runs so each lookup
    is a binary search (long, fact-dense answers stay fast)."""

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []

    def add(self, start: int, end: int) -> None:
        if end <= start:
            return
        i = bisect.bisect_right(self.ends, start)  # runs that end after `start` ...
        j = bisect.bisect_left(self.starts, end)  # ... and start before `end` overlap it

        if i < j:
            start, end = min(start, self.starts[i]), max(end, self.ends[j - 1])
        del self.starts[i:j], self.ends[i:j]
        self.starts.insert(i, start)
        self.ends.insert(i, end)

    def overlaps(self, start: int, end: int) -> bool:
        k = bisect.bisect_left(self.starts, end) - 1
        return k >= 0 and self.ends[k] > start

    def contains(self, pos: int) -> bool:
        k = bisect.bisect_right(self.starts, pos) - 1
        return k >= 0 and pos < self.ends[k]


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    chars = list(text)
    for s, e in spans:
        for i in range(max(0, s), min(e, len(chars))):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _date_value(kind: str, m: re.Match) -> str | None:
    try:
        if kind == "ymd":
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        elif kind == "mdy":
            mo, d = _MONTHS[m.group(1).lower()[:3]], int(m.group(2))
            y = int(m.group(3)) if m.group(3) else None
        elif kind == "dmy":
            d, mo = int(m.group(1)), _MONTHS[m.group(2).lower()[:3]]
            y = int(m.group(3)) if m.group(3) else None
        elif kind == "numeric":
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y = y + 2000 if y < 100 else y
            mo, d = (a, b) if a <= 12 else (b, a)  # US month/day unless that is impossible
        else:
            mo, y, d = _MONTHS[m.group(1).lower()[:3]], int(m.group(2)), None
    except (KeyError, ValueError):
        return None
    if not 1 <= mo <= 12 or (d is not None and not 1 <= d <= 31):
        return None
    if y is not None and d is not None:
        return f"{y:04d}-{mo:02d}-{d:02d}"
    if y is not None:
        return f"{y:04d}-{mo:02d}"
    return f"--{mo:02d}-{d:02d}"


def _dates(text: str) -> list[dict]:
    out: list[dict] = []
    taken = _Taken()
    for pattern, kind in _DATE_RES:
        for m in pattern.finditer(text):
            if taken.overlaps(m.start(), m.end()):
                continue
            value = _date_value(kind, m)
            if value:
                taken.add(m.start(), m.end())
                out.append({"kind": "date", "text": m.group(0).strip(), "value": value, "span": (m.start(), m.end())})
    return out


def _times(text: str) -> list[dict]:
    out: list[dict] = []
    taken = _Taken()
    for pattern in _TIME_RES:
        for m in pattern.finditer(text):
            if taken.overlaps(m.start(), m.end()):
                continue
            h, mi = int(m.group(1)), int(m.group(2) or 0)
            ampm = (m.group(3) or "").lower()
            if ampm == "p" and h < 12:
                h += 12
            if ampm == "a" and h == 12:
                h = 0
            if h > 23 or mi > 59:
                continue
            taken.add(m.start(), m.end())
            out.append({"kind": "time", "text": m.group(0).strip().rstrip("."), "value": f"{h:02d}:{mi:02d}", "span": (m.start(), m.end())})
    return out


def _context_words(text: str, pos: int, n: int = 3) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", text[max(0, pos - 60) : pos].lower())[-n:]


def _numbers(text: str) -> list[dict]:
    out = []
    for m in _NUM_RE.finditer(text):
        tail = (m.group("tail") or "").lower()
        unit = ""
        if m.group("sunit"):
            unit = _UNITS.get(m.group("sunit").lower(), m.group("sunit").lower())
        if tail:
            if tail in _ORDINALS:
                pass
            elif tail in _UNITS and not unit:
                unit = _UNITS[tail]
            elif m.group("scalew") or m.group("scalel") or m.group("pct") or m.group("sunit"):
                pass  # e.g. "5 millionth": keep the number, ignore the letters
            else:
                continue  # glued to letters, e.g. 'H2O', '4o', 'v2', '3D'
        raw = m.group("num")
        try:
            base = float(raw.replace(",", ""))
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        scale_word = (m.group("scalew") or m.group("scalel") or "").lower()
        scale = _SCALES.get(scale_word, 1.0)
        if m.group("cur"):
            unit = m.group("cur")
        elif m.group("pct"):
            unit = "%"
        value = base * scale * (-1 if m.group("sign") else 1)
        number = {
            "kind": "number",
            "text": m.group(0).strip(),
            "value": value,
            "tol": 0.5 * (10 ** -decimals) * scale,
            "unit": unit,
            "span": (m.start(), m.end()),
            "context": _context_words(text, m.start()),
        }
        if scale_word:
            number["scale"] = {"k": "thousand", "m": "million", "b": "billion", "bn": "billion"}.get(scale_word, scale_word)
        out.append(number)
    return out


def _scan(text: str) -> dict:
    """Locate all hard facts in text. Shared by extract() and match()."""
    text = _FENCED_RE.sub(lambda m: " " * len(m.group(0)), text or "")
    urls, emails, codes = [], [], []
    spans: list[tuple[int, int]] = []
    taken = _Taken()

    def claim(start: int, end: int) -> None:
        spans.append((start, end))
        taken.add(start, end)

    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?")
        urls.append({"kind": "url", "text": url})
        claim(m.start(), m.start() + len(url))
    for m in _EMAIL_RE.finditer(text):
        if not taken.contains(m.start()):
            emails.append({"kind": "email", "text": m.group(0)})
            claim(m.start(), m.end())
    for m in _CODE_RE.finditer(text):
        code = m.group(1).strip()
        if code and not taken.contains(m.start()):
            codes.append({"kind": "code", "text": code})
        claim(m.start(), m.end())
    blanked = _blank(text, spans)
    dates = _dates(blanked)  # before IDs, so "03/14/2026" is a date, not a phone-like ID
    for d in dates:
        claim(*d["span"])
    blanked = _blank(text, spans)
    ids = []
    for pattern in _ID_RES:
        for m in pattern.finditer(blanked):
            if taken.overlaps(m.start(), m.end()):
                continue
            ids.append({"kind": "id", "text": m.group(0).strip()})
            claim(m.start(), m.end())
    blanked = _blank(text, spans)
    times = _times(blanked)
    blanked = _blank(blanked, [t["span"] for t in times])
    blanked = _LIST_MARKER_RE.sub(lambda m: m.group(1) + " " * len(m.group(2)) + m.group(3), blanked)
    numbers = _numbers(blanked)
    return {"text": text, "blanked": blanked, "urls": urls, "emails": emails, "codes": codes, "ids": ids, "dates": dates, "times": times, "numbers": numbers}


def _is_sentence_start(text: str, pos: int) -> bool:
    j = pos - 1
    while j >= 0 and text[j] in " \t*_\"'([":
        j -= 1
    if j < 0 or text[j] in ".!?:\n":
        return True
    line_start = text.rfind("\n", 0, pos) + 1
    prefix = text[line_start:pos].strip()
    return bool(re.fullmatch(r"(?:[-*+#>•]+|\d{1,3}[.)])?", prefix))


def _is_capitalized(token: str) -> bool:
    if len(token) < 2 or token.lower() in _STOP_CAPS:
        return False
    if token[0].isupper():
        return True
    return token[0].isalpha() and any(c.isupper() for c in token[1:])  # iPhone, eBay


def extract_entities(text: str, limit: int = MAX_ENTITIES) -> list[dict]:
    """Named entities: runs of capitalised words, e.g. 'Marie Curie', 'Bank of America'."""
    groups: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    pending: tuple[str, int, int] | None = None
    for m in _ENTITY_TOKEN_RE.finditer(text):
        tok, s, e = m.group(0), m.start(), m.end()
        clean = re.sub(r"['’]s$", "", tok).strip(".-'")
        contiguous = bool(current) and re.fullmatch(r"[ \t]+", text[current[-1][2] : s]) is not None
        if clean and not clean.isdigit() and _is_capitalized(clean):
            if current and (contiguous or pending):
                if pending:
                    current.append(pending)
                    pending = None
                current.append((clean, s, e))
            else:
                if current:
                    groups.append(current)
                current, pending = [(clean, s, e)], None
            continue
        if current and contiguous and pending is None and tok.lower() in _CONNECTORS:
            pending = (tok, s, e)
            continue
        if current:
            groups.append(current)
        current, pending = [], None
    if current:
        groups.append(current)

    lower_words = set(re.findall(r"\b[a-z][a-z'’-]*\b", text))
    out: list[dict] = []
    seen = set()
    for g in groups:
        name = " ".join(t for t, _, _ in g)
        first = g[0][0]
        if len(g) == 1 and first.lower() in lower_words:
            # Capitalised here but written in lower case elsewhere: an ordinary word
            # that happens to start a sentence ("Shipping is free ... free shipping").
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": "entity", "text": name})
        if len(out) >= limit:
            break
    return out


def _strip(f: dict) -> dict:
    return {k: v for k, v in f.items() if k != "span"}


def extract(text: str, *, max_hard: int = MAX_HARD, max_entities: int = MAX_ENTITIES) -> list[dict]:
    """Extract checkable facts from a text (fenced code blocks are ignored)."""
    found = _scan(text)
    ordered = found["urls"] + found["emails"] + found["ids"] + found["codes"] + found["dates"] + found["times"] + found["numbers"]
    hard: list[dict] = []
    seen = set()
    for f in ordered:
        key = (f["kind"], f.get("value", normalize(f["text"])), f.get("unit", ""))
        if key in seen:
            continue
        seen.add(key)
        hard.append(_strip(f))
    hard = hard[:max_hard]
    entity_text = _blank(found["text"], [(m.start(), m.end()) for m in _CODE_RE.finditer(found["text"])])
    hard_texts = {normalize(h["text"]) for h in hard}
    entities = [e for e in extract_entities(entity_text, limit=max_entities * 2) if normalize(e["text"]) not in hard_texts]
    return hard + entities[:max_entities]


# ------------------------------------------------------------------ matching


def _num_match(fact: dict, cand: dict) -> bool:
    fu, cu = fact.get("unit") or "", cand.get("unit") or ""
    fv, cv = float(fact["value"]), float(cand["value"])
    tol = float(fact.get("tol", 0.5)) + 1e-9 * max(1.0, abs(fv))
    if fu in _CURRENCIES and cu in _CURRENCIES and fu != cu:
        return False
    if fu == "%" and cu == "":
        # "85%" written as a fraction ("0.85") or as a bare number ("85")
        return abs(cv - fv) < tol or abs(cv * 100 - fv) < tol
    if fu == "" and cu == "%":
        return abs(cv - fv) < tol or abs(cv - fv * 100) < tol * 100
    if fu and cu and fu != cu:
        return False
    return abs(cv - fv) < tol


def _contains_number_text(norm_text: str, text: str) -> bool:
    """Literal fallback with numeric boundaries: '10' is not found inside '10,000'."""
    t = normalize(text)
    if not t:
        return False
    return re.search(r"(?<![\w.,\-])" + re.escape(t) + r"(?![\w]|[.,]\d)", norm_text) is not None


def _id_key(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _id_found(fact_text: str, candidate: str) -> bool:
    """IDs and phone numbers match on their letters/digits: '(555) 123-4567' == '555-123-4567'."""
    key = _id_key(fact_text)
    if not key:
        return False
    tokens = candidate.split()
    for i in range(len(tokens)):
        joined = ""
        for j in range(i, min(i + 4, len(tokens))):
            joined += _id_key(tokens[j])
            if joined == key:
                return True
            if len(joined) >= len(key):
                break
    return False


def _entity_found(entity: str, norm_text: str, raw_text: str) -> bool:
    e = normalize(entity)
    if re.search(r"(?<!\w)" + re.escape(e) + r"(?!\w)", norm_text):
        return True
    parts = [p for p in entity.split() if p.lower() not in _CONNECTORS]
    if len(parts) < 2:
        return False
    # "Marie Curie" -> "Curie" is fine, but "South Korea" -> "North Korea" is not.
    qualifier = parts[-2].lower()
    for m in re.finditer(r"(?<!\w)" + re.escape(parts[-1]) + r"(?!\w)", raw_text, re.I):
        before = re.findall(r"[\w'’-]+", raw_text[max(0, m.start() - 40) : m.start()])
        prev = before[-1] if before else ""
        if prev and prev[0].isupper() and prev.lower() != qualifier and prev.lower() not in _STOP_CAPS:
            continue
        return True
    return False


def _date_found(value: str, cand_dates: set) -> bool:
    if value in cand_dates:
        return True
    if len(value) == 10:  # YYYY-MM-DD: accept the same day written without a year
        return "--" + value[5:] in cand_dates
    if value.startswith("--"):  # month-day: accept any year
        return any(len(c) == 10 and c[4:] == value[1:] for c in cand_dates)
    if len(value) == 7:  # YYYY-MM
        return any(c.startswith(value) for c in cand_dates)
    return False


def contains_term(norm_text: str, term: str) -> bool:
    """Substring test that respects word boundaries at the term's edges."""
    t = normalize(term)
    if not t:
        return False
    left = r"(?<![\w.])" if re.match(r"\w", t) else ""
    right = r"(?!\w)" if re.search(r"\w$", t) else ""
    return re.search(left + re.escape(t) + right, norm_text) is not None


def _words_to_digits(text: str) -> str:
    return _NUMBER_WORD_RE.sub(lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), text)


def match(facts: list[dict], candidate: str) -> dict:
    """Check which facts appear in `candidate`.

    Returns {"found", "missing", "changed": [{"from", "to"}], "hard_recall", "entity_recall"}.
    """
    candidate = candidate or ""
    found_all = _scan(candidate)
    norm = normalize(candidate)
    cand_numbers = list(found_all["numbers"])
    # "two" counts for 2 (only when matching, so a baseline never gains word facts)
    worded = _words_to_digits(found_all["blanked"])
    if worded != found_all["blanked"]:
        seen_spans = {n["span"] for n in cand_numbers}
        cand_numbers += [n for n in _numbers(worded) if n["span"] not in seen_spans]
    cand_dates = {d["value"] for d in found_all["dates"]}
    for m in _SHORT_NUMERIC_DATE_RE.finditer(candidate):
        a, b = int(m.group(1)), int(m.group(2))
        for mo, d in ((a, b), (b, a)):
            if 1 <= mo <= 12 and 1 <= d <= 31:
                cand_dates.add(f"--{mo:02d}-{d:02d}")
    cand_times = {t["value"] for t in found_all["times"]}
    found, missing = [], []
    for f in facts:
        kind = f.get("kind")
        text = f.get("text") or ""
        if kind == "number":
            ok = any(_num_match(f, n) for n in cand_numbers) or _contains_number_text(norm, text)
        elif kind == "date":
            ok = _date_found(str(f.get("value", "")), cand_dates) or (bool(text) and contains_term(norm, text))
        elif kind == "time":
            ok = f.get("value") in cand_times or (bool(text) and contains_term(norm, text))
        elif kind == "url":
            ok = normalize(text).rstrip("/") in norm
        elif kind == "id":
            ok = _id_found(text, candidate) or (bool(text) and contains_term(norm, text))
        elif kind == "entity":
            ok = _entity_found(text, norm, candidate)
        else:
            ok = bool(text) and contains_term(norm, text)
        (found if ok else missing).append(f)

    # A missing number "changed" only with evidence: shared wording, or the same unit/scale and a similar size.
    changed = []
    base_numbers = [f for f in facts if f.get("kind") == "number"]
    unmatched = [n for n in cand_numbers if not any(_num_match(b, n) for b in base_numbers)]
    used: set[int] = set()
    for f in missing:
        if f.get("kind") != "number":
            continue
        best, best_score = None, 0.0
        for i, n in enumerate(unmatched):
            if i in used:
                continue
            context = len(set(n.get("context", [])) & set(f.get("context", [])))
            fu = f.get("unit") or f.get("scale") or ""
            nu = n.get("unit") or n.get("scale") or ""
            same_unit = bool(fu) and fu == nu
            fv, nv = abs(float(f["value"])), abs(float(n["value"]))
            similar = bool(fv and nv and 0.5 <= nv / fv <= 2.0)
            if not context and not (same_unit and similar):
                continue
            score = context + (0.5 if same_unit else 0.0) + (0.75 if similar else 0.0)
            if score > best_score:
                best, best_score = i, score
        if best is not None and best_score >= 1.0:
            used.add(best)
            changed.append({"from": f["text"], "to": unmatched[best]["text"]})

    hard = [f for f in facts if f.get("kind") != "entity"]
    ents = [f for f in facts if f.get("kind") == "entity"]
    return {
        "found": found,
        "missing": missing,
        "changed": changed,
        "hard_recall": sum(1 for f in found if f.get("kind") != "entity") / len(hard) if hard else 1.0,
        "entity_recall": sum(1 for f in found if f.get("kind") == "entity") / len(ents) if ents else 1.0,
    }


def public(fact: dict) -> dict:
    """Keep only the keys worth storing in a test file."""
    keep = ("kind", "text", "value", "unit", "scale", "tol", "context")
    return {k: fact[k] for k in keep if k in fact and fact[k] not in (None, "", [])}
