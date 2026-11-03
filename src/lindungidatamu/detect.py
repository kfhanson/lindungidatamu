"""Finding Indonesian personal data in text.

Three sources, most certain first:
  pattern   regular expressions with checks (patterns.json): NIK, NPWP, KK, cards, keys, ...
  term      your own words: client names, project codenames, internal systems
  model     names, organisations and addresses from a name model (lindungidatamu.names, optional)

`find` returns non-overlapping entities: where two overlap, the more certain source wins, then the
longer match, then the earlier pattern in patterns.json.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from .validators import VALIDATORS

SOURCES = ("pattern", "term", "model")


@dataclass(frozen=True)
class Entity:
    type: str        # NIK, NPWP, PERSON, ...
    start: int
    end: int
    text: str
    source: str      # pattern | term | model
    detector: str = ""   # the pattern's id, the term, or the model's name


@dataclass(frozen=True)
class Format:
    """One of your own number formats, e.g. a policy number: Format("POLICY", "POL-########").
    # is a digit, @ a letter, * either; anything else as written (any case). Built into a pattern
    here, never written as a regular expression, so a format can't make matching slow."""
    type: str
    format: str

    @property
    def regex(self) -> re.Pattern:
        tokens = {"#": r"\d", "@": "[A-Za-z]", "*": "[A-Za-z0-9]"}
        body = "".join(tokens.get(c) or re.escape(c) for c in self.format.strip())
        return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class Term:
    """One of your own words, matched as a whole word in any case: Term("CLIENT", "PT Sinar Abadi")."""
    type: str
    text: str
    aliases: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _compiled() -> tuple:
    doc = json.loads(resources.files(__package__).joinpath("patterns.json").read_text())
    out = []
    for p in doc["patterns"]:
        flags = re.IGNORECASE if "i" in p.get("flags", "") else 0
        allow = frozenset(_normal(v) for v in p.get("allow", []))
        out.append((p, re.compile(p["pattern"], flags), VALIDATORS.get(p.get("validator", "")), allow))
    return tuple(out)


@lru_cache(maxsize=1)
def _public() -> dict:
    doc = json.loads(resources.files(__package__).joinpath("public.json").read_text())
    titles = sorted(doc["titles"], key=len, reverse=True)
    return {"organisations": tuple(doc["organisations"]),
            "not_names": frozenset(w.lower() for w in doc["not_names"]),
            "public_prefixes": frozenset(doc.get("public_prefixes", [])),
            "facilities": frozenset(w.lower() for w in doc["facilities"]),
            "facility_suffixes": frozenset(w.lower() for w in doc["facility_suffixes"]),
            "phrase_breaks": frozenset(w.lower() for w in doc["phrase_breaks"]),
            "title_words": frozenset(t.rstrip(".").lower() for t in doc["titles"]),
            "disease_before": frozenset(w.lower() for w in doc["disease_before"]),
            "disease_after": frozenset(w.lower() for w in doc["disease_after"]),
            "titles": re.compile(r"^(?:(?:" + "|".join(re.escape(t) for t in titles) + r")\s+)+", re.IGNORECASE)}


# After a street (or Gg., Komplek, Dusun) the rest of an Indonesian address follows in parts:
# RT/RW, kelurahan or desa, kecamatan, kabupaten or kota, a city, a postcode. A known city
# (public.json `places`) counts wherever it is; another capitalised name only after a comma and
# before a comma, full stop, line end or postcode, so "Jl. Kenanga No. 7, Budi minta dikirim"
# doesn't take "Budi".
_ADDRESS_PART = re.compile(
    r"\s*,?\s*(?:"
    r"RT\.?\s*\d{1,3}\s*[/ ]?\s*RW\.?\s*\d{1,3}|RT\.?\s*\d{1,3}|RW\.?\s*\d{1,3}"
    r"|(?:Kel(?:urahan)?|Desa|Ds|Dusun|Dsn|Kec(?:amatan)?|Kab(?:upaten)?|Kota|Kodya|Prov(?:insi)?)\.?\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2}"
    r"|\d{5}\b)")
_CITY_PART = re.compile(r"\s*,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?=\s*(?:[,.;\n]|\d{5}\b|$))")


@lru_cache(maxsize=1)
def _places() -> re.Pattern:
    names = sorted(json.loads(resources.files(__package__).joinpath("public.json").read_text())["places"], key=len, reverse=True)
    return re.compile(r"\s*,?\s*(?:" + "|".join(re.escape(n) for n in names) + r")\b")


def _extend_address(text: str, end: int) -> int:
    cities = 0
    while True:
        m = _ADDRESS_PART.match(text, end)
        if m and m.end() > end:
            end = m.end()
            continue
        m = _places().match(text, end) or _CITY_PART.match(text, end)
        if m and cities < 2:
            end, cities = m.end(), cities + 1
            continue
        return end


def patterns_version() -> str:
    return json.loads(resources.files(__package__).joinpath("patterns.json").read_text())["version"]


def _normal(v: str) -> str:
    d = re.sub(r"\D", "", v)
    return d if len(d) >= 8 and re.fullmatch(r"[\d\s.\-+()]+", v) else v.strip()


def _words(text: str, word: str) -> list[tuple[int, int]]:
    if not word.strip():
        return []
    rx = re.compile(rf"(?<![^\W_]){re.escape(word.strip())}(?![^\W_])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in rx.finditer(text)]


def pattern_entities(text: str, types: set[str] | None = None) -> list[Entity]:
    out = []
    for spec, rx, check, allow in _compiled():
        if types and spec["type"] not in types:
            continue
        g = spec.get("group", 0)
        for m in rx.finditer(text):
            value = m.group(g)
            if (check is None or check(value)) and _normal(value) not in allow:
                start, end = m.start(g), m.end(g)
                if spec["type"] == "ADDRESS":
                    end = _extend_address(text, end)
                out.append(Entity(spec["type"], start, end, text[start:end], "pattern", spec["id"]))
    return out


def _not_an_organisation(name: str, public: tuple[str, ...]) -> bool:
    """Real organisation names are capitalised: "akun aku dibajak" isn't one. Nor is a public one
    with filler around it ("VA BTN")."""
    words = [w for w in name.split() if len(w) > 3]          # "the", "of", "dan" don't decide it
    if words and sum(w[:1].islower() for w in words) * 2 >= len(words):
        return True
    rest = name
    for p in sorted(public, key=len, reverse=True):
        rest = re.sub(rf"(?<![^\W_]){re.escape(p)}(?![^\W_])", " ", rest)
    return rest != name and not any(len(w) >= 4 and w[:1].isupper() for w in rest.split())


_TOKEN = re.compile(r"[^\W_][\w'’]*\.?|[^\w\s]")
# Words that make an organisation a health facility wherever they stand ("Siloam Hospitals Medan").
# Not "Medika" or "Farma": "PT Sinar Medika" may be a client whose name must stay masked.
_FACILITY_ANYWHERE = {"hospital", "hospitals", "clinic", "klinik", "puskesmas"}
# An ICD-10 code written onto the name: "G20-PARKINSON". Not "A12 - Budi" (a bed, a room).
_ICD_BEFORE = re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,3})?[-–:]$")


def _first_word(s: str) -> str:
    m = _TOKEN.search(s)
    return m.group(0).rstrip(".").lower() if m else ""


def _namesake(e: Entity, text: str) -> bool:
    """A person's name that is part of an institution's name: RS Hermina, RSUD Dr. Soetomo, Siloam
    Hospitals Medan (Budi), Hermina Hospital. Indonesian hospitals are often named after people; the
    name model reads them as patients. Looks back along the same name phrase (capitalised words,
    titles, brackets) for a facility word; a lowercase word, a comma or a linking word ("oleh",
    "a.n.") ends the phrase, so "RS Hermina, dr. Budi" keeps Budi a person. A doctor after a smaller
    facility ("Klinik dr. Budi") stays a person: that is usually the doctor's own practice."""
    pub = _public()
    after = text[e.end:e.end + 30]
    m = re.match(r"\s*([^\W\d_]+)", after)
    if m and m.group(1).lower() in pub["facility_suffixes"]:
        return True
    line = text[text.rfind("\n", 0, e.start) + 1:e.start]
    line = re.sub(r"(?i)rumah\s+sakit", "RS", line)
    titled = False
    for tok in reversed(_TOKEN.findall(line)[-8:]):
        w = tok.rstrip(".").lower()
        if w in pub["facilities"]:
            # "RSUD Dr. Soetomo" is a namesake; "Klinik dr. Budi" is usually Dr Budi's own practice
            return not titled or w.startswith("rs") or w.startswith("hospital")
        if w in pub["phrase_breaks"] or tok in (",", ";", "/", "|", ":"):
            return False
        if w in pub["title_words"]:
            titled = True
            continue
        if not tok[0].isalnum() or tok[0].isupper() or tok[0].isdigit():
            continue
        return False
    return False


def _trim_facility(e: Entity) -> Entity | None:
    """A name the model ran together with a facility word and title ("KLINIK DR. HENDRA KUSUMA",
    "RSUD DR. SOETOMO"): the name alone, so the facility rules can judge it. None if nothing is left."""
    pub, skipped_facility, cut = _public(), False, 0
    text = re.sub(r"(?i)^rumah\s+sakit", "RS   ", e.text[:10]) + e.text[10:]   # same length
    for m in _TOKEN.finditer(text):
        w = m.group(0).rstrip(".").lower()
        if w in pub["facilities"]:
            skipped_facility = True
        elif not (skipped_facility and (w in pub["title_words"] or not m.group(0)[0].isalnum())):
            cut = m.start()
            break
    else:
        return None
    if not skipped_facility or cut == 0:
        return e
    return Entity(e.type, e.start + cut, e.end, e.text[cut:], e.source, e.detector)


def _eponym(e: Entity, text: str) -> bool:
    """A disease named after a person (Parkinson, Crohn, Hodgkin): an ICD code written onto it, or
    an unambiguous disease word before or after it."""
    pub = _public()
    before = text[max(0, e.start - 20):e.start]
    if _ICD_BEFORE.search(before):
        return True
    words = re.findall(r"[^\W\d_]+", before)
    if words and words[-1].lower() in pub["disease_before"]:
        return True
    m = re.match(r"(?:['’]s)?\s*([^\W\d_]+)", text[e.end:e.end + 30])
    return bool(m and m.group(1).lower() in pub["disease_after"])


def _clean_names(found: list[Entity], text: str, public: bool = True) -> list[Entity]:
    """A name model's people without forms of address ("kak", "Customer") and titles, and its
    organisations without ordinary phrases and public bodies dressed up ("VA BTN"). People who are
    really an institution's or a disease's name are dropped; so are health facilities (public)."""
    pub, out = _public(), []
    cased = any(c.isupper() for c in text)      # all-lowercase text has no capitals to go by
    for e in found:
        if e.type in ("PERSON", "ORG") and " ".join(e.text.strip(" .,").lower().split()) in pub["not_names"]:
            continue
        if e.type == "PERSON":
            e = _trim_facility(e)
            if e is None or _namesake(e, text) or _eponym(e, text):
                continue
        if e.type == "ORG" and public and _namesake(e, text):
            continue                             # "RS Pondok Indah": the model's ORG left out the "RS"
        if e.type == "ORG" and public and (_first_word(re.sub(r"(?i)^rumah\s+sakit", "RS", e.text)) in pub["facilities"]
                                           or any(w.lower() in _FACILITY_ANYWHERE for w in re.findall(r"[^\W\d_]+", e.text))):
            continue                             # a hospital, clinic or pharmacy: public, and what an
                                                 # analysis of claims or referrals needs to see
        if e.type == "ORG" and cased and _not_an_organisation(e.text, pub["organisations"] if public else ()):
            continue
        if e.type == "ORG" and public and len(e.text.split()) <= 4 and e.text.split()[0].lower() in pub["public_prefixes"]:
            continue                             # a ministry, court, council or party: public. Only a
                                                 # short name: a long phrase may hold a private one
        if e.type == "ADDRESS" and len(e.text.split()) < 2 and not any(c.isdigit() for c in e.text):
            continue                             # "rumahnya": a place word, not an address
        if e.type == "PERSON":
            t = pub["titles"].match(e.text)
            if t and t.end() < len(e.text):
                e = Entity(e.type, e.start + t.end(), e.end, e.text[t.end():], e.source, e.detector)
            elif t:
                continue
        out.append(e)
    return out


def find(text: str, *, types: set[str] | None = None, terms: list[Term] = (), formats: list[Format] = (),
         names=None, keep: list[str] = (), public: bool = True) -> list[Entity]:
    """Personal data in `text`, non-overlapping, in order.

    types    only these types (default: all)
    terms    your own words to find (client names, codenames)
    formats  your own number formats
    names    a name finder (lindungidatamu.names.NameFinder) for PERSON, ORG and ADDRESS
    keep     public words the name finder must not report (your brand, a regulator)
    public   also keep Indonesia's public banks, payment apps, couriers, platforms and government
             bodies (public.json), which a name model often reports as organisations
    """
    found = pattern_entities(text, types)
    for t in terms:
        for word in (t.text, *t.aliases):
            found += [Entity(t.type, a, b, text[a:b], "term", t.text) for a, b in _words(text, word)]
    for f in formats:
        found += [Entity(f.type, m.start(), m.end(), m.group(0), "pattern", f.format) for m in f.regex.finditer(text)]
    if names is not None:
        kept = [*keep, *(_public()["organisations"] if public else ())]
        ranges = [r for w in kept for r in _words(text, w)]
        found += [e for e in _clean_names(names.find(text), text, public)
                  if not any(a <= e.start and e.end <= b for a, b in ranges)]
    if types:
        found = [e for e in found if e.type in types]
    return _resolve(found, text)


# Placeholders already in the text ([NIK_1]) are never found again.
_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z_]{1,15}_\d{1,7}\]")


def _resolve(found: list[Entity], text: str) -> list[Entity]:
    taken = [(m.start(), m.end()) for m in _PLACEHOLDER.finditer(text)]
    order = {id(e): i for i, e in enumerate(found)}
    ranked = sorted((e for e in found if e.end > e.start and not any(e.start < b and a < e.end for a, b in taken)),
                    # An address competes on length whoever found it: the model's full address beats
                    # a pattern that stopped at the street.
                    key=lambda e: (0 if e.type == "ADDRESS" else SOURCES.index(e.source), -(e.end - e.start), order[id(e)]))
    kept: list[Entity] = []
    for e in ranked:
        if not any(e.start < k.end and k.start < e.end for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)
