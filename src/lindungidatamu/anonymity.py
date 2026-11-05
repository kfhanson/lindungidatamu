"""Re-identification risk: can a masked table still single people out?

Masking takes out names and numbers, but the columns that stay can still point to one person
together: a woman born on 14 March 1987, living in kecamatan Cinere, admitted on 3 February, is
probably the only one in the table. Those columns are quasi-identifiers. This module measures how
identifying they are and finds the smallest change that fixes it.

    report = measure(columns, quasi=["birth_date", "sex", "kecamatan"], k=5)
    plan = suggest(columns, {"birth_date": "date", "sex": "sex", "kecamatan": "wilayah"}, k=5)
    plan.levels                     # {'birth_date': 'year', 'sex': 'value', 'kecamatan': 'kabupaten'}
    safe = apply(columns, plan.levels, plan.hierarchies, k=5)

The measures are the standard ones:
    k             the smallest group of rows that share their quasi-identifiers (k-anonymity). An
                  attacker who knows those facts about someone can't narrow them below 1 in k
    unique rows   rows that are alone in their group
    average risk  the mean of 1 / group size: the chance of picking the right row, averaged over everyone
    l             for a sensitive column (a diagnosis), the fewest different values in any group. l = 1
                  means some group all shares one diagnosis: you learn it without knowing the row.
                  `suggest` and `apply` take an `l` to reach as well as k

Dates are most useful, and least identifying, as intervals: age at admission and days from admission
to discharge say what the analysis needs, the calendar dates mostly say who it was. `plan_intervals`
picks the date the others hang off (the anchor) and `interval` turns a date into whole years (age) or
days from it; the anchor itself stays a quasi-identifier, generalised like any date.

Groups count people, not rows, when the table says which rows belong to one person (`person`, a
tokenized member or patient column): five claims of one patient are a group of one.

Generalisation steps a column up its hierarchy (date -> month -> year -> 5 years -> *; kelurahan code
-> kecamatan -> kabupaten -> provinsi -> *; ICD-10 code -> category -> chapter -> *). `suggest` searches
every combination of levels for the one that reaches k with the least lost: fewest steps up, fewest
rows suppressed. Rows still below k after that (at most `max_suppress` of them) have their
quasi-identifiers replaced by "*". Everything this module reports is counts: safe to share.
"""

import bisect
import datetime as dt
import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

SUPPRESSED = "*"

# Each hierarchy's levels, from the value as it is to fully suppressed.
LEVELS = {
    "date": ("value", "month", "year", "5 years", "*"),
    "age": ("value", "5 years", "10 years", "20 years", "*"),
    "days": ("value", "7 days", "30 days", "90 days", "365 days", "*"),
    "wilayah": ("value", "kecamatan", "kabupaten", "provinsi", "*"),
    "postal": ("value", "3 digits", "2 digits", "*"),
    "icd10": ("value", "category", "chapter", "*"),
    "sex": ("value", "*"),
    "category": ("value", "common", "*"),
}
HIERARCHIES = tuple(LEVELS)

_YMD = re.compile(r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_DMY = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})")
_ICD = re.compile(r"^\s*([A-Z])(\d{2})")
_WILAYAH = re.compile(r"^\s*\d{2}(?:\.?\d{2}(?:\.?\d{2}(?:\.?\d{4})?)?)?\s*$")
_POSTAL = re.compile(r"^\s*\d{5}\s*$")

# ICD-10 (WHO) chapters by their first code.
_CHAPTERS = [("A00", "A00-B99"), ("C00", "C00-D48"), ("D50", "D50-D89"), ("E00", "E00-E90"), ("F00", "F00-F99"),
             ("G00", "G00-G99"), ("H00", "H00-H59"), ("H60", "H60-H95"), ("I00", "I00-I99"), ("J00", "J00-J99"),
             ("K00", "K00-K93"), ("L00", "L00-L99"), ("M00", "M00-M99"), ("N00", "N00-N99"), ("O00", "O00-O99"),
             ("P00", "P00-P96"), ("Q00", "Q00-Q99"), ("R00", "R00-R99"), ("S00", "S00-T98"), ("U00", "U00-U99"),
             ("V01", "V01-Y98"), ("Z00", "Z00-Z99")]
_CHAPTER_STARTS = [c for c, _ in _CHAPTERS]

COMMON_SHARE = 0.01        # "common": values held by fewer than 1% of rows become "other"


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip()) or \
        type(v).__name__ in ("NAType", "NaTType")


def _text(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _ymd(v):
    if hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day"):
        return v.year, v.month, v.day
    s = _text(v)
    if m := _YMD.match(s):
        return int(m[1]), int(m[2]), int(m[3])
    if m := _DMY.match(s):
        return int(m[3]), int(m[2]), int(m[1])
    return None


def _number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(_text(v).replace(",", "."))
    except ValueError:
        return None


def _step(v, hierarchy: str, level: str, common: set | None = None):
    """One value, generalised to `level`. Empty stays empty."""
    if _empty(v):
        return None
    if level == "value":
        return v
    if level == "*":
        return SUPPRESSED
    if hierarchy == "date":
        d = _ymd(v)
        if d is None:
            return "other"
        y, m, _ = d
        if level == "month":
            return f"{y:04d}-{m:02d}"
        if level == "year":
            return str(y)
        lo = y - y % 5
        return f"{lo}-{lo + 4}"
    if hierarchy == "age":
        a = _number(v)
        if a is None:
            return "other"
        width = int(level.split()[0])
        lo = int(a // width * width)
        return f"{lo}-{lo + width - 1}"
    if hierarchy == "days":
        d = _number(v)
        if d is None:
            return "other"
        width = int(level.split()[0])
        lo = int(d // width * width)
        return f"{lo}..{lo + width - 1}"
    if hierarchy == "wilayah":
        digits = re.sub(r"\D", "", _text(v))
        n = {"kecamatan": 6, "kabupaten": 4, "provinsi": 2}[level]
        if len(digits) < n:
            return digits or "other"
        d = digits[:n]
        return ".".join(d[i:i + 2] for i in range(0, n, 2))
    if hierarchy == "postal":
        s = _text(v)
        n = int(level.split()[0])
        return s[:n] + "x" * (5 - n) if _POSTAL.match(s) else "other"
    if hierarchy == "icd10":
        m = _ICD.match(_text(v).upper())
        if not m:
            return "other"
        code = m[1] + m[2]
        if level == "category":
            return code
        i = bisect.bisect_right(_CHAPTER_STARTS, code) - 1
        return _CHAPTERS[i][1] if i >= 0 else "other"
    if hierarchy == "category":
        return v if common is not None and _text(v) in common else "other"
    raise ValueError(f"unknown hierarchy {hierarchy!r}")


def generalize(values: list, hierarchy: str, level: str) -> list:
    """A column, generalised to one level of its hierarchy."""
    if hierarchy not in LEVELS:
        raise ValueError(f"unknown hierarchy {hierarchy!r}: one of {', '.join(HIERARCHIES)}")
    if level not in LEVELS[hierarchy]:
        raise ValueError(f"{hierarchy} has no level {level!r}: one of {', '.join(LEVELS[hierarchy])}")
    common = None
    if hierarchy == "category" and level == "common":
        counts = Counter(_text(v) for v in values if not _empty(v))
        floor = COMMON_SHARE * len(values)
        common = {t for t, n in counts.items() if n >= floor}
    return [_step(v, hierarchy, level, common) for v in values]


def _ordinal(v):
    d = _ymd(v)
    if d is None:
        return None
    try:
        return dt.date(*d).toordinal()
    except ValueError:
        return None


def interval(values: list, anchor: list, unit: str) -> list:
    """A date column as an interval from the anchor date of the same row.

    unit "age"   whole years from the date to the anchor (birth date -> age at admission)
         "days"  days from the anchor to the date (discharge -> days after admission; earlier is negative)
    Empty, or no anchor, gives None.
    """
    if unit not in ("age", "days"):
        raise ValueError(f"unknown interval {unit!r}: age or days")
    out = []
    for v, a in zip(values, anchor):
        d, b = (None if _empty(v) else _ymd(v)), (None if _empty(a) else _ymd(a))
        if d is None or b is None:
            out.append(None)
        elif unit == "age":
            out.append(b[0] - d[0] - ((b[1], b[2]) < (d[1], d[2])))
        else:
            x, y = _ordinal(v), _ordinal(a)
            out.append(None if x is None or y is None else x - y)
    return out


def plan_intervals(columns: dict[str, list], dates: list[str], birth: tuple[str, ...] = ()) -> dict[str, tuple]:
    """Which date to anchor on, and each other date's interval: {column: (anchor, "age" | "days")}.

    The anchor is the date the others are closest to (the least spread of days from it, by entropy), the
    earliest of those that are about as close: in a claims table, admission rather than discharge or
    policy start. Birth dates (named in `birth`, or 15+ years
    before the latest date on the median) become age, never the anchor; the rest days. Fewer than two dates: nothing.
    """
    if len(dates) < 2:
        return {}
    ords = {c: [_ordinal(v) if not _empty(v) else None for v in columns[c]] for c in dates}

    def median(a):
        xs = sorted(x for x in ords[a] if x is not None)
        return xs[len(xs) // 2] if xs else math.inf

    latest = max((median(c) for c in dates if median(c) != math.inf), default=math.inf)
    birth = set(birth) | {c for c in dates if latest != math.inf and median(c) <= latest - 15 * 365}
    events = [c for c in dates if c not in birth]
    if not events:
        return {}

    def spread(a: str) -> float:
        total = 0.0
        for c in events:
            if c != a:
                total += _entropy([None if x is None or y is None else x - y for x, y in zip(ords[c], ords[a])])
        present = sum(x is not None for x in ords[a])
        return total - 1e-9 * present          # ties: the anchor with the fewest empties

    # Admission and discharge are equally close to each other: of the closest, the earliest.
    scores = {a: spread(a) for a in events}
    best = min(scores.values())
    close = [a for a in events if scores[a] <= best + 0.05 * max(abs(best), 1)]
    anchor = min(close, key=median)
    plan = {}
    for c in dates:
        if c == anchor:
            continue
        plan[c] = (anchor, "age" if c in birth else "days")
    return plan


def guess_hierarchy(values: list, sample: int = 500) -> str:
    """The hierarchy a quasi-identifier column's values fit, from a sample."""
    vals = [v for v in values if not _empty(v)][:sample]
    if not vals:
        return "category"
    share = lambda f: sum(1 for v in vals if f(v)) / len(vals)
    if share(lambda v: _ymd(v) is not None) >= 0.9:
        return "date"
    texts = [_text(v) for v in vals]
    if sum(bool(_POSTAL.match(t)) for t in texts) / len(texts) >= 0.9:
        return "postal"
    if sum(bool(_WILAYAH.match(t)) and len(re.sub(r"\D", "", t)) >= 4 for t in texts) / len(texts) >= 0.9:
        return "wilayah"
    if sum(bool(_ICD.match(t.upper())) for t in texts) / len(texts) >= 0.8:
        return "icd10"
    if share(lambda v: (a := _number(v)) is not None and 0 <= a <= 130) >= 0.9 and len(set(texts)) > 3:
        return "age"
    if len({t.lower() for t in texts}) <= 3:
        return "sex"
    return "category"


def _codes(values: list) -> list[int]:
    """A column as small ints: equal values, equal ints; empty is a value of its own."""
    ids: dict = {}
    return [ids.setdefault(None if _empty(v) else _text(v), len(ids)) for v in values]


def _numpy():
    try:
        import numpy
        return numpy
    except ImportError:
        return None


def _group(code_cols: list, owners=None):
    """Rows grouped by the combination of encoded columns: (group id per row, group size per row).

    The size counts rows, or people when `owners` (encoded) says whose rows are whose. Uses numpy
    when it is installed (it is with the files extra), plain Python otherwise.
    """
    n = len(code_cols[0]) if code_cols else 0
    np = _numpy()
    if np is not None and n:
        key = np.zeros(n, dtype=np.int64)
        span = 1
        for col in code_cols:
            a = np.asarray(col, dtype=np.int64)
            card = int(a.max()) + 1
            if span * card >= 2 ** 62:
                key = np.unique(key, return_inverse=True)[1].astype(np.int64).ravel()
                span = int(key.max()) + 1
            key = key * card + a
            span *= card
        key = np.unique(key, return_inverse=True)[1].astype(np.int64).ravel()
        if owners is None:
            sizes = np.bincount(key)[key]
        else:
            o = np.asarray(owners, dtype=np.int64)
            width = int(o.max()) + 1
            people = np.bincount(np.unique(key * width + o) // width, minlength=int(key.max()) + 1)
            sizes = people[key]
        return key, sizes
    ids: dict = {}
    key = [ids.setdefault(t, len(ids)) for t in zip(*code_cols)] if code_cols else []
    if owners is None:
        counts = Counter(key)
    else:
        counts = Counter(g for g, _ in set(zip(key, owners)))
    return key, [counts[g] for g in key]


def _below(sizes, k: int) -> int:
    return int((sizes < k).sum()) if hasattr(sizes, "dtype") else sum(1 for s in sizes if s < k)


def _failing(key, sizes, k: int, sensitive: list = (), l: int = 1):
    """Per row: is its group smaller than k, or with fewer than l different values in a sensitive column?"""
    np = _numpy()
    if np is not None and hasattr(key, "dtype"):
        bad = sizes < k
        if l > 1 and len(key):
            groups = int(key.max()) + 1
            for s in sensitive:
                s = np.asarray(s, dtype=np.int64)
                width = int(s.max()) + 1
                distinct = np.bincount(np.unique(key * width + s) // width, minlength=groups)
                bad |= distinct[key] < l
        return bad
    key, sizes = _list(key), _list(sizes)
    bad = [x < k for x in sizes]
    if l > 1:
        for s in sensitive:
            seen = defaultdict(set)
            for g, v in zip(key, s):
                seen[g].add(v)
            bad = [b or len(seen[g]) < l for b, g in zip(bad, key)]
    return bad


def _count(bad) -> int:
    return int(bad.sum()) if hasattr(bad, "dtype") else sum(bad)


def _list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _entropy(codes) -> float:
    n = len(codes)
    return -sum(c / n * math.log(c / n) for c in Counter(_list(codes)).values()) if n else 0.0


def measure(columns: dict[str, list], quasi: list[str], sensitive: list[str] = (), *, k: int = 5,
            person: str | None = None, skip: list[bool] | None = None) -> dict:
    """How identifying the quasi-identifier columns are, together. Counts only.

    person    a column that says which rows are one person (a tokenized member number): groups then
              count people, not rows
    skip      rows to leave out (suppressed rows, whose quasi-identifiers are all "*")
    """
    n_all = max((len(v) for v in columns.values()), default=0)
    keep = [i for i in range(n_all) if not (skip and skip[i])]
    pick = (lambda vs: [vs[i] for i in keep]) if skip else list
    rows = len(keep)
    out = {"rows": rows, "quasi": list(quasi), "target_k": k}
    if skip:
        out["suppressed_rows"] = n_all - rows
    if not quasi or not rows:
        return out | {"groups": 0, "k": None, "unique_rows": 0, "rows_below_k": 0}
    codes = {c: _codes(pick(columns[c])) for c in quasi}
    owners = _codes(pick(columns[person])) if person else None
    key, sizes = _group([codes[c] for c in quasi], owners)
    key, sizes = _list(key), _list(sizes)
    unique = sum(1 for s in sizes if s == 1)
    below = sum(1 for s in sizes if s < k)
    out |= {
        "counted": "people" if person else "rows",
        "groups": len(set(key)),
        "k": min(sizes),
        "unique_rows": unique, "unique_share": round(unique / rows, 4),
        "rows_below_k": below, "below_k_share": round(below / rows, 4),
        "average_risk": round(sum(1 / s for s in sizes) / rows, 4),
        "highest_risk": round(1 / min(sizes), 4),
    }
    # Which column makes rows unique: the unique share without it.
    if len(quasi) > 1:
        out["unique_share_without"] = {}
        for c in quasi:
            _, s = _group([codes[q] for q in quasi if q != c], owners)
            out["unique_share_without"][c] = round(sum(1 for x in _list(s) if x == 1) / rows, 4)
    # l-diversity: the fewest different sensitive values in any group of at least k.
    for c in sensitive:
        values = defaultdict(set)
        for g, v, s in zip(key, _codes(pick(columns[c])), sizes):
            if s >= k:
                values[g].add(v)
        if values:
            one = {g for g, v in values.items() if len(v) == 1}
            out.setdefault("sensitive", {})[c] = {
                "l": min(len(v) for v in values.values()), "groups_one_value": len(one),
                "rows_in_groups_one_value": sum(1 for g, s in zip(key, sizes) if s >= k and g in one),
            }
    return out


@dataclass
class Plan:
    levels: dict[str, str]                 # column -> level
    hierarchies: dict[str, str]            # column -> hierarchy
    k: int
    rows_suppressed: int
    loss: float                            # information lost: mean entropy lost per column (0..1) + share suppressed
    reached: bool                          # False: even the most general levels don't reach k
    tried: int = 0
    l: int = 1                             # and at least this many different sensitive values per group
    lost: dict = field(default_factory=dict)                     # column -> share of its information lost
    candidates: list = field(default_factory=list, repr=False)   # other minimal plans: {loss, levels, lost, rows_suppressed}


def suggest(columns: dict[str, list], hierarchies: dict[str, str], *, k: int = 5, max_suppress: float = 0.01,
            person: str | None = None, keep_value: tuple[str, ...] = (), limit: dict[str, str] | None = None,
            max_nodes: int = 3000, sensitive: dict[str, list] | None = None, l: int = 1) -> Plan:
    """The least generalisation that reaches k, suppressing at most `max_suppress` of the rows.

    Levels are searched from the least general up. A combination that reaches k makes every more general
    one reach it too (monotone), so those are skipped. Among the minimal ones, the one that loses the
    least information wins: per column, the share of its entropy lost (birth date -> year keeps more
    than birth date -> *), plus the share of rows suppressed.
    sensitive   column -> values, for l: groups with fewer than l different values in any of them are
                suppressed too (they count against the budget like groups below k)
    keep_value  columns that must stay as they are
    limit       column -> the most general level allowed ({"birth_date": "5 years"})
    max_nodes   above this many combinations (many quasi-identifiers), a greedy search instead:
                step up the column that removes the most risk per information lost, until k is
                reached, then step back down wherever k still holds. Not always the least loss, but
                minimal: no column can be made more precise without breaking k
    """
    quasi = list(hierarchies)
    n = max((len(v) for v in columns.values()), default=0)
    budget = int(max_suppress * n)
    np = _numpy()
    owners = _codes(columns[person]) if person else None
    enc: dict[str, list] = {}
    lost: dict[str, list[float]] = {}
    for c in quasi:
        levels = [_codes(generalize(columns[c], hierarchies[c], lvl)) for lvl in LEVELS[hierarchies[c]]]
        h = [_entropy(e) for e in levels]
        lost[c] = [round(1 - x / h[0], 4) if h[0] > 0 else 0.0 for x in h]
        enc[c] = [np.asarray(e, dtype=np.int64) for e in levels] if np is not None else levels
    if np is not None and owners is not None:
        owners = np.asarray(owners, dtype=np.int64)
    sens = [_codes(v) for v in (sensitive or {}).values()] if l > 1 else []
    if np is not None:
        sens = [np.asarray(s, dtype=np.int64) for s in sens]
    tops = [len(LEVELS[hierarchies[c]]) - 1 for c in quasi]
    most = dict.fromkeys(keep_value, "value") | (limit or {})
    for c, lvl in most.items():
        if c not in hierarchies or lvl not in LEVELS[hierarchies[c]]:
            raise ValueError(f"limit {c}={lvl}: not a quasi-identifier, or not a level of its hierarchy")
    caps = [LEVELS[hierarchies[c]].index(most[c]) if c in most else t for c, t in zip(quasi, tops)]
    ranges = [range(cap + 1) for cap in caps]
    found: list[tuple] = []
    results = []
    tried = 0
    seen: dict[tuple, int] = {}

    def below(node) -> int:
        node = tuple(node)
        if node not in seen:
            key, sizes = _group([enc[c][lvl] for c, lvl in zip(quasi, node)], owners)
            seen[node] = _count(_failing(key, sizes, k, sens, l))
        return seen[node]

    def loss(node, b) -> float:
        return round(sum(lost[c][lvl] for c, lvl in zip(quasi, node)) / len(quasi) + b / max(n, 1), 4)

    name = lambda nd: {c: LEVELS[hierarchies[c]][lvl] for c, lvl in zip(quasi, nd)}
    share = lambda nd: {c: lost[c][lvl] for c, lvl in zip(quasi, nd)}
    if math.prod(len(r) for r in ranges) > max_nodes:
        plan = _greedy(quasi, caps, lost, below, loss, budget, k, hierarchies, name, share)
        plan.l = l
        return plan
    nodes = sorted(itertools.product(*ranges), key=lambda x: (sum(x), x))
    for node in nodes:
        if any(all(a >= b for a, b in zip(node, f)) for f in found):
            continue                           # more general than a plan that already works
        tried += 1
        b = below(node)
        if b <= budget:
            found.append(node)
            results.append((loss(node, b), b, node))
    if not results:
        node = tuple(caps)
        b = below(node)
        return Plan(name(node), dict(hierarchies), k, b, loss(node, b), False, tried, l, share(node))
    results.sort()
    best, b, node = results[0]
    return Plan(name(node), dict(hierarchies), k, b, best, True, tried, l, share(node),
                [{"loss": l_, "levels": name(nd), "lost": share(nd), "rows_suppressed": b_}
                 for l_, b_, nd in results[1:6]])


def _greedy(quasi, caps, lost, below, loss, budget, k, hierarchies, name, share) -> Plan:
    node = [0] * len(quasi)
    b = below(node)
    tried = 1
    while b > budget:
        best = None
        for i, c in enumerate(quasi):
            if node[i] >= caps[i]:
                continue
            cand = node.copy()
            cand[i] += 1
            bc = below(cand)
            tried += 1
            cost = lost[c][cand[i]] - lost[c][node[i]] + 1e-6
            score = ((b - bc) / cost, -cost)
            if best is None or score > best[0]:
                best = (score, cand, bc)
        if best is None:                 # every column fully generalised
            break
        _, node, b = best
    reached = b <= budget
    if reached:
        # Back down: make each column as precise as k allows, most information-costly first.
        for i in sorted(range(len(quasi)), key=lambda i: -lost[quasi[i]][node[i]]):
            while node[i] > 0:
                cand = node.copy()
                cand[i] -= 1
                bc = below(cand)
                tried += 1
                if bc > budget:
                    break
                node, b = cand, bc
    return Plan(name(node), dict(hierarchies), k, b, loss(node, b), reached, tried, 1, share(node))


def apply(columns: dict[str, list], levels: dict[str, str], hierarchies: dict[str, str], *, k: int = 5,
          person: str | None = None, quasi: list[str] | None = None, sensitive: list[str] = (),
          l: int = 1) -> tuple[dict[str, list], list[bool]]:
    """Generalise the quasi-identifiers, then suppress ("*") those of rows still below k (or below l
    different values in a sensitive column).

    quasi   all the quasi-identifier columns, including those kept as they are (default: those in levels)
    Returns the new columns (others unchanged) and which rows were suppressed.
    """
    out = dict(columns)
    for c, lvl in levels.items():
        out[c] = generalize(columns[c], hierarchies[c], lvl)
    quasi = list(quasi) if quasi is not None else list(levels)
    n = max((len(v) for v in columns.values()), default=0)
    if not quasi:
        return out, [False] * n
    key, sizes = _group([_codes(out[c]) for c in quasi], _codes(columns[person]) if person else None)
    suppressed = [bool(x) for x in _list(_failing(key, sizes, k, [_codes(columns[c]) for c in sensitive], l))]
    if any(suppressed):
        for c in quasi:
            out[c] = [SUPPRESSED if s and not _empty(v) else v for v, s in zip(out[c], suppressed)]
    return out, suppressed
