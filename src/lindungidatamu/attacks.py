"""The attack gallery: known attacks on an aggregate question service, run with the gate off and on.

    python -m lindungidatamu.attacks

Each attack is something an AI model (or a person behind it) could try through `aggregate`: subtract
two answers, ask the same thing many ways, narrow a filter down to one person, read a sum one person
dominates, ask for an identifier column. "Gate off" is the same question language answered exactly,
the way a plain SQL or pandas tool would; "gate on" is `Analysis`. Every attack runs on a made-up
claims table, so the gallery can be published and rerun by anyone.

A row that says "not blocked" is a limit of the gate, not a bug in the gallery: it is there so the
gallery shows what the gate does not do as well as what it does.
"""

import collections
import datetime as dt
import random
from dataclasses import dataclass

from .analysis import Analysis, Answer, QueryError, _pandas


def claims(n_members: int = 1500, seed: int = 7) -> dict[str, list]:
    """A made-up claims table with realistic skew: crowded and sparse areas, common and rare diagnoses,
    members with one to six claims."""
    rng = random.Random(seed)
    kelurahan = [f"{p}.{k:02d}.{c:02d}.{1000 + v}" for p in (31, 32, 35) for k in (1, 2, 71) for c in (1, 2, 3)
                 for v in range(4)]
    area_w = [rng.paretovariate(1.2) for _ in kelurahan]
    common = ["J18.9", "E11.9", "I10", "K29.7", "A09", "J06.9", "N39.0", "M54.5"]
    rare = ["G20", "C50.9", "Q90.9", "B20", "F20.0", "D57.1"]
    cols = {c: [] for c in ("claim_no", "member_no", "birth_date", "sex", "kelurahan", "admitted", "diagnosis",
                            "amount_paid")}
    claim = 0
    for m in range(n_members):
        member = f"{100000000 + m * 37:013d}"
        born = dt.date(1945, 1, 1) + dt.timedelta(days=rng.randrange(0, 65 * 365))
        sex = rng.choice("LP")
        area = rng.choices(kelurahan, area_w)[0]
        for _ in range(rng.choice((1, 1, 1, 2, 2, 3, 6))):
            claim += 1
            cols["claim_no"].append(f"CLM-{claim:07d}")
            cols["member_no"].append(member)
            cols["birth_date"].append(born)
            cols["sex"].append(sex)
            cols["kelurahan"].append(area)
            cols["admitted"].append(dt.date(2024, 1, 1) + dt.timedelta(days=rng.randrange(366)))
            cols["diagnosis"].append(rng.choice(rare) if rng.random() < 0.03 else rng.choice(common))
            cols["amount_paid"].append(rng.randrange(100, 20000) * 1000)
    return cols


class Open:
    """The same questions answered exactly, with no gate: what a plain SQL or pandas tool would give
    an AI model. Every column can be used, the person column included."""

    def __init__(self, data: dict, person: str = "member_no"):
        pd, _ = _pandas()
        self.df, self.person = pd.DataFrame(data), person

    def aggregate(self, group_by=(), measures=("people",), where=()) -> Answer:
        group_by = [group_by] if isinstance(group_by, str) else list(group_by)
        measures = [measures] if isinstance(measures, (str, dict)) else list(measures)
        where = [where] if isinstance(where, dict) else list(where)
        df = self.df
        for cond in where:
            (op, value), = [(k, v) for k, v in cond.items() if k != "column"]
            s = df[cond["column"]]
            text = s.astype(str)
            hit = {"=": lambda: text == str(value), "!=": lambda: text != str(value),
                   "in": lambda: text.isin([str(v) for v in value]),
                   "not_in": lambda: ~text.isin([str(v) for v in value]),
                   ">=": lambda: s >= value, "<=": lambda: s <= value, ">": lambda: s > value, "<": lambda: s < value,
                   "between": lambda: (text >= str(value[0])) & (text <= str(value[1])) if s.dtype == object
                   else (s >= value[0]) & (s <= value[1])}[op]()
            df = df[hit]
        groups = list(df.groupby(group_by, sort=True)) if group_by else [((), df)]
        names = [m if isinstance(m, str) else f"{next(iter(m))}({next(iter(m.values()))})" for m in measures]
        rows = []
        for key, g in groups:
            row = [str(v) for v in (key if isinstance(key, tuple) else (key,))]
            for m in measures:
                if m == "people":
                    row.append(int(g[self.person].nunique()))
                elif m == "rows":
                    row.append(len(g))
                else:
                    (op, col), = m.items()
                    row.append(float(g[col].sum() if op == "sum" else g[col].mean()))
            rows.append(row)
        return Answer(columns=list(group_by) + names, rows=rows)


@dataclass
class Result:
    attack: str
    how: str
    got: dict           # system -> what the attacker got
    wins: dict          # system -> whether the attack worked
    note: str = ""


SYSTEMS = ("Gate off", "Gate on", "DP mode")
DP_QUESTION = 0.1
DP_UNLIMITED = 1e6          # the gallery measures the noise: no total budget, so every attack runs to the end


def _systems(t):
    """Fresh systems for one attack: no gate, the gate, and DP mode at epsilon 0.1 a question."""
    return {"Gate off": Open(t), "Gate on": Analysis(t, k=5, secret="gallery"),
            "DP mode": Analysis(t, dp=DP_UNLIMITED, dp_question=DP_QUESTION)}


def _dp(system) -> bool:
    return getattr(system, "budget", None) is not None


def _ask(system, **q):
    try:
        return system.aggregate(**q).rows
    except QueryError:
        return None


def _first(rows, i=0):
    return rows[0][i] if rows and rows[0][i] is not None else None


def _sum(system, col, upper):
    """{"sum": col}, with the bound DP mode asks for: the attacker picks it to suit the attack."""
    return {"sum": col, "upper": upper} if _dp(system) else {"sum": col}


def _unique_amounts(t):
    """(kelurahan, amount, member, diagnosis) for claims whose amount appears once in the table: a
    filter `amount_paid != amount` then drops exactly that one claim."""
    counts = collections.Counter(t["amount_paid"])
    seen, out = set(), []
    for kel, amt, m, d in zip(t["kelurahan"], t["amount_paid"], t["member_no"], t["diagnosis"]):
        if counts[amt] == 1 and kel not in seen:
            seen.add(kel)
            out.append((kel, amt, m, d))
    return out


def subtraction(t) -> Result:
    """Sum over a kelurahan, minus the same sum without one claim: the difference is that claim."""
    targets = _unique_amounts(t)
    n = len(targets)
    mean = sum(t["amount_paid"]) / len(t["amount_paid"])
    chance = sum(abs(mean - amt) <= 0.05 * amt for _, amt, _, _ in targets)   # guessing the average claim
    got, wins = {}, {}
    for name, system in _systems(t).items():
        hits = 0
        for kel, amt, _, _ in targets:
            w = [{"column": "kelurahan", "=": kel}]
            m = [_sum(system, "amount_paid", 20_000_000)]
            a = _first(_ask(system, measures=m, where=w))
            b = _first(_ask(system, measures=m, where=w + [{"column": "amount_paid", "!=": amt}]))
            hits += a is not None and b is not None and abs((a - b) - amt) <= 0.05 * amt
        got[name] = f"within 5% {hits}/{n}" + ("" if name == "Gate off" else f" (guessing the average: {chance}/{n})")
        wins[name] = hits > (0.9 * n if name == "Gate off" else chance)
    return Result("Subtraction", "sum(kelurahan) − sum(kelurahan without one claim), to get that claim", got, wins,
                  "Scored against guessing the average claim: no better means nothing was learned.")


def diagnosis_guess(t) -> Result:
    """Does the target's claim carry J18.9? Count J18.9 claims in their kelurahan with and without it."""
    targets = _unique_amounts(t)
    n, base = len(targets), sum(d != "J18.9" for *_, d in targets)
    got, wins = {}, {}
    for name, system in _systems(t).items():
        right = 0
        for kel, amt, _, d in targets:
            w = [{"column": "kelurahan", "=": kel}, {"column": "diagnosis", "=": "J18.9"}]
            a = _first(_ask(system, measures="rows", where=w)) or 0
            b = _first(_ask(system, measures="rows", where=w + [{"column": "amount_paid", "!=": amt}])) or 0
            right += (a - b >= 1) == (d == "J18.9")
        got[name] = f"right {right}/{n}" + ("" if name == "Gate off" else f" (always \"no\": {base}/{n})")
        wins[name] = right == n if name == "Gate off" else right > base
    return Result("Diagnosis guess", "count of J18.9 claims with and without the target's claim", got, wins,
                  "Scored against always guessing \"no\": no better means nothing was learned.")


def averaging(t) -> Result:
    """Ask the same count many ways and average the answers, so random noise cancels out."""
    pd, _ = _pandas()
    df = pd.DataFrame(t)
    common = ["J18.9", "E11.9", "I10", "K29.7"]
    others = sorted(set(df["diagnosis"]))
    groups = [(k, d) for k in sorted(set(df["kelurahan"])) for d in common]
    ways = lambda d: [[{"column": "diagnosis", "=": d}], [{"column": "diagnosis", "in": [d]}],
                      [{"column": "diagnosis", "in": [d, "Z99.9"]}],
                      [{"column": "diagnosis", "not_in": [o for o in others if o != d]}],
                      [{"column": "diagnosis", "=": d}, {"column": "amount_paid", ">=": 0}],
                      [{"column": "diagnosis", "=": d}, {"column": "amount_paid", "!=": -1}],
                      [{"column": "diagnosis", "in": [d]}, {"column": "admitted", "between": ["2024-01-01", "2024-12-31"]}],
                      [{"column": "diagnosis", "=": d}, {"column": "sex", "in": ["L", "P"]}]]
    got, wins = {}, {}
    for name, system in _systems(t).items():
        exact = once = asked = questions = 0
        for kel, d in groups:
            true = int(df[(df["kelurahan"] == kel) & (df["diagnosis"] == d)]["member_no"].nunique())
            if true < 5:
                continue
            questions += 8
            answers = [_first(_ask(system, measures="people", where=[{"column": "kelurahan", "=": kel}] + w))
                       for w in ways(d)]
            answers = [x for x in answers if x is not None]
            if not answers:
                continue
            asked += 1
            once += answers[0] == true
            exact += round(sum(answers) / len(answers)) == true
        got[name] = f"exact {exact}/{asked} after averaging, {once}/{asked} from one answer"
        wins[name] = exact > once if name != "Gate off" else exact == asked
    return Result("Averaging", "the same count asked 8 ways, answers averaged", got, wins,
                  "Averaging only helps if it beats a single answer. The gate's noise is fixed by who is in the "
                  "group, so every way gets the same answer. In DP mode every way gets new noise, so averaging "
                  "does help, and the 8 answers cost epsilon 0.8 together: that is what the total budget is for. "
                  "With a budget of 1 the attacker averages one count, and has nothing left.")


def narrowing(t) -> Result:
    """Filter on what you know about someone (birth date, sex, kelurahan) until only they are left,
    then read their diagnoses."""
    pd, _ = _pandas()
    df = pd.DataFrame(t)
    df["birth_date"] = df["birth_date"].astype(str)
    key = ["birth_date", "sex", "kelurahan"]
    alone = df.groupby(key)["member_no"].nunique()
    people = df.drop_duplicates("member_no")
    targets = [r for r in people.itertuples() if alone[(r.birth_date, r.sex, r.kelurahan)] == 1][:100]
    diagnoses = sorted(set(df["diagnosis"]))
    n = len(targets)
    got, wins = {}, {}
    for name, system in _systems(t).items():
        if name == "Gate off":
            system = Open(t | {"birth_date": [str(b) for b in t["birth_date"]]})
        hits = 0
        for r in targets:
            w = [{"column": c, "=": getattr(r, c)} for c in key]
            group = {"column": "diagnosis", "values": diagnoses} if _dp(system) else "diagnosis"
            rows = _ask(system, group_by=[group], measures="rows", where=w)
            truth = set(df[df["member_no"] == r.member_no]["diagnosis"])
            hits += bool(rows) and {row[0] for row in rows if row[1] > 0} == truth
        got[name] = f"their diagnoses read {hits}/{n}"
        wins[name] = hits == n if name == "Gate off" else hits > 0
    return Result("Narrowing", "filter on birth date + sex + kelurahan until one person is left", got, wins)


def dominance(t) -> Result:
    """One very large claim in a kelurahan: the kelurahan's total is mostly that one claim, so
    total − (claims − 1) × the usual claim gives it."""
    rng = random.Random(3)
    t = {c: list(v) for c, v in t.items()}
    size = collections.Counter(t["kelurahan"])
    kels = sorted(size, key=lambda k: (size[k], k))[:20]          # the quiet kelurahan: where one claim stands out
    big = {}
    for i, kel in enumerate(kels):
        amt = rng.randrange(1_000, 5_000) * 1_000_000
        big[kel] = amt
        for c, v in (("claim_no", f"CLM-BIG{i}"), ("member_no", f"BIG{i}"), ("birth_date", dt.date(1960, 1, 1)),
                     ("sex", "L"), ("kelurahan", kel), ("admitted", dt.date(2024, 6, 1)), ("diagnosis", "I21.9"),
                     ("amount_paid", amt)):
            t[c].append(v)
    n = len(big)
    got, wins = {}, {}
    for name, system in _systems(t).items():
        mean = {"mean": "amount_paid", "upper": 20_000_000} if _dp(system) else {"mean": "amount_paid"}
        usual = _first(_ask(system, measures=[mean]))
        hits = 0
        for kel, amt in big.items():
            rows = _ask(system, measures=["rows", _sum(system, "amount_paid", 5_000_000_000)],
                        where={"column": "kelurahan", "=": kel})
            total = _first(rows, 1)
            if total is not None and usual is not None:
                hits += abs(total - (rows[0][0] - 1) * usual - amt) <= 0.1 * amt
        got[name] = f"big claim within 10% {hits}/{n}"
        wins[name] = hits > (0.9 * n if name == "Gate off" else 0)
    return Result("Dominance", "a quiet kelurahan's total when one claim of Rp 1–5 billion is most of it", got, wins)


def identifiers(t) -> Result:
    """Group by the member number or the claim number: one row per person or per claim."""
    got, wins = {}, {}
    for name, system in _systems(t).items():
        answered = sum(bool(_ask(system, group_by=col, measures="rows")) for col in ("member_no", "claim_no"))
        got[name] = "every person's rows" if answered == 2 else f"refused, {answered}/2 answered"
        wins[name] = answered > 0
    return Result("Identifier columns", "group by member_no or claim_no", got, wins)


def everyone_has_it(t) -> Result:
    """Kelurahan where every member has the same diagnosis (say around a dialysis unit): the count of
    people there about equals the count with N18.5, so knowing someone lives there tells their
    diagnosis. Ten such kelurahan and ten where no one has it, 12 people each: the attacker says which
    is which. The gate does not block this: it is true of the group, not of one person, and every count
    service shows it. In DP mode a group this small is lost in the noise."""
    t = {c: list(v) for c, v in t.items()}
    areas = {f"36.01.01.{1001 + i}": i < 10 for i in range(20)}
    for area, all_ill in areas.items():
        for i in range(12):
            for c, v in (("claim_no", f"CLM-{area}-{i}"), ("member_no", f"{area}-{i}"),
                         ("birth_date", dt.date(1950 + i, 2, 1)), ("sex", "LP"[i % 2]), ("kelurahan", area),
                         ("admitted", dt.date(2024, 3, 1 + i)), ("diagnosis", "N18.5" if all_ill else "J06.9"),
                         ("amount_paid", 900_000 + i * 1000)):
                t[c].append(v)
    got, wins = {}, {}
    for name, system in _systems(t).items():
        right = 0
        for area, all_ill in areas.items():
            everyone = _first(_ask(system, measures="people", where={"column": "kelurahan", "=": area}))
            ill = _first(_ask(system, measures="people", where=[{"column": "kelurahan", "=": area},
                                                                {"column": "diagnosis", "=": "N18.5"}]))
            says = everyone is not None and ill is not None and everyone >= 5 and ill >= 0.9 * everyone
            right += says == all_ill
        wins[name] = right >= 18
        got[name] = f"right {right}/20 (chance: 10/20)" + (" — not blocked" if wins[name] and name == "Gate on" else "")
    return Result("Everyone has it", "which kelurahan all share N18.5: people there vs people there with it", got,
                  wins, "The gate doesn't block it: a fact about the whole group. `risk --l 2` finds such groups "
                        "before release; DP mode hides groups this small in its noise.")


ATTACKS = (subtraction, diagnosis_guess, averaging, narrowing, dominance, identifiers, everyone_has_it)


def run(table: dict | None = None) -> list[Result]:
    table = table or claims()
    return [attack(table) for attack in ATTACKS]


def report(results: list[Result]) -> str:
    lines = ["| Attack | What it tries | " + " | ".join(SYSTEMS) + " |", "|---|---|" + "---|" * len(SYSTEMS)]
    for r in results:
        cells = [r.got[s] if s == "Gate off" else ("✗ " if r.wins[s] else "✓ ") + r.got[s] for s in SYSTEMS]
        lines.append(f"| {r.attack} | {r.how} | " + " | ".join(cells) + " |")
    notes = [f"- {r.attack}: {r.note}" for r in results if r.note]
    return "\n".join(lines + ["", f"DP mode: epsilon {DP_QUESTION:g} a question, with no total budget so every "
                                  "attack runs to the end (a budget of 1 would stop each one after 10 questions); "
                                  "the attacker supplies sum bounds and lists the groups it wants."] + notes)


if __name__ == "__main__":
    print(report(run()))
