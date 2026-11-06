"""Differential privacy mode for safe analysis: a provable limit on what answers say about one person.

    a = Analysis(df, dp=1.0)                    # a total budget of epsilon = 1 for this table
    a.aggregate(group_by="sex", measures=["people", {"mean": "amount_paid", "upper": 50_000_000}])
    lindungidatamu serve claims.parquet --dp 1.0

Whatever is asked, and however questions are combined, everything released about the table changes
by at most a factor of e^epsilon (plus a chance delta) whether or not any one person is in it. Each
answer spends part of the budget; when it is gone, questions are refused. The same question asked
again gets the same answer and spends nothing.

How an answer is made, per question with budget eps:
    bounding     each person counts in at most `groups_per_person` groups (G, default 1) and with
                 at most `rows_per_person` rows in each (R, default 3); which ones is random. Sums
                 and means need bounds: values are clipped to [lower, upper] (lower defaults to 0).
    noise        discrete Laplace, drawn with integer arithmetic (no floating-point leaks), scaled
                 to what one person can change: people G, rows G*R, sums G*R*max(|lower|, |upper|).
                 eps is split evenly over the people count and each measure (a mean is two: its sum
                 and its count).
    groups       a group appears only if its noisy people count clears a threshold set by delta,
                 so a group of one person shows up with probability at most delta. Unless the
                 question lists them: {"column": "sex", "values": ["L", "P"]} on every group_by
                 column makes the groups public, so every listed group is shown (empty ones too),
                 with no threshold and no delta. Without group_by the one total always appears.
    describe     lists columns and how to group them, never values: values come from the data,
                 so they cost budget like any answer (ask with group_by).

The budget, and the answers already given, are kept in the state folder (budget.json): a new folder
is a new budget, so the folder is what the data custodian guards. Mixing modes on one folder is
refused: answers from the other mode are not counted in this budget.
"""

import json
import math
import secrets
from fractions import Fraction
from pathlib import Path

GROUPS_PER_PERSON = 1
ROWS_PER_PERSON = 3
UNITS = 1000                      # a clipped value is rounded to 1/UNITS of its bound before summing


# ---- the sampler (Canonne, Kamath and Steinke 2020, exact) ------------------------------------

def _bernoulli(p: Fraction) -> bool:
    return secrets.randbelow(p.denominator) < p.numerator


def _bernoulli_exp1(gamma: Fraction) -> bool:
    """True with probability exp(-gamma), 0 <= gamma <= 1."""
    k = 1
    while _bernoulli(gamma / k):
        k += 1
    return k % 2 == 1


def _bernoulli_exp(gamma: Fraction) -> bool:
    """True with probability exp(-gamma), gamma >= 0."""
    while gamma > 1:
        if not _bernoulli_exp1(Fraction(1)):
            return False
        gamma -= 1
    return _bernoulli_exp1(gamma)


def discrete_laplace(scale: Fraction) -> int:
    """An integer y with probability proportional to exp(-|y| / scale)."""
    scale = Fraction(scale)
    t, s = scale.numerator, scale.denominator
    while True:
        u = secrets.randbelow(t)
        if not _bernoulli_exp(Fraction(u, t)):
            continue
        v = 0
        while _bernoulli_exp(Fraction(1)):
            v += 1
        y = (u + t * v) // s
        negative = secrets.randbelow(2)
        if negative and y == 0:
            continue
        return -y if negative else y


def _frac(x: float) -> Fraction:
    return Fraction(x).limit_denominator(10 ** 6)


def margin(scale: Fraction, p: float = 0.05) -> float:
    """Half-width that the noise stays within with probability 1 - p."""
    return float(scale) * math.log(1 / p)


# ---- the budget -------------------------------------------------------------------------------

class Budget:
    """Epsilon and delta spent so far, and the answers already given, for one table."""

    def __init__(self, epsilon: float, per_question: float | None = None, delta: float = 1e-6,
                 path: Path | None = None):
        if not epsilon > 0:
            raise ValueError("the privacy budget (epsilon) must be more than 0")
        self.epsilon, self.delta = float(epsilon), float(delta)
        self.per_question = float(per_question) if per_question else self.epsilon / 10
        self.spent, self.spent_delta, self.questions = 0.0, 0.0, 0
        self.cache: dict[str, dict] = {}
        self.path = path
        if path is not None and path.exists():
            kept = json.loads(path.read_text())
            if abs(kept["epsilon"] - self.epsilon) > 1e-12 or abs(kept["delta"] - self.delta) > 1e-18:
                raise ValueError(f"{path.parent} has a budget of epsilon {kept['epsilon']:g}, delta {kept['delta']:g}: "
                                 "ask with those, or use a new state folder")
            self.spent, self.spent_delta = kept["spent"], kept["spent_delta"]
            self.questions, self.cache = kept["questions"], kept["cache"]

    @property
    def left(self) -> float:
        return max(self.epsilon - self.spent, 0.0)

    def charge(self, epsilon: float, delta: float):
        from .analysis import QueryError
        if epsilon > self.left + 1e-12:
            raise QueryError(f"this question needs epsilon {epsilon:g} and the budget has {self.left:g} left "
                             f"of {self.epsilon:g}" + (": it is used up" if self.left < 1e-12 else
                                                       f'; ask with "epsilon": {self.left:g} or less'))
        self.spent += epsilon
        self.spent_delta += delta
        self.questions += 1
        self.save()

    def save(self):
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.touch(mode=0o600)
        tmp.write_text(json.dumps({"epsilon": self.epsilon, "delta": self.delta, "spent": self.spent,
                                   "spent_delta": self.spent_delta, "questions": self.questions,
                                   "cache": self.cache}, ensure_ascii=False))
        tmp.replace(self.path)

    def to_dict(self) -> dict:
        return {"mode": "differential privacy", "epsilon": self.epsilon, "spent": round(self.spent, 6),
                "left": round(self.left, 6), "per_question": self.per_question, "delta": self.delta,
                "questions": self.questions}


# ---- answering ---------------------------------------------------------------------------------

def _bounds(m: dict, col: str):
    from .analysis import QueryError
    if "upper" not in m:
        raise QueryError(f'in DP mode a sum or mean needs bounds: {{"sum": "{col}", "upper": ...}} (and "lower", '
                         "default 0). Values outside are clipped; pick bounds from what you know of the column, "
                         "not by asking: a tight upper bound means less noise")
    try:
        lo, hi = float(m.get("lower", 0)), float(m["upper"])
    except (TypeError, ValueError):
        raise QueryError("lower and upper are numbers") from None
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise QueryError("lower must be below upper")
    return lo, hi


def aggregate(a, group_by, measures, where, epsilon=None, groups_per_person=None, rows_per_person=None):
    """`Analysis.aggregate` in DP mode (see the module docstring)."""
    import datetime as dt
    import time

    from .analysis import Answer, QueryError, _pandas, _round
    pd, np = _pandas()
    budget = a.budget
    started = time.time()
    G = GROUPS_PER_PERSON if groups_per_person is None else groups_per_person
    R = ROWS_PER_PERSON if rows_per_person is None else rows_per_person
    if not (isinstance(G, int) and 1 <= G <= 50 and isinstance(R, int) and 1 <= R <= 50):
        raise QueryError("groups_per_person and rows_per_person are whole numbers from 1 to 50")
    if not group_by:
        G = 1                                           # one total: everyone is in it once
    eps = budget.per_question if epsilon is None else epsilon
    if not isinstance(eps, (int, float)) or not 0 < eps <= budget.epsilon:
        raise QueryError(f"epsilon is a number above 0 and at most the budget, {budget.epsilon:g}")
    query = {"group_by": group_by, "measures": measures, "where": where}
    key = json.dumps([query, eps, G, R], sort_keys=True, default=str)
    entry = {"at": dt.datetime.now().isoformat(timespec="seconds"), "query": query}
    if key in budget.cache:
        answer = Answer(**budget.cache[key])
        answer.notes = answer.notes + ["asked before: the same answer, no budget spent"]
        a.log.append(entry | {"cached": True})
        if a.state is not None:
            a._save(a.log[-1], history=False)
        return answer

    # Everything that can be refused is checked before anything is spent or read.
    mask = pd.Series(True, index=a.df.index)
    for cond in where:
        mask &= a._condition(cond)
    names, keys = [], []
    for g in group_by:
        name, series = a._group({k: v for k, v in g.items() if k != "values"} if isinstance(g, dict) else g)
        names.append(name)
        keys.append(series)
    declared = [g.get("values") if isinstance(g, dict) else None for g in group_by]
    public = bool(names) and all(v is not None for v in declared)
    if any(v is not None for v in declared) and not public:
        raise QueryError('list "values" for every group_by column, or for none')
    if public:
        if not all(isinstance(v, list) and v and all(isinstance(x, (str, int, float)) for x in v) for v in declared):
            raise QueryError('"values" is a list of the groups to show')
        combos = math.prod(len(v) for v in declared)
        if combos > a.max_groups:
            raise QueryError(f"{combos:,} listed groups: at most {a.max_groups} per answer")
    specs = []
    for m in measures:
        label, kind, col = a._measure(m)
        specs.append((label, kind, col, _bounds(m, col) if kind in ("sum", "mean") else None))
    selecting = bool(names) and not public
    shares = (1 if selecting or any(k == "people" for _, k, _, _ in specs) else 0) + \
        sum({"people": 0, "rows": 1, "sum": 1, "mean": 2}[k] for _, k, _, _ in specs)
    delta = budget.delta if selecting else 0.0
    budget.charge(eps, delta)
    part = _frac(eps / shares)

    # bound what one person contributes: G groups, R rows in each, chosen at random
    pos = np.flatnonzero(mask.to_numpy())
    if public:
        import itertools
        labels = list(itertools.product(*[[str(x) for x in v] for v in declared]))
        index = {lab: i for i, lab in enumerate(labels)}
        found = zip(*[k.to_numpy()[pos] for k in keys])
        gid = np.array([index.get(tuple("(missing)" if x is None else str(x) for x in t), -1) for t in found],
                       dtype=int)
        pos, gid = pos[gid >= 0], gid[gid >= 0]
    elif names:
        frame = pd.DataFrame({f"g{i}": k.to_numpy()[pos] for i, k in enumerate(keys)}).fillna("(missing)")
        gid, labels = pd.factorize(pd.MultiIndex.from_frame(frame)) if len(pos) else (np.zeros(0, int), [])
        labels = [tuple(str(v) for v in lab) for lab in labels]
    else:
        gid, labels = np.zeros(len(pos), dtype=int), [()]
    rng = np.random.default_rng(secrets.randbits(128))
    rows = pd.DataFrame({"pos": pos, "pid": a._pid[pos], "gid": gid, "r": rng.random(len(pos))})
    pairs = rows[["pid", "gid"]].drop_duplicates()
    pairs = pairs.assign(p=rng.random(len(pairs)))
    pairs = pairs[pairs.groupby("pid")["p"].rank(method="first") <= G]
    rows = rows.merge(pairs[["pid", "gid"]], on=["pid", "gid"])
    rows = rows[rows.groupby(["pid", "gid"])["r"].rank(method="first") <= R]
    by = rows.groupby("gid")
    people = by["pid"].nunique().reindex(range(len(labels)), fill_value=0).to_numpy()
    count = by.size().reindex(range(len(labels)), fill_value=0).to_numpy()

    noisy = lambda true, sens: int(true) + discrete_laplace(Fraction(sens) / part)
    keep = list(range(len(labels)))
    noisy_people = None
    if selecting or any(k == "people" for _, k, _, _ in specs):
        noisy_people = [noisy(n, G) for n in people]
    threshold = None
    if selecting:
        scale = Fraction(G) / part
        threshold = 1 + float(scale) * math.log(G / delta)
        keep = [i for i in keep if noisy_people[i] >= threshold]
        if len(keep) > a.max_groups:                    # post-processing: the largest groups
            keep = sorted(keep, key=lambda i: -noisy_people[i])[:a.max_groups]
        keep.sort(key=lambda i: labels[i])

    answer = Answer(columns=names + [label for label, _, _, _ in specs], rows=[])
    margins = {}
    for label, kind, col, bnd in specs:
        if kind == "people":
            margins[label] = margin(Fraction(G) / part)
        elif kind == "rows":
            margins[label] = margin(Fraction(G * R) / part)
        else:
            lo, hi = bnd
            unit = max(abs(lo), abs(hi)) / UNITS
            m = margin(Fraction(G * R * UNITS) / part) * unit
            margins[label] = f"±{_round(m):,}" if kind == "sum" else f"its sum ±{_round(m):,}, then divided by the count"
        margins[label] = margins[label] if isinstance(margins[label], str) else f"±{_round(margins[label]):,}"
    values = {}
    for label, kind, col, bnd in specs:
        if kind not in ("sum", "mean"):
            continue
        lo, hi = bnd
        unit = max(abs(lo), abs(hi)) / UNITS
        v = a.df[col].to_numpy()[rows["pos"].to_numpy()]
        present = ~np.isnan(v)
        units = np.where(present, np.rint(np.clip(np.nan_to_num(v), lo, hi) / unit), 0).astype("int64")
        s = pd.Series(units).groupby(rows["gid"].to_numpy()).sum().reindex(range(len(labels)), fill_value=0)
        n = pd.Series(present.astype(int)).groupby(rows["gid"].to_numpy()).sum().reindex(range(len(labels)),
                                                                                         fill_value=0)
        values[label] = (s.to_numpy(), n.to_numpy(), unit, lo, hi)
    for i in keep:
        row = list(labels[i])
        for label, kind, col, bnd in specs:
            if kind == "people":
                row.append(max(noisy_people[i], 0))
            elif kind == "rows":
                row.append(max(noisy(count[i], G * R), 0))
            else:
                s, n, unit, lo, hi = values[label]
                total = noisy(s[i], G * R * UNITS) * unit
                if kind == "sum":
                    row.append(_round(total))
                else:
                    c = noisy(n[i], G * R)
                    row.append(_round(min(max(total / c, lo), hi)) if c > 0 else None)
        answer.rows.append(row)

    spread = "; ".join(f"{label} {m}" for label, m in margins.items())
    answer.notes.append(f"differential privacy: epsilon {eps:g} spent ({budget.left:g} of {budget.epsilon:g} left). "
                        f"Noise, 95% of the time within: {spread}")
    if selecting:
        answer.notes.append(f"groups appear only when their noisy count of people is at least {threshold:.0f} "
                            '(list "values" to see every group you name); '
                            f"each person counts in at most {G} group(s), with at most {R} row(s) in each")
    elif public:
        answer.notes.append(f"the groups you listed, every one shown (counts below 0 shown as 0); each person "
                            f"counts in at most {G} group(s), with at most {R} row(s) in each")
    else:
        answer.notes.append(f"each person counts with at most {R} row(s)")
    answer.privacy = {"epsilon": eps, "delta": delta, "left": round(budget.left, 6),
                      "groups_per_person": G, "rows_per_person": R}
    budget.cache[key] = answer.to_dict()
    budget.save()
    a.log.append(entry | {"groups_shown": len(answer.rows), "epsilon": eps, "delta": delta,
                          "seconds": round(time.time() - started, 3)})
    if a.state is not None:
        a._save(a.log[-1], history=False)
    return answer
