"""Safe analysis: aggregate answers about a table, never its rows.

An AI model (or anyone) asks questions; only gated aggregates come back. Identifier columns, free
text and the column that says whose row it is are not reachable at all.

    a = Analysis(df, k=5)                       # or Analysis("claims.parquet", policy="claims.policy.json")
    a.describe()                                # columns that can be asked about, and how
    a.aggregate(group_by=["sex", {"column": "admitted", "by": "month"}],
                measures=["people", {"mean": "paid"}],
                where=[{"column": "diagnosis", "in": ["E11.9", "I10"]}])

What the gate does to every answer:
    small groups     a group of fewer than k people is left out (and counted as left out)
    dominance        a sum or mean that one person makes more than half of, or two people more than
                     85%, is withheld: it would say roughly what they have
    differencing     a group whose rows differ from an earlier group's by rows of fewer than k people
                     is answered as that earlier group (left out if it was): "everyone in Cinere" and "everyone in
                     Cinere but the one claim of 7,350,000" get the same numbers, so their difference
                     gives nothing, whichever is asked first. (Refusing such a group instead would
                     itself tell: it would be refused exactly when that claim is in the group.)
    noise            counts move by up to 2, sums by up to a fifth of the largest person's share.
                     The noise comes from the people in the group (a keyed hash of who they are), so
                     the same group always gets the same answer: asking again doesn't average it
                     away, and subtracting two answers that differ by one person doesn't give that
                     person's value. (This is the cell-key method national statistics offices use.)
    limits           numbers are grouped only in bins; at most `max_groups` groups per answer

Groups count people, not rows, when the table says whose row it is (the policy's tokenized member,
patient or NIK column, or `person`).

`state` (a folder) keeps what must outlive one session: the secret the noise comes from, the groups
already answered and a log of every question (counts only). Without it a new secret is made for each
Analysis and the differencing check starts empty, so ask a restarted session nothing you would not
answer twice. The MCP server (`lindungidatamu serve`) always keeps one.
"""

import datetime as dt
import hashlib
import hmac
import math
import secrets as _secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import anonymity

DATE_BINS = ("day", "week", "month", "quarter", "year")
OPS = ("=", "!=", "in", "not_in", "<", "<=", ">", ">=", "between", "missing")
DOMINANCE = ((1, 0.5), (2, 0.85))      # (n, share): the top n people may not make up more than share
COUNT_NOISE = ((0.05, -2), (0.25, -1), (0.75, 0), (0.95, 1), (1.0, 2))   # cumulative probability -> noise
SUM_NOISE = 0.2                         # sums move by up to this times the largest person's contribution


def _pandas():
    try:
        import numpy
        import pandas
        return pandas, numpy
    except ImportError:
        raise ImportError("safe analysis needs pandas: pip install 'lindungidatamu[files]'") from None


@dataclass
class Answer:
    """Rows of a gated aggregate, and what the gate held back (counts only)."""
    columns: list[str]
    rows: list[list]
    hidden_groups: int = 0          # groups of fewer than k people, left out
    reused: int = 0                 # groups answered from an earlier group that differs by fewer than k people
    withheld: int = 0               # sums or means left empty because one or two people dominate them
    notes: list[str] = field(default_factory=list)
    privacy: dict | None = None     # DP mode: what this answer spent

    def to_dict(self) -> dict:
        d = {"columns": self.columns, "rows": self.rows, "hidden_groups": self.hidden_groups,
             "reused": self.reused, "withheld": self.withheld, "notes": self.notes}
        return d | {"privacy": self.privacy} if self.privacy is not None else d

    def frame(self):
        pd, _ = _pandas()
        return pd.DataFrame(self.rows, columns=self.columns)

    def __str__(self) -> str:
        lines = ["| " + " | ".join(self.columns) + " |", "|" + "---|" * len(self.columns)]
        lines += ["| " + " | ".join("" if v is None else f"{v:,}" if isinstance(v, (int, float)) else str(v)
                                    for v in r) + " |" for r in self.rows]
        return "\n".join(lines + self.notes)


class QueryError(ValueError):
    """A question the gate won't answer, with the reason."""


class Analysis:
    def __init__(self, data, policy=None, *, k: int = 5, person: str | None = None, secret: bytes | str | None = None,
                 max_groups: int = 500, names=None, state: str | Path | None = None, dp: float | None = None,
                 dp_question: float | None = None, delta: float = 1e-6):
        from . import table
        pd, np = _pandas()
        if isinstance(data, (str, Path)):
            data = pd.read_parquet(data) if str(data).endswith(".parquet") else \
                pd.DataFrame(__import__("lindungidatamu.files", fromlist=["read_table"]).read_table(data))
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        data = data.reset_index(drop=True)
        if isinstance(policy, (str, Path)):
            policy = table.load_policy(policy)
        if policy is None:
            columns = {str(c): [None if table._missing(v) else v for v in data[c].tolist()] for c in data.columns}
            policy = table.profile(columns, names=names)
        self.k, self.max_groups = k, max_groups
        self.person = person if person is not None else table.person_column(policy)
        self.state = Path(state) if state is not None else None
        self.budget = None
        if self.state is not None:
            other = self.state / ("history.npz" if dp else "budget.json")
            if other.exists():
                raise ValueError(f"{self.state} was used {'without' if dp else 'in'} DP mode: its answers are not "
                                 "counted the same way, so use a new state folder for this mode")
        if dp:
            from . import dp as _dp
            if self.state is not None:
                self.state.mkdir(parents=True, exist_ok=True)
            self.budget = _dp.Budget(dp, dp_question, delta, self.state / "budget.json" if self.state else None)
        if self.state is not None and secret is None:
            secret = self._state_secret()
        self._secret = (secret.encode() if isinstance(secret, str) else secret) or _secrets.token_bytes(32)
        self.log: list[dict] = []
        self._released: dict[int, list] = {}    # people -> (people, rows, left out) of every group asked
        self.hidden: dict[str, str] = {}
        self.kinds: dict[str, str] = {}
        self.roles: dict[str, str] = {}
        self.levels: dict[str, tuple] = {}      # category columns with a hierarchy: the levels to group by
        frame = {}
        for c in data.columns:
            c = str(c)
            rule = policy.get(c)
            if c == self.person:
                self.hidden[c] = "says whose row it is: used only to count people"
                continue
            if rule is None:
                self.hidden[c] = "not in the policy"
                continue
            if rule.action in ("tokenize", "drop"):
                self.hidden[c] = "identifier" if rule.action == "tokenize" else "dropped by the policy"
                continue
            if rule.action == "scan_text" and not rule.role and not rule.text_quasi:
                self.hidden[c] = "free text"
                continue
            kind, values = self._kind(data[c], rule)
            if kind is None:
                self.hidden[c] = "free text"
                continue
            self.kinds[c], frame[c] = kind, values
            if rule.role:
                self.roles[c] = rule.role
            if kind == "category":
                h = rule.hierarchy if rule.hierarchy in ("wilayah", "postal", "icd10") else \
                    anonymity.guess_hierarchy(values.dropna().tolist())
                if h in ("wilayah", "postal", "icd10"):
                    self.levels[c] = (h, tuple(l for l in anonymity.LEVELS[h] if l not in ("value", "*")))
        self.df = pd.DataFrame(frame, index=data.index)
        ids = data[self.person].astype(str) if self.person else pd.Series(range(len(data))).astype(str)
        codes, uniq = pd.factorize(ids)
        keys = np.array([int.from_bytes(hmac.new(self._secret, u.encode(), hashlib.sha256).digest()[:8], "little")
                         for u in uniq], dtype="uint64")
        self._pid = codes                                   # row -> person number
        self._pkey = keys[codes]                            # row -> its person's key
        with np.errstate(over="ignore"):                    # row keys: the person's key mixed with the row
            x = self._pkey ^ (np.arange(len(codes), dtype="uint64") * np.uint64(0x9E3779B97F4A7C15))
            x = (x ^ (x >> np.uint64(31))) * np.uint64(0xBF58476D1CE4E5B9)
            self._rkey = x ^ (x >> np.uint64(29))
        # what this table is, for the state: the same rows in the same order, the same people
        with np.errstate(over="ignore"):
            self._fingerprint = f"{len(codes)}:{int(self._rkey.sum(dtype='uint64'))}"
        if self.state is not None and self.budget is None:
            self._load_history()

    # ---- state ----------------------------------------------------------------------------------

    def _state_secret(self) -> bytes:
        self.state.mkdir(parents=True, exist_ok=True)
        key = self.state / "secret"
        if not key.exists():
            key.touch(mode=0o600)
            key.write_bytes(_secrets.token_bytes(32))
        return key.read_bytes()

    def _load_history(self):
        _, np = _pandas()
        f = self.state / "history.npz"
        if not f.exists():
            return
        h = np.load(f)
        if str(h["fingerprint"]) != self._fingerprint:
            raise ValueError(f"{self.state} was kept for a different table (or the same one changed): use a new "
                             "state folder for it")
        who, pos, sizes, small = h["who"], h["pos"], h["sizes"], h["small"]
        w = p = 0
        for (nw, np_), sm in zip(sizes, small):
            self._released.setdefault(int(nw), []).append((who[w:w + nw], pos[p:p + np_], bool(sm)))
            w, p = w + nw, p + np_

    def _save(self, entry: dict, history: bool = True):
        import json
        _, np = _pandas()
        if not history:
            with open(self.state / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return
        groups = [g for gs in self._released.values() for g in gs]
        empty = np.zeros(0, dtype="int64")
        tmp = self.state / "history.tmp.npz"
        np.savez_compressed(tmp, fingerprint=np.array(self._fingerprint),
                            who=np.concatenate([g[0] for g in groups]) if groups else empty,
                            pos=np.concatenate([g[1] for g in groups]) if groups else empty,
                            sizes=np.array([(len(g[0]), len(g[1])) for g in groups], dtype="int64").reshape(-1, 2),
                            small=np.array([g[2] for g in groups], dtype=bool))
        tmp.replace(self.state / "history.npz")
        with open(self.state / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- columns --------------------------------------------------------------------------------

    def _kind(self, s, rule):
        pd, np = _pandas()
        if pd.api.types.is_bool_dtype(s):
            return "category", s.astype("object").where(s.notna(), None)
        if pd.api.types.is_datetime64_any_dtype(s):
            return "date", s.dt.tz_localize(None) if getattr(s.dt, "tz", None) else s
        if pd.api.types.is_numeric_dtype(s):
            return "number", s.astype("float64")
        present = s.dropna()
        if present.empty:
            return "category", s
        sample = present.sample(min(len(present), 500), random_state=0)
        if sample.map(lambda v: anonymity._ymd(v) is not None).mean() >= 0.9:
            ords = {u: anonymity._ordinal(u) for u in present.unique()}
            days = s.map(lambda v: ords.get(v))
            return "date", pd.to_datetime(days.map(lambda o: dt.date.fromordinal(int(o)) if o else None))
        nums = pd.to_numeric(sample.astype(str).str.replace(r"^Rp\.?\s*", "", regex=True), errors="coerce")
        if nums.notna().mean() >= 0.95:
            return "number", pd.to_numeric(s.astype(str).str.replace(r"^Rp\.?\s*", "", regex=True), errors="coerce")
        if rule.action == "scan_text" and not rule.role and present.nunique() > 0.5 * len(present):
            return None, None
        return "category", s.astype("object").where(s.notna(), None).map(lambda v: v if v is None else str(v))

    def describe(self) -> dict:
        """What can be asked: each column's kind and role; for a category, the values held by at
        least k people (the rest are counted, not named). Counts are noisy like every answer."""
        pd, np = _pandas()
        cols = []
        for c, kind in self.kinds.items():
            entry = {"name": c, "kind": kind}
            if c in self.roles:
                entry["role"] = self.roles[c]
            s = self.df[c]
            if self.budget is not None:            # DP: values come from the data, so they cost budget
                entry["group_by"] = {"category": "as is" + (f", or by one of {', '.join(self.levels[c][1])}"
                                                             if c in self.levels else ""),
                                     "number": 'bins: {"column": c, "width": w}',
                                     "date": f"by one of {', '.join(DATE_BINS)}"}[kind]
            elif kind == "category":
                people = pd.Series(self._pid).groupby(s.fillna("\0")).nunique()
                shown = sorted(str(v) for v, n in people.items() if n >= self.k and v != "\0")
                entry["values"] = shown[:100]
                if len(shown) > 100:
                    entry["more_values"] = len(shown) - 100
                if c in self.levels:
                    entry["group_by"] = f"as is, or by one of {', '.join(self.levels[c][1])}"
                rare = int((people < self.k).sum())
                if rare:
                    entry["rare_values"] = rare
            elif kind == "number":
                entry["group_by"] = 'bins: {"column": c, "width": w}'
            else:
                entry["group_by"] = f"by one of {', '.join(DATE_BINS)}"
            cols.append(entry)
        if self.budget is not None:
            return {"privacy": self.budget.to_dict(), "columns": cols,
                    "hidden": [{"name": c, "why": why} for c, why in self.hidden.items()],
                    "measures": ["people", "rows", {"sum": "<number column>", "upper": "<bound>", "lower": 0},
                                 {"mean": "<number column>", "upper": "<bound>", "lower": 0}],
                    "options": {"epsilon": f"budget this question spends (default {self.budget.per_question:g})",
                                "groups_per_person": "groups one person may count in (default 1)",
                                "rows_per_person": "rows one person may add to a group (default 3)"},
                    "where_ops": list(OPS)}
        everyone = np.arange(len(self._pid))
        total = self._noisy_count(len(np.unique(self._pid)), self._cell_key(everyone)[0], "people")
        return {"people": total, "k": self.k, "columns": cols,
                "hidden": [{"name": c, "why": why} for c, why in self.hidden.items()],
                "measures": ["people", "rows", {"sum": "<number column>"}, {"mean": "<number column>"}],
                "where_ops": list(OPS)}

    # ---- the gate -------------------------------------------------------------------------------

    def _cell_key(self, pos) -> tuple[int, int]:
        """Two numbers fixed by who is in a group (its people, its rows), not by the question asked."""
        _, np = _pandas()
        _, first = np.unique(self._pid[pos], return_index=True)
        with np.errstate(over="ignore"):
            return int(self._pkey[pos[first]].sum(dtype="uint64")), int(self._rkey[pos].sum(dtype="uint64"))

    def _draw(self, cell: int, what: str) -> float:
        h = hmac.new(self._secret, f"{cell}:{what}".encode(), hashlib.sha256).digest()
        return int.from_bytes(h[:8], "little") / 2 ** 64

    def _noisy_count(self, n: int, cell: int, what: str) -> int:
        u = self._draw(cell, what)
        noise = next(d for p, d in COUNT_NOISE if u < p)
        return max(n + noise, 0)

    def aggregate(self, group_by=(), measures=("people",), where=(), **dp_options) -> Answer:
        """A gated aggregate. group_by: column names, {"column": date, "by": "month"} or
        {"column": number, "width": 10}. measures: "people", "rows", {"sum": col}, {"mean": col}.
        where: {"column": c, "<op>": value} with op one of OPS. In DP mode also epsilon,
        groups_per_person and rows_per_person (see dp.py), and sums and means take "upper"."""
        group_by = [group_by] if isinstance(group_by, (str, dict)) else list(group_by)
        measures = [measures] if isinstance(measures, (str, dict)) else list(measures)
        where = [where] if isinstance(where, dict) else list(where)
        try:
            if self.budget is not None:
                from . import dp
                unknown = set(dp_options) - {"epsilon", "groups_per_person", "rows_per_person"}
                if unknown:
                    raise QueryError(f"unknown option(s) {', '.join(sorted(unknown))}")
                return dp.aggregate(self, group_by, measures, where, **dp_options)
            if any(v is not None for v in dp_options.values()):
                raise QueryError(f"{', '.join(sorted(dp_options))}: only in DP mode (serve --dp)")
            if any(isinstance(g, dict) and "values" in g for g in group_by):
                raise QueryError('"values" in group_by: only in DP mode (serve --dp)')
            return self._aggregate(group_by, measures, where)
        except QueryError as e:
            self.log.append({"at": dt.datetime.now().isoformat(timespec="seconds"),
                             "query": {"group_by": group_by, "measures": measures, "where": where},
                             "refused": str(e)})
            if self.state is not None:
                self._save(self.log[-1], history=self.budget is None)
            raise

    def _aggregate(self, group_by, measures, where) -> Answer:
        pd, np = _pandas()
        started = time.time()
        mask = pd.Series(True, index=self.df.index)
        for cond in where:
            mask &= self._condition(cond)
        keys, names = [], []
        for g in group_by:
            name, series = self._group(g)
            names.append(name)
            keys.append(series[mask].fillna("(missing)"))
        specs = [self._measure(m) for m in measures]
        rows_idx = self.df.index[mask]
        if names:
            frame = pd.DataFrame({f"g{i}": k for i, k in enumerate(keys)})
            groups = frame.groupby(list(frame.columns), sort=True, dropna=False).groups
            if len(groups) > self.max_groups:
                raise QueryError(f"{len(groups):,} groups: at most {self.max_groups} per answer. Filter, or use "
                                 "wider bins")
            items = [((k if isinstance(k, tuple) else (k,)), idx) for k, idx in groups.items()]
        else:
            items = [((), rows_idx)]
        answer = Answer(columns=names + [label for label, _, _ in specs], rows=[])
        for labels, idx in items:
            pos = np.sort(np.asarray(idx, dtype="int64"))
            who = np.unique(self._pid[pos])
            near = self._near(who, pos)
            if near is not None:
                who, pos, small = near
                answer.reused += not small
            else:
                small = len(who) < self.k
                self._released.setdefault(len(who), []).append((who, pos, small))
            if small:
                answer.hidden_groups += 1
                continue
            people = len(who)
            cell = self._cell_key(pos)
            row = [str(v) for v in labels]
            for label, kind, col in specs:
                row.append(self._value(kind, col, pos, people, cell, answer))
            answer.rows.append(row)
        if answer.hidden_groups:
            answer.notes.append(f"{answer.hidden_groups} group(s) of fewer than {self.k} people left out")
        if answer.reused:
            answer.notes.append(f"{answer.reused} group(s) answered as an earlier group with the same people, or "
                                f"fewer than {self.k} different (so no two answers single anyone out)")
        if answer.withheld:
            answer.notes.append(f"{answer.withheld} value(s) withheld: one or two people make up most of them")
        self.log.append({"at": dt.datetime.now().isoformat(timespec="seconds"),
                         "query": {"group_by": group_by, "measures": measures, "where": where},
                         "groups_shown": len(answer.rows), "groups_hidden": answer.hidden_groups,
                         "reused": answer.reused,
                         "withheld": answer.withheld, "seconds": round(time.time() - started, 3)})
        if self.state is not None:
            self._save(self.log[-1])
        return answer

    def _value(self, kind, col, pos, people, cell, answer):
        _, np = _pandas()
        pcell, cell = cell
        if kind == "people":
            return self._noisy_count(people, pcell, "people")
        if kind == "rows":
            return self._noisy_count(len(pos), cell, "rows")
        values = self.df[col].to_numpy()[pos]
        present = ~np.isnan(values)
        owners = self._pid[pos][present]
        if len(np.unique(owners)) < self.k:
            answer.withheld += 1
            return None
        per = __import__("pandas").Series(values[present]).groupby(owners).sum()
        mags = per.abs().sort_values(ascending=False)
        whole = mags.sum()
        if whole > 0 and any(mags.iloc[:n].sum() > share * whole for n, share in DOMINANCE):
            answer.withheld += 1
            return None
        total = float(per.sum())
        noise = (2 * self._draw(cell, f"sum:{col}") - 1) * SUM_NOISE * float(mags.iloc[0])
        noisy = total + noise
        if kind == "sum":
            return _round(noisy)
        count = self._noisy_count(int(present.sum()), cell, f"n:{col}")
        return _round(noisy / count) if count else None

    def _near(self, who, pos):
        """An earlier asked group whose rows differ from this one's by rows of fewer than k people (or
        none), as (people, rows, left out); None if there is none. Groups left out for being small
        count too: otherwise "the group without her" being left out would say she was in it."""
        _, np = _pandas()
        n = len(who)
        for size in range(max(n - self.k + 1, 1), n + self.k):
            for prev_who, prev_pos, small in self._released.get(size, ()):
                if len(np.setxor1d(prev_who, who, assume_unique=True)) >= self.k:
                    continue
                differ = np.setxor1d(prev_pos, pos, assume_unique=True)
                if len(np.unique(self._pid[differ])) < self.k:
                    return prev_who, prev_pos, small
        return None

    def _measure(self, m):
        if m in ("people", "rows"):
            return m, m, None
        if isinstance(m, dict):
            ops = [k for k in m if k not in ("lower", "upper")]
            op, col = (ops[0], m[ops[0]]) if len(ops) == 1 else (None, None)
            if op in ("sum", "mean"):
                self._column(col, "number", f"{op} needs a number column")
                return f"{op}({col})", op, col
        raise QueryError(f"unknown measure {m!r}: \"people\", \"rows\", {{\"sum\": column}} or {{\"mean\": column}}")

    def _column(self, col, kind=None, why=""):
        if col in self.hidden:
            raise QueryError(f"{col} can't be used: {self.hidden[col]}")
        if col not in self.kinds:
            raise QueryError(f"no column {col!r}")
        if kind and self.kinds[col] != kind:
            raise QueryError(f"{col} is a {self.kinds[col]} column: {why}")
        return self.df[col]

    def _group(self, g):
        pd, np = _pandas()
        if isinstance(g, str):
            s = self._column(g)
            if self.kinds[g] == "number":
                raise QueryError(f"{g} is a number: group it in bins, {{\"column\": \"{g}\", \"width\": ...}}")
            if self.kinds[g] == "date":
                raise QueryError(f"{g} is a date: group it by {', '.join(DATE_BINS)}, "
                                 f"{{\"column\": \"{g}\", \"by\": \"month\"}}")
            return g, s.astype("object")
        if not isinstance(g, dict) or "column" not in g:
            raise QueryError(f"unknown group {g!r}")
        col = g["column"]
        s = self._column(col)
        if self.kinds[col] == "date":
            by = g.get("by", "month")
            if by not in DATE_BINS:
                raise QueryError(f"dates group by one of {', '.join(DATE_BINS)}")
            codes, uniques = pd.factorize(s)
            u = pd.DatetimeIndex(uniques)
            if by == "quarter":
                labels = [f"{y}-Q{q}" for y, q in zip(u.year, u.quarter)]
            elif by == "week":
                iso = u.isocalendar()
                labels = [f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)]
            else:
                labels = list(u.strftime({"day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}[by]))
            out = pd.Series(np.array(labels + [None], dtype=object)[codes], index=s.index)
            return f"{col} ({by})", out.where(s.notna(), None)
        if col in self.levels and "by" in g:
            h, levels = self.levels[col]
            if g["by"] not in levels:
                raise QueryError(f"{col} groups by one of {', '.join(levels)}")
            codes, uniques = pd.factorize(s)
            labels = anonymity.generalize(list(uniques), h, g["by"]) + [None]
            return f"{col} ({g['by']})", pd.Series(np.array(labels, dtype=object)[codes], index=s.index)
        if self.kinds[col] == "number":
            width = g.get("width")
            if not isinstance(width, (int, float)) or width <= 0:
                raise QueryError(f"{col}: give a bin width, {{\"column\": \"{col}\", \"width\": 10}}")
            lo = np.floor(s / width) * width
            return f"{col} (bins of {width:g})", lo.map(lambda v: None if v != v else f"{v:g} to <{v + width:g}")
        return col, s.astype("object")

    def _condition(self, cond):
        pd, np = _pandas()
        if not isinstance(cond, dict) or "column" not in cond:
            raise QueryError(f"unknown condition {cond!r}: {{\"column\": c, \"<op>\": value}}")
        ops = [k for k in cond if k != "column"]
        if len(ops) != 1 or ops[0] not in OPS:
            raise QueryError(f"a condition takes one of {', '.join(OPS)}")
        col, op = cond["column"], ops[0]
        value = cond[op]
        s = self._column(col)
        kind = self.kinds[col]
        conv = lambda v: self._convert(col, kind, v)
        if op == "missing":
            return s.isna() if value else s.notna()
        if op in ("in", "not_in"):
            if not isinstance(value, list):
                raise QueryError(f"{op} takes a list")
            hit = s.isin([conv(v) for v in value])
            return hit if op == "in" else ~hit & s.notna()
        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise QueryError("between takes [low, high]")
            return (s >= conv(value[0])) & (s <= conv(value[1]))
        v = conv(value)
        if op in ("<", "<=", ">", ">=") and kind == "category":
            raise QueryError(f"{col} is a category: use =, !=, in or not_in")
        return {"=": s == v, "!=": (s != v) & s.notna(), "<": s < v, "<=": s <= v, ">": s > v, ">=": s >= v}[op]

    def _convert(self, col, kind, v):
        pd, _ = _pandas()
        if kind == "number":
            try:
                return float(v)
            except (TypeError, ValueError):
                raise QueryError(f"{col} is a number: {v!r} isn't") from None
        if kind == "date":
            d = anonymity._ymd(v)
            if d is None:
                raise QueryError(f"{col} is a date: write {v!r} as YYYY-MM-DD")
            return pd.Timestamp(*d)
        return str(v)


def _round(x: float) -> float:
    """Three significant figures: more would only be noise."""
    if x == 0 or x != x:
        return x
    digits = 3 - int(math.floor(math.log10(abs(x)))) - 1
    r = round(x, digits)
    return int(r) if digits <= 0 else r
