"""A local, offline HTML page to check masking by eye: sampled rows of a table, what was masked in
each cell and why, with Right / Wrong marks that export without any values.

The page holds the real values: it is for the data's owner, on their machine. It loads nothing
from the internet.
"""

import json
import random
import time
from importlib import resources
from pathlib import Path

from dataclasses import replace

from .table import Rule, _anonymize, _empty, _text, _top_code, mask_table, person_column


def build(columns: dict[str, list], policy: dict[str, Rule], *, file_name: str, rows: int = 300, names=None,
          seed: int = 0, k: int | None = None, l: int = 1, person: str | None = None) -> str:
    """The review page for `rows` random rows, masked by `policy`.

    Generalising, suppressing and pooling large amounts depend on the whole table (a group of five is
    five in the table, not in the sample), so they run on all rows first; the sample is then masked.
    """
    cols = list(columns)
    total = max((len(v) for v in columns.values()), default=0)
    idx = list(range(total))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:rows])
    t0 = time.perf_counter()
    person = person if person is not None else person_column(policy)
    if person is not None and person not in columns:
        person = None
    published, risk = _anonymize(columns, policy, k, person, l)
    published, top_coded = _top_code(published, policy)
    plain = {c: replace(r, role=None, generalize=None, relative_to=None, interval=None, top_code=None)
             for c, r in policy.items()}
    sample = {c: [columns[c][i] for i in idx] for c in cols}
    pub = {c: [published[c][i] for i in idx] for c in cols}
    done = mask_table(pub, plain, names=names, keep_details=True)
    seconds = round(time.perf_counter() - t0, 1)
    changed_by_risk = {c for c in cols if published[c] is not columns[c]}

    out_rows = []
    for k, i in enumerate(idx):
        lines, masked_lines, cells, ents, pos = [], [], [], [], 0
        for ci, c in enumerate(cols):
            v = sample[c][k]
            if _empty(v):
                continue
            s = _text(v)
            head = f"{c}: "
            start = pos + len(head)
            rule = policy.get(c) or Rule("scan_patterns")
            hits = []
            if rule.action in ("tokenize", "drop"):
                ph = done.columns[c][k] if rule.action == "tokenize" else "(dropped)"
                hits = [(0, len(s), rule.type or "DROPPED", "column", f"policy: {rule.action}", ph)]
            elif c in changed_by_risk and _text(pub[c][k]) != s:
                hits = [(0, len(s), "GENERALIZED", "risk", _why(c, rule, risk, top_coded), _text(done.columns[c][k]))]
            elif rule.action in ("scan_text", "scan_patterns"):
                for e in done.details[c][s][1]:
                    hits.append((e.start, e.end, e.type, e.source, e.detector, done.vault.placeholder(e.type, e.text)))
            cell_spans = []
            for a, b, t, src, det, ph in hits:
                eid = len(ents)
                ents.append({"id": eid, "type": t, "start": start + a, "end": start + b, "source": src, "detector": det,
                             "ph": ph, "cells": [ci], "header": False})
                cell_spans.append([a, b, eid])
            cell = {"c": ci, "v": s, "e": cell_spans}
            if rule.text_quasi and rule.action in ("scan_text", "scan_patterns"):
                cell["m"] = _text(done.columns[c][k])        # dates and ages generalised inside the text
            cells.append(cell)
            lines.append(head + s)
            masked_v = "(dropped)" if rule.action == "drop" else _text(done.columns[c][k])
            masked_lines.append(head + masked_v)
            pos = start + len(s) + 1
        text = "\n".join(lines)
        out_rows.append({"i": i, "text": text, "masked": "\n".join(masked_lines), "words": len(text.split()),
                         "ents": ents, "cells": cells})

    data = {
        "file": file_name, "total_rows": total, "columns": cols,
        "dtypes": [_kind(columns[c]) for c in cols],
        "names_model": getattr(names, "name", None) if names is not None else None, "names_max_words": None,
        "decision_model": None, "questions": {}, "seconds": {"mask": seconds, "decision": 0},
        "policy": {c: {"action": r.action, "type": r.type, "reason": r.reason} for c, r in policy.items()},
        "rows": out_rows,
        "risk": _risk_data(columns, published, policy, risk, top_coded, idx, done)
                if (risk or top_coded or any(r.text_quasi for r in policy.values())) else None,
    }
    page = resources.files(__package__).joinpath("review.html").read_text(encoding="utf-8")
    return page.replace("__DATA__", json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/"))


def _kind(values: list) -> str:
    for v in values:
        if not _empty(v):
            return type(v).__name__
    return "empty"


def write(path: str | Path, page: str) -> None:
    Path(path).write_text(page, encoding="utf-8")


def _why(c: str, rule: Rule, risk: dict | None, top_coded: dict) -> str:
    if c in top_coded:
        return f"large amount: pooled into the mean of the values at or above {top_coded[c]['at_or_above']:,.0f}"
    parts = []
    if rule.relative_to:
        parts.append(f"{'age at' if rule.interval == 'age' else 'days from'} {rule.relative_to}")
    if rule.generalize:
        parts.append(f"generalised to {rule.generalize}")
    return "; ".join(parts) or "suppressed: still in a group smaller than k"


def _risk_data(columns, published, policy, risk, top_coded, idx, done) -> dict:
    """The Risk tab: measures over the whole table, and per changed column a few examples."""
    examples = {}
    for c in columns:
        if published[c] is columns[c]:
            continue
        pairs = []
        for i in idx:
            a, b = columns[c][i], published[c][i]
            if not _empty(a) and _text(a) != _text(b) and (_text(a), _text(b)) not in pairs:
                pairs.append((_text(a), _text(b)))
            if len(pairs) == 4:
                break
        r = policy.get(c) or Rule("keep")
        examples[c] = {"role": r.role, "hierarchy": r.hierarchy, "generalize": r.generalize,
                       "relative_to": r.relative_to, "interval": r.interval, "top_code": r.top_code,
                       "changed": sum(1 for a, b in zip(columns[c], published[c]) if _text(a) != _text(b)
                                      and not _empty(a)),
                       "examples": pairs}
    text = {c: done.report["columns"][c].get("quasi_in_text", {}) for c, r in policy.items()
            if r.text_quasi and c in done.report["columns"]}
    return {"text_quasi": text, "before": (risk or {}).get("before"), "after": (risk or {}).get("after"),
            "intervals": (risk or {}).get("intervals") or {}, "generalized": (risk or {}).get("generalized") or {},
            "top_coded": top_coded, "columns": examples}
