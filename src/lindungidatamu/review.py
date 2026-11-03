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

from .table import Rule, _empty, _text, mask_table


def build(columns: dict[str, list], policy: dict[str, Rule], *, file_name: str, rows: int = 300, names=None,
          seed: int = 0) -> str:
    """The review page for `rows` random rows, masked by `policy`."""
    cols = list(columns)
    total = max((len(v) for v in columns.values()), default=0)
    idx = list(range(total))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:rows])
    sample = {c: [columns[c][i] for i in idx] for c in cols}
    t0 = time.perf_counter()
    done = mask_table(sample, policy, names=names, keep_details=True)
    seconds = round(time.perf_counter() - t0, 1)

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
            elif rule.action in ("scan_text", "scan_patterns"):
                for e in done.details[c][s][1]:
                    hits.append((e.start, e.end, e.type, e.source, e.detector, done.vault.placeholder(e.type, e.text)))
            cell_spans = []
            for a, b, t, src, det, ph in hits:
                eid = len(ents)
                ents.append({"id": eid, "type": t, "start": start + a, "end": start + b, "source": src, "detector": det,
                             "ph": ph, "cells": [ci], "header": False})
                cell_spans.append([a, b, eid])
            cells.append({"c": ci, "v": s, "e": cell_spans})
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
