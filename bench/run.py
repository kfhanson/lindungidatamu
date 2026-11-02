"""Scores lindungidatamu on the benchmark (bench/data/id-pii-v1.jsonl).

    uv run python bench/run.py                    # patterns only, v1 data
    uv run --extra names python bench/run.py --names   # + the GLiNER name model (downloads ~1 GB once)
    uv run python bench/run.py --data v2          # the second set

For masking, what matters most is that nothing slips through, so the headline is **protected**:
the share of labelled items at least 90% hidden by something lindungidatamu found (any type).
**strict** asks more: the right type and the exact span. **over-masking** counts what it hid
that isn't labelled at all (an invoice number, a date): harmless to privacy, but it costs the
AI context. Results go to bench/results/<data>-<system>[-<tag>].json and a table on stdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))
import lindungidatamu as ld  # noqa: E402

COVER = 0.9
NAME_TYPES = {"PERSON", "ORG"}


def covered(gold: dict, found: list) -> float:
    chars = set(range(gold["start"], gold["end"]))
    hit = {i for e in found for i in range(e.start, e.end)} & chars
    return len(hit) / len(chars)


def score(rows: list[dict], finder) -> dict:
    per: dict[str, dict] = {}
    over, over_texts, total_found, t0 = 0, 0, 0, time.perf_counter()
    misses: list[dict] = []
    for r in rows:
        found = ld.find(r["text"], names=finder)
        total_found += len(found)
        for g in r["spans"]:
            p = per.setdefault(g["type"], {"labelled": 0, "protected": 0, "strict": 0})
            p["labelled"] += 1
            if covered(g, found) >= COVER:
                p["protected"] += 1
            else:
                misses.append({"id": r["id"], "type": g["type"], "value": r["text"][g["start"]:g["end"]]})
            if any(e.start == g["start"] and e.end == g["end"] and e.type == g["type"] for e in found):
                p["strict"] += 1
        stray = [e for e in found if not any(e.start < g["end"] and g["start"] < e.end for g in r["spans"])]
        over += len(stray)
        over_texts += bool(stray)
    ms = (time.perf_counter() - t0) * 1000 / len(rows)
    labelled = sum(p["labelled"] for p in per.values())
    return {
        "texts": len(rows), "labelled": labelled,
        "protected": sum(p["protected"] for p in per.values()) / labelled,
        "strict": sum(p["strict"] for p in per.values()) / labelled,
        "over_masked_items": over, "over_masked_texts": over_texts, "found": total_found,
        "ms_per_text": round(ms, 2), "per_type": per, "misses": misses,
    }


def table(name: str, res: dict, skip_names: bool) -> str:
    lines = [f"### {name}", "",
             f"Protected **{res['protected']:.1%}** of {res['labelled']} labelled items · strict {res['strict']:.1%} · "
             f"over-masked {res['over_masked_items']} items in {res['over_masked_texts']} of {res['texts']} texts · "
             f"{res['ms_per_text']} ms per text", "",
             "| Type | Labelled | Protected | Strict |", "|---|---:|---:|---:|"]
    for t, p in sorted(res["per_type"].items(), key=lambda kv: -kv[1]["labelled"]):
        note = " (needs a name model)" if skip_names and t in NAME_TYPES else ""
        lines.append(f"| {t}{note} | {p['labelled']} | {p['protected'] / p['labelled']:.0%} | {p['strict'] / p['labelled']:.0%} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", action="store_true", help="add the GLiNER name model")
    ap.add_argument("--data", default="v1", choices=["v1", "v2"], help="which benchmark set")
    ap.add_argument("--tag", default="", help="added to the results file name (e.g. before, after)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data" / f"id-pii-{args.data}.jsonl").read_text().splitlines()]
    finder, name = None, "patterns"
    if args.names:
        from lindungidatamu.names import GlinerNames
        finder, name = GlinerNames(), "patterns+gliner"
    res = score(rows, finder)
    (ROOT / "results").mkdir(exist_ok=True)
    out = "-".join(x for x in (args.data, name, args.tag) if x)
    (ROOT / "results" / f"{out}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(table(out, res, finder is None))
    by_type: dict[str, list] = {}
    for m in res["misses"]:
        by_type.setdefault(m["type"], []).append(m["value"])
    print("\nMissed, a few per type:")
    for t, values in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if finder is None and t in NAME_TYPES:
            continue
        print(f"  {t} ({len(values)}): " + " | ".join(values[:4]))


if __name__ == "__main__":
    main()
