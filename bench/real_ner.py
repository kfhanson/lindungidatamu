"""Names on real Indonesian text: people and organisations in two public NER test sets from IndoNLU
(github.com/IndoNLP/indonlu, Apache-2.0), text we didn't write.

    uv run --extra names python bench/real_ner.py --split valid   # tune on this
    uv run --extra names python bench/real_ner.py                 # report on the test sets

- NERGrit: 209 sentences (encyclopedic), labelled PERSON, PLACE, ORGANISATION.
- NERP: 840 news sentences, all lowercase, labelled people, places, products, events, food.

The files are fetched into bench/external/ on first use and never committed. Scored: the share of
labelled people and organisations at least 90% hidden, for GLiNER alone and for lindungidatamu
(GLiNER + patterns + the Indonesian layer in public.json), and what each hides that isn't a
labelled entity at all (over-masking). Places, products and events are labelled in these sets but
aren't personal data, so hiding one isn't counted either way.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))
import lindungidatamu as ld  # noqa: E402

BASE = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset"
SETS = {"NERGrit": "nergrit_ner-grit", "NERP": "nerp_ner-prosa"}
PEOPLE, ORGS = {"PERSON", "PPL"}, {"ORGANISATION"}
COVER = 0.9


def load(name: str, split: str = "test") -> list[dict]:
    path = ROOT / "external" / f"{SETS[name]}-{split}.txt"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{SETS[name]}/{split}_preprocess.txt", path)
    rows, tokens = [], []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip():
            tok, lab = line.split("\t")
            tokens.append((tok, lab))
            continue
        if not tokens:
            continue
        text, spans, pos, cur = "", [], 0, None
        for tok, lab in tokens:
            start = len(text) + (1 if text else 0)
            text = f"{text} {tok}" if text else tok
            if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur["label"] != lab[2:])):
                cur = {"start": start, "end": len(text), "label": lab[2:]}
                spans.append(cur)
            elif lab.startswith("I-"):
                cur["end"] = len(text)
            else:
                cur = None
        rows.append({"text": text, "spans": spans})
        tokens = []
    return rows


def covered(span: dict, found: list) -> bool:
    chars = set(range(span["start"], span["end"]))
    return len(chars & {i for e in found for i in range(e.start, e.end)}) / len(chars) >= COVER


def score(rows: list[dict], find) -> dict:
    out = {"people": [0, 0], "orgs": [0, 0], "over": 0, "over_examples": {}}
    t0 = time.perf_counter()
    for r in rows:
        found = find(r["text"])
        for s in r["spans"]:
            key = "people" if s["label"] in PEOPLE else "orgs" if s["label"] in ORGS else None
            if key:
                out[key][1] += 1
                out[key][0] += covered(s, found)
        for e in found:
            if not any(e.start < s["end"] and s["start"] < e.end for s in r["spans"]):
                out["over"] += 1
                out["over_examples"][f"{e.type}: {e.text}"] = out["over_examples"].get(f"{e.type}: {e.text}", 0) + 1
    out["ms_per_text"] = round((time.perf_counter() - t0) * 1000 / len(rows), 1)
    out["over_examples"] = dict(sorted(out["over_examples"].items(), key=lambda kv: -kv[1])[:15])
    return out


def pct(pair: list[int]) -> str:
    return f"{pair[0] / pair[1]:.0%} ({pair[0]}/{pair[1]})" if pair[1] else "–"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"],
                    help="valid for tuning; test only to report (never tune on it)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from lindungidatamu.names import GlinerNames
    model = GlinerNames()
    systems = {"GLiNER alone": model.find, "lindungidatamu": lambda t: ld.find(t, names=model)}
    results = {}
    print("| Set | System | People protected | Organisations protected | Over-masked | ms per text |")
    print("|---|---|---:|---:|---:|---:|")
    for name in SETS:
        rows = load(name, args.split)
        for sys_name, find in systems.items():
            res = score(rows, find)
            results[f"{name} · {sys_name}"] = res
            print(f"| {name} ({len(rows)}) | {sys_name} | {pct(res['people'])} | {pct(res['orgs'])} | {res['over']} | {res['ms_per_text']} |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "-".join(x for x in ("real-ner", args.split, args.tag) if x)).with_suffix(".json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    for k, res in results.items():
        print(f"\n{k}: over-masked most often: {res['over_examples']}")


if __name__ == "__main__":
    main()
