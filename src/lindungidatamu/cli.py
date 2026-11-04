"""lindungidatamu on the command line: mask personal data in tables and text files.

    lindungidatamu scan claims.parquet               # writes claims.policy.json: one action per column, with why
    lindungidatamu mask claims.parquet               # writes claims.masked.parquet and a report (counts only)
    lindungidatamu mask claims.parquet --vault claims.vault.json   # reversible: keep the vault private
    lindungidatamu restore findings.md --vault claims.vault.json   # real values back, exact placeholders only
    lindungidatamu review claims.parquet             # writes claims.review.html: check the masking by eye
    lindungidatamu risk claims.parquet               # can rows still be singled out? writes the fix into the policy

Masking is one-way unless you ask for a vault. Add --names to use the name model for people and
organisations in text (pip install "lindungidatamu[names]"). Tables need pip install "lindungidatamu[files]".
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .files import is_table, read_table, read_text, write_table, write_text
from .detect import find
from .mask import Vault, mask


def _beside(path: Path, suffix: str, ext: str | None = None) -> Path:
    return path.with_name(f"{path.stem}.{suffix}{ext if ext is not None else path.suffix}")


def _names(on: bool):
    if not on:
        return None
    from .names import GlinerNames
    print("loading the name model...", file=sys.stderr)
    return GlinerNames()


def _label(hide: bool):
    seen = {}
    return (lambda c: seen.setdefault(c, f"C{len(seen) + 1}")) if hide else (lambda c: c)


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def cmd_scan(a) -> int:
    from .table import profile, save_policy
    path = Path(a.file)
    if not is_table(path):
        found = Counter(e.type for e in find(read_text(path), names=_names(a.names)))
        _say(f"{path.name}: {sum(found.values())} found " + (json.dumps(dict(found)) if found else ""))
        return 0
    columns = read_table(path)
    names = _names(a.names)
    policy = profile(columns, sample=a.sample, names=names)
    out = Path(a.policy) if a.policy else _beside(path, "policy", ".json")
    save_policy(policy, out, file=path.name, rows=max((len(v) for v in columns.values()), default=0),
                names_model=getattr(names, "name", None))
    label = _label(a.hide_columns)
    _say(f"{path.name}: {len(columns)} columns, {max((len(v) for v in columns.values()), default=0):,} rows\n")
    _say("| Column | Action | Type | Why |\n|---|---|---|---|")
    for c, r in policy.items():
        _say(f"| {label(c)} | {r.action} | {r.type or ''} | {r.reason} |")
    _say(f"\npolicy: {out}\nCheck it (change any action: keep, tokenize, scan_text, scan_patterns, drop), then: "
         f"lindungidatamu mask {path.name}{' --names' if a.names else ''}")
    return 0


def cmd_mask(a) -> int:
    path = Path(a.file)
    out = Path(a.out) if a.out else _beside(path, "masked")
    vault = None
    if a.vault and Path(a.vault).exists():
        vault = Vault.load(a.vault)
    names = _names(a.names)
    label = _label(a.hide_columns)

    if not is_table(path):
        vault = vault or Vault()
        m = mask(read_text(path), vault=vault, names=names)
        write_text(m.text, out)
        report = {"file": path.name, "found": dict(Counter(e.type for e in m.entities))}
    else:
        from .table import load_policy, mask_table, profile, save_policy
        columns = read_table(path)
        policy_path = Path(a.policy) if a.policy else _beside(path, "policy", ".json")
        settings = {}
        if policy_path.exists():
            from .table import load_settings
            policy = load_policy(policy_path)
            settings = load_settings(policy_path).get("risk") or {}
            _say(f"policy: {policy_path}")
        else:
            policy = profile(columns, names=names)
            save_policy(policy, policy_path, file=path.name)
            _say(f"no policy yet: made one without review, {policy_path}")
        for c in columns:                       # C1, C2... in file order, as scan prints them
            label(c)
        done = mask_table(columns, policy, vault=vault or Vault(merge_partial=False), names=names,
                          k=settings.get("k"), person=settings.get("person"),
                          progress=lambda m: print(f"  {label(m.split(':')[0])}:{m.split(':', 1)[1]}", file=sys.stderr))
        vault = done.vault
        report = {"file": path.name} | done.report
        residual = done.report["checks"]["residual"]
        if not residual["clean"] and not a.force:
            report_path = Path(a.report) if a.report else _beside(out, "report", ".json")
            report_path.write_text(json.dumps(_hide(report, label), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            _say("NOT written: personal data is still visible after masking (counts per column):")
            for k, v in {**residual["identifier_values_visible"], **residual["pattern_hits"]}.items():
                _say(f"  {label(k.split(': ')[0])}{': ' + k.split(': ', 1)[1] if ': ' in k else ''}: {v}")
            _say(f"Change those columns' actions in {policy_path} and run again, or add --force.\nreport: {report_path}")
            return 2
        write_table(done.columns, out)
        _summary(report, label)
        if "risk" in done.report:
            rk = done.report["risk"]
            _say(f"\nre-identification risk, before: {_risk_line(rk['before'])}")
            _say(f"                         after:  {_risk_line(rk['after'])}")
            if not rk["generalized"]:
                _say("  (nothing generalised: run lindungidatamu risk " + path.name + " to find the least fix)")

    report["reversible"] = bool(a.vault)
    report_path = Path(a.report) if a.report else _beside(out, "report", ".json")
    report_path.write_text(json.dumps(_hide(report, label) if is_table(path) else report, indent=1, ensure_ascii=False)
                           + "\n", encoding="utf-8")
    _say(f"masked: {out}\nreport: {report_path} (counts only)")
    if a.vault:
        vault.save(a.vault)
        _say(f"vault:  {a.vault} (the real values: keep it private, never send it with the data)")
    else:
        _say("one-way: no vault kept, the placeholders can't be turned back into values")
    return 0


def _pct(x) -> str:
    return f"{x:.1%}" if x is not None else "-"


def _risk_line(m: dict) -> str:
    if not m.get("groups"):
        return "no quasi-identifiers"
    unit = m.get("counted", "rows")
    line = (f"k = {m['k']}; {_pct(m['unique_share'])} of rows unique; {_pct(m['below_k_share'])} in groups of fewer "
            f"than {m['target_k']} {unit}; average risk {m['average_risk']:.3f}")
    if m.get("suppressed_rows"):
        line += f"; {m['suppressed_rows']:,} rows suppressed"
    return line


def cmd_risk(a) -> int:
    from .table import load_policy, load_settings, profile, risk, save_policy
    path = Path(a.file)
    if not is_table(path):
        _say(f"{path.name}: risk is for tables (CSV, Parquet, JSON Lines, Excel)")
        return 1
    columns = read_table(path)
    rows = max((len(v) for v in columns.values()), default=0)
    policy_path = Path(a.policy) if a.policy else _beside(path, "policy", ".json")
    settings = {}
    if policy_path.exists():
        policy = load_policy(policy_path)
        settings = {k: v for k, v in load_settings(policy_path).items()
                    if k not in ("lindungidatamu", "patterns", "risk")}
        _say(f"policy: {policy_path}")
    else:
        policy = profile(columns, sample=a.sample)
        settings = {"file": path.name, "rows": rows}
        _say(f"no policy yet: made one, {policy_path} (check it, as for scan)")
    label = _label(a.hide_columns)
    for c in columns:
        label(c)
    keep = dict((x.split("=", 1) + ["value"])[:2] for x in a.keep)
    keep = {c.strip(): lvl.strip() for c, lvl in keep.items()}
    unknown = [c for c in keep if c not in columns]
    if unknown:
        _say(f"--keep: no column {', '.join(unknown)}")
        return 1
    try:
        result = risk(columns, policy, k=a.k, max_suppress=a.max_suppress, keep=keep)
    except ValueError as e:
        _say(f"--keep: {e}")
        return 1
    quasi = result.before.get("quasi", [])
    if not quasi:
        _say("No quasi-identifiers (birth date, sex, area, dates...) in the policy. Mark columns with "
             '"role": "quasi" in the policy to measure them.')
        return 0
    who = f", counting people ({label(result.person)})" if result.person else ""
    _say(f"{path.name}: {rows:,} rows; quasi-identifiers: {', '.join(label(c) for c in quasi)}{who}\n")
    _say(f"Now:   {_risk_line(result.before)}")
    without = result.before.get("unique_share_without") or {}
    if without:
        _say("       unique rows without each column: " +
             ", ".join(f"{label(c)} {_pct(v)}" for c, v in sorted(without.items(), key=lambda x: x[1])))
    for c, s in (result.before.get("sensitive") or {}).items():
        _say(f"       {label(c)} (sensitive): l = {s['l']}; {s['groups_one_value']:,} groups share one value "
             f"({s['rows_in_groups_one_value']:,} rows)")
    plan = result.plan
    _say(f"\nLeast change that reaches k = {a.k} (at most {a.max_suppress:.1%} of rows suppressed):\n")
    _say("| Column | Hierarchy | Publish as | Information kept |\n|---|---|---|---:|")
    for c, lvl in plan.levels.items():
        _say(f"| {label(c)} | {plan.hierarchies[c]} | {lvl} | {_pct(1 - plan.lost.get(c, 0))} |")
    if plan.reached and plan.candidates:
        _say("\nOther ways to reach it (choose one with --keep COLUMN, or edit the policy):")
        for alt in plan.candidates[:3]:
            _say("  " + ", ".join(f"{label(c)} {lvl}" for c, lvl in alt["levels"].items()) +
                 f"  (information lost {alt['loss']:.2f} vs {plan.loss:.2f})")
    if not plan.reached:
        _say(f"\nk = {a.k} is NOT reached even fully generalised within the suppression budget: drop a column, "
             "raise --max-suppress, or lower --k. Nothing written.")
        return 2
    _say(f"\nAfter: {_risk_line(result.after)}")
    for c, s in (result.after.get("sensitive") or {}).items():
        _say(f"       {label(c)} (sensitive): l = {s['l']}; {s['groups_one_value']:,} groups share one value")
    save_policy(result.policy, policy_path, **settings,
                risk={"k": a.k, "max_suppress": a.max_suppress, "person": result.person})
    _say(f"\npolicy: {policy_path} (generalize levels written; change any, then: lindungidatamu mask {path.name})")
    return 0


def _hide(report: dict, label) -> dict:
    """The report with column names replaced, when --hide-columns is on."""
    if "columns" not in report:
        return report
    r = dict(report)
    r["columns"] = {label(c): v for c, v in report["columns"].items()}
    checks = report["checks"]
    r["checks"] = {
        "residual": {"identifier_values_visible": {label(k): v for k, v in checks["residual"]["identifier_values_visible"].items()},
                     "pattern_hits": {f"{label(k.split(': ')[0])}: {k.split(': ', 1)[1]}": v
                                      for k, v in checks["residual"]["pattern_hits"].items()},
                     "clean": checks["residual"]["clean"]},
        "readable_share": {label(k): v for k, v in checks["readable_share"].items()},
    }
    if "risk" in report:
        def one(m: dict) -> dict:
            m = dict(m)
            m["quasi"] = [label(c) for c in m.get("quasi", [])]
            for key in ("unique_share_without", "sensitive"):
                if key in m:
                    m[key] = {label(c): v for c, v in m[key].items()}
            return m
        rk = report["risk"]
        r["risk"] = {"before": one(rk["before"]), "after": one(rk["after"]),
                     "generalized": {label(c): v for c, v in rk["generalized"].items()}}
    return r


def _summary(report: dict, label) -> None:
    _say(f"\n| Column | Action | Values changed | Found |\n|---|---|---:|---|")
    for c, r in report["columns"].items():
        if r["action"] == "keep":
            continue
        changed = f"{r.get('changed', 0):,} of {r['rows']:,}" if r["action"] != "drop" else "dropped"
        _say(f"| {label(c)} | {r['action']}{' ' + r['type'] if r.get('type') else ''} | {changed} | "
             f"{', '.join(f'{t} {n:,}' for t, n in r.get('found', {}).items())} |")
    _say(f"\nplaceholders: {json.dumps(report['placeholders'])}")
    _say(f"residual check: {'clean' if report['checks']['residual']['clean'] else 'NOT clean (forced)'}; "
         f"{report['seconds']}s\n")


def cmd_restore(a) -> int:
    path = Path(a.file)
    vault = Vault.load(a.vault)
    out = Path(a.out) if a.out else _beside(path, "restored")
    if is_table(path):
        columns = read_table(path)
        write_table({c: [vault.restore(v, strict=True) if isinstance(v, str) else v for v in vals]
                     for c, vals in columns.items()}, out)
    else:
        write_text(vault.restore(read_text(path), strict=True), out)
    _say(f"restored: {out} (exact placeholders only)")
    return 0


def cmd_review(a) -> int:
    from .review import build, write
    from .table import load_policy, profile
    path = Path(a.file)
    columns = read_table(path)
    names = _names(a.names)
    policy_path = Path(a.policy) if a.policy else _beside(path, "policy", ".json")
    policy = load_policy(policy_path) if policy_path.exists() else profile(columns, names=names)
    out = Path(a.out) if a.out else _beside(path, "review", ".html")
    write(out, build(columns, policy, file_name=path.name, rows=a.rows, names=names))
    _say(f"review: {out}\nOpen it in your browser. It holds the real values: keep it on this machine.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lindungidatamu", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, names=True):
        if names:
            p.add_argument("--names", action="store_true", help="use the name model for people and organisations in text")
        p.add_argument("--hide-columns", action="store_true", help="print columns as C1, C2... (when others see the output)")

    p = sub.add_parser("scan", help="decide an action per column, written to a policy file to check")
    p.add_argument("file")
    p.add_argument("--policy", help="where to write the policy (default: FILE.policy.json)")
    p.add_argument("--sample", type=int, default=2000, help="rows sampled per column (default 2000)")
    common(p)
    p.set_defaults(run=cmd_scan)

    p = sub.add_parser("mask", help="mask a table or a text file")
    p.add_argument("file")
    p.add_argument("--policy", help="the policy to use (default: FILE.policy.json, made if missing)")
    p.add_argument("--out", help="default: FILE.masked.EXT")
    p.add_argument("--vault", help="keep the values behind the placeholders in this file, to restore later")
    p.add_argument("--report", help="default: OUT.report.json")
    p.add_argument("--force", action="store_true", help="write the output even if the residual check finds data")
    common(p)
    p.set_defaults(run=cmd_mask)

    p = sub.add_parser("restore", help="put the real values back, from a vault")
    p.add_argument("file")
    p.add_argument("--vault", required=True)
    p.add_argument("--out", help="default: FILE.restored.EXT")
    p.set_defaults(run=cmd_restore)

    p = sub.add_parser("review", help="an offline HTML page to check the masking by eye")
    p.add_argument("file")
    p.add_argument("--policy")
    p.add_argument("--rows", type=int, default=300)
    p.add_argument("--out", help="default: FILE.review.html")
    common(p)
    p.set_defaults(run=cmd_review)

    p = sub.add_parser("risk", help="can rows still be singled out? measure, and write the least fix into the policy")
    p.add_argument("file")
    p.add_argument("--policy", help="default: FILE.policy.json (made if missing)")
    p.add_argument("--k", type=int, default=5, help="smallest group allowed (default 5)")
    p.add_argument("--max-suppress", type=float, default=0.01, help="share of rows that may be suppressed (default 0.01)")
    p.add_argument("--keep", action="append", default=[], metavar="COLUMN[=LEVEL]",
                   help='what the analysis needs: a column as it is, or no coarser than a level '
                        '("birth_date=5 years"); repeatable')
    p.add_argument("--sample", type=int, default=2000, help="rows sampled per column when profiling (default 2000)")
    common(p, names=False)
    p.set_defaults(run=cmd_risk)

    a = ap.parse_args(argv)
    try:
        return a.run(a)
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
