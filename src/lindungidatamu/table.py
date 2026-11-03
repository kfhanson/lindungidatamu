"""Tables: decide what to do with each column, then mask the whole table.

A table is not chat text. Most of its personal data sits in identifier columns whose label is the
column name (`account_no`, `member_no`), not a word before the number, and a column usually holds
one kind of thing. So the work is done per column:

    policy = profile(columns)                 # column -> action, with a reason; a person reviews it
    result = mask_table(columns, policy)      # masked columns, and a report with checks

Actions:
    tokenize       every value becomes a placeholder of one type ([MEMBER_12]); no scanning. The
                   same value gets the same placeholder in every column of that type, so joins survive
    scan_text      each distinct value is masked with patterns and the name model (if given)
    scan_patterns  each distinct value is masked with patterns only (categories, codes, amounts as text)
    keep           unchanged (numbers, dates, true/false)
    drop           the column is removed

Every text column is at least pattern-scanned, whatever its profile says: a column that looks like
a category can still hold a NIK in a few rows.

Masking is one-way unless you keep the vault: the placeholders are consistent within the table, and
the values behind them exist only in the vault you pass (and save) yourself.

`columns` is a dict of column name -> list of values (None or NaN for empty), e.g.
{c: df[c].tolist() for c in df.columns}. Nothing here needs pandas.
"""

import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .detect import Entity, find, patterns_version
from .mask import Vault, mask

ACTIONS = ("keep", "tokenize", "scan_text", "scan_patterns", "drop")

# Column-name words -> the type an identifier column is tokenized as. Order matters: the first type
# whose word appears wins ("patient_member_no" is MEMBER, so it joins with "emp_member_no").
TYPE_WORDS = [
    ("NIK", {"nik", "ktp", "nomorinduk"}),
    ("NPWP", {"npwp"}),
    ("KK", {"kk", "kartukeluarga"}),
    ("BPJS", {"bpjs", "jkn"}),
    ("SIM", {"sim"}),
    ("PASSPORT", {"passport", "paspor"}),
    ("BANK", {"rek", "rekening", "norek", "account", "acct", "acc", "iban"}),
    ("EMAIL", {"email", "surel", "mail"}),
    ("PHONE", {"phone", "hp", "telp", "telepon", "tel", "mobile", "wa", "whatsapp", "handphone", "msisdn"}),
    ("ADDRESS", {"address", "alamat", "street", "jalan", "domisili"}),
    ("MEMBER", {"member", "peserta", "participant", "insured", "tertanggung"}),
    ("CARD", {"card", "kartu"}),
    ("POLICY", {"policy", "polis"}),
    ("CLAIM", {"claim", "klaim"}),
    ("EMPLOYEE", {"employee", "karyawan", "pegawai", "nip", "emp", "staff"}),
    ("PATIENT", {"patient", "pasien", "mrn", "rm"}),
]
# Types whose patterns need a label before the number; in a table the label is the column name.
LABELLED = {"NIK": "NIK", "NPWP": "NPWP", "KK": "KK", "BPJS": "BPJS Kesehatan", "SIM": "SIM",
            "PASSPORT": "paspor", "BANK": "no. rek."}
ID_WORDS = {"no", "nomor", "number", "num", "nr", "id", "ref", "reference", "code", "kode"}
AMOUNT_WORDS = {"amount", "amt", "jumlah", "nominal", "billed", "bill", "paid", "bayar", "tagihan", "excess", "rate",
                "tarif", "biaya", "cost", "price", "harga", "total", "sum", "fee", "premi", "premium", "qty",
                "quantity", "count", "plafon", "limit", "saldo", "balance", "value", "nilai", "idr", "rp", "usd"}
PERSON_WORDS = {"nama", "name", "payee", "penerima", "holder", "pemegang", "beneficiary", "ahliwaris", "contact",
                "kontak", "pic", "suami", "istri", "ayah", "ibu", "wali", "dokter", "doctor"}
ORG_WORDS = {"provider", "faskes", "hospital", "rs", "rumahsakit", "klinik", "clinic", "company", "perusahaan",
             "employer", "insurer", "asuransi", "bank", "cabang", "branch", "vendor", "supplier", "merchant",
             "toko", "store", "instansi", "institution", "unit", "dept", "department", "divisi"}
BIRTH_WORDS = {"lahir", "birth", "dob", "tgllahir", "birthdate", "tanggallahir"}
_ICD = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,3})?\b")
_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$")
_NUMBER = re.compile(r"^[-+]?(?:Rp\.?\s*)?\d[\d.,]*$")


@dataclass
class Rule:
    action: str
    type: str | None = None          # for tokenize
    reason: str = ""
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}: one of {', '.join(ACTIONS)}")
        if self.action == "tokenize" and not self.type:
            raise ValueError("tokenize needs a type")


def _words(name: str) -> list[str]:
    """Column-name words: "patientMemberNo" and "patient_member_no" -> patient, member, no."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    words = [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]
    joined = "".join(words)
    return words + [joined] + ["".join(words[i:i + 2]) for i in range(len(words) - 1)]


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip()) or \
        (type(v).__name__ in ("NAType", "NaTType"))


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) or type(v).__module__ == "numpy" and \
        hasattr(v, "dtype") and v.dtype.kind in "iuf"


def _is_date(v) -> bool:
    return hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day")


def _text(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _shape(s: str) -> str:
    return re.sub(r"[A-Za-z]+", "A", re.sub(r"\d+", "9", s))


def _covered(text: str, types: set[str] | None = None) -> str | None:
    """The one type that covers at least 90% of `text`, if any."""
    found = [e for e in find(text) if types is None or e.type in types]
    by_type = Counter()
    for e in found:
        by_type[e.type] += e.end - e.start
    if by_type:
        t, n = by_type.most_common(1)[0]
        if n >= 0.9 * len(text.replace(" ", "")) or n >= 0.9 * len(text):
            return t
    return None


def _type_from_name(words: list[str]) -> str | None:
    for t, keys in TYPE_WORDS:
        if keys & set(words):
            return t
    return None


def profile_column(name: str, values: list, *, sample: int = 2000, names=None, seed: int = 0) -> Rule:
    """Decide one column's action from its name, a row-weighted sample of its values and their shape."""
    words = _words(name)
    present = [v for v in values if not _empty(v)]
    stats = {"rows": len(values), "empty": len(values) - len(present)}
    if not present:
        return Rule("keep", reason="empty", stats=stats)
    distinct = len({_text(v) for v in present})
    stats |= {"distinct": distinct, "distinct_share": round(distinct / len(present), 4)}
    rng = random.Random(seed)
    picked = present if len(present) <= sample else rng.sample(present, sample)
    by_name = _type_from_name(words)
    is_amount = bool(AMOUNT_WORDS & set(words))
    is_birth = bool(BIRTH_WORDS & set(words))

    if all(isinstance(v, bool) or type(v).__name__ == "bool_" for v in picked):
        return Rule("keep", reason="true/false", stats=stats)
    if all(_is_date(v) for v in picked):
        if is_birth:
            return Rule("keep", reason="date of birth: REVIEW (identifying with other columns; consider keeping the year only)",
                        stats=stats)
        return Rule("keep", reason="dates", stats=stats)

    texts = [_text(v) for v in picked]
    numeric = all(_is_number(v) for v in picked)
    long_digits = sum(t.isdigit() and len(t) >= 8 for t in texts) / len(texts)

    if numeric:
        if is_amount:
            return Rule("keep", reason="numbers, amount-like name", stats=stats)
        if (by_name or ID_WORDS & set(words)) and (long_digits >= 0.5 or distinct >= 50):
            return Rule("tokenize", by_name or "ID", f"identifier-like name; numbers, {distinct:,} distinct", stats)
        if by_name or ID_WORDS & set(words):
            return Rule("keep", reason=f"identifier-like name, but only {distinct} distinct numbers: a code", stats=stats)
        if long_digits >= 0.8 and stats["distinct_share"] >= 0.3:
            nik = sum(_covered(t, {"NIK"}) == "NIK" for t in texts) / len(texts)
            if nik >= 0.5:
                return Rule("tokenize", "NIK", f"{nik:.0%} of a sample are valid NIKs", stats)
            return Rule("tokenize", by_name or "ID", "long numbers, mostly unique: an identifier", stats)
        return Rule("keep", reason="numbers", stats=stats)

    # Text (or mixed): never plain "keep"; patterns at least.
    if by_name in LABELLED:
        label = LABELLED[by_name]
        hit = sum(_covered(f"{label} {t}", {by_name}) == by_name or _covered(t) == by_name for t in texts) / len(texts)
        if hit >= 0.8:
            return Rule("tokenize", by_name, f"name says {by_name}; {hit:.0%} of a sample match it after its label", stats)

    kinds = Counter(_covered(t) for t in texts)
    top, n = kinds.most_common(1)[0]
    if top and n / len(texts) >= 0.8:
        return Rule("tokenize", top, f"{n / len(texts):.0%} of a sample are {top}", stats)

    no_space = sum(" " not in t for t in texts) / len(texts)
    has_digit = sum(any(c.isdigit() for c in t) for t in texts) / len(texts)
    if is_amount and sum(bool(_NUMBER.match(t)) for t in texts) / len(texts) >= 0.9:
        return Rule("scan_patterns", reason="amounts written as text", stats=stats)
    if sum(bool(_DATE.match(t)) for t in texts) / len(texts) >= 0.9:
        reason = "dates written as text" + (": date of birth, REVIEW" if is_birth else "")
        return Rule("scan_patterns", reason=reason, stats=stats)
    if by_name in ("EMAIL", "PHONE", "ADDRESS"):
        return Rule("tokenize", by_name, f"name says {by_name}", stats)

    if sum(bool(_ICD.match(t)) for t in texts) / len(texts) >= 0.8:
        return Rule("scan_patterns", reason="coded values (ICD-10): no name model, eponyms like Parkinson stay", stats=stats)

    # Identifiers repeat (one member, many claims), so a type from the column name is enough; a
    # generic "no"/"id" name needs many distinct values, or it is a code list.
    id_shaped = no_space >= 0.9 and has_digit >= 0.9
    shapes = Counter(_shape(t) for t in texts)
    if id_shaped and by_name:
        return Rule("tokenize", by_name, f"name says {by_name}; one-word values with digits, {distinct:,} distinct, "
                    f"{len(shapes)} shape(s)", stats)
    if id_shaped and ID_WORDS & set(words) and distinct >= 50:
        return Rule("tokenize", "ID", f"identifier-like name; one-word values with digits, {distinct:,} distinct", stats)
    if id_shaped and stats["distinct_share"] >= 0.5:
        return Rule("tokenize", "ID", f"one-word values with digits, {stats['distinct_share']:.0%} distinct", stats)

    person = bool(PERSON_WORDS & set(words)) and not (ORG_WORDS & set(words))
    if person:
        if names is None:
            return Rule("tokenize", "PERSON", "name says a person; no name model to check the values", stats)
        check = texts[:200]
        raw = names.find_many(check) if hasattr(names, "find_many") else [names.find(t) for t in check]
        known = _Known(dict(zip(check, raw)))           # cleaned as masking cleans it: "RS Hermina" is no one
        found = [find(t, names=known) for t in check]
        people = sum(sum(e.end - e.start for e in f if e.type == "PERSON") >= 0.6 * len(t) for t, f in zip(check, found))
        if people / len(check) >= 0.9:
            return Rule("tokenize", "PERSON", f"name says a person; the name model finds one in {people / len(check):.0%} "
                        "of a sample", stats)
        return Rule("scan_text", reason=f"name says a person, but only {people / len(check):.0%} of a sample are "
                    "names: masked value by value", stats=stats)

    if distinct <= 20 and max(len(t.split()) for t in texts) <= 3:
        return Rule("scan_patterns", reason=f"a short list of categories ({distinct} values)", stats=stats)
    if names is None:
        return Rule("scan_patterns", reason="text; no name model, so patterns only", stats=stats)
    return Rule("scan_text", reason=f"text, {distinct} distinct values: masked value by value", stats=stats)


def profile(columns: dict[str, list], *, sample: int = 2000, names=None, seed: int = 0) -> dict[str, Rule]:
    """Every column's rule. Write it out with save_policy, have a person check it, then mask_table."""
    return {c: profile_column(c, v, sample=sample, names=names, seed=seed) for c, v in columns.items()}


def save_policy(policy: dict[str, Rule], path: str | Path, **meta) -> None:
    doc = {"lindungidatamu": _version(), "patterns": patterns_version(), **meta,
           "columns": {c: {k: v for k, v in asdict(r).items() if v not in (None, "", {})} for c, r in policy.items()}}
    Path(path).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_policy(path: str | Path) -> dict[str, Rule]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c: Rule(r["action"], r.get("type"), r.get("reason", ""), r.get("stats", {})) for c, r in doc["columns"].items()}


def _version() -> str:
    from . import __version__
    return __version__


class _Known:
    """A name finder that answers from names found earlier, in a batch."""
    def __init__(self, found: dict):
        self.found = found

    def find(self, text):
        return self.found[text]


@dataclass
class MaskedTable:
    columns: dict[str, list]     # the masked table (dropped columns left out)
    report: dict                 # counts only: safe to share
    vault: Vault                 # the values behind the placeholders: save it only to make this reversible
    details: dict = field(default_factory=dict, repr=False)   # column -> value -> (masked, entities), if asked
    frame: object = field(default=None, repr=False)            # the masked pandas DataFrame, from mask_frame


def mask_table(columns: dict[str, list], policy: dict[str, Rule], *, vault: Vault | None = None, names=None,
               batch_size: int = 32, keep_details: bool = False, progress=None) -> MaskedTable:
    """Mask a table by its policy. Columns missing from the policy are pattern-scanned.

    vault     pass one to reuse placeholders from an earlier table; a new one otherwise (table mode:
              no merging of "Budi" into "Budi Santoso"). Save it only if the masking must be reversible
    names     a name finder, used by scan_text columns
    progress  called with a short message per column
    """
    t0 = time.perf_counter()
    vault = vault if vault is not None else Vault(merge_partial=False)
    out, cols_report, details = {}, {}, {}
    tokenized = set()
    known: dict[str, tuple[str, str]] = {}      # a value tokenized in one column -> (placeholder, type)
    rules = {c: policy.get(c) or Rule("scan_patterns", reason="not in the policy") for c in columns}
    # Tokenized columns first: a scanned column holding the same value whole (a payee who is also the
    # patient) then gets the same placeholder, even where the name model misses it.
    order = sorted(columns, key=lambda c: rules[c].action != "tokenize")
    for c in order:
        values, rule = columns[c], rules[c]
        if progress:
            progress(f"{c}: {rule.action}")
        rep = {"action": rule.action, **({"type": rule.type} if rule.type else {})}
        if rule.action == "drop":
            cols_report[c] = rep | {"rows": len(values)}
            continue
        if rule.action == "keep":
            out[c] = list(values)
            cols_report[c] = rep | {"rows": len(values), "changed": 0}
            continue
        present = [v for v in values if not _empty(v)]
        distinct = list(dict.fromkeys(_text(v) for v in present))
        cache: dict[str, tuple[str, list]] = {}
        if rule.action == "tokenize":
            for t in distinct:
                cache[t] = (vault.placeholder(rule.type, t), [])
                if len(t) >= 6:          # shorter values ("2024") appear by chance elsewhere
                    tokenized.add(t)
                if len(t) >= 4:
                    known.setdefault(" ".join(t.lower().split()), (cache[t][0], rule.type))
        else:
            finder = None
            if rule.action == "scan_text" and names is not None:
                found = {}
                for i in range(0, len(distinct), batch_size * 8):
                    chunk = distinct[i:i + batch_size * 8]
                    res = names.find_many(chunk, batch_size=batch_size) if hasattr(names, "find_many") else \
                        [names.find(t) for t in chunk]
                    found.update(zip(chunk, res))
                finder = _Known(found)
            for t in distinct:
                if (hit := known.get(" ".join(t.lower().split()))) is not None:
                    cache[t] = (hit[0], [Entity(hit[1], 0, len(t), t, "column", "same value as a tokenized column")])
                    continue
                m = mask(t, vault=vault, names=finder)
                cache[t] = (m.text, m.entities)
        col, changed, types = [], 0, Counter()
        for v in values:
            if _empty(v):
                col.append(v)
                continue
            masked, ents = cache[_text(v)]
            col.append(masked)
            if masked != _text(v):
                changed += 1
                if rule.action == "tokenize":
                    types[rule.type] += 1
                for e in ents:
                    types[e.type] += 1
        out[c] = col
        cols_report[c] = rep | {"rows": len(values), "changed": changed, "distinct": len(distinct),
                                "distinct_changed": sum(cache[t][0] != t for t in distinct), "found": dict(types)}
        if keep_details:
            details[c] = cache
    out = {c: out[c] for c in columns if c in out}
    cols_report = {c: cols_report[c] for c in columns}
    report = {
        "lindungidatamu": _version(), "patterns": patterns_version(),
        "names_model": getattr(names, "name", None) if names is not None else None,
        "names_revision": getattr(names, "revision", None) if names is not None else None,
        "rows": max((len(v) for v in columns.values()), default=0), "columns": cols_report,
        "placeholders": dict(Counter(i.type for i in vault.items)),
        "checks": check(columns, out, policy, tokenized),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return MaskedTable(out, report, vault, details)


def _missing(v) -> bool:
    if v is None or isinstance(v, (str, list, tuple, dict)):
        return v is None
    try:
        import pandas as pd
        return bool(pd.isna(v))
    except (ImportError, TypeError, ValueError):
        return False


def mask_frame(df, policy: "dict[str, Rule] | str | Path | None" = None, *, names=None, vault: Vault | None = None,
               **options) -> MaskedTable:
    """Mask a pandas DataFrame: table mode with a frame in and a frame out (`.frame`).

    policy    a policy (dict, or a policy file to load); profiled from the frame if not given. For data
              that matters, profile once, check the policy, and pass it every time.
    Columns whose values did not change keep their dtype; the others become text (object).
    """
    import pandas as pd
    names_of = {c: str(c) for c in df.columns}
    if len(set(names_of.values())) < len(names_of):
        raise ValueError("mask_frame: column names must be unique")
    columns = {names_of[c]: [None if _missing(v) else v for v in df[c].tolist()] for c in df.columns}
    if policy is None:
        policy = profile(columns, names=names)
    elif isinstance(policy, (str, Path)):
        policy = load_policy(policy)
    done = mask_table(columns, policy, vault=vault, names=names, **options)
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        values = done.columns.get(names_of[c])
        if values is None:
            continue                                         # dropped
        out[c] = df[c] if values == columns[names_of[c]] else pd.Series(values, index=df.index, dtype=object)
    done.frame = out
    return done


def check(original: dict[str, list], masked: dict[str, list], policy: dict[str, Rule], tokenized: set[str]) -> dict:
    """Counts only. residual: what personal data is still visible (must be all zero to release);
    readable: how much was left as it was (masking too much costs the analysis)."""
    leftover_ids, pattern_hits = Counter(), Counter()
    readable = {}
    for c, values in masked.items():
        rule = policy.get(c)
        strings = set() if rule and rule.action == "tokenize" else {v for v in values if isinstance(v, str) and v}
        for s in strings:
            if s in tokenized or any(w in tokenized for w in re.findall(r"[\w.\-/]{6,}", s)):
                leftover_ids[c] += 1
            for e in find(s):
                pattern_hits[f"{c}: {e.type}"] += 1
        before = original[c]
        same = sum(1 for a, b in zip(before, values) if not _empty(a) and _text(a) == (b if isinstance(b, str) else _text(b)))
        present = sum(1 for a in before if not _empty(a))
        readable[c] = round(same / present, 4) if present else None
    return {
        "residual": {"identifier_values_visible": dict(leftover_ids), "pattern_hits": dict(pattern_hits),
                     "clean": not leftover_ids and not pattern_hits},
        "readable_share": readable,
    }
