"""Indonesian area names (provinsi, kabupaten/kota, kecamatan) and their Kemendagri codes.

A column of area names generalises like a column of codes: kecamatan -> kabupaten -> provinsi.

    areas = Areas(["Cinere", "Kec. Beji", "Sukmajaya, Depok"])
    areas.generalize("Kec. Beji", "kabupaten")      # 'Kota Depok'
    areas.generalize("Kec. Beji", "provinsi")       # 'Jawa Barat'

Names match loosely: case, punctuation, "Kab." / "Kota Adm." / "Kec." / "Prov." prefixes and the
usual province short forms (Jabar, DKI, DIY, NTT). A name used in more than one place (291 kecamatan
names are; Bogor is a kabupaten and a kota) is settled by:
    a second part after a comma            "Sukamaju, Kab. Bogor"
    the column's level                     in a column of kabupaten, "Gorontalo" is not the province
    the rest of the column                 when at least half of its other values are in one kabupaten
                                           (or provinsi), a shared name is taken to be the one there
What still can't be settled keeps only what its candidates share (Bogor: Jawa Barat) and is never
published finer than that; a value that matches nothing becomes "other".

Data: cahyadsn/wilayah (MIT; see wilayah-LICENSE.txt), Kepmendagri No 300.2.2-2138 Tahun 2025.
Villages (desa/kelurahan) are left out: 84 thousand names, too many of them shared.
"""

import gzip
import re
from collections import Counter, defaultdict
from functools import lru_cache
from importlib import resources

DEPTH = {"provinsi": 0, "kabupaten": 1, "kecamatan": 2}

# Short forms people write for provinces, by code.
_ALIASES = {
    "11": ["NAD", "NANGGROE ACEH DARUSSALAM"], "12": ["SUMUT"], "13": ["SUMBAR"], "16": ["SUMSEL"],
    "19": ["BABEL", "BANGKA BELITUNG"], "21": ["KEPRI"],
    "31": ["DKI", "DKI JAKARTA", "JAKARTA", "DAERAH KHUSUS JAKARTA"], "32": ["JABAR"], "33": ["JATENG"],
    "34": ["DIY", "DI YOGYAKARTA", "YOGYAKARTA", "JOGJA", "JOGJAKARTA", "YOGYA"], "35": ["JATIM"],
    "52": ["NTB"], "53": ["NTT"], "61": ["KALBAR"], "62": ["KALTENG"], "63": ["KALSEL"], "64": ["KALTIM"],
    "65": ["KALTARA"], "71": ["SULUT"], "72": ["SULTENG"], "73": ["SULSEL"], "74": ["SULTRA"],
    "76": ["SULBAR"], "82": ["MALUT"],
}
# A prefix that says which level a name is, normalised.
_PREFIX = re.compile(
    r"^(?:(?P<provinsi>PROVINSI|PROPINSI|PROV)|(?P<kabupaten>KABUPATEN ADMINISTRASI|KABUPATEN|KAB ADM|KAB)"
    r"|(?P<kota>KOTA ADMINISTRASI|KOTA ADM|KOTAMADYA|KODYA|KOTA)|(?P<kecamatan>KECAMATAN|KEC)) ")


def _norm(s: str) -> str:
    s = re.sub(r"['`’]", "", str(s).upper())
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", s).split())


def _kind(code: str, name: str) -> str:
    depth = code.count(".")
    if depth == 1:
        return "kota" if name.startswith("Kota") else "kabupaten"
    return ("provinsi", "kabupaten", "kecamatan")[depth]


@lru_cache(maxsize=1)
def _index():
    """(names by code, kinds by code, codes by normalised name)."""
    raw = resources.files(__package__).joinpath("wilayah.tsv.gz").read_bytes()
    names, kinds, keys = {}, {}, defaultdict(list)
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        code, name = line.split("\t")
        names[code], kinds[code] = name, _kind(code, name)
        key = _norm(name)
        if code.count(".") == 1:
            m = _PREFIX.match(key + " ")
            key = key[m.end():].strip() if m else key
        keys[key].append(code)
    for code, aliases in _ALIASES.items():
        for a in aliases:
            if code not in keys[a]:
                keys[a].append(code)
    return names, kinds, dict(keys)


def name(code: str) -> str:
    """The name of a code ('32.76' -> 'Kota Depok')."""
    return _index()[0][code]


def _lookup(part: str) -> list[str]:
    _, kinds, keys = _index()
    key = _norm(part)
    m = _PREFIX.match(key + " ")
    if m:
        kind = m.lastgroup
        found = [c for c in keys.get(key[m.end():].strip(), []) if kinds[c] == kind]
        if found:
            return found
    return list(keys.get(key, []))


def candidates(text: str) -> list[str]:
    """Every code a written area name could be. "Name, parent" keeps the ones under that parent."""
    parts = [p for p in re.split(r"[,;]", str(text)) if p.strip()]
    if not parts:
        return []
    found = _lookup(parts[0])
    for part in parts[1:]:
        parents = _lookup(part)
        narrowed = [c for c in found if any(c.startswith(p + ".") for p in parents)]
        if narrowed:
            found = narrowed
    return found


def _shared(codes: list[str]) -> str:
    """The deepest code every one of `codes` is under or equal to ('' when they share nothing)."""
    split = [c.split(".") for c in codes]
    common = []
    for segs in zip(*split):
        if len(set(segs)) > 1:
            break
        common.append(segs[0])
    return ".".join(common)


def _narrow(codes: list[str], parent, votes: Counter) -> list[str]:
    """The codes under the parent the column's other values clearly favour, else all of them."""
    groups = defaultdict(list)
    for c in codes:
        groups[parent(c)].append(c)
    if len(groups) < 2:
        return codes
    ranked = sorted(groups, key=lambda g: votes[g], reverse=True)
    top, second = votes[ranked[0]], votes[ranked[1]]
    clear = top > 0 and top >= 2 * second and top >= 0.5 * sum(votes.values())
    return groups[ranked[0]] if clear else codes


class Areas:
    """A column of area names, settled together. `generalize` publishes one value at a level."""

    def __init__(self, values):
        counts = Counter(str(v).strip() for v in values)
        cands = {t: candidates(t) for t in counts}
        unique = {t: c[0] for t, c in cands.items() if len(c) == 1}
        depths = Counter()
        for t, code in unique.items():
            depths[code.count(".")] += counts[t]
        depth = depths.most_common(1)[0][0] if depths else None
        kab, prov = Counter(), Counter()
        for t, code in unique.items():
            prov[code[:2]] += counts[t]
            if code.count(".") >= 1:
                kab[code[:5]] += counts[t]
        names = _index()[0]
        self._known = {}
        for t, cs in cands.items():
            if not cs:
                continue
            if len(cs) > 1:
                cs = [c for c in cs if c.count(".") == depth] or cs
            if len(cs) > 1 and all(c.count(".") == 2 for c in cs):
                cs = _narrow(cs, lambda c: c[:5], kab)
            if len(cs) > 1:
                cs = _narrow(cs, lambda c: c[:2], prov)
            own = max(c.count(".") for c in cs)
            if len(cs) == 1:
                label = names[cs[0]]
            elif len({names[c] for c in cs}) == 1:
                label = names[cs[0]]
            else:
                label = re.split(r"[,;]", t)[0].strip()
            self._known[t] = (_shared(cs), own, label)

    def resolved(self, value) -> bool:
        return str(value).strip() in self._known

    def generalize(self, value, level: str) -> str:
        """`value` published at kecamatan, kabupaten or provinsi: never finer than the value itself,
        nor finer than what its name settles."""
        known = self._known.get(str(value).strip())
        if known is None:
            return "other"
        code, own, label = known
        target = DEPTH[level]
        if target >= own:
            return label
        if not code:
            return "other"
        segs = code.split(".")
        return name(".".join(segs[:target + 1]))


def share_named(values, sample: int = 500) -> float:
    """The share of a sample of values that are area names."""
    texts = [str(v).strip() for v in values][:sample]
    return sum(bool(candidates(t)) for t in texts) / len(texts) if texts else 0.0
