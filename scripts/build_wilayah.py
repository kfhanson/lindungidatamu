"""Rebuild src/lindungidatamu/wilayah.tsv.gz from cahyadsn/wilayah's db/wilayah.sql.

    curl -LO https://raw.githubusercontent.com/cahyadsn/wilayah/<commit>/db/wilayah.sql
    python scripts/build_wilayah.py wilayah.sql <commit>

Keeps provinsi, kabupaten/kota and kecamatan (codes with up to two dots); villages are left out.
The source is MIT licensed: its notice ships as src/lindungidatamu/wilayah-LICENSE.txt.
"""

import gzip
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "lindungidatamu" / "wilayah.tsv.gz"


def main(sql: str, commit: str) -> None:
    text = Path(sql).read_text(encoding="utf-8")
    source = re.search(r"Kepmendagri[^\n]*", text)
    rows = re.findall(r"\('([0-9.]+)','((?:[^'\\]|\\.|'')*)'\)", text)
    keep = sorted((code, re.sub(r"\s+", " ", name.replace("''", "'")).strip())
                  for code, name in rows if code.count(".") <= 2)
    if not keep or any("\t" in n or "\n" in n for _, n in keep):
        raise SystemExit("unexpected input")
    lines = [
        "# Kode wilayah: provinsi, kabupaten/kota, kecamatan",
        f"# {source[0].strip() if source else 'Kemendagri'}",
        f"# from https://github.com/cahyadsn/wilayah db/wilayah.sql at {commit}",
        "# Copyright (c) 2017-2025 Cahya DSN, MIT License (see wilayah-LICENSE.txt)",
    ] + [f"{c}\t{n}" for c, n in keep]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    OUT.write_bytes(gzip.compress(data, mtime=0))
    levels = [sum(c.count(".") == d for c, _ in keep) for d in range(3)]
    print(f"{OUT.name}: {levels[0]} provinsi, {levels[1]} kabupaten/kota, {levels[2]} kecamatan, "
          f"{OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
