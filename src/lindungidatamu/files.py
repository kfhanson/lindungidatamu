"""Reading and writing the files the command line works on (pip install "lindungidatamu[files]").

Tables: CSV, TSV, Parquet, JSON Lines, Excel. Text: .txt, .md, .log and anything else read as text.
CSV and Excel are read as text, so identifiers keep their leading zeros (0812..., 0001234567890).
"""

from pathlib import Path

TABLES = {".csv", ".tsv", ".parquet", ".pq", ".jsonl", ".ndjson", ".xlsx", ".xls"}


def is_table(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TABLES


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError('tables need pandas: pip install "lindungidatamu[files]"') from e
    return pd


def read_table(path: str | Path) -> dict[str, list]:
    """Column name -> values (None where empty)."""
    pd = _pandas()
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(path, dtype=str, keep_default_na=False, sep="\t" if ext == ".tsv" else ",")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True, dtype=False)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"not a table: {path.name}")
    df.columns = [str(c) for c in df.columns]
    return {c: [None if v == "" else v for v in df[c].tolist()] for c in df.columns}


def _as_text(v):
    if v is None or isinstance(v, str) or type(v).__name__ in ("NAType", "NaTType") or (isinstance(v, float) and v != v):
        return None if not isinstance(v, str) else v
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def write_table(columns: dict[str, list], path: str | Path) -> None:
    pd = _pandas()
    path = Path(path)
    ext = path.suffix.lower()
    df = pd.DataFrame(columns)
    if ext in (".csv", ".tsv"):
        df.to_csv(path, index=False, sep="\t" if ext == ".tsv" else ",")
    elif ext in (".parquet", ".pq"):
        # a column that mixes placeholders (text) with numbers is written as text
        for c in df.columns:
            if df[c].dtype == object and df[c].map(type).nunique() > 1:
                df[c] = df[c].map(_as_text)
        df.to_parquet(path, index=False)
    elif ext in (".jsonl", ".ndjson"):
        df.to_json(path, orient="records", lines=True, force_ascii=False, date_format="iso")
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"not a table: {path.name}")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(text: str, path: str | Path) -> None:
    Path(path).write_text(text, encoding="utf-8")
