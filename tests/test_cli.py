import json

import pytest

from lindungidatamu.cli import main

pytest.importorskip("pandas")

CSV = """claim_no,member_no,account_no,amount_paid,notes
CLM-001,0012345,1234567890,150000,klaim susulan
CLM-002,0067890,9876543210,250000,hubungi 081234567890
CLM-003,0012345,1234567890,90000,
"""


def test_scan_mask_restore(tmp_path, capsys):
    src = tmp_path / "claims.csv"
    src.write_text(CSV)
    assert main(["scan", str(src)]) == 0
    policy = json.loads((tmp_path / "claims.policy.json").read_text())["columns"]
    assert policy["member_no"]["type"] == "MEMBER" and policy["amount_paid"]["action"] == "scan_patterns"

    assert main(["mask", str(src), "--vault", str(tmp_path / "v.json")]) == 0
    out = (tmp_path / "claims.masked.csv").read_text()
    assert "0012345" not in out and "1234567890" not in out and "081234567890" not in out
    assert "[MEMBER_1]" in out and "150000" in out
    report = json.loads((tmp_path / "claims.masked.report.json").read_text())
    assert report["checks"]["residual"]["clean"] and report["reversible"] is True

    note = tmp_path / "findings.md"
    note.write_text("[MEMBER_1] claimed twice; member 2 once.")
    assert main(["restore", str(note), "--vault", str(tmp_path / "v.json")]) == 0
    assert (tmp_path / "findings.restored.md").read_text() == "0012345 claimed twice; member 2 once."


def test_one_way_by_default(tmp_path, capsys):
    src = tmp_path / "c.csv"
    src.write_text(CSV)
    assert main(["mask", str(src)]) == 0
    assert "one-way" in capsys.readouterr().out
    assert not list(tmp_path.glob("*vault*"))


def test_text_file(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("NIK 3174091203880004, hp 081234567890")
    assert main(["mask", str(src)]) == 0
    assert (tmp_path / "note.masked.txt").read_text() == "NIK [NIK_1], hp [PHONE_1]"


def test_hide_columns(tmp_path, capsys):
    src = tmp_path / "c.csv"
    src.write_text(CSV)
    assert main(["scan", str(src), "--hide-columns"]) == 0
    out = capsys.readouterr().out
    assert "member_no" not in out.split("policy:")[0] and "| C2 |" in out


def test_hidden_column_numbers_match_between_scan_and_mask(tmp_path, capsys):
    src = tmp_path / "c.csv"
    src.write_text(CSV)
    main(["scan", str(src), "--hide-columns"])
    main(["mask", str(src), "--hide-columns"])
    out = capsys.readouterr().out
    assert "| C2 | tokenize MEMBER |" in out and "| C3 | tokenize BANK |" in out
