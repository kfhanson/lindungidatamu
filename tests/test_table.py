import datetime as dt
import random

import lindungidatamu as ld
from lindungidatamu.table import Rule, load_policy, mask_table, profile, save_policy

r = random.Random(0)
PEOPLE = ["BAMBANG SUTRISNO", "SITI RAHAYU", "AGUS WIJAYA", "DEWI LESTARI", "BUDI SANTOSO"]
MEMBERS = [str(r.randint(1000000, 9999999)) for _ in range(60)]
N = 400


def table():
    rows = [{
        "claim_no": f"CLM-2024-{i:06d}",
        "patient_member_no": (m := r.choice(MEMBERS)),
        "emp_member_no": r.choice(MEMBERS),
        "patient_name": PEOPLE[MEMBERS.index(m) % len(PEOPLE)],
        "account_no": str(r.randint(10**9, 10**12)),
        "amount_paid": round(r.uniform(1e5, 5e7), 2),
        "approved": r.random() > .5,
        "service_date": dt.date(2024, 1 + i % 12, 1 + i % 28),
        "relation": r.choice(["EMPLOYEE", "SPOUSE", "CHILD"]),
        "diagnosis": r.choice(["J18.9-PNEUMONIA", "G20-PARKINSON DISEASE", "I10-HIPERTENSI"]),
        "ref_no": "3174091203880004" if i == 7 else r.choice(["BATCH-A", "BATCH-B"]),
        "notes": r.choice([None, "klaim susulan", f"hubungi 0812{r.randint(10**7, 10**8 - 1)}"]),
    } for i in range(N)]
    return {c: [row[c] for row in rows] for c in rows[0]}


class People:
    """A stand-in name model: finds the PEOPLE names."""
    name = "fake"

    def find(self, text):
        return [ld.Entity("PERSON", text.index(p), text.index(p) + len(p), p, "model", "fake") for p in PEOPLE if p in text]


def test_profile_decides_by_name_shape_and_values():
    p = profile(table(), names=People())
    assert (p["claim_no"].action, p["claim_no"].type) == ("tokenize", "CLAIM")
    assert (p["patient_member_no"].type, p["emp_member_no"].type) == ("MEMBER", "MEMBER")   # shared: joins survive
    assert (p["patient_name"].action, p["patient_name"].type) == ("tokenize", "PERSON")
    assert (p["account_no"].action, p["account_no"].type) == ("tokenize", "BANK")
    for c in ("amount_paid", "approved", "service_date"):
        assert p[c].action == "keep", c
    assert p["relation"].action == "scan_patterns"          # a short list of categories
    assert p["diagnosis"].action == "scan_patterns"         # ICD-coded: no name model
    assert p["notes"].action == "scan_text"


def test_mask_table_is_consistent_and_checked():
    cols = table()
    done = mask_table(cols, profile(cols, names=People()), names=People())
    out = done.columns
    assert all(v.startswith("[CLAIM_") for v in out["claim_no"])
    same = [i for i in range(N) if cols["patient_member_no"][i] == cols["emp_member_no"][i]]
    assert all(out["patient_member_no"][i] == out["emp_member_no"][i] for i in same)
    assert out["amount_paid"] == cols["amount_paid"] and out["relation"] == cols["relation"]
    assert out["ref_no"][7] == "[NIK_1]"                    # patterns run on every text column
    assert not any("0812" in (v or "") for v in out["notes"])
    assert done.report["checks"]["residual"]["clean"]
    assert done.report["checks"]["readable_share"]["amount_paid"] == 1.0
    assert "BAMBANG" not in str(done.report)                # the report holds counts, not values


def test_one_way_unless_the_vault_is_kept(tmp_path):
    cols = {"member_no": ["1234567", "7654321", "1234567"]}
    done = mask_table(cols, {"member_no": Rule("tokenize", "MEMBER")})
    assert done.columns["member_no"] == ["[MEMBER_1]", "[MEMBER_2]", "[MEMBER_1]"]
    done.vault.save(tmp_path / "v.json")
    assert ld.Vault.load(tmp_path / "v.json").restore("[MEMBER_2]", strict=True) == "7654321"


def test_a_value_already_tokenized_gets_the_same_placeholder_elsewhere():
    cols = {"patient_name": ["Paijo Sastro", "Siti Aminah"], "payee_name": ["Paijo Sastro", "RS Hermina"]}
    policy = {"patient_name": Rule("tokenize", "PERSON"), "payee_name": Rule("scan_text")}
    done = mask_table(cols, policy)                          # no name model: found because the patient column had it
    assert done.columns["payee_name"] == ["[PERSON_1]", "RS Hermina"]


def test_residual_check_catches_what_masking_left():
    cols = {"member_no": ["M0012345"], "note": ["see M0012345"]}
    done = mask_table(cols, {"member_no": Rule("tokenize", "MEMBER"), "note": Rule("keep")})
    assert done.report["checks"]["residual"] == {"identifier_values_visible": {"note": 1}, "pattern_hits": {},
                                                  "clean": False}


def test_policy_round_trip(tmp_path):
    p = {"a": Rule("tokenize", "MEMBER", "why"), "b": Rule("drop")}
    save_policy(p, tmp_path / "p.json")
    q = load_policy(tmp_path / "p.json")
    assert (q["a"].action, q["a"].type, q["a"].reason, q["b"].action) == ("tokenize", "MEMBER", "why", "drop")
    done = mask_table({"a": ["x1234567"], "b": ["secret"]}, q)
    assert list(done.columns) == ["a"]


def test_a_numeric_code_with_an_identifier_name_is_kept():
    assert profile({"member_type": [1, 2, 1, 1, 2] * 50})["member_type"].action == "keep"
    assert profile({"member_no": list(range(1000, 1200))})["member_no"].type == "MEMBER"
