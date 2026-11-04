import datetime as dt
import json
import random

import pytest

from lindungidatamu import anonymity as an
from lindungidatamu.table import Rule, mask_table, profile, risk


def claims(n_members: int = 1500, seed: int = 7) -> dict[str, list]:
    """A made-up claims table with realistic skew: crowded and sparse areas, common and rare diagnoses,
    members with one to six claims."""
    rng = random.Random(seed)
    kelurahan = [f"{p}.{k:02d}.{c:02d}.{1000 + v}" for p in (31, 32, 35) for k in (1, 2, 71) for c in (1, 2, 3)
                 for v in range(4)]
    area_w = [rng.paretovariate(1.2) for _ in kelurahan]
    common = ["J18.9", "E11.9", "I10", "K29.7", "A09", "J06.9", "N39.0", "M54.5"]
    rare = ["G20", "C50.9", "Q90.9", "B20", "F20.0", "D57.1"]
    cols = {c: [] for c in ("claim_no", "member_no", "birth_date", "sex", "kelurahan", "admitted", "diagnosis",
                            "amount_paid")}
    claim = 0
    for m in range(n_members):
        member = f"{100000000 + m * 37:013d}"
        born = dt.date(1945, 1, 1) + dt.timedelta(days=rng.randrange(0, 65 * 365))
        sex = rng.choice("LP")
        area = rng.choices(kelurahan, area_w)[0]
        for _ in range(rng.choice((1, 1, 1, 2, 2, 3, 6))):
            claim += 1
            cols["claim_no"].append(f"CLM-{claim:07d}")
            cols["member_no"].append(member)
            cols["birth_date"].append(born)
            cols["sex"].append(sex)
            cols["kelurahan"].append(area)
            cols["admitted"].append(dt.date(2024, 1, 1) + dt.timedelta(days=rng.randrange(366)))
            cols["diagnosis"].append(rng.choice(rare) if rng.random() < 0.03 else rng.choice(common))
            cols["amount_paid"].append(rng.randrange(100, 20000) * 1000)
    return cols


def test_measure_counts_groups():
    cols = {"sex": ["L", "L", "P", "P", "P", "L"], "year": ["1980", "1980", "1990", "1990", "1990", "1975"],
            "dx": ["A", "B", "C", "C", "C", "D"]}
    m = an.measure(cols, ["sex", "year"], ["dx"], k=2)
    assert m["groups"] == 3 and m["k"] == 1 and m["unique_rows"] == 1 and m["rows_below_k"] == 1
    assert m["average_risk"] == round((2 * 0.5 + 3 / 3 + 1) / 6, 4) and m["highest_risk"] == 1.0
    assert m["unique_share_without"]["year"] == 0.0
    # The group of three women born in 1990 all have C: you learn it without knowing the row.
    assert m["sensitive"]["dx"] == {"l": 1, "groups_one_value": 1, "rows_in_groups_one_value": 3}


def test_groups_count_people_not_rows():
    cols = {"member": ["m1"] * 5 + ["m2", "m3"], "year": ["1980"] * 5 + ["1990", "1990"]}
    assert an.measure(cols, ["year"], k=2)["k"] == 2
    by_person = an.measure(cols, ["year"], k=2, person="member")
    assert by_person["k"] == 1 and by_person["counted"] == "people"     # five claims of one patient


def test_hierarchies():
    d = [dt.date(1987, 3, 14), "14/03/1987", "1987-03-14", None]
    assert an.generalize(d, "date", "month") == ["1987-03", "1987-03", "1987-03", None]
    assert an.generalize(d, "date", "5 years")[:3] == ["1985-1989"] * 3
    assert an.generalize([34, 35.0, "61"], "age", "10 years") == ["30-39", "30-39", "60-69"]
    assert an.generalize(["32.76.06.1004", "3276061004"], "wilayah", "kabupaten") == ["32.76", "32.76"]
    assert an.generalize(["32.76.06.1004"], "wilayah", "kecamatan") == ["32.76.06"]
    assert an.generalize(["16423"], "postal", "3 digits") == ["164xx"]
    assert an.generalize(["G20-PARKINSON", "J18.9 PNEUMONIA", "S72.0", "T14", "x"], "icd10", "chapter") == \
        ["G00-G99", "J00-J99", "S00-T98", "S00-T98", "other"]
    assert an.generalize(["J18.9 PNEUMONIA"], "icd10", "category") == ["J18"]
    vals = ["a"] * 200 + ["b"]
    assert an.generalize(vals, "category", "common")[-1] == "other"
    assert an.generalize(["L", None], "sex", "*") == ["*", None]
    with pytest.raises(ValueError):
        an.generalize(d, "date", "decade")


def test_guess_hierarchy():
    t = claims(50)
    assert an.guess_hierarchy(t["birth_date"]) == "date"
    assert an.guess_hierarchy(t["kelurahan"]) == "wilayah"
    assert an.guess_hierarchy(t["diagnosis"]) == "icd10"
    assert an.guess_hierarchy(t["sex"]) == "sex"
    assert an.guess_hierarchy(["16423", "16424"] * 10) == "postal"


def test_suggest_reaches_k_with_the_least_change():
    t = claims()
    h = {"birth_date": "date", "sex": "sex", "kelurahan": "wilayah", "admitted": "date"}
    before = an.measure(t, list(h), k=5, person="member_no")
    assert before["k"] == 1 and before["unique_share"] > 0.9
    plan = an.suggest(t, h, k=5, max_suppress=0.01, person="member_no")
    assert plan.reached and plan.rows_suppressed <= 0.01 * len(t["member_no"])
    done, suppressed = an.apply(t, plan.levels, h, k=5, person="member_no")
    after = an.measure(done, list(h), k=5, person="member_no", skip=suppressed)
    assert after["k"] >= 5 and after["unique_rows"] == 0
    assert sum(suppressed) == plan.rows_suppressed
    # Minimal: one step less general on any column no longer reaches k within the budget.
    budget = int(0.01 * len(t["member_no"]))
    for c, lvl in plan.levels.items():
        i = an.LEVELS[h[c]].index(lvl)
        if i == 0:
            continue
        lower = dict(plan.levels, **{c: an.LEVELS[h[c]][i - 1]})
        cols = [an._codes(an.generalize(t[q], h[q], lower[q])) for q in h]
        _, sizes = an._group(cols, an._codes(t["member_no"]))
        assert an._below(sizes, 5) > budget, c


def test_suggest_respects_keep_value_and_reports_unreachable():
    t = claims(200)
    h = {"birth_date": "date", "sex": "sex"}
    plan = an.suggest(t, h, k=5, keep_value=("birth_date",), max_suppress=0.0, person="member_no")
    assert plan.levels["birth_date"] == "value" and not plan.reached


def test_profile_marks_quasi_and_sensitive():
    t = claims(100)
    p = profile(t)
    assert (p["birth_date"].role, p["birth_date"].hierarchy) == ("quasi", "date")
    assert (p["admitted"].role, p["admitted"].hierarchy) == ("quasi", "date")
    assert (p["kelurahan"].role, p["kelurahan"].hierarchy) == ("quasi", "wilayah")
    assert (p["sex"].role, p["sex"].hierarchy) == ("quasi", "sex")
    assert p["diagnosis"].role == "sensitive"
    assert p["member_no"].role is None and p["amount_paid"].role is None


def test_risk_then_mask_applies_the_plan():
    t = claims()
    r = risk(t, k=5)
    assert r.person == "member_no" and r.plan.reached
    assert r.before["k"] == 1 and r.after["k"] >= 5
    assert r.policy["birth_date"].generalize in an.LEVELS["date"]
    done = mask_table(t, r.policy, k=5)
    rk = done.report["risk"]
    assert rk["after"]["k"] >= 5 and rk["generalized"] == {c: v for c, v in r.plan.levels.items() if v != "value"}
    assert done.report["checks"]["residual"]["clean"]
    born = done.columns["birth_date"]
    lvl = r.plan.levels["birth_date"]
    if lvl == "year":
        assert all(v in ("*",) or len(v) == 4 for v in born)
    assert all(str(m).startswith("[") for m in done.columns["member_no"])
    assert done.columns["amount_paid"] == t["amount_paid"]          # analysis columns untouched


def test_mask_table_measures_without_changing_when_nothing_generalised():
    t = claims(100)
    p = profile(t)
    done = mask_table(t, p)
    assert done.columns["birth_date"] == t["birth_date"]
    assert done.report["risk"]["generalized"] == {} and done.report["risk"]["before"]["k"] == 1


def test_rule_validates_levels():
    with pytest.raises(ValueError):
        Rule("keep", role="quasi", hierarchy="date", generalize="kabupaten")
    with pytest.raises(ValueError):
        Rule("keep", role="maybe")


def test_cli_risk_writes_the_plan(tmp_path, capsys):
    pytest.importorskip("pandas")
    from lindungidatamu.cli import main
    import pandas as pd
    t = claims(600)
    src = tmp_path / "claims.csv"
    pd.DataFrame(t).to_csv(src, index=False)
    assert main(["risk", str(src)]) == 0
    out = capsys.readouterr().out
    assert "Now:   k = 1" in out and "After: k = " in out
    doc = json.loads((tmp_path / "claims.policy.json").read_text())
    assert doc["risk"] == {"k": 5, "max_suppress": 0.01, "person": "member_no"}
    assert any(c.get("generalize") for c in doc["columns"].values())
    assert main(["mask", str(src)]) == 0
    report = json.loads((tmp_path / "claims.masked.report.json").read_text())
    assert report["risk"]["after"]["k"] >= 5 and report["risk"]["generalized"]


def test_limit_keeps_what_the_analysis_needs():
    t = claims()
    h = {"birth_date": "date", "sex": "sex", "kelurahan": "wilayah", "admitted": "date"}
    plan = an.suggest(t, h, k=5, person="member_no", limit={"birth_date": "5 years"})
    assert plan.levels["birth_date"] in ("value", "month", "year", "5 years") and plan.reached
    with pytest.raises(ValueError):
        an.suggest(t, h, limit={"birth_date": "kabupaten"})
    r = risk(t, keep={"birth_date": "year"})
    assert r.policy["birth_date"].generalize in (None, "month", "year")
