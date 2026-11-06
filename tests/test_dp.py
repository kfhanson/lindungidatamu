import collections
import math
from fractions import Fraction

import pytest

pytest.importorskip("pandas")

from lindungidatamu import dp
from lindungidatamu.analysis import Analysis, QueryError
from lindungidatamu.attacks import claims


@pytest.fixture(scope="module")
def table():
    return claims()


def test_discrete_laplace_has_the_right_shape():
    """P(y) is proportional to exp(-|y|/b): check P(0), symmetry and the variance."""
    b = Fraction(3, 2)
    draws = [dp.discrete_laplace(b) for _ in range(20000)]
    q = math.exp(-1 / float(b))
    p0 = (1 - q) / (1 + q)
    var = 2 * q / (1 - q) ** 2
    counts = collections.Counter(draws)
    assert abs(counts[0] / len(draws) - p0) < 0.02
    assert abs(sum(draws) / len(draws)) < 0.1
    assert abs(sum(d * d for d in draws) / len(draws) - var) < 0.15 * var
    assert abs(counts[2] - counts[-2]) < 0.1 * counts[2]


def test_budget_runs_out_and_repeats_are_free(table):
    a = Analysis(table, dp=0.3, dp_question=0.1)
    q = dict(measures=["people"], where={"column": "sex", "=": "L"})
    first = a.aggregate(**q)
    assert a.aggregate(**q).rows == first.rows and a.budget.spent == pytest.approx(0.1)
    a.aggregate(measures="rows")
    a.aggregate(measures="rows", where={"column": "sex", "=": "P"})
    with pytest.raises(QueryError, match="used up"):
        a.aggregate(measures="rows", where={"column": "sex", "=": "L"})
    assert "refused" in a.log[-1]
    with pytest.raises(QueryError, match="epsilon"):
        Analysis(table, dp=0.3).aggregate(measures="rows", epsilon=0.5)


def test_budget_outlives_a_session(table, tmp_path):
    state = tmp_path / "claims.analysis"
    a = Analysis(table, dp=1.0, state=state)
    first = a.aggregate(measures="people")
    b = Analysis(table, dp=1.0, state=state)
    assert b.budget.spent == pytest.approx(0.1) and b.aggregate(measures="people").rows == first.rows
    assert b.budget.spent == pytest.approx(0.1)
    assert (state / "budget.json").stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="budget of epsilon 1"):
        Analysis(table, dp=2.0, state=state)
    with pytest.raises(ValueError, match="DP mode"):
        Analysis(table, state=state)                     # the SDC gate's answers aren't in the budget
    sdc = tmp_path / "sdc.analysis"
    Analysis(table, state=sdc).aggregate(group_by="sex")
    with pytest.raises(ValueError, match="DP mode"):
        Analysis(table, dp=1.0, state=sdc)


def test_listed_groups_all_show(table):
    a = Analysis(table, dp=5.0, dp_question=1.0)
    r = a.aggregate(group_by=[{"column": "diagnosis", "values": ["J18.9", "I10", "XX.X"]}], measures="people",
                    groups_per_person=3)
    assert [row[0] for row in r.rows] == ["J18.9", "I10", "XX.X"] and r.rows[2][1] < 15
    assert r.privacy["delta"] == 0
    with pytest.raises(QueryError, match="every group_by column"):
        a.aggregate(group_by=[{"column": "diagnosis", "values": ["I10"]}, "sex"])


def test_unlisted_groups_need_many_people(table):
    a = Analysis(table, dp=5.0, dp_question=1.0)
    r = a.aggregate(group_by=["kelurahan"], measures="people")
    threshold = 1 + 1 * math.log(1 / 1e-6)               # scale G/eps = 1
    assert r.privacy["delta"] == 1e-6 and all(people >= threshold for _, people in r.rows)
    assert len(r.rows) < 108


def test_bounding_one_group_per_person(table):
    a = Analysis(table, dp=10.0, dp_question=2.0)
    quarters = {"column": "admitted", "by": "quarter", "values": ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]}
    one = sum(r[1] for r in a.aggregate(group_by=[quarters], measures="rows").rows)
    four = sum(r[1] for r in a.aggregate(group_by=[quarters], measures="rows", groups_per_person=4,
                                         rows_per_person=6).rows)
    assert abs(four - len(table["claim_no"])) < 200 and one < 0.7 * four


def test_sums_need_bounds_and_clip(table):
    a = Analysis(table, dp=10.0, dp_question=5.0)
    with pytest.raises(QueryError, match="upper"):
        a.aggregate(measures=[{"sum": "amount_paid"}])
    r = a.aggregate(measures=["rows", {"mean": "amount_paid", "upper": 1_000_000}])
    assert r.rows[0][1] <= 1_000_000                       # everything above was clipped


def test_describe_shows_no_values(table):
    d = Analysis(table, dp=1.0).describe()
    assert d["privacy"]["left"] == 1.0 and "people" not in d
    assert all("values" not in c and "rare_values" not in c for c in d["columns"])


def test_dp_options_only_in_dp_mode(table):
    a = Analysis(table, secret="x")
    with pytest.raises(QueryError, match="only in DP mode"):
        a.aggregate(measures="rows", epsilon=0.1)
    with pytest.raises(QueryError, match="only in DP mode"):
        a.aggregate(group_by=[{"column": "sex", "values": ["L"]}])


def test_mcp_server_in_dp_mode(table):
    from lindungidatamu import serve
    a = Analysis(table, dp=1.0)
    init = serve.handle(a, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert "differential\nprivacy" in init["result"]["instructions"]
    r = serve.handle(a, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "aggregate", "arguments": {"group_by": [{"column": "sex", "values": ["L", "P"]}],
                                           "measures": ["people"], "epsilon": 0.2}}})
    assert not r["result"]["isError"] and a.budget.spent == pytest.approx(0.2)
