import collections
import random

import pytest

pytest.importorskip("pandas")

from lindungidatamu.analysis import Analysis, QueryError
from test_risk import claims


@pytest.fixture(scope="module")
def table():
    return claims()


@pytest.fixture
def a(table):
    return Analysis(table, k=5, secret="test")


def test_describe_hides_identifiers(a):
    d = a.describe()
    hidden = {h["name"]: h["why"] for h in d["hidden"]}
    assert hidden["claim_no"] == "identifier" and "count people" in hidden["member_no"]
    kinds = {c["name"]: c["kind"] for c in d["columns"]}
    assert kinds == {"birth_date": "date", "sex": "category", "kelurahan": "category", "admitted": "date",
                     "diagnosis": "category", "amount_paid": "number"}
    diag = next(c for c in d["columns"] if c["name"] == "diagnosis")
    assert "I10" in diag["values"] and "chapter" in diag["group_by"]
    assert abs(d["people"] - 1500) <= 2


def test_aggregate_counts_people(a, table):
    r = a.aggregate(group_by="sex", measures=["people", "rows", {"mean": "amount_paid"}])
    assert r.columns == ["sex", "people", "rows", "mean(amount_paid)"]
    true = collections.defaultdict(set)
    for s, m in zip(table["sex"], table["member_no"]):
        true[s].add(m)
    for sex, people, rows, mean in r.rows:
        assert abs(people - len(true[sex])) <= 2 and rows > people and mean > 0


def test_groups_bins_and_filters(a):
    r = a.aggregate(group_by=[{"column": "kelurahan", "by": "provinsi"}, {"column": "admitted", "by": "quarter"}],
                    where=[{"column": "amount_paid", ">=": 1_000_000}, {"column": "sex", "=": "P"}])
    assert {row[0] for row in r.rows} <= {"31", "32", "35"} and {row[1] for row in r.rows} <= \
        {"2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"}
    r = a.aggregate(group_by={"column": "amount_paid", "width": 5_000_000}, measures="rows")
    assert r.rows[0][0] == "0 to <5e+06" or r.rows[0][0].startswith("0 to")
    r = a.aggregate(group_by={"column": "diagnosis", "by": "chapter"}, where={"column": "diagnosis",
                                                                                "in": ["G20", "F20.0", "I10"]})
    assert {row[0] for row in r.rows} <= {"G00-G99", "F00-F99", "I00-I99"}


@pytest.mark.parametrize("q, why", [
    ({"group_by": "member_no"}, "count people"),
    ({"group_by": "claim_no"}, "identifier"),
    ({"group_by": "amount_paid"}, "bins"),
    ({"group_by": "admitted"}, "month"),
    ({"measures": {"sum": "sex"}}, "number column"),
    ({"measures": "max"}, "unknown measure"),
    ({"where": {"column": "sex", "<": "L"}}, "category"),
    ({"where": {"column": "admitted", ">": "soon"}}, "YYYY-MM-DD"),
])
def test_refusals(a, q, why):
    with pytest.raises(QueryError, match=why):
        a.aggregate(**q)


def test_too_many_groups(table):
    a = Analysis(table, k=5, secret="test", max_groups=20)
    with pytest.raises(QueryError, match="at most 20"):
        a.aggregate(group_by="kelurahan")


def test_small_groups_left_out(a):
    r = a.aggregate(group_by=["kelurahan", "sex"])
    assert r.hidden_groups > 0 and all(people >= 3 for *_, people in r.rows)
    assert "fewer than 5 people" in r.notes[0]


def test_dominance_withholds_sums():
    t = {"member_no": [f"M{i}" for i in range(20)], "region": ["A"] * 20, "paid": [1000] * 19 + [10_000_000]}
    r = Analysis(t, k=5, secret="s").aggregate(group_by="region", measures=["people", {"sum": "paid"}, {"mean": "paid"}])
    assert r.rows == [["A", r.rows[0][1], None, None]] and r.withheld == 2


def test_same_people_same_answer(table):
    a = Analysis(table, k=5, secret="test")
    one = a.aggregate(measures=["people", {"sum": "amount_paid"}], where={"column": "sex", "=": "L"}).rows
    two = a.aggregate(measures=["people", {"sum": "amount_paid"}], where={"column": "sex", "in": ["L"]}).rows
    assert one == two
    again = Analysis(table, k=5, secret="test").aggregate(measures=["people", {"sum": "amount_paid"}],
                                                          where={"column": "sex", "=": "L"}).rows
    assert again == one
    other = Analysis(table, k=5, secret="other").aggregate(measures=["people", {"sum": "amount_paid"}],
                                                           where={"column": "sex", "=": "L"}).rows
    assert other != one


def test_differencing_gives_nothing(table):
    """Everyone in a kelurahan, then the same without one claim: the two answers are equal, whether or
    not that claim was in the group, so the difference says nothing about it."""
    a = Analysis(table, k=5, secret="test")
    amounts = collections.Counter(table["amount_paid"])
    seen = set()
    pairs = 0
    for kel, amt, diag in zip(table["kelurahan"], table["amount_paid"], table["diagnosis"]):
        if kel in seen or amounts[amt] != 1:
            continue
        seen.add(kel)
        base = [{"column": "kelurahan", "=": kel}]
        without = [{"column": "amount_paid", "!=": amt}]
        q1 = a.aggregate(measures=["rows", {"sum": "amount_paid"}], where=base)
        q2 = a.aggregate(measures=["rows", {"sum": "amount_paid"}], where=base + without)
        assert q1.rows == q2.rows
        # a diagnosis the target may or may not have, asked in either order: the same answers either way
        w = base + [{"column": "diagnosis", "=": "J18.9"}]
        order = (w, w + without) if len(seen) % 2 else (w + without, w)
        c1, c2 = (a.aggregate(measures="rows", where=q).rows for q in order)
        assert c1 == c2
        pairs += 1
    assert pairs > 20


def test_log_is_counts_only(a):
    a.aggregate(group_by="sex")
    entry = a.log[-1]
    assert set(entry) == {"at", "query", "groups_shown", "groups_hidden", "reused", "withheld", "seconds"}


def test_refusals_are_logged(a):
    with pytest.raises(QueryError):
        a.aggregate(group_by="member_no")
    assert "refused" in a.log[-1]


def test_state_outlives_a_session(table, tmp_path):
    """The secret and the groups answered are kept: a restarted session answers the same, and
    "the same group without one claim" asked after a restart still gets the earlier answer."""
    state = tmp_path / "claims.analysis"
    kel = table["kelurahan"][0]
    amt = table["amount_paid"][0]
    first = Analysis(table, k=5, state=state).aggregate(measures=["rows", {"sum": "amount_paid"}],
                                                        where={"column": "kelurahan", "=": kel})
    assert (state / "secret").stat().st_mode & 0o077 == 0
    again = Analysis(table, k=5, state=state)
    assert again.aggregate(measures=["rows", {"sum": "amount_paid"}],
                           where=[{"column": "kelurahan", "=": kel}, {"column": "amount_paid", "!=": amt}]).rows == \
        first.rows
    assert len((state / "log.jsonl").read_text().splitlines()) == 2
    other = dict(table, amount_paid=[v + 1 for v in table["amount_paid"]], member_no=table["member_no"][::-1])
    with pytest.raises(ValueError, match="different table"):
        Analysis(other, k=5, state=state)


def test_mcp_server(table):
    import io
    import json
    from lindungidatamu import serve
    a = Analysis(table, k=5, secret="test")
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "describe_table"}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "aggregate", "arguments": {"group_by": ["sex"], "measures": ["people"]}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "aggregate", "arguments": {"group_by": ["claim_no"]}}},
            {"jsonrpc": "2.0", "id": 6, "method": "nope"}]
    out = io.StringIO()
    serve.run(a, io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\nnot json\n"), out)
    replies = [json.loads(l) for l in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, 2, 3, 4, 5, 6, None]
    assert replies[0]["result"]["capabilities"] == {"tools": {}}
    assert [t["name"] for t in replies[1]["result"]["tools"]] == ["describe_table", "aggregate"]
    assert "member_no" not in replies[2]["result"]["content"][0]["text"].split('"columns"')[1].split('"hidden"')[0]
    assert "| sex | people |" in replies[3]["result"]["content"][0]["text"]
    assert replies[4]["result"]["isError"] and "identifier" in replies[4]["result"]["content"][0]["text"]
    assert replies[5]["error"]["code"] == -32601 and replies[6]["error"]["code"] == -32700
