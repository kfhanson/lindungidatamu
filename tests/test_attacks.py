import pytest

pytest.importorskip("pandas")

from lindungidatamu import attacks


@pytest.fixture(scope="module")
def results():
    return {r.attack: r for r in attacks.run()}


def test_every_attack_works_without_the_gate(results):
    assert all(r.wins["Gate off"] for r in results.values())


@pytest.mark.parametrize("name", ["Subtraction", "Diagnosis guess", "Averaging", "Narrowing", "Dominance",
                                  "Identifier columns"])
def test_the_gate_stops_it(results, name):
    assert not results[name].wins["Gate on"], results[name]


@pytest.mark.parametrize("name", ["Subtraction", "Diagnosis guess", "Narrowing", "Dominance", "Identifier columns",
                                  "Everyone has it"])
def test_dp_mode_stops_it(results, name):
    """Averaging is left out on purpose: with no total budget it helps in DP mode (see its note)."""
    assert not results[name].wins["DP mode"], results[name]


def test_the_known_limit_is_shown(results):
    """Not blocked by the gate, and the gallery says so: if this starts passing, update the README."""
    assert results["Everyone has it"].wins["Gate on"]


def test_report(results):
    text = attacks.report(list(results.values()))
    assert text.count("\n| ") == len(results) and "not blocked" in text and "DP mode" in text
