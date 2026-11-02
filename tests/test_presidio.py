import pytest

pytest.importorskip("presidio_analyzer")


def test_presidio_recognisers_report_indonesian_types():
    # The recogniser directly: an AnalyzerEngine would also load a large spaCy model.
    from lindungidatamu.presidio import recognizers
    r = recognizers(("en",))[0]
    found = r.analyze("NIK 3174091203880004, NPWP 01.234.567.8-901.000, kartu 4532 0151 8372 9646", entities=[])
    assert sorted((x.entity_type, x.start, x.end) for x in found) == \
        [("CREDIT_CARD", 55, 74), ("ID_NIK", 4, 20), ("ID_NPWP", 27, 47)]
    assert [x.entity_type for x in r.analyze("NIK 3174091203880004 kartu 4532 0151 8372 9646", entities=["ID_NIK"])] == ["ID_NIK"]
