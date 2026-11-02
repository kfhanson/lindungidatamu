import json
from pathlib import Path

import pytest

import lindungidatamu as ld

CASES = json.loads((Path(__file__).parent / "fixtures.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:40] for c in CASES])
def test_fixture(case):
    assert [[e.type, e.text] for e in ld.find(case["text"])] == case["expect"]


def test_types_filter_and_positions():
    text = "NIK 3174091203880004, email budi@contoh.co.id"
    found = ld.find(text, types={"EMAIL"})
    assert [(e.type, text[e.start:e.end]) for e in found] == [("EMAIL", "budi@contoh.co.id")]


def test_terms_and_formats():
    text = "Klaim pol-12345678 dari PT Sinar Abadi (Sinar Abadi), bukan POL-1234 atau XPOL-12345678"
    found = ld.find(text, terms=[ld.Term("CLIENT", "PT Sinar Abadi", aliases=("Sinar Abadi",))],
                    formats=[ld.Format("POLICY", "POL-########")])
    assert [(e.type, e.text) for e in found] == [("POLICY", "pol-12345678"), ("CLIENT", "PT Sinar Abadi"),
                                                  ("CLIENT", "Sinar Abadi")]


def test_names_from_a_finder_and_the_keep_list():
    class Fake:
        def find(self, text):
            return [ld.Entity(t, text.index(w), text.index(w) + len(w), w, "model", "fake")
                    for t, w in (("PERSON", "Budi Santoso"), ("ORG", "Bank Indonesia")) if w in text]
    found = ld.find("Budi Santoso lapor ke Bank Indonesia", names=Fake(), keep=["Bank Indonesia"])
    assert [(e.type, e.text) for e in found] == [("PERSON", "Budi Santoso")]


def test_placeholders_are_never_found_again():
    assert ld.find("NIK [NIK_1] dan email [EMAIL_1]") == []


def found(text, **kw):
    return [(e.type, e.text) for e in ld.find(text, **kw)]


def test_addresses_take_their_parts_but_not_the_next_sentence():
    assert found("kirim ke Jl. Melati No. 2, RT 006/RW 003, Kel. Sukajadi, Jakarta Selatan, 40162 ya") == \
        [("ADDRESS", "Jl. Melati No. 2, RT 006/RW 003, Kel. Sukajadi, Jakarta Selatan, 40162")]
    assert found("Dusun Krajan RT 002/RW 003, Desa Sukamaju, Kec. Ciawi, Kab. Bogor") == \
        [("ADDRESS", "Dusun Krajan RT 002/RW 003, Desa Sukamaju, Kec. Ciawi, Kab. Bogor")]
    assert found("Jl. Kenanga No. 7, Budi minta dikirim besok") == [("ADDRESS", "Jl. Kenanga No. 7")]


def test_phone_formats():
    for p in ("0814.6502.45997", "+62 878 7675 911", "+62 812-3456-7890", "62 8123456789"):
        assert found(f"hp {p} ya") == [("PHONE", p)], p
    assert found("tanggal 0812 dan 2026") == []


def test_words_between_a_label_and_its_number():
    assert found("rek gaji Mandiri 028-160-4247651") == [("BANK", "028-160-4247651")]
    assert found("ke rekening BSI nomor 408-972-8046464 atas nama PT") == [("BANK", "408-972-8046464")]
    assert found("bpjs aku 0001871888514 masih aktif") == [("BPJS", "0001871888514")]
    assert found("Tax ID (NPWP): 567688406185182") == [("NPWP", "567688406185182")]
    assert found("rekening sebesar Rp 1.500.000") == []


class Names:
    """A stand-in for a name model, reporting what GLiNER reported on the benchmark."""
    def __init__(self, *found):
        self.found = found

    def find(self, text):
        return [ld.Entity(t, text.index(w), text.index(w) + len(w), w, "model", "fake") for t, w in self.found if w in text]


def test_forms_of_address_titles_and_public_organisations_are_not_masked():
    text = "kak, Bapak Budi Santoso transfer via CIMB Niaga, dilaporkan ke OJK oleh PT Maju Jaya"
    names = Names(("PERSON", "kak"), ("PERSON", "Bapak Budi Santoso"), ("ORG", "CIMB Niaga"), ("ORG", "OJK"),
                  ("ORG", "PT Maju Jaya"))
    assert found(text, names=names) == [("PERSON", "Budi Santoso"), ("ORG", "PT Maju Jaya")]
    assert ("ORG", "OJK") in found(text, names=names, public=False)


def test_ordinary_phrases_and_dressed_up_public_bodies_are_not_organisations():
    text = "min akun aku dibajak, bayar ke VA BTN, sy pesan lewat PT Maju Jaya"
    names = Names(("ORG", "akun aku dibajak"), ("ORG", "VA BTN"), ("PERSON", "sy"), ("ORG", "PT Maju Jaya"))
    assert found(text, names=names) == [("ORG", "PT Maju Jaya")]
