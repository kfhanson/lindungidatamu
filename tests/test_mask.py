import lindungidatamu as ld

TEXT = "Kirim kontrak ke Budi Santoso, NIK 3174091203880004, rek. BCA 1234567890, email budi@contoh.co.id"


class Names:
    def find(self, text):
        out = []
        for w in ("Budi Santoso", "Budi"):
            i = text.find(w)
            if i >= 0 and not any(e.start <= i < e.end for e in out):
                out.append(ld.Entity("PERSON", i, i + len(w), w, "model", "fake"))
        return out


def test_mask_and_restore():
    m = ld.mask(TEXT, names=Names())
    assert m.text == "Kirim kontrak ke [PERSON_1], NIK [NIK_1], rek. BCA [BANK_1], email [EMAIL_1]"
    assert m.restore("Sudah dikirim ke [PERSON_1] ([EMAIL_1]).") == "Sudah dikirim ke Budi Santoso (budi@contoh.co.id)."


def test_restore_tolerates_rewritten_placeholders():
    m = ld.mask(TEXT, names=Names())
    assert m.restore("Person 1 and nik-1") == "Budi Santoso and 3174091203880004"
    assert m.restore("Version 2 of PERSON_9 stays") == "Version 2 of PERSON_9 stays"


def test_one_vault_per_conversation():
    v = ld.Vault()
    first = ld.mask("Budi Santoso, NIK 3174091203880004", vault=v, names=Names())
    second = ld.mask("Tolong balas ke Budi soal NIK 3174 0912 0388 0004", vault=v, names=Names())
    assert first.text == "[PERSON_1], NIK [NIK_1]"
    assert second.text == "Tolong balas ke [PERSON_1] soal NIK [NIK_1]"    # same person, same number
    assert len(v.items) == 2


def test_nothing_to_mask():
    m = ld.mask("Tolong buatkan ringkasan rapat kemarin")
    assert m.text == "Tolong buatkan ringkasan rapat kemarin" and m.entities == []


def test_placeholders_beyond_999_restore():
    v = ld.Vault()
    for i in range(1, 1201):
        v.placeholder("MEMBER", f"M{i:06d}")
    assert v.restore("[MEMBER_12] and [MEMBER_1200]") == "M000012 and M001200"
    assert ld.find("member [MEMBER_1200]") == []


def test_strict_restore_only_takes_exact_placeholders():
    m = ld.mask(TEXT, names=Names())
    assert m.restore("Person 1 wrote to [PERSON_1]", strict=True) == "Person 1 wrote to Budi Santoso"


def test_a_table_vault_keeps_two_budis_apart():
    v = ld.Vault(merge_partial=False)
    assert v.placeholder("PERSON", "Budi Santoso") == "[PERSON_1]"
    assert v.placeholder("PERSON", "Budi") == "[PERSON_2]"
    assert v.placeholder("PERSON", "budi  santoso") == "[PERSON_1]"


def test_the_earliest_partial_match_wins():
    v = ld.Vault()
    v.placeholder("PERSON", "Budi Santoso")
    v.placeholder("PERSON", "Budi Hartono")
    assert v.placeholder("PERSON", "Budi") == "[PERSON_1]"


def test_save_and_load(tmp_path):
    import stat
    v = ld.Vault(merge_partial=False)
    v.placeholder("NIK", "3174091203880004")
    v.placeholder("PERSON", "Budi")
    path = tmp_path / "vault.json"
    v.save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    w = ld.Vault.load(path)
    assert w.merge_partial is False and w.restore("[NIK_1] [PERSON_1]") == "3174091203880004 Budi"
    assert w.placeholder("PERSON", "Siti") == "[PERSON_2]"            # numbering carries on


def test_items_added_from_outside_are_seen():
    v = ld.Vault()
    v.items.append(ld.Item("[NIK_7]", "NIK", "3174091203880004"))
    assert v.placeholder("NIK", "3174 0912 0388 0004") == "[NIK_7]"
    assert v.placeholder("NIK", "3174091203880005") == "[NIK_8]"
