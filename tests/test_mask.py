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
