import logging

import pytest

import lindungidatamu as ld


def _logger(handler_filter):
    stream = []

    class Keep(logging.Handler):
        def emit(self, record):
            stream.append(self.format(record))
    h = Keep()
    h.setFormatter(logging.Formatter("%(message)s"))
    h.addFilter(handler_filter)
    log = logging.getLogger(f"t{id(stream)}")
    log.propagate = False
    log.addHandler(h)
    log.setLevel(logging.INFO)
    return log, stream


def test_log_filter_masks_message_args_and_tracebacks():
    log, out = _logger(ld.LogFilter())
    log.info("refund ke rek. %s untuk NIK %s", "1234567890", "3174091203880004")
    try:
        raise ValueError("hp 081234567890 tidak aktif")
    except ValueError:
        log.exception("gagal")
    assert out[0] == "refund ke rek. [BANK_1] untuk NIK [NIK_1]"
    assert "081234567890" not in out[1] and "[PHONE_1]" in out[1] and "Traceback" in out[1]


def test_log_filter_one_way_unless_given_a_vault():
    log, out = _logger(ld.LogFilter())
    log.info("hp 081234567890")
    log.info("hp 081299998888")
    assert out == ["hp [PHONE_1]", "hp [PHONE_1]"]            # nothing kept, so no numbering across lines
    vault = ld.Vault()
    log, out = _logger(ld.LogFilter(vault=vault))
    log.info("hp 081234567890")
    log.info("hp 081299998888")
    assert out == ["hp [PHONE_1]", "hp [PHONE_2]"] and vault.value("PHONE", 2) == "081299998888"


def test_mask_frame_keeps_dtypes_of_untouched_columns():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "member_no": ["0012345", "0067890", "0012345"] * 20,
        "amount_paid": [150000.0, 250000.5, 90000.0] * 20,
        "notes": ["klaim susulan", "hubungi 081234567890", None] * 20,
    }, index=range(100, 160))
    done = ld.mask_frame(df)
    f = done.frame
    assert list(f.columns) == list(df.columns) and list(f.index) == list(df.index)
    assert f["amount_paid"].dtype == df["amount_paid"].dtype
    assert f["member_no"].iloc[0] == f["member_no"].iloc[2] and f["member_no"].iloc[0].startswith("[MEMBER_")
    assert "081234567890" not in " ".join(f["notes"].dropna()) and f["notes"].iloc[2] is None
    assert done.report["checks"]["residual"]["clean"]


def test_mask_frame_with_a_policy_file(tmp_path):
    pd = pytest.importorskip("pandas")
    from lindungidatamu.table import Rule, save_policy
    save_policy({"secret": Rule("drop"), "member_no": Rule("tokenize", "MEMBER")}, tmp_path / "p.json")
    df = pd.DataFrame({"secret": ["x"], "member_no": ["0012345"]})
    assert list(ld.mask_frame(df, tmp_path / "p.json").frame.columns) == ["member_no"]
