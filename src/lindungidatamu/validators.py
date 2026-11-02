"""Checks that turn a pattern match into a real finding: a NIK's province and birth date, a card's
Luhn digit, and so on. Named by `validator` in patterns.json."""

import re

_NON_DIGIT = re.compile(r"\D")

# Indonesian province codes (first two digits of a NIK), Kemendagri.
_NIK_PROVINCES = {
    11, 12, 13, 14, 15, 16, 17, 18, 19, 21,
    31, 32, 33, 34, 35, 36,
    51, 52, 53,
    61, 62, 63, 64, 65,
    71, 72, 73, 74, 75, 76,
    81, 82,
    91, 92, 93, 94, 95, 96, 97,
}


def digits(s: str) -> str:
    return _NON_DIGIT.sub("", s)


def luhn_card(match: str) -> bool:
    """13-19 digit PAN with a known issuer prefix and a valid Luhn check digit."""
    d = digits(match)
    if not 13 <= len(d) <= 19:
        return False
    # Visa 4, Mastercard 51-55 / 2221-2720, Amex 34/37, Discover 6011/65, JCB 35, UnionPay 62.
    two, four = int(d[:2]), int(d[:4])
    if not (d[0] == "4" or 51 <= two <= 55 or 2221 <= four <= 2720 or two in (34, 37, 35, 62, 65)
            or d.startswith("6011")):
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_CARD_GROUPS = re.compile(r"\d{4}([ -])\d{4}\1\d{4}\1\d{4}")


def nik(match: str) -> bool:
    """Indonesian NIK: PP KK CC DDMMYY SSSS. Women's day-of-birth is +40. Written in groups of
    four with a valid card checksum, it's a card number (52xx is both NTB and Mastercard)."""
    if _CARD_GROUPS.fullmatch(match.strip()) and luhn_card(match):
        return False
    d = digits(match)
    if len(d) != 16 or int(d[:2]) not in _NIK_PROVINCES:
        return False
    day, month = int(d[6:8]), int(d[8:10])
    if day > 40:
        day -= 40
    return 1 <= day <= 31 and 1 <= month <= 12 and d[12:] != "0000"


def nik_labelled(match: str) -> bool:
    """A number labelled as a NIK that isn't a valid one (a typo, a digit short or extra). Valid
    NIKs are id_nik's; this only takes the rest, so one number is never reported twice."""
    return 14 <= len(digits(match)) <= 18 and not nik(match)


def bank_account(match: str) -> bool:
    """A labelled bank account number: 9-17 digits, not one digit repeated (a placeholder)."""
    d = digits(match)
    return 9 <= len(d) <= 17 and len(set(d)) > 1


def ipv4(match: str) -> bool:
    """Four numbers 0-255."""
    parts = match.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 for p in parts)


def phone_id(match: str) -> bool:
    """An Indonesian mobile number: 10-13 digits written with a leading 0 (+62 and 62 count as 0)."""
    d = digits(match)
    d = "0" + d[2:] if d.startswith("62") else d
    return d.startswith("08") and 10 <= len(d) <= 13


VALIDATORS = {"luhn_card": luhn_card, "phone_id": phone_id, "nik": nik, "nik_labelled": nik_labelled, "bank_account": bank_account,
              "ipv4": ipv4}
