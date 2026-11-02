"""Masking and restoring: personal data becomes a placeholder ([PERSON_1], [NIK_1]) before text goes
to an AI model or a log, and comes back in the answer.

The same value gets the same placeholder every time in one Vault (one conversation), so the model
can still tell who is who; "Budi" after "Budi Santoso" is the same person. The vault holds the
real values: keep it on your side, never send it with the text.
"""

import re
from dataclasses import dataclass, field

from .detect import Entity, find

NAME_TYPES = {"PERSON", "ORG", "CLIENT", "PROJECT", "SYSTEM", "ADDRESS"}

# A placeholder as an AI model may write it back: [PERSON_1], PERSON_1, [Person 1], person-1.
_BACK = re.compile(r"\[?\b([A-Za-z][A-Za-z_]{1,15}?)[ _-]?(\d{1,3})\b\]?")


def _norm(type_: str, value: str) -> str:
    if type_ in NAME_TYPES:
        return " ".join(value.lower().split())
    return re.sub(r"[\s.\-]", "", value)


@dataclass
class Item:
    placeholder: str
    type: str
    value: str


@dataclass
class Vault:
    """Placeholder <-> real value, for one conversation."""
    items: list[Item] = field(default_factory=list)

    def placeholder(self, type_: str, value: str) -> str:
        n = _norm(type_, value)
        for i in self.items:
            if i.type == type_ and _norm(type_, i.value) == n:
                return i.placeholder
        if type_ in ("PERSON", "ORG", "CLIENT"):          # "Budi" and "Budi Santoso": one person
            words = set(n.split())
            for i in self.items:
                other = set(_norm(type_, i.value).split())
                if i.type == type_ and (other <= words or words <= other):
                    return i.placeholder
        ph = f"[{type_}_{sum(1 for i in self.items if i.type == type_) + 1}]"
        self.items.append(Item(ph, type_, value))
        return ph

    def value(self, type_: str, n: int | str) -> str | None:
        ph = f"[{type_.upper()}_{int(n)}]"
        return next((i.value for i in self.items if i.placeholder == ph), None)

    def restore(self, text: str) -> str:
        """The real values back in place of placeholders, also as the model rewrote them ("Person 1")."""
        return _BACK.sub(lambda m: self.value(m.group(1), m.group(2)) or m.group(0), text)


@dataclass
class Masked:
    text: str                 # what to send
    entities: list[Entity]    # what was found
    vault: Vault              # keep this; restore() with it

    def restore(self, text: str) -> str:
        return self.vault.restore(text)


def mask(text: str, vault: Vault | None = None, **find_options) -> Masked:
    """Mask `text`. Pass the same `vault` for every message of one conversation. Options are
    those of lindungidatamu.find (types, terms, formats, names, keep)."""
    vault = vault if vault is not None else Vault()
    entities = find(text, **find_options)
    out, last = [], 0
    for e in entities:
        out += [text[last:e.start], vault.placeholder(e.type, e.text)]
        last = e.end
    return Masked("".join(out) + text[last:], entities, vault)
