"""Masking and restoring: personal data becomes a placeholder ([PERSON_1], [NIK_1]) before text goes
to an AI model or a log, and comes back in the answer.

The same value gets the same placeholder every time in one Vault (one conversation, or one
dataset), so the model can still tell who is who; in a conversation "Budi" after "Budi Santoso" is
the same person. The vault holds the real values: keep it on your side, never send it with the text.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .detect import Entity, find

NAME_TYPES = {"PERSON", "ORG", "CLIENT", "PROJECT", "SYSTEM", "ADDRESS"}
PARTIAL_TYPES = ("PERSON", "ORG", "CLIENT")

# A placeholder exactly as written: [PERSON_1], [MEMBER_1200].
_EXACT = re.compile(r"\[([A-Z][A-Z_]{1,15})_(\d{1,7})\]")
# A placeholder as an AI model may write it back: [PERSON_1], PERSON_1, [Person 1], person-1.
_BACK = re.compile(r"\[?\b([A-Za-z][A-Za-z_]{1,15}?)[ _-]?(\d{1,7})\b\]?")


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
    """Placeholder <-> real value, for one conversation or one dataset.

    merge_partial  "Budi" and "Budi Santoso" get one placeholder (right for a conversation). Turn it
                   off for a table, where two different people called Budi must stay apart.
    """
    items: list[Item] = field(default_factory=list)
    merge_partial: bool = True
    _key: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _value: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _words: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _count: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _indexed: int = field(default=0, init=False, repr=False, compare=False)

    def _index(self) -> None:
        """Lookups by dict; brings the index up to date if items were added from outside."""
        if self._indexed > len(self.items):
            self._key, self._value, self._words, self._count, self._indexed = {}, {}, {}, {}, 0
        for pos, i in enumerate(self.items[self._indexed:], self._indexed):
            n = _norm(i.type, i.value)
            self._key.setdefault((i.type, n), i)
            self._value[i.placeholder] = i.value
            if i.type in PARTIAL_TYPES:
                for w in set(n.split()):
                    self._words.setdefault((i.type, w), []).append((pos, i))
            m = _EXACT.fullmatch(i.placeholder)
            if m and m.group(1) == i.type:
                self._count[i.type] = max(self._count.get(i.type, 0), int(m.group(2)))
        self._indexed = len(self.items)

    def placeholder(self, type_: str, value: str) -> str:
        self._index()
        n = _norm(type_, value)
        if (hit := self._key.get((type_, n))) is not None:
            return hit.placeholder
        if self.merge_partial and type_ in PARTIAL_TYPES:     # "Budi" and "Budi Santoso": one person
            words = set(n.split())
            candidates = sorted({pos: i for w in words for pos, i in self._words.get((type_, w), ())}.items())
            for _, i in candidates:                           # the earliest match, as before
                other = set(_norm(type_, i.value).split())
                if other <= words or words <= other:
                    return i.placeholder
        self._count[type_] = self._count.get(type_, 0) + 1
        self.items.append(Item(f"[{type_}_{self._count[type_]}]", type_, value))
        self._index()
        return self.items[-1].placeholder

    def value(self, type_: str, n: int | str) -> str | None:
        self._index()
        return self._value.get(f"[{type_.upper()}_{int(n)}]")

    def restore(self, text: str, strict: bool = False) -> str:
        """The real values back in place of placeholders.

        strict  only placeholders exactly as written ([PERSON_1]). Use it for reports and files:
                without it, rewritten forms ("Person 1", "member-12") are restored too, which suits a
                chat answer but turns ordinary words like "member 812" into someone's data.
        """
        rx = _EXACT if strict else _BACK
        return rx.sub(lambda m: self.value(m.group(1), m.group(2)) or m.group(0), text)

    def save(self, path: str | Path) -> None:
        """Write the vault to a file only you can read (mode 0600). It holds the real values."""
        data = json.dumps({"version": 1, "merge_partial": self.merge_partial,
                           "items": [[i.placeholder, i.type, i.value] for i in self.items]}, ensure_ascii=False)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.chmod(path, 0o600)            # also when the file already existed

    @classmethod
    def load(cls, path: str | Path) -> "Vault":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Item(*i) for i in doc["items"]], merge_partial=doc.get("merge_partial", True))


@dataclass
class Masked:
    text: str                 # what to send
    entities: list[Entity]    # what was found
    vault: Vault              # keep this; restore() with it

    def restore(self, text: str, strict: bool = False) -> str:
        return self.vault.restore(text, strict=strict)


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
