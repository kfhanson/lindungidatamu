"""Names, organisations and addresses, from a name model (optional: pip install "lindungidatamu[names]").

Patterns catch numbers; names need a model. GlinerNames wraps GLiNER's multilingual PII model,
which runs on your machine (the first use downloads it, about 1 GB). Anything with a
`find(text) -> list[Entity]` method works in its place, a hosted model included.
"""

import os
import re
import sys
from contextlib import contextmanager

from .detect import Entity

# GLiNER labels -> lindungidatamu types
LABELS = {"person": "PERSON", "organization": "ORG", "address": "ADDRESS"}

DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"
DEFAULT_REVISION = "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"   # pinned: the version benchmarked


@contextmanager
def _offline():
    """Hugging Face offline mode while the model loads, then back as it was. Without it, a cached
    model still waits on the network at every load (transformers asks the hub about the tokenizer
    even with local_files_only). The libraries must already be imported: they read the setting at
    import, and the caller's own downloads afterwards must not stay offline."""
    saved_env = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    saved = []
    for module, attr in (("huggingface_hub.constants", "HF_HUB_OFFLINE"), ("transformers.utils.hub", "_is_offline_mode")):
        m = sys.modules.get(module)
        if m is not None and hasattr(m, attr):
            saved.append((m, attr, getattr(m, attr)))
            setattr(m, attr, True)
    try:
        yield
    finally:
        for m, attr, value in saved:
            setattr(m, attr, value)
        if saved_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = saved_env


def _mostly_upper(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) >= 0.7 * len(letters)


def model_view(text: str) -> str:
    """The text as the name model should see it: lines that are mostly upper case in title case
    ("BAMBANG SUTRISNO" -> "Bambang Sutrisno"), which the model reads as names. Same length, so the
    model's positions hold for the original text; a line whose length would change stays as it is."""
    def word(m: re.Match) -> str:
        w = m.group(0)
        t = w[0] + w[1:].lower() if w.isupper() else w
        return t if len(t) == len(w) else w

    def fix(m: re.Match) -> str:
        line = m.group(0)
        return re.sub(r"[^\W\d_]{2,}", word, line) if _mostly_upper(line) else line
    return re.sub(r"[^\n]+", fix, text)


class GlinerNames:
    """GLiNER on your machine.

    revision  the model version; the default model is pinned to the version benchmarked
    offline   True: load from the local cache only. False: may contact Hugging Face. None (default):
              the cache if the model is there, otherwise download it once
    case      "auto" (default): mostly upper-case lines go to the model in title case. "as_is": unchanged
    """

    def __init__(self, model: str = DEFAULT_MODEL, threshold: float = 0.4, device: str = "cpu",
                 revision: str | None = None, offline: bool | None = None, case: str = "auto"):
        try:
            from gliner import GLiNER
        except ImportError as e:      # an extra, so the core stays dependency-free
            raise ImportError('names need GLiNER: pip install "lindungidatamu[names]"') from e
        if case not in ("auto", "as_is"):
            raise ValueError('case is "auto" or "as_is"')
        revision = revision or (DEFAULT_REVISION if model == DEFAULT_MODEL else None)
        if offline is False:
            loaded = GLiNER.from_pretrained(model, revision=revision)
        else:
            try:
                with _offline():
                    loaded = GLiNER.from_pretrained(model, revision=revision, local_files_only=True)
            except Exception:
                if offline:
                    raise
                loaded = GLiNER.from_pretrained(model, revision=revision)   # not cached yet: download once
        self.model = loaded.to(device)
        self.threshold = threshold
        self.case = case
        self.name = model

    def find(self, text: str) -> list[Entity]:
        seen = model_view(text) if self.case == "auto" else text
        out = []
        for e in self.model.predict_entities(seen, list(LABELS), threshold=self.threshold):
            out.append(Entity(LABELS[e["label"]], e["start"], e["end"], text[e["start"]:e["end"]], "model", self.name))
        return out
