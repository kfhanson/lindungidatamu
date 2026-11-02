"""Names, organisations and addresses, from a name model (optional: pip install "lindungidatamu[names]").

Patterns catch numbers; names need a model. GlinerNames wraps GLiNER's multilingual PII model,
which runs on your machine (the first use downloads it, about 1 GB). Anything with a
`find(text) -> list[Entity]` method works in its place, a hosted model included.
"""

from .detect import Entity

# GLiNER labels -> lindungidatamu types
LABELS = {"person": "PERSON", "organization": "ORG", "address": "ADDRESS"}


class GlinerNames:
    def __init__(self, model: str = "urchade/gliner_multi_pii-v1", threshold: float = 0.4, device: str = "cpu"):
        try:
            from gliner import GLiNER
        except ImportError as e:      # an extra, so the core stays dependency-free
            raise ImportError('names need GLiNER: pip install "lindungidatamu[names]"') from e
        self.model = GLiNER.from_pretrained(model).to(device)
        self.threshold = threshold
        self.name = model

    def find(self, text: str) -> list[Entity]:
        out = []
        for e in self.model.predict_entities(text, list(LABELS), threshold=self.threshold):
            out.append(Entity(LABELS[e["label"]], e["start"], e["end"], text[e["start"]:e["end"]], "model", self.name))
        return out
