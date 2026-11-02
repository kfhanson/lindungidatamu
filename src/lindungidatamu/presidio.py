"""Indonesian recognisers for Microsoft Presidio (pip install "lindungidatamu[presidio]").

    from presidio_analyzer import AnalyzerEngine
    from lindungidatamu.presidio import add_to

    analyzer = add_to(AnalyzerEngine())
    analyzer.analyze("NIK 3174091203880004", language="en")

Each recogniser reports lindungidatamu's type with an "ID_" prefix where Presidio has no such entity
(ID_NIK, ID_NPWP, ID_KK, ID_BPJS, ...), and Presidio's own names where it does (CREDIT_CARD,
EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS). Matches pass the same checks as lindungidatamu.find.
"""

from .detect import pattern_entities

PRESIDIO_NAMES = {"CARD": "CREDIT_CARD", "EMAIL": "EMAIL_ADDRESS", "PHONE": "PHONE_NUMBER", "IP": "IP_ADDRESS",
                  "PASSPORT": "ID_PASSPORT", "SECRET": "ID_SECRET"}
TYPES = ("NIK", "NPWP", "KK", "PASSPORT", "BPJS", "SIM", "BANK", "CARD", "EMAIL", "PHONE", "ADDRESS", "IP", "HOST", "SECRET")


def entity_name(type_: str) -> str:
    return PRESIDIO_NAMES.get(type_) or f"ID_{type_}"


def recognizers(languages=("en", "id")) -> list:
    """One recogniser per language (Presidio picks recognisers by the language you analyse in)."""
    try:
        from presidio_analyzer import EntityRecognizer, RecognizerResult
    except ImportError as e:
        raise ImportError('the Presidio add-on needs Presidio: pip install "lindungidatamu[presidio]"') from e

    class IndonesianRecognizer(EntityRecognizer):
        def __init__(self, language: str):
            super().__init__(supported_entities=[entity_name(t) for t in TYPES], name="LindungidatamuRecognizer",
                             supported_language=language)

        def load(self) -> None:
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            wanted = set(entities or [])
            return [RecognizerResult(entity_name(e.type), e.start, e.end, 0.9)
                    for e in pattern_entities(text) if not wanted or entity_name(e.type) in wanted]

    return [IndonesianRecognizer(lang) for lang in languages]


def add_to(analyzer, languages=("en", "id")):
    """Register the recognisers on an AnalyzerEngine; returns it."""
    for r in recognizers(languages):
        analyzer.registry.add_recognizer(r)
    return analyzer
