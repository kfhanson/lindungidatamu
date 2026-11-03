"""lindungidatamu: find and mask Indonesian personal data (NIK, NPWP, KK, BPJS, bank accounts,
addresses, names) before text reaches an AI model, a log or a dataset.

    import lindungidatamu as ld

    m = ld.mask("Kirim kontrak ke Budi Santoso, NIK 3174091203880004")
    m.text                      # 'Kirim kontrak ke Budi Santoso, NIK [NIK_1]'  (names need a name model)
    m.restore("[NIK_1] valid")  # '3174091203880004 valid'
"""

from .detect import Entity, Format, Term, find, patterns_version
from .mask import Item, Masked, Vault, mask
from . import table

__version__ = "0.1.0.dev0"
__all__ = ["Entity", "Format", "Item", "Masked", "Term", "Vault", "find", "mask", "patterns_version", "table"]
