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
from .llm import wrap
from .logs import LogFilter
from .table import mask_frame, risk

__version__ = "0.1.0.dev0"
__all__ = ["Entity", "Format", "Item", "LogFilter", "Masked", "Term", "Vault", "find", "mask", "mask_frame",
           "patterns_version", "risk", "table", "wrap"]
