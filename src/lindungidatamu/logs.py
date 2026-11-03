"""Personal data out of your logs: a logging filter that masks each record before it is written.

    import logging, lindungidatamu as ld
    handler = logging.StreamHandler()
    handler.addFilter(ld.LogFilter())        # on the handler: covers records from every logger
    logging.basicConfig(handlers=[handler])
    logging.getLogger("app").info("refund to rek. %s", "1234567890")   # 'refund to rek. [BANK_1]'

Patterns only by default (fast enough for every line). One-way: each record gets its own
placeholders, nothing is kept. Pass vault= to number values across records (the vault then holds
them; it grows as long as the process runs).
"""

import logging
import threading

from .mask import Vault, mask


class LogFilter(logging.Filter):
    def __init__(self, name: str = "", *, vault: Vault | None = None, **find_options):
        super().__init__(name)
        self.vault = vault
        self.options = find_options
        self._lock = threading.Lock()

    def _mask(self, text: str) -> str:
        if self.vault is None:
            return mask(text, **self.options).text
        with self._lock:
            return mask(text, vault=self.vault, **self.options).text

    def filter(self, record: logging.LogRecord) -> bool:
        if not super().filter(record):
            return False
        try:
            message = record.getMessage()
        except Exception:               # a broken format string: let logging report it as usual
            return True
        record.msg, record.args = self._mask(message), None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._mask(record.exc_text)
        if record.stack_info:
            record.stack_info = self._mask(record.stack_info)
        return True
