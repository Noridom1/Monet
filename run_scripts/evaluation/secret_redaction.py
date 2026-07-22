"""Small helpers that keep injected judge credentials out of logs and metadata."""

from __future__ import annotations

import logging
from collections.abc import Sequence


def redact_cli_secret(argv: Sequence[str], option: str = "--judge-key") -> list[str]:
    """Redact an option's value in a command-argument sequence."""
    redacted = list(argv)
    for index, value in enumerate(redacted):
        if value == option and index + 1 < len(redacted):
            redacted[index + 1] = "[REDACTED]"
        elif value.startswith(f"{option}="):
            redacted[index] = f"{option}=[REDACTED]"
    return redacted


def install_log_secret_redaction(secret: str) -> None:
    """Redact a secret from all subsequently created standard logging records."""
    if not secret:
        return
    original_factory = logging.getLogRecordFactory()

    def redacting_factory(*args, **kwargs):
        record = original_factory(*args, **kwargs)
        message = record.getMessage()
        if secret in message:
            record.msg = message.replace(secret, "[REDACTED]")
            record.args = ()
        return record

    logging.setLogRecordFactory(redacting_factory)
