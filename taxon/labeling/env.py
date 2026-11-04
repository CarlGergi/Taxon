"""Load API keys from .env or the environment, keeping them out of anything printed."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from taxon.pins import ROOT

KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


class MissingKey(RuntimeError):
    """A required key is not set."""


class Secrets:
    def __init__(self, values: Mapping[str, str]):
        self._values = {name: value for name, value in values.items() if value}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def require(self, name: str) -> str:
        value = self._values.get(name)
        if value is None:
            raise MissingKey(f"{name} is not set; add it to .env")
        return value

    def redact(self, text: str) -> str:
        """Replace every known key in `text`, for error messages that may quote one."""
        for name, value in self._values.items():
            text = text.replace(value, f"<{name}>")
        return text

    def __repr__(self) -> str:
        return f"Secrets(set={sorted(self._values)})"


def load_secrets(
    path: Path = ROOT / ".env", environ: Mapping[str, str] | None = None
) -> Secrets:
    """Read keys without touching os.environ. A value in .env wins over the environment."""
    environ = os.environ if environ is None else environ
    from_file = dotenv_values(path) if path.exists() else {}
    return Secrets({n: from_file.get(n) or environ.get(n) or "" for n in KEY_NAMES})
