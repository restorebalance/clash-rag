"""Configuration and secrets.

Reads .env from the project root so keys never live in tracked files or in
shell history. Written against the standard library rather than pulling in
python-dotenv, per the project's dependency rule.
"""

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(path=ENV_PATH):
    """Load KEY=value lines from .env into the environment.

    A variable already set in the real environment wins, so an explicit
    export can override the file for a one-off run.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def require(name):
    """Return an environment variable, loading .env first."""
    load_env()
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Add it to {ENV_PATH.name} in the project root "
            f"or export it in your shell."
        )
    return value
