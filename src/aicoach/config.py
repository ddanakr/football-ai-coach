"""Project settings read from the `.env` file in the repository root.

Secrets live only in `.env` (ignored by git, D-057); `.env.example` in the repository
lists the names with empty values. Secrets are returned as `SecretStr`, so repr and
str hide the value, and they are never logged or put into error messages.
"""

from pathlib import Path

from dotenv import dotenv_values
from pydantic import SecretStr

from aicoach.errors import ConfigError
from aicoach.i18n import ru

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # pragma: allowlist secret (a variable name)

# src/aicoach/config.py -> repository root.
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def read_gemini_api_key(env_file: Path = DEFAULT_ENV_FILE) -> SecretStr:
    """Read `GEMINI_API_KEY` from the `.env` file.

    Only the file is read: a variable of the same name in the shell is ignored, and
    the process environment is not changed (child processes do not inherit the key).
    `${VAR}` interpolation is off. Surrounding whitespace is stripped.

    Args:
        env_file: path to the `.env` file; the repository root by default.

    Returns:
        The key wrapped in `SecretStr`.

    Raises:
        ConfigError: the file is missing or unreadable, or the key is absent, empty
            or only whitespace. The message never contains file contents.
    """
    if not env_file.is_file():
        raise ConfigError(ru.MISSING_GEMINI_KEY)
    try:
        values = dotenv_values(env_file, interpolate=False, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as err:
        # The OS and codec messages carry the path or a byte offset, never the content.
        raise ConfigError(ru.MISSING_GEMINI_KEY) from err
    value = (values.get(GEMINI_API_KEY_ENV) or "").strip()
    if not value:
        raise ConfigError(ru.MISSING_GEMINI_KEY)
    return SecretStr(value)
