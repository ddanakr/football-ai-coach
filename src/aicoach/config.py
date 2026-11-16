"""Project settings in one place (D-057, D-060).

Priority, highest first: CLI flag, environment variable, `.env` in the repository root, default
(the usual order for Docker and CI). Only the five settings below live here. Thresholds, clip
windows, limits and prices stay in the code of their modules and are not read from `.env`.

The process environment is never changed: `pydantic-settings` reads `.env` into the settings
object only, so the key is not inherited by child processes (FFmpeg, Streamlit, ingest).
The key is a `SecretStr`: repr and str hide it, and it is never logged or put into errors.
"""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aicoach.errors import ConfigError
from aicoach.i18n import ru

# Flash-class model with a free tier and function calling, recommended for new projects in the
# Gemini API docs (models page, checked 2026-11-14). A setting, so it can be changed in `.env`.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# The speech-to-text model: Stable, with a free tier, word timestamps and speaker labels (models
# and pricing pages of the Gemini API docs, checked 2026-11-16).
DEFAULT_TRANSCRIBE_MODEL = "gemini-3.5-transcribe"

# src/aicoach/config.py -> repository root.
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """The settings; every field is read from the environment variable of the same name.

    Build it with `load_settings`: a bare `Settings()` does not read `.env`. An empty
    environment variable counts as not set (`.env`, then the default, applies); a value of only
    spaces counts as not set too, but ends the search there (the default applies, `.env` is not
    consulted). `${VAR}` in `.env` is expanded by the dotenv library (D-060).
    `gemini_api_key` is optional here: only the agent needs it (`require_gemini_api_key`).
    """

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8-sig",  # a BOM from Notepad is harmless
        env_ignore_empty=True,
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL
    aicoach_data_root: Path | None = None
    tesseract_cmd: Path | None = None

    @field_validator("gemini_api_key", "aicoach_data_root", "tesseract_cmd", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Whitespace only means not set."""
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("gemini_model", mode="after")
    @classmethod
    def _blank_model_is_default(cls, value: str) -> str:
        return value.strip() or DEFAULT_GEMINI_MODEL

    @field_validator("gemini_transcribe_model", mode="after")
    @classmethod
    def _blank_transcribe_model_is_default(cls, value: str) -> str:
        return value.strip() or DEFAULT_TRANSCRIBE_MODEL

    def require_gemini_api_key(self) -> SecretStr:
        """The Gemini key.

        Raises:
            ConfigError: No key is set. The message names no file content and no value.
        """
        if self.gemini_api_key is None:
            raise ConfigError(ru.MISSING_GEMINI_KEY)
        return self.gemini_api_key


def load_settings(
    *,
    env_file: Path | None = None,
    data_root: Path | None = None,
    gemini_model: str | None = None,
) -> Settings:
    """Settings from the environment and `.env`; given arguments (CLI flags) win over both.

    Args:
        env_file: The dotenv file; the one in the repository root by default (a missing file is
            fine: the environment and the defaults still apply).
        data_root: CLI value for `AICOACH_DATA_ROOT`; `None` leaves the lower sources in charge.
        gemini_model: CLI value for `GEMINI_MODEL`; `None` or blank leaves them in charge.

    Raises:
        ConfigError: `.env` cannot be read (encoding, permissions). The message never quotes
            the content.
    """
    given: dict[str, Path | str] = {}
    if data_root is not None:
        given["aicoach_data_root"] = data_root
    if gemini_model is not None and gemini_model.strip():
        given["gemini_model"] = gemini_model
    try:
        # `_env_file` is the documented per-instance way to choose the file; mypy does not see it
        # in the synthesized signature, nor the field values passed as a mapping.
        return Settings(_env_file=env_file or DEFAULT_ENV_FILE, **given)  # type: ignore[call-arg,arg-type]
    except (OSError, UnicodeDecodeError) as err:
        # The OS and codec messages carry the path or a byte offset, never the content.
        raise ConfigError(ru.UNREADABLE_ENV_FILE) from err


def read_gemini_api_key(settings: Settings | None = None) -> SecretStr:
    """The Gemini key from the settings (loaded with the defaults when none is given).

    Raises:
        ConfigError: No key is set, or the settings cannot be loaded.
    """
    return (settings or load_settings()).require_gemini_api_key()
