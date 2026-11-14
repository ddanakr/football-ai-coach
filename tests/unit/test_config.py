"""Reading the Gemini API key from the `.env` file."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from pydantic import SecretStr
from structlog.testing import capture_logs

from aicoach.config import DEFAULT_ENV_FILE, GEMINI_API_KEY_ENV, read_gemini_api_key
from aicoach.errors import ConfigError
from aicoach.logging_setup import configure_logging

FAKE_KEY = "AIzaFakeKeyForTests0123456789abcdefXYZ"  # pragma: allowlist secret
ENV_HINT = "скопируйте .env.example в .env и впишите ключ"


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """structlog configuration is global: restore the defaults even if a test fails."""
    yield
    structlog.reset_defaults()


@pytest.fixture(autouse=True)
def _no_key_in_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start without the variable and restore the real environment after the test.

    `setenv` first so that monkeypatch records the original state even when the
    variable is absent; a plain `delenv(raising=False)` would not undo a later change.
    """
    monkeypatch.setenv(GEMINI_API_KEY_ENV, "placeholder")
    monkeypatch.delenv(GEMINI_API_KEY_ENV)


def _env_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
def test_default_env_file_is_dot_env_in_repository_root() -> None:
    assert DEFAULT_ENV_FILE.name == ".env"
    assert (DEFAULT_ENV_FILE.parent / "pyproject.toml").is_file()


@pytest.mark.unit
def test_read_gemini_api_key_file_missing_raises_config_error_with_hint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc_info:
        read_gemini_api_key(tmp_path / ".env")
    assert ENV_HINT in str(exc_info.value)
    assert "setx" not in str(exc_info.value)


@pytest.mark.unit
def test_read_gemini_api_key_file_missing_ignores_variable_from_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GEMINI_API_KEY_ENV, FAKE_KEY)
    with pytest.raises(ConfigError, match=r"\.env\.example"):
        read_gemini_api_key(tmp_path / ".env")


@pytest.mark.unit
def test_read_gemini_api_key_path_is_a_directory_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / ".env").mkdir()
    with pytest.raises(ConfigError, match=r"\.env\.example"):
        read_gemini_api_key(tmp_path / ".env")


@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    [
        "",
        "GEMINI_API_KEY=\n",
        "GEMINI_API_KEY=   \n",
        'GEMINI_API_KEY=" \t "\n',
        "GEMINI_API_KEY\n",
        "# only a comment\nOTHER=value\n",
    ],
)
def test_read_gemini_api_key_missing_or_blank_in_file_raises_config_error(
    tmp_path: Path, content: str
) -> None:
    with pytest.raises(ConfigError, match=r"\.env\.example"):
        read_gemini_api_key(_env_file(tmp_path, content))


@pytest.mark.unit
def test_read_gemini_api_key_blank_in_file_ignores_variable_from_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GEMINI_API_KEY_ENV, FAKE_KEY)
    with pytest.raises(ConfigError):
        read_gemini_api_key(_env_file(tmp_path, "GEMINI_API_KEY=\n"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    [
        f"GEMINI_API_KEY={FAKE_KEY}\n",
        f'GEMINI_API_KEY="{FAKE_KEY}"\r\n',
        f"# comment\nOTHER=1\nGEMINI_API_KEY = '  {FAKE_KEY}  '\n",
        f"\ufeffGEMINI_API_KEY={FAKE_KEY}",
    ],
)
def test_read_gemini_api_key_present_returns_secret_with_the_value(
    tmp_path: Path, content: str
) -> None:
    key = read_gemini_api_key(_env_file(tmp_path, content))
    assert isinstance(key, SecretStr)
    assert key.get_secret_value() == FAKE_KEY


@pytest.mark.unit
def test_read_gemini_api_key_does_not_change_the_process_environment(tmp_path: Path) -> None:
    read_gemini_api_key(_env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\nOTHER=1\n"))
    assert GEMINI_API_KEY_ENV not in os.environ
    assert os.environ.get("OTHER") != "1"


@pytest.mark.unit
def test_read_gemini_api_key_not_utf8_raises_config_error_without_content(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"GEMINI_API_KEY=" + FAKE_KEY.encode() + b"\xff\xfe\n")
    with pytest.raises(ConfigError, match=r"\.env\.example") as exc_info:
        read_gemini_api_key(path)
    assert FAKE_KEY not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


@pytest.mark.unit
def test_read_gemini_api_key_unreadable_file_raises_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _deny(*_args: object, **_kwargs: object) -> dict[str, str | None]:
        raise PermissionError

    monkeypatch.setattr("aicoach.config.dotenv_values", _deny)
    with pytest.raises(ConfigError, match=r"\.env\.example") as exc_info:
        read_gemini_api_key(_env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\n"))
    assert isinstance(exc_info.value.__cause__, PermissionError)


@pytest.mark.unit
def test_read_gemini_api_key_value_is_hidden_in_repr_and_str(tmp_path: Path) -> None:
    key = read_gemini_api_key(_env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\n"))
    assert FAKE_KEY not in repr(key)
    assert FAKE_KEY not in str(key)
    assert FAKE_KEY not in f"{key}"


@pytest.mark.unit
def test_read_gemini_api_key_error_text_has_no_file_content_and_no_cause(tmp_path: Path) -> None:
    other_value = "unrelated-value-0123456789"
    path = _env_file(tmp_path, f"OTHER={other_value}\nGEMINI_API_KEY=\n")
    with pytest.raises(ConfigError) as exc_info:
        read_gemini_api_key(path)
    assert other_value not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
def test_read_gemini_api_key_never_reaches_the_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\nbroken line without equals\n")
    configure_logging("DEBUG")
    caplog.set_level(logging.DEBUG)
    with capture_logs() as logs:
        key = read_gemini_api_key(path)
        structlog.get_logger().info("key_loaded", key=key)
    # Logging the returned object itself must not reveal the value either.
    logging.getLogger("aicoach").info("key loaded: %s %r", key, key)
    structlog.get_logger().info("key_loaded", key=key)
    captured = capsys.readouterr()
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in captured.out
    assert FAKE_KEY not in captured.err
    assert all(FAKE_KEY not in repr(entry) for entry in logs)
