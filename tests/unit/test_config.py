"""Settings: priority CLI > environment > `.env` > default, and the secret key."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from pydantic import SecretStr
from structlog.testing import capture_logs

from aicoach import config
from aicoach.config import DEFAULT_GEMINI_MODEL, Settings, load_settings, read_gemini_api_key
from aicoach.errors import ConfigError
from aicoach.logging_setup import configure_logging

FAKE_KEY = "AIzaFakeKeyForTests0123456789abcdefXYZ"  # pragma: allowlist secret
OTHER_KEY = "AIzaOtherFakeKey9876543210zyxwvutsrq"  # pragma: allowlist secret
ENV_HINT = "скопируйте .env.example в .env и впишите ключ"


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """structlog configuration is global: restore the defaults even if a test fails."""
    yield
    structlog.reset_defaults()


def _env_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
def test_default_env_file_is_dot_env_in_repository_root() -> None:
    # The autouse fixture replaces the constant; the real one is the module's own computation.
    real = Path(config.__file__).resolve().parents[2] / ".env"
    assert real.name == ".env"
    assert (real.parent / "pyproject.toml").is_file()


@pytest.mark.unit
def test_load_settings_without_anything_gives_the_defaults() -> None:
    settings = load_settings()
    assert settings.gemini_api_key is None
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.aicoach_data_root is None
    assert settings.tesseract_cmd is None


@pytest.mark.unit
def test_env_file_is_used_when_the_environment_is_silent(tmp_path: Path) -> None:
    path = _env_file(
        tmp_path,
        f"GEMINI_API_KEY={FAKE_KEY}\nGEMINI_MODEL=model-from-file\n"
        "AICOACH_DATA_ROOT=data-from-file\n"
        "TESSERACT_CMD=tess-from-file\n",  # pragma: allowlist secret
    )
    settings = load_settings(env_file=path)
    assert settings.require_gemini_api_key().get_secret_value() == FAKE_KEY
    assert settings.gemini_model == "model-from-file"
    assert settings.aicoach_data_root == Path("data-from-file")
    assert settings.tesseract_cmd == Path("tess-from-file")


@pytest.mark.unit
def test_environment_beats_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "model-from-env")
    monkeypatch.setenv("GEMINI_API_KEY", OTHER_KEY)
    path = _env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\nGEMINI_MODEL=model-from-file\n")
    settings = load_settings(env_file=path)
    assert settings.gemini_model == "model-from-env"
    assert settings.require_gemini_api_key().get_secret_value() == OTHER_KEY


@pytest.mark.unit
def test_cli_value_beats_environment_and_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICOACH_DATA_ROOT", "root-from-env")
    monkeypatch.setenv("GEMINI_MODEL", "model-from-env")
    path = _env_file(tmp_path, "AICOACH_DATA_ROOT=root-from-file\nGEMINI_MODEL=model-from-file\n")
    settings = load_settings(
        env_file=path, data_root=Path("root-from-cli"), gemini_model="model-from-cli"
    )
    assert settings.aicoach_data_root == Path("root-from-cli")
    assert settings.gemini_model == "model-from-cli"


@pytest.mark.unit
def test_absent_cli_values_leave_the_lower_sources_in_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICOACH_DATA_ROOT", "root-from-env")
    settings = load_settings(data_root=None, gemini_model=None)
    assert settings.aicoach_data_root == Path("root-from-env")
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL


@pytest.mark.unit
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_cli_model_is_ignored(blank: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "model-from-env")
    assert load_settings(gemini_model=blank).gemini_model == "model-from-env"


@pytest.mark.unit
@pytest.mark.parametrize("blank", ["", "   ", "' \t '"])
def test_blank_values_in_env_file_mean_the_default(tmp_path: Path, blank: str) -> None:
    path = _env_file(
        tmp_path,
        f"GEMINI_API_KEY={blank}\nGEMINI_MODEL={blank}\n"
        f"AICOACH_DATA_ROOT={blank}\nTESSERACT_CMD={blank}\n",
    )
    settings = load_settings(env_file=path)
    assert settings.gemini_api_key is None
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.aicoach_data_root is None
    assert settings.tesseract_cmd is None


@pytest.mark.unit
def test_blank_environment_value_falls_through_to_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "")
    path = _env_file(tmp_path, "GEMINI_MODEL=model-from-file\n")
    assert load_settings(env_file=path).gemini_model == "model-from-file"


@pytest.mark.unit
def test_blank_environment_value_without_file_value_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "  ")
    monkeypatch.setenv("TESSERACT_CMD", "")
    settings = load_settings()
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.tesseract_cmd is None


@pytest.mark.unit
def test_unknown_names_in_env_file_are_ignored(tmp_path: Path) -> None:
    path = _env_file(tmp_path, "SOMETHING_ELSE=1\nGEMINI_MODEL=m\n")
    assert load_settings(env_file=path).gemini_model == "m"


@pytest.mark.unit
def test_default_env_file_is_read_when_no_file_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", _env_file(tmp_path, "GEMINI_MODEL=m\n"))
    assert load_settings().gemini_model == "m"


@pytest.mark.unit
def test_path_that_is_a_directory_is_treated_as_no_file(tmp_path: Path) -> None:
    (tmp_path / ".env").mkdir()
    assert load_settings(env_file=tmp_path / ".env").gemini_api_key is None


@pytest.mark.unit
def test_require_gemini_api_key_missing_raises_config_error_with_hint() -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_settings().require_gemini_api_key()
    assert ENV_HINT in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    ["", "GEMINI_API_KEY=\n", "GEMINI_API_KEY=   \n", "GEMINI_API_KEY\n", "# comment\nOTHER=1\n"],
)
def test_read_gemini_api_key_missing_or_blank_raises_config_error(
    tmp_path: Path, content: str
) -> None:
    with pytest.raises(ConfigError, match=r"\.env\.example"):
        read_gemini_api_key(load_settings(env_file=_env_file(tmp_path, content)))


@pytest.mark.unit
def test_read_gemini_api_key_without_settings_loads_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert read_gemini_api_key().get_secret_value() == FAKE_KEY


@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    [
        f"GEMINI_API_KEY={FAKE_KEY}\n",
        f'GEMINI_API_KEY="{FAKE_KEY}"\r\n',
        f"# comment\nOTHER=1\nGEMINI_API_KEY = '  {FAKE_KEY}  '\n",
        f"\N{BYTE ORDER MARK}GEMINI_API_KEY={FAKE_KEY}",
    ],
)
def test_read_gemini_api_key_present_returns_secret_with_the_value(
    tmp_path: Path, content: str
) -> None:
    key = read_gemini_api_key(load_settings(env_file=_env_file(tmp_path, content)))
    assert isinstance(key, SecretStr)
    assert key.get_secret_value() == FAKE_KEY


@pytest.mark.unit
def test_load_settings_does_not_change_the_process_environment(tmp_path: Path) -> None:
    before = dict(os.environ)
    load_settings(env_file=_env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\nOTHER=1\n"))
    assert dict(os.environ) == before
    assert "GEMINI_API_KEY" not in os.environ


@pytest.mark.unit
def test_not_utf8_env_file_raises_config_error_without_content(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"GEMINI_API_KEY=" + FAKE_KEY.encode() + b"\xff\xfe\n")
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=path)
    assert FAKE_KEY not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


@pytest.mark.unit
def test_unreadable_env_file_raises_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _deny(*_args: object, **_kwargs: object) -> Settings:
        raise PermissionError

    monkeypatch.setattr(config, "Settings", _deny)
    with pytest.raises(ConfigError, match=r"\.env") as exc_info:
        load_settings(env_file=_env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\n"))
    assert isinstance(exc_info.value.__cause__, PermissionError)


@pytest.mark.unit
def test_secret_is_hidden_in_repr_and_str_of_key_and_settings(tmp_path: Path) -> None:
    settings = load_settings(env_file=_env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\n"))
    key = settings.require_gemini_api_key()
    for text in (repr(key), str(key), f"{key}", repr(settings), str(settings), f"{settings}"):
        assert FAKE_KEY not in text


@pytest.mark.unit
def test_error_text_has_no_file_content_and_no_cause(tmp_path: Path) -> None:
    other_value = "unrelated-value-0123456789"
    path = _env_file(tmp_path, f"OTHER={other_value}\nGEMINI_API_KEY=\n")
    with pytest.raises(ConfigError) as exc_info:
        read_gemini_api_key(load_settings(env_file=path))
    assert other_value not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.unit
def test_key_never_reaches_the_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _env_file(tmp_path, f"GEMINI_API_KEY={FAKE_KEY}\nbroken line without equals\n")
    configure_logging("DEBUG")
    caplog.set_level(logging.DEBUG)
    with capture_logs() as logs:
        settings = load_settings(env_file=path)
        key = read_gemini_api_key(settings)
        structlog.get_logger().info("loaded", key=key, settings=settings)
    # Logging the objects themselves must not reveal the value either.
    logging.getLogger("aicoach").info("loaded: %s %r %s", key, key, settings)
    structlog.get_logger().info("loaded", key=key, settings=settings)
    captured = capsys.readouterr()
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in captured.out
    assert FAKE_KEY not in captured.err
    assert all(FAKE_KEY not in repr(entry) for entry in logs)


@pytest.mark.unit
def test_whitespace_environment_value_is_the_default_and_hides_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "  ")
    path = _env_file(tmp_path, "GEMINI_MODEL=model-from-file\n")
    assert load_settings(env_file=path).gemini_model == DEFAULT_GEMINI_MODEL


@pytest.mark.unit
def test_dotenv_expands_variables_in_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Known deviation from D-057, recorded in D-060: the library expands ${VAR} in `.env`.
    monkeypatch.setenv("SOME_OTHER_NAME", "expanded")
    path = _env_file(tmp_path, "GEMINI_MODEL=${SOME_OTHER_NAME}\n")
    assert load_settings(env_file=path).gemini_model == "expanded"
