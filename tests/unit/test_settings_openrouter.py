"""The OpenRouter key, the budget and the provider among the settings (D-067, D-068)."""

from pathlib import Path

import pytest

from aicoach.config import load_settings
from aicoach.errors import ConfigError

FAKE_KEY = "sk-or-v1-fake-key-for-tests-0123456789"  # pragma: allowlist secret


def _env_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
def test_new_settings_default_to_openrouter_without_a_key_and_without_a_budget() -> None:
    settings = load_settings()

    assert settings.aicoach_llm_provider == "openrouter"
    assert settings.openrouter_api_key is None
    assert settings.aicoach_budget_usd is None


@pytest.mark.unit
def test_openrouter_key_budget_and_provider_come_from_the_env_file(tmp_path: Path) -> None:
    path = _env_file(
        tmp_path,
        f"OPENROUTER_API_KEY={FAKE_KEY}\nAICOACH_BUDGET_USD=12.5\nAICOACH_LLM_PROVIDER=Gemini\n",
    )

    settings = load_settings(env_file=path)

    assert settings.require_openrouter_api_key().get_secret_value() == FAKE_KEY
    assert settings.aicoach_budget_usd == 12.5
    assert settings.aicoach_llm_provider == "gemini"  # the case does not matter


@pytest.mark.unit
def test_empty_budget_and_provider_values_mean_not_set(tmp_path: Path) -> None:
    path = _env_file(
        tmp_path, "AICOACH_BUDGET_USD=\nAICOACH_LLM_PROVIDER=  \nOPENROUTER_API_KEY=\n"
    )

    settings = load_settings(env_file=path)

    assert settings.aicoach_budget_usd is None
    assert settings.aicoach_llm_provider == "openrouter"
    assert settings.openrouter_api_key is None


@pytest.mark.unit
def test_a_budget_of_zero_is_a_real_ceiling_not_unset(tmp_path: Path) -> None:
    settings = load_settings(env_file=_env_file(tmp_path, "AICOACH_BUDGET_USD=0\n"))

    assert settings.aicoach_budget_usd == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["abc", "-1", "nan", "inf"])
def test_a_bad_budget_is_a_config_error_that_names_the_setting_not_the_value(
    tmp_path: Path, bad: str
) -> None:
    with pytest.raises(ConfigError, match="AICOACH_BUDGET_USD") as caught:
        load_settings(env_file=_env_file(tmp_path, f"AICOACH_BUDGET_USD={bad}\n"))

    assert bad not in str(caught.value)


@pytest.mark.unit
def test_an_unknown_provider_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="AICOACH_LLM_PROVIDER"):
        load_settings(env_file=_env_file(tmp_path, "AICOACH_LLM_PROVIDER=bing\n"))


@pytest.mark.unit
def test_provider_flag_beats_the_environment_and_a_blank_flag_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICOACH_LLM_PROVIDER", "gemini")

    assert load_settings(llm_provider="openrouter").aicoach_llm_provider == "openrouter"
    assert load_settings(llm_provider=" ").aicoach_llm_provider == "gemini"
    assert load_settings(llm_provider=None).aicoach_llm_provider == "gemini"


@pytest.mark.unit
def test_missing_openrouter_key_is_a_config_error_with_a_hint() -> None:
    with pytest.raises(ConfigError, match=r"OPENROUTER_API_KEY.*--provider gemini"):
        load_settings().require_openrouter_api_key()


@pytest.mark.unit
def test_openrouter_key_is_hidden_in_repr_and_str(tmp_path: Path) -> None:
    settings = load_settings(env_file=_env_file(tmp_path, f"OPENROUTER_API_KEY={FAKE_KEY}\n"))

    for text in (repr(settings), str(settings), repr(settings.openrouter_api_key)):
        assert FAKE_KEY not in text
