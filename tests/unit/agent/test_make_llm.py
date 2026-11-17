"""The client factories: provider, model, key, ledger (no request is sent)."""

from pathlib import Path

import pytest

from aicoach.agent import service
from aicoach.config import DEFAULT_AGENT_MODEL, load_settings
from aicoach.errors import ConfigError
from aicoach.speech.service import make_transcriber
from aicoach.spend import MeteredLLM, MeteredTranscriber


@pytest.mark.unit
def test_make_llm_wraps_the_chosen_provider_in_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")
    monkeypatch.setenv("AICOACH_BUDGET_USD", "3")

    llm = service.make_llm(load_settings(), tmp_path / "match01", "openai/gpt-6-luna")

    assert isinstance(llm, MeteredLLM)
    assert llm.model == "openai/gpt-6-luna"


@pytest.mark.unit
def test_make_llm_defaults_to_gpt_6_luna_through_openrouter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")

    assert service.make_llm(load_settings(), tmp_path).model == DEFAULT_AGENT_MODEL
    assert DEFAULT_AGENT_MODEL == "openai/gpt-6-luna"


@pytest.mark.unit
def test_make_llm_takes_the_agent_model_from_the_environment_and_the_flag_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")
    monkeypatch.setenv("AICOACH_AGENT_MODEL", "anthropic/claude-haiku-5.5")
    settings = load_settings()

    assert service.make_llm(settings, tmp_path).model == "anthropic/claude-haiku-5.5"
    assert service.make_llm(settings, tmp_path, "google/gemini-3.7-flash").model == (
        "google/gemini-3.7-flash"
    )


@pytest.mark.unit
def test_make_llm_takes_the_agent_model_from_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")
    env_file = tmp_path / ".env"
    content = "AICOACH_AGENT_MODEL=model-from-file\n"  # pragma: allowlist secret
    env_file.write_text(content, encoding="utf-8")

    assert service.make_llm(load_settings(env_file=env_file), tmp_path).model == "model-from-file"


@pytest.mark.unit
def test_make_llm_environment_beats_the_env_file_for_the_agent_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")
    monkeypatch.setenv("AICOACH_AGENT_MODEL", "model-from-env")
    env_file = tmp_path / ".env"
    content = "AICOACH_AGENT_MODEL=model-from-file\n"  # pragma: allowlist secret
    env_file.write_text(content, encoding="utf-8")

    assert service.make_llm(load_settings(env_file=env_file), tmp_path).model == "model-from-env"


@pytest.mark.unit
@pytest.mark.parametrize("blank", ["", "   "])
def test_make_llm_blank_agent_model_is_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")
    monkeypatch.setenv("AICOACH_AGENT_MODEL", blank)

    assert service.make_llm(load_settings(), tmp_path).model == DEFAULT_AGENT_MODEL


@pytest.mark.unit
def test_make_llm_blank_flag_and_blank_env_file_value_give_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")
    env_file = tmp_path / ".env"
    env_file.write_text("AICOACH_AGENT_MODEL=   \n", encoding="utf-8")
    settings = load_settings(env_file=env_file)

    assert service.make_llm(settings, tmp_path).model == DEFAULT_AGENT_MODEL
    assert service.make_llm(settings, tmp_path, "  ").model == DEFAULT_AGENT_MODEL


@pytest.mark.unit
def test_make_llm_with_the_gemini_provider_uses_the_gemini_model_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "key-for-the-fake-client")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
    monkeypatch.setenv("AICOACH_AGENT_MODEL", "openai/not-for-gemini")  # OpenRouter slug only
    settings = load_settings(llm_provider="gemini")

    assert service.make_llm(settings, tmp_path).model == "gemini-from-env"
    assert service.make_llm(settings, tmp_path, "gemini-from-flag").model == "gemini-from-flag"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "hint"), [("openrouter", "OPENROUTER_API_KEY"), ("gemini", "GEMINI_API_KEY")]
)
def test_make_llm_without_the_key_of_its_provider_is_a_config_error(
    tmp_path: Path, provider: str, hint: str
) -> None:
    with pytest.raises(ConfigError, match=hint):
        service.make_llm(load_settings(llm_provider=provider), tmp_path)


@pytest.mark.unit
def test_make_transcriber_picks_the_provider_and_its_default_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")
    monkeypatch.setenv("GEMINI_API_KEY", "key-for-the-fake-client")

    via_router = make_transcriber(load_settings(), tmp_path)
    direct = make_transcriber(load_settings(llm_provider="gemini"), tmp_path)
    chosen = make_transcriber(load_settings(), tmp_path, "elevenlabs/scribe-v2")

    assert isinstance(via_router, MeteredTranscriber)
    assert via_router.model == "google/gemini-3.5-transcribe"
    assert direct.model == "gemini-3.5-transcribe"
    assert chosen.model == "elevenlabs/scribe-v2"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "hint"), [("openrouter", "OPENROUTER_API_KEY"), ("gemini", "GEMINI_API_KEY")]
)
def test_make_transcriber_without_the_key_is_a_config_error(
    tmp_path: Path, provider: str, hint: str
) -> None:
    with pytest.raises(ConfigError, match=hint):
        make_transcriber(load_settings(llm_provider=provider), tmp_path)
