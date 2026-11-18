"""Shared fixtures: tests never see the developer's real settings."""

import pytest

SETTING_VARIABLES = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_TRANSCRIBE_MODEL",
    "OPENROUTER_API_KEY",
    "AICOACH_AGENT_MODEL",
    "AICOACH_VLM_MODEL",
    "AICOACH_LLM_PROVIDER",
    "AICOACH_BUDGET_USD",
    "AICOACH_DATA_ROOT",
    "TESSERACT_CMD",
    "AICOACH_FONT_FILE",
    "TDEED_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def _isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """No setting comes from the shell or from the real `.env` of the repository.

    `setenv` before `delenv` so that monkeypatch restores a variable that existed before.
    """
    for name in SETTING_VARIABLES:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    empty = tmp_path_factory.mktemp("no_env") / ".env"  # never created
    monkeypatch.setattr("aicoach.config.DEFAULT_ENV_FILE", empty)
