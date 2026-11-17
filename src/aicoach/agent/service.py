"""Wiring of the agent for one match: used by the `agent-eval` command and the viewer."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aicoach.agent.context import load_context
from aicoach.agent.llm import GeminiClient, LLMClient
from aicoach.agent.loop import Agent, load_system_prompt
from aicoach.agent.openrouter_llm import OpenRouterClient
from aicoach.agent.tools import ClipMaker, EpisodeDescriber, ToolBox
from aicoach.clips import export_event_clip, find_current_event
from aicoach.config import Settings
from aicoach.episode import EpisodeDescription, describe_episode
from aicoach.errors import ConfigError
from aicoach.openrouter import OpenRouterHttp
from aicoach.spend import MeteredLLM, MeteredVideoModel, SpendLedger, ledger_path
from aicoach.vlm import OpenRouterVideoModel, VideoModel


@dataclass(frozen=True)
class AgentRuntime:
    """An agent and the factory of its tools (a fresh `ToolBox` with the current events for
    every question, so edits made in the viewer are seen at once)."""

    agent: Agent
    box_factory: Callable[[], ToolBox]


def gemini_client(settings: Settings) -> GeminiClient:
    """The Gemini client for the model and key of the settings.

    Raises:
        ConfigError: No key is set.
    """
    return GeminiClient(settings.require_gemini_api_key(), settings.gemini_model)


def openrouter_client(settings: Settings, model: str | None = None) -> OpenRouterClient:
    """The OpenRouter client for the key of the settings and `model` (default: the setting
    `AICOACH_AGENT_MODEL`, then the default of `config.py`).

    Raises:
        ConfigError: No key is set.
    """
    http = OpenRouterHttp(settings.require_openrouter_api_key())
    return OpenRouterClient(http, (model or "").strip() or settings.aicoach_agent_model)


def make_llm(settings: Settings, match_dir: Path, model: str | None = None) -> LLMClient:
    """The client of the provider in the settings, wrapped in the spending ledger and the budget.

    Args:
        settings: Provider (`AICOACH_LLM_PROVIDER` or `--provider`), keys, budget, data root.
        match_dir: The match directory (the ledger goes to the data root, or above this one).
        model: The model (`--model`); when `None`, `AICOACH_AGENT_MODEL` for OpenRouter and
            `GEMINI_MODEL` for Gemini.

    Raises:
        ConfigError: The key of the chosen provider is not set.
    """
    provider = settings.aicoach_llm_provider
    inner: LLMClient
    if provider == "gemini":
        inner = GeminiClient(
            settings.require_gemini_api_key(), (model or "").strip() or settings.gemini_model
        )
    else:
        inner = openrouter_client(settings, model)
    ledger = SpendLedger(ledger_path(settings, match_dir), settings.aicoach_budget_usd)
    return MeteredLLM(inner, ledger, provider)


def make_video_model(
    settings: Settings, match_dir: Path, model: str | None = None
) -> MeteredVideoModel:
    """The video model of the episode description (FR-11, D-073): OpenRouter only (the one path
    with video input that is built), wrapped in the spending ledger and the budget.

    Args:
        settings: Key, budget, data root and the default model (`AICOACH_VLM_MODEL`).
        match_dir: The match directory (the ledger goes to the data root, or above this one).
        model: The model (`--model`); when `None` or blank, `AICOACH_VLM_MODEL`.

    Raises:
        ConfigError: No OpenRouter key is set.
    """
    http = OpenRouterHttp(settings.require_openrouter_api_key())
    inner = OpenRouterVideoModel(http, (model or "").strip() or settings.aicoach_vlm_model)
    ledger = SpendLedger(ledger_path(settings, match_dir), settings.aicoach_budget_usd)
    return MeteredVideoModel(inner, ledger, "openrouter")


def optional_video_model(settings: Settings, match_dir: Path) -> VideoModel | None:
    """The video model, or `None` when there is no OpenRouter key: the agent then simply has no
    `describe_episode` tool (it needs the key, the agent itself may run on the free Gemini)."""
    try:
        return make_video_model(settings, match_dir)
    except ConfigError:
        return None


def build_runtime(
    llm: LLMClient,
    db_path: Path,
    match_id: str,
    match_dir: Path,
    video: Path | None = None,
    vlm: VideoModel | None = None,
) -> AgentRuntime:
    """The agent of a match.

    Args:
        llm: The language-model client.
        db_path: The events database.
        match_id: Match id in the database.
        match_dir: Directory with the match files; the journal is written there.
        video: The match video; without it the `make_clip` and `describe_episode` tools are not
            offered.
        vlm: The video model; without it the `describe_episode` tool is not offered.

    Raises:
        StorageError: The database does not exist.
    """
    system = load_system_prompt(load_context(db_path, match_id, match_dir))
    clip_maker: ClipMaker | None = None
    if video is not None:

        def clip_maker(event_id: str, before_s: float, after_s: float) -> Path:
            return export_event_clip(db_path, match_id, video, event_id, before_s, after_s)

    describer: EpisodeDescriber | None = None
    if video is not None and vlm is not None:

        def describer(event_id: str) -> EpisodeDescription:
            event = find_current_event(db_path, match_id, event_id)
            return describe_episode(video, event, vlm)

    return AgentRuntime(
        Agent(llm, system, match_dir=match_dir),
        lambda: ToolBox(load_context(db_path, match_id, match_dir), clip_maker, describer),
    )
