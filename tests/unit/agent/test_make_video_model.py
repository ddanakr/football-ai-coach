"""The video model of the episode description: the setting `AICOACH_VLM_MODEL`, the factory, and
the agent that gets (or does not get) the `describe_episode` tool."""

from pathlib import Path

import pytest

from aicoach.agent import service
from aicoach.agent.service import build_runtime
from aicoach.config import DEFAULT_VLM_MODEL, load_settings
from aicoach.errors import ConfigError
from aicoach.spend import MeteredVideoModel
from tests.fixtures.agent_synth import FakeLLM
from tests.fixtures.episode_synth import FakeClipper, FakeVideoModel
from tests.fixtures.ui_synth import write_match_dir

KEY = "key-for-the-fake-client"  # pragma: allowlist secret


@pytest.mark.unit
def test_the_default_is_a_video_model_slug_and_the_factory_wraps_it_in_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)

    model = service.make_video_model(load_settings(), tmp_path / "match01")

    assert isinstance(model, MeteredVideoModel)
    assert model.model == DEFAULT_VLM_MODEL == "google/gemini-3.7-flash"


@pytest.mark.unit
def test_the_setting_beats_the_default_and_the_flag_beats_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setenv("AICOACH_VLM_MODEL", "qwen/qwen3.8-omni-flash")
    settings = load_settings()

    assert service.make_video_model(settings, tmp_path).model == "qwen/qwen3.8-omni-flash"
    assert service.make_video_model(settings, tmp_path, "x/y").model == "x/y"
    assert service.make_video_model(settings, tmp_path, "  ").model == "qwen/qwen3.8-omni-flash"


@pytest.mark.unit
def test_the_setting_is_read_from_the_env_file_and_blank_means_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    env_file = tmp_path / ".env"
    env_file.write_text("AICOACH_VLM_MODEL=model-from-file\n", encoding="utf-8")
    assert load_settings(env_file=env_file).aicoach_vlm_model == "model-from-file"

    env_file.write_text("AICOACH_VLM_MODEL=   \n", encoding="utf-8")
    assert load_settings(env_file=env_file).aicoach_vlm_model == DEFAULT_VLM_MODEL
    monkeypatch.setenv("AICOACH_VLM_MODEL", "   ")
    assert load_settings().aicoach_vlm_model == DEFAULT_VLM_MODEL


@pytest.mark.unit
def test_the_video_model_needs_the_openrouter_key_even_when_the_agent_runs_on_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    settings = load_settings(llm_provider="gemini")

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        service.make_video_model(settings, tmp_path)
    assert service.optional_video_model(settings, tmp_path) is None


@pytest.mark.unit
def test_optional_video_model_is_the_model_when_the_key_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)

    assert service.optional_video_model(load_settings(), tmp_path) is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("with_video", "with_model", "offered"),
    [(True, True, True), (False, True, False), (True, False, False), (False, False, False)],
)
def test_the_tool_needs_both_the_video_and_the_video_model(
    tmp_path: Path, with_video: bool, with_model: bool, offered: bool
) -> None:
    db = write_match_dir(tmp_path, "m1")
    runtime = build_runtime(
        FakeLLM([]),
        db,
        "m1",
        tmp_path,
        tmp_path / "src.mp4" if with_video else None,
        FakeVideoModel() if with_model else None,
    )

    names = [s.name for s in runtime.box_factory().specs()]

    assert ("describe_episode" in names) is offered


@pytest.mark.unit
def test_the_tool_describes_the_current_event_through_the_cache_and_the_video_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = write_match_dir(tmp_path, "m1")
    clipper = FakeClipper()
    monkeypatch.setattr("aicoach.episode.export_clip", clipper, raising=True)
    vlm = FakeVideoModel()
    runtime = build_runtime(FakeLLM([]), db, "m1", tmp_path, tmp_path / "src.mp4", vlm)
    box = runtime.box_factory()
    event_id = box.ctx.events[0].event.event_id

    result = box.call("describe_episode", {"event_id": event_id})

    assert result["status"] == "described_from_clip_unverified"
    assert len(vlm.requests) == 1
    assert result["description"]
    assert [c[1].event_id for c in clipper.calls] == [event_id]


@pytest.mark.unit
def test_an_unknown_event_id_of_the_tool_is_an_error_and_the_video_model_is_not_asked(
    tmp_path: Path,
) -> None:
    db = write_match_dir(tmp_path, "m1")
    vlm = FakeVideoModel()
    runtime = build_runtime(FakeLLM([]), db, "m1", tmp_path, tmp_path / "src.mp4", vlm)

    result = runtime.box_factory().call("describe_episode", {"event_id": "p:999"})

    assert "нет в данных матча" in result["error"]
    assert vlm.requests == []


@pytest.mark.unit
def test_without_the_video_file_the_tool_reports_that_no_description_was_made(
    tmp_path: Path,
) -> None:
    db = write_match_dir(tmp_path, "m1")
    vlm = FakeVideoModel()
    runtime = build_runtime(FakeLLM([]), db, "m1", tmp_path, tmp_path / "missing.mp4", vlm)
    box = runtime.box_factory()

    result = box.call("describe_episode", {"event_id": box.ctx.events[0].event.event_id})

    assert result == {"error": "описание эпизода не получено"}  # the clip could not be cut
    assert vlm.requests == []
