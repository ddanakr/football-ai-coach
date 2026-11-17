"""The episode description: the request, the cache, the failures (no network, no FFmpeg)."""

import json
from pathlib import Path

import pytest

from aicoach.domain.events import EventType
from aicoach.episode import (
    DESCRIPTIONS_DIR,
    MAX_TEXT_CHARS,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt,
    cache_path,
    describe_episode,
)
from aicoach.errors import ExportError, ExternalServiceError
from aicoach.storage import CurrentEvent
from aicoach.vlm import VideoReply
from tests.fixtures.episode_synth import TEXT, FakeClipper, FakeVideoModel, reply
from tests.fixtures.ui_synth import make_event

EVENT = make_event("p:12", EventType.GOAL, 1055.0)
MODEL = "test/vlm-1"


def setup(tmp_path: Path) -> tuple[Path, FakeClipper, FakeVideoModel]:
    return tmp_path / "src.mp4", FakeClipper(), FakeVideoModel()


def stored_path(tmp_path: Path) -> Path:
    return cache_path(tmp_path, EVENT, MODEL, 8.0, 12.0)


@pytest.mark.unit
def test_the_request_forbids_names_and_numbers_of_players_and_invention() -> None:
    text = SYSTEM_PROMPT.casefold()

    assert "не называй фамилии" in text
    assert "номера игроков" in text
    assert "не придумывай" in text
    assert "не видно" in text  # the way out when in doubt
    assert "не инструкции" in text  # on-screen text and commentary are data
    assert "по-русски" in text
    assert "2–4 предложения" in text


# Stems of the words that name an event type or hint at one (D-074).
TYPE_STEMS = ("гол", "удар", "угловой", "фол", "замен", "карточк", "нарушен", "штраф")


@pytest.mark.unit
@pytest.mark.parametrize("kind", list(EventType))
def test_the_request_does_not_tell_the_type_of_the_event(tmp_path: Path, kind: EventType) -> None:
    video, clipper, model = setup(tmp_path)

    describe_episode(video, make_event("p:3", kind, 500.0), model, clipper=clipper)

    ((system, prompt, _, _),) = model.requests
    request = f"{system} {prompt}".casefold()
    assert [stem for stem in TYPE_STEMS if stem in request] == []
    assert kind.value not in request


@pytest.mark.unit
def test_the_user_part_has_only_the_second_of_the_moment_and_no_names() -> None:
    prompt = build_prompt(8.0)

    assert "около 8-й секунды" in prompt
    assert len(prompt) < 200  # nothing from plates or lineups fits in


@pytest.mark.unit
def test_the_prompt_version_was_raised_so_old_descriptions_are_stale() -> None:
    assert PROMPT_VERSION >= 2


@pytest.mark.unit
def test_cache_name_has_no_colon_or_slash_and_no_names() -> None:
    path = cache_path(Path("m"), EVENT, "google/gemini-3.7-flash", 8.0, 12.0)

    expected = "p12_goal_t1055000ms_google_gemini-3.7-flash_b8_a12.json"
    assert path == Path("m") / DESCRIPTIONS_DIR / expected


@pytest.mark.unit
def test_first_call_cuts_the_clip_asks_the_model_and_stores_the_description(
    tmp_path: Path,
) -> None:
    video, clipper, model = setup(tmp_path)

    result = describe_episode(video, EVENT, model, clipper=clipper)

    assert result.text == TEXT
    assert (result.model, result.event_id, result.cached) == (MODEL, "p:12", False)
    assert (result.tokens_in, result.tokens_out, result.cost_usd) == (2100, 480, 0.0031)
    assert len(clipper.calls) == 1
    assert clipper.calls[0][2:] == (8.0, 12.0)
    ((system, prompt, data, seconds),) = model.requests
    assert system == SYSTEM_PROMPT
    assert "около 8-й секунды" in prompt
    assert data == b"fake mp4"
    assert seconds == 20.0
    stored = json.loads(stored_path(tmp_path).read_text("utf-8"))
    assert stored["text"] == TEXT
    assert "cached" not in stored
    assert stored["prompt_version"] == PROMPT_VERSION


@pytest.mark.unit
def test_second_call_comes_from_the_cache_without_a_clip_or_a_request(tmp_path: Path) -> None:
    video, clipper, model = setup(tmp_path)
    describe_episode(video, EVENT, model, clipper=clipper)

    again = describe_episode(video, EVENT, model, clipper=clipper)

    assert again.cached
    assert again.text == TEXT
    assert len(model.requests) == 1
    assert len(clipper.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event", "model_name", "before", "after"),
    [
        (make_event("p:12", EventType.GOAL, 1056.0), MODEL, 8.0, 12.0),  # the event was moved
        (EVENT, "test/vlm-2", 8.0, 12.0),  # another model
        (EVENT, MODEL, 15.0, 12.0),  # another window
        (EVENT, MODEL, 8.0, 5.0),
    ],
    ids=["moved", "model", "before", "after"],
)
def test_a_changed_key_part_makes_a_new_request(
    tmp_path: Path, event: CurrentEvent, model_name: str, before: float, after: float
) -> None:
    video, clipper, model = setup(tmp_path)
    describe_episode(video, EVENT, model, clipper=clipper)
    other = FakeVideoModel(model=model_name)

    result = describe_episode(video, event, other, before_s=before, after_s=after, clipper=clipper)

    assert not result.cached
    assert len(other.requests) == 1


@pytest.mark.unit
def test_a_description_of_an_older_prompt_is_not_returned(tmp_path: Path) -> None:
    video, clipper, model = setup(tmp_path)
    describe_episode(video, EVENT, model, clipper=clipper)
    path = stored_path(tmp_path)
    stored = json.loads(path.read_text("utf-8"))
    stored["prompt_version"] = PROMPT_VERSION - 1
    path.write_text(json.dumps(stored), encoding="utf-8")

    result = describe_episode(video, EVENT, model, clipper=clipper)

    assert not result.cached
    assert len(model.requests) == 2


@pytest.mark.unit
@pytest.mark.parametrize("garbage", ["", "not json", "[]", '{"text": 5}', "{"])
def test_an_unreadable_cache_file_is_a_miss_that_is_overwritten(
    tmp_path: Path, garbage: str
) -> None:
    video, clipper, model = setup(tmp_path)
    path = stored_path(tmp_path)
    path.parent.mkdir()
    path.write_text(garbage, encoding="utf-8")

    result = describe_episode(video, EVENT, model, clipper=clipper)

    assert not result.cached
    assert json.loads(path.read_text("utf-8"))["text"] == TEXT


@pytest.mark.unit
def test_text_is_one_paragraph_and_cut_to_the_limit(tmp_path: Path) -> None:
    video, clipper, _ = setup(tmp_path)
    model = FakeVideoModel([reply("Слово  \n" * 600)])

    result = describe_episode(video, EVENT, model, clipper=clipper)

    assert "\n" not in result.text
    assert "  " not in result.text
    assert len(result.text) == MAX_TEXT_CHARS
    assert result.text.endswith("…")


@pytest.mark.unit
@pytest.mark.parametrize("bad", [reply(""), reply("   \n "), reply("Игрок бьёт", truncated=True)])
def test_an_empty_or_cut_off_answer_is_an_error_and_is_not_cached(
    tmp_path: Path, bad: VideoReply
) -> None:
    video, clipper, _ = setup(tmp_path)

    with pytest.raises(ExternalServiceError, match="no usable description"):
        describe_episode(video, EVENT, FakeVideoModel([bad]), clipper=clipper)

    assert not stored_path(tmp_path).exists()


@pytest.mark.unit
def test_a_model_failure_propagates_and_nothing_is_cached(tmp_path: Path) -> None:
    video, clipper, _ = setup(tmp_path)
    model = FakeVideoModel([ExternalServiceError("OpenRouter request failed (HTTP 500)")])

    with pytest.raises(ExternalServiceError, match="500"):
        describe_episode(video, EVENT, model, clipper=clipper)

    assert not (tmp_path / DESCRIPTIONS_DIR).exists()


@pytest.mark.unit
def test_a_clip_failure_stops_before_any_request(tmp_path: Path) -> None:
    model = FakeVideoModel()

    with pytest.raises(ExportError, match="ffmpeg"):
        describe_episode(
            tmp_path / "src.mp4", EVENT, model, clipper=FakeClipper(ExportError("ffmpeg gone"))
        )

    assert model.requests == []


@pytest.mark.unit
def test_a_clip_that_cannot_be_read_is_an_export_error(tmp_path: Path) -> None:
    def vanished(video: Path, event: CurrentEvent, before: float, after: float) -> Path:  # noqa: ARG001
        return video.parent / "clips" / "gone.mp4"

    model = FakeVideoModel()

    with pytest.raises(ExportError, match="cannot read the clip"):
        describe_episode(tmp_path / "src.mp4", EVENT, model, clipper=vanished)

    assert model.requests == []


@pytest.mark.unit
def test_near_the_start_of_the_video_the_hint_uses_the_real_offset(tmp_path: Path) -> None:
    video, clipper, model = setup(tmp_path)
    early = make_event("p:1", EventType.CORNER, 3.0)

    describe_episode(video, early, model, clipper=clipper)

    assert "около 3-й секунды" in model.requests[0][1]  # the clip starts at 0, not at -5


@pytest.mark.unit
@pytest.mark.parametrize(
    ("before", "after"),
    [(-1.0, 12.0), (8.0, -0.5), (float("nan"), 12.0), (8.0, float("inf")), (40.0, 30.0)],
)
def test_a_negative_nan_or_huge_window_is_refused_before_any_clip_or_request(
    tmp_path: Path, before: float, after: float
) -> None:
    video, clipper, model = setup(tmp_path)

    with pytest.raises(ExportError, match="window"):
        describe_episode(video, EVENT, model, before_s=before, after_s=after, clipper=clipper)

    assert clipper.calls == []
    assert model.requests == []


@pytest.mark.unit
def test_a_window_exactly_at_the_limit_is_allowed(tmp_path: Path) -> None:
    video, clipper, model = setup(tmp_path)

    result = describe_episode(video, EVENT, model, before_s=30.0, after_s=30.0, clipper=clipper)

    assert not result.cached
