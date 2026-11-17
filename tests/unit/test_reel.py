"""Reel export (`aicoach.reel`): the choice of events, captions, the file name and cache, the
FFmpeg commands, the join and its fallback. FFmpeg is not run: `ReelRunner` records the commands
and writes the output files. The real FFmpeg run is in `tests/integration/test_reel_ffmpeg.py`."""

import json
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.agent.context import EventFacts
from aicoach.clips import CLIPS_DIR
from aicoach.domain.events import EventType
from aicoach.errors import ExportError, StorageError
from aicoach.reel import (
    CARD,
    KINDS,
    REELS_DIR,
    ReelFilter,
    ReelOptions,
    ReelResult,
    Selection,
    build_concat_command,
    build_reel,
    caption_text,
    concat_list_text,
    export_event_reel,
    find_font,
    media_duration,
    reel_filename,
    select_events,
    stream_signature,
)
from tests.fixtures.reel_synth import ReelRunner, fake_probe, make_facts, match_facts
from tests.fixtures.ui_synth import write_match_dir


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.fixture
def font(tmp_path: Path) -> Path:
    path = tmp_path / "font.ttf"
    path.write_bytes(b"not a real font: the fake FFmpeg never reads it")
    return path


def ids(selection: Selection) -> list[str]:
    return [f.event.event_id for f in selection.events]


def build(
    tmp_path: Path,
    facts: list[EventFacts],
    flt: ReelFilter,
    runner: ReelRunner,
    options: ReelOptions,
) -> ReelResult:
    return build_reel(
        tmp_path / "src.mp4", tmp_path, facts, flt, options, runner=runner, probe=fake_probe
    )


# --- select_events -------------------------------------------------------------------------


@pytest.mark.unit
def test_select_events_kind_returns_events_in_time_order() -> None:
    selection = select_events(match_facts(), ReelFilter("goal"))

    assert ids(selection) == ["p:1", "e:5", "p:9"]


@pytest.mark.unit
def test_select_events_half_keeps_only_that_half_and_drops_unknown_half() -> None:
    first = select_events(match_facts(), ReelFilter("goal", half=1))
    second = select_events(match_facts(), ReelFilter("goal", half=2))

    assert ids(first) == ["p:1"]  # e:5 has no half: it is in neither
    assert ids(second) == ["p:9"]


@pytest.mark.unit
def test_select_events_side_skips_unknown_team_and_counts_it() -> None:
    selection = select_events(match_facts(), ReelFilter("goal", side="home"))

    assert ids(selection) == ["p:1"]
    assert selection.team_unknown == 1  # e:5 has no team: not guessed


@pytest.mark.unit
def test_select_events_card_returns_only_fouls_with_a_card() -> None:
    assert ids(select_events(match_facts(), ReelFilter(CARD))) == ["p:2"]
    assert ids(select_events(match_facts(), ReelFilter("foul"))) == ["p:2", "p:3"]


@pytest.mark.unit
def test_select_events_card_never_takes_a_card_on_another_type() -> None:
    odd = make_facts("p:7", EventType.GOAL, 10.0, card="yellow")

    assert ids(select_events([odd], ReelFilter(CARD))) == []


@pytest.mark.unit
def test_select_events_empty_input_returns_empty_selection() -> None:
    assert select_events([], ReelFilter("goal")) == Selection(())


@pytest.mark.unit
def test_select_events_equal_times_are_ordered_by_id() -> None:
    twins = [
        make_facts("p:8", EventType.SHOT, 50.0),
        make_facts("p:3", EventType.SHOT, 50.0),
    ]

    assert ids(select_events(twins, ReelFilter("shot"))) == ["p:3", "p:8"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "flt",
    [ReelFilter("penalty"), ReelFilter("goal", half=3), ReelFilter("goal", side="both")],
)
def test_select_events_bad_filter_raises_value_error(flt: ReelFilter) -> None:
    with pytest.raises(ValueError, match="reel"):
        select_events(match_facts(), flt)


@pytest.mark.unit
def test_kinds_are_the_five_categories_and_card() -> None:
    assert KINDS == ("goal", "shot", "corner", "foul", "subst", "card")


# --- captions and names --------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "event_type", "minute", "expected"),
    [
        ("goal", EventType.GOAL, 5, "Гол · 5'"),
        ("subst", EventType.SUBST, 30, "Замена · 30'"),
        (CARD, EventType.FOUL, 7, "Карточка · 7'"),
        ("goal", EventType.GOAL, None, "Гол"),
    ],
)
def test_caption_text_has_category_and_minute(
    kind: str, event_type: EventType, minute: int | None, expected: str
) -> None:
    assert caption_text(make_facts("p:1", event_type, 10.0, minute=minute), kind) == expected


@pytest.mark.unit
def test_caption_text_stoppage_time_is_written_as_in_the_viewer() -> None:
    facts = make_facts("p:1", EventType.GOAL, 10.0, half=1, minute=47)

    assert caption_text(facts, "goal") == "Гол · 45+2'"


@pytest.mark.unit
def test_find_font_picks_the_first_existing_candidate(tmp_path: Path) -> None:
    second = tmp_path / "b.ttf"
    second.write_bytes(b"x")

    assert find_font(None, [tmp_path / "a.ttf", second]) == second
    assert find_font(None, [tmp_path / "a.ttf"]) is None
    assert find_font(None, []) is None


@pytest.mark.unit
def test_find_font_configured_file_is_not_replaced_by_a_system_font(tmp_path: Path) -> None:
    system = tmp_path / "system.ttf"
    system.write_bytes(b"x")

    assert find_font(tmp_path / "missing.ttf", [system]) is None
    assert find_font(system, []) == system


@pytest.mark.unit
def test_reel_filename_is_stable_and_follows_events_and_options() -> None:
    flt = ReelFilter("goal", half=1, side="home")
    chosen = select_events(match_facts(), ReelFilter("goal"))
    moved_facts = match_facts()
    moved_facts[1] = make_facts("p:1", EventType.GOAL, 601.0, team="home", minute=5)
    moved = select_events(moved_facts, ReelFilter("goal"))
    options = ReelOptions()

    name = reel_filename(flt, chosen, options, captioned=True)

    assert name == reel_filename(flt, chosen, options, captioned=True)
    assert name.startswith("reel_goal_h1_home_")
    assert name.endswith(".mp4")
    assert name != reel_filename(flt, moved, options, captioned=True)
    assert name != reel_filename(flt, chosen, options, captioned=False)
    assert name != reel_filename(flt, chosen, ReelOptions(before_s=5.0), captioned=True)


@pytest.mark.unit
def test_concat_list_text_quotes_each_path_and_escapes_apostrophes(tmp_path: Path) -> None:
    text = concat_list_text([tmp_path / "a.mp4", tmp_path / "it's.mp4"])

    lines = text.splitlines()
    assert lines[0].startswith("file '") and lines[0].endswith("a.mp4'")
    assert lines[1].endswith("it'\\''s.mp4'")


@pytest.mark.unit
def test_build_concat_command_copies_streams_or_reencodes() -> None:
    copy = build_concat_command(Path("l.txt"), Path("o.mp4"), reencode=False)
    encode = build_concat_command(Path("l.txt"), Path("o.mp4"), reencode=True)

    assert copy[copy.index("-c") + 1] == "copy"
    assert "libx264" not in copy
    assert "-c" not in encode
    assert encode[encode.index("-c:v") + 1] == "libx264"
    assert copy[:1] == encode[:1] == ["ffmpeg"]
    assert copy[copy.index("-safe") + 1] == "0"
    assert copy[-1] == "o.mp4"


# --- FFprobe helpers -----------------------------------------------------------------------


@pytest.mark.unit
def test_stream_signature_same_streams_in_any_order_are_equal(tmp_path: Path) -> None:
    video = {"codec_type": "video", "codec_name": "h264"}
    audio = {"codec_type": "audio", "codec_name": "aac"}
    one = ReelRunner(signatures={"a.mp4": [video, audio], "b.mp4": [audio, video]})

    assert stream_signature(tmp_path / "a.mp4", one) == stream_signature(tmp_path / "b.mp4", one)


@pytest.mark.unit
def test_stream_signature_without_ffprobe_is_none(tmp_path: Path) -> None:
    assert stream_signature(tmp_path / "a.mp4", ReelRunner(ffprobe_missing=True)) is None


@pytest.mark.unit
def test_stream_signature_garbage_output_is_none(tmp_path: Path) -> None:
    def garbage(command: list[str], _t: float) -> "subprocess.CompletedProcess[bytes]":
        return subprocess.CompletedProcess(command, 0, b"not json", b"")

    def failing(command: list[str], _t: float) -> "subprocess.CompletedProcess[bytes]":
        return subprocess.CompletedProcess(command, 1, b"", b"error")

    assert stream_signature(tmp_path / "a.mp4", garbage) is None
    assert stream_signature(tmp_path / "a.mp4", failing) is None
    assert media_duration(tmp_path / "a.mp4", garbage) is None
    for value in ("NaN", "Infinity", "-1"):
        odd = _duration_runner(value)
        assert media_duration(tmp_path / "a.mp4", odd) is None
    assert media_duration(tmp_path / "a.mp4", ReelRunner(ffprobe_missing=True)) is None


def _duration_runner(
    value: str,
) -> Callable[[list[str], float], "subprocess.CompletedProcess[bytes]"]:
    def run(command: list[str], _t: float) -> "subprocess.CompletedProcess[bytes]":
        body = json.dumps({"format": {"duration": value}}).encode()
        return subprocess.CompletedProcess(command, 0, body, b"")

    return run


# --- build_reel ----------------------------------------------------------------------------


@pytest.mark.unit
def test_build_reel_cuts_every_event_in_time_order_and_joins_without_encoding(
    tmp_path: Path, font: Path
) -> None:
    runner = ReelRunner()
    result = build(tmp_path, match_facts(), ReelFilter("goal"), runner, ReelOptions(font_file=font))

    assert result.n_events == 3
    assert result.path.parent == tmp_path / CLIPS_DIR / REELS_DIR
    assert result.path.is_file()
    cuts = runner.cuts()
    assert [c[c.index("-ss") + 1] for c in cuts] == ["592.000", "992.000", "2992.000"]
    assert all("-vf" in c and "drawtext" in c[c.index("-vf") + 1] for c in cuts)
    (join,) = runner.joins()
    assert join[join.index("-c") + 1] == "copy"
    assert result.captioned and not result.reencoded and not result.cached
    assert result.duration_s == 40.0
    assert result.warnings == ()


@pytest.mark.unit
def test_build_reel_without_a_font_has_no_captions_and_a_warning(tmp_path: Path) -> None:
    runner = ReelRunner()
    options = ReelOptions(font_file=tmp_path / "missing.ttf")

    result = build(tmp_path, match_facts(), ReelFilter("goal"), runner, options)

    assert not result.captioned
    assert str(tmp_path / "missing.ttf") in result.warnings[0]
    assert all("-vf" not in c for c in runner.cuts())
    assert len(runner.joins()) == 1


@pytest.mark.unit
def test_build_reel_second_call_returns_the_finished_file_without_ffmpeg(
    tmp_path: Path, font: Path
) -> None:
    runner = ReelRunner()
    options = ReelOptions(font_file=font)
    first = build(tmp_path, match_facts(), ReelFilter("goal"), runner, options)
    n_calls = len(runner.calls)

    second = build(tmp_path, match_facts(), ReelFilter("goal"), runner, options)

    assert second.cached and second.path == first.path
    assert [c for c in runner.calls[n_calls:] if c[0] == "ffmpeg"] == []


@pytest.mark.unit
def test_build_reel_moved_event_gives_another_reel(tmp_path: Path, font: Path) -> None:
    runner = ReelRunner()
    options = ReelOptions(font_file=font)
    first = build(tmp_path, match_facts(), ReelFilter("goal"), runner, options)
    facts = match_facts()
    facts[1] = make_facts("p:1", EventType.GOAL, 650.0, team="home", minute=5)

    second = build(tmp_path, facts, ReelFilter("goal"), runner, options)

    assert not second.cached and second.path != first.path


@pytest.mark.unit
def test_build_reel_more_events_than_max_raises_with_the_counts_and_cuts_nothing(
    tmp_path: Path, font: Path
) -> None:
    runner = ReelRunner()

    with pytest.raises(ExportError, match=r"3 events.*--max 2"):
        build(
            tmp_path,
            match_facts(),
            ReelFilter("goal"),
            runner,
            ReelOptions(font_file=font, max_events=2),
        )

    assert runner.calls == []
    assert not (tmp_path / CLIPS_DIR).exists()


@pytest.mark.unit
def test_build_reel_exactly_max_events_is_allowed(tmp_path: Path, font: Path) -> None:
    result = build(
        tmp_path,
        match_facts(),
        ReelFilter("goal"),
        ReelRunner(),
        ReelOptions(font_file=font, max_events=3),
    )

    assert result.n_events == 3


@pytest.mark.unit
def test_build_reel_no_matching_event_raises(tmp_path: Path, font: Path) -> None:
    runner = ReelRunner()

    with pytest.raises(ExportError, match="no events"):
        build(tmp_path, match_facts(), ReelFilter("subst"), runner, ReelOptions(font_file=font))
    with pytest.raises(ExportError, match="no events"):
        build(tmp_path, [], ReelFilter("goal"), runner, ReelOptions(font_file=font))

    assert runner.calls == []


@pytest.mark.unit
def test_build_reel_reports_events_skipped_for_an_unknown_team(tmp_path: Path, font: Path) -> None:
    result = build(
        tmp_path,
        match_facts(),
        ReelFilter("goal", side="away"),
        ReelRunner(),
        ReelOptions(font_file=font),
    )

    assert result.n_events == 1 and result.team_unknown == 1


@pytest.mark.unit
def test_build_reel_pieces_with_different_streams_are_reencoded(tmp_path: Path, font: Path) -> None:
    other = [{"codec_type": "video", "codec_name": "h264", "width": "128"}]
    runner = ReelRunner(signatures={"p1_goal": other})  # one piece differs from the others

    result = build(tmp_path, match_facts(), ReelFilter("goal"), runner, ReelOptions(font_file=font))

    assert result.reencoded
    (join,) = runner.joins()
    assert join[join.index("-c:v") + 1] == "libx264"


@pytest.mark.unit
def test_build_reel_without_ffprobe_reencodes_the_join(tmp_path: Path, font: Path) -> None:
    runner = ReelRunner(ffprobe_missing=True)

    result = build(tmp_path, match_facts(), ReelFilter("goal"), runner, ReelOptions(font_file=font))

    assert result.reencoded
    assert result.duration_s is None
    assert runner.joins()[0].count("libx264") == 1


@pytest.mark.unit
def test_build_reel_reports_progress_before_and_after_every_piece(
    tmp_path: Path, font: Path
) -> None:
    seen: list[tuple[int, int]] = []

    build_reel(
        tmp_path / "src.mp4",
        tmp_path,
        match_facts(),
        ReelFilter("goal"),
        ReelOptions(font_file=font),
        progress=lambda done, total: seen.append((done, total)),
        runner=ReelRunner(),
        probe=fake_probe,
    )

    assert seen == [(0, 3), (1, 3), (2, 3), (3, 3)]


@pytest.mark.unit
def test_build_reel_failed_cut_in_the_middle_stops_before_the_join(
    tmp_path: Path, font: Path
) -> None:
    runner = ReelRunner(fail_cut_number=2)  # the second of the three pieces fails

    with pytest.raises(ExportError, match="cut failed"):
        build(tmp_path, match_facts(), ReelFilter("goal"), runner, ReelOptions(font_file=font))

    assert runner.joins() == []
    assert len(list((tmp_path / CLIPS_DIR).glob("*.mp4"))) == 1  # the first piece stays cached
    assert list((tmp_path / CLIPS_DIR).glob("*.part*")) == []


@pytest.mark.unit
def test_build_reel_failed_join_raises_and_leaves_no_partial_files(
    tmp_path: Path, font: Path
) -> None:
    runner = ReelRunner(fail_join=True)

    with pytest.raises(ExportError, match="join failed"):
        build(tmp_path, match_facts(), ReelFilter("goal"), runner, ReelOptions(font_file=font))

    leftovers = [p.name for p in (tmp_path / CLIPS_DIR / REELS_DIR).iterdir()]
    assert leftovers == []  # no half-written reel and no list file
    assert len(list((tmp_path / CLIPS_DIR).glob("*.mp4"))) == 3  # the pieces stay cached


@pytest.mark.unit
def test_build_reel_event_outside_the_video_raises_export_error(tmp_path: Path, font: Path) -> None:
    far = [make_facts("p:1", EventType.GOAL, 9000.0, minute=5)]  # the fake video is 4000 s

    with pytest.raises(ExportError, match="outside the video"):
        build(tmp_path, far, ReelFilter("goal"), ReelRunner(), ReelOptions(font_file=font))


# --- from the database ---------------------------------------------------------------------


@pytest.mark.unit
def test_export_event_reel_card_takes_the_foul_that_a_card_plate_made(
    tmp_path: Path, font: Path
) -> None:
    db = write_match_dir(tmp_path)
    runner = ReelRunner()

    result = export_event_reel(
        db,
        "m1",
        tmp_path,
        tmp_path / "src.mp4",
        ReelFilter(CARD),
        ReelOptions(font_file=font),
        runner=runner,
        probe=fake_probe,
    )

    assert result.n_events == 1
    captions = [
        p.read_text(encoding="utf-8") for p in (tmp_path / CLIPS_DIR).glob("*.txt")
    ]  # removed after each cut
    assert captions == []
    (cut,) = runner.cuts()
    assert "drawtext" in cut[cut.index("-vf") + 1]


@pytest.mark.unit
def test_export_event_reel_missing_database_raises_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="does not exist"):
        export_event_reel(
            tmp_path / "none.db", "m1", tmp_path, tmp_path / "src.mp4", ReelFilter("goal")
        )
