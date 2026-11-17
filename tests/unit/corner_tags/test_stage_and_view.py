"""The `corner_tags` stage, the views and filters built from its file (D-078)."""

from pathlib import Path

import pytest

from aicoach.corner_tags.schemas import RESULT_FILE, CornerFields, CornerTagsFile, read_tags_file
from aicoach.corner_tags.stage import CornerTagsConfig, run_corner_tags
from aicoach.corner_tags.view import (
    CornerFilter,
    CornerView,
    apply_filter,
    build_views,
    describe,
    load_views,
    matches,
    team_label,
    texts,
    value_label,
)
from aicoach.domain.events import EventType
from aicoach.errors import ConfigError, ExportError, ExternalServiceError
from aicoach.status import read_status
from aicoach.teams.store import EffectiveTeam, Side
from tests.fixtures.corner_synth import (
    CORNER_JSON,
    DEFAULT_SPECS,
    MATCH_ID,
    corner_reply,
    write_db,
    write_video,
)
from tests.fixtures.episode_synth import FakeClipper, FakeVideoModel, reply
from tests.fixtures.ui_synth import make_event

TEAMS: dict[Side, EffectiveTeam] = {
    "home": EffectiveTeam(name="Актобе", name_source="lineup", kit="white", kit_source="vlm"),
    "away": EffectiveTeam(name="Кайрат", name_source="lineup", kit="red", kit_source="vlm"),
}


class Setup:
    def __init__(self, tmp_path: Path, script: list[object] | None = None) -> None:
        self.dir = tmp_path / "match"
        self.db = write_db(self.dir)
        self.video = write_video(self.dir)
        self.clipper = FakeClipper()
        self.model = FakeVideoModel(
            [corner_reply(), corner_reply(), corner_reply()] if script is None else script  # type: ignore[arg-type]
        )

    def run(self, **kwargs: object) -> CornerTagsFile | None:
        return run_corner_tags(
            self.dir,
            MATCH_ID,
            db_path=self.db,
            video=self.video,
            model=self.model,
            clipper=self.clipper,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_run_corner_tags_tags_every_corner_in_time_order_with_its_half(tmp_path: Path) -> None:
    s = Setup(tmp_path, [corner_reply(attacking_kit="red"), corner_reply(), corner_reply()])

    result = s.run()

    assert result is not None
    assert [t.event_id for t in result.tags] == ["p:3", "p:5", "p:7"]
    assert [t.half for t in result.tags] == [1, 2, 2]
    assert result.tags[0].fields.attacking_kit == "red"
    assert {t.status for t in result.tags} == {"ok"}
    assert read_tags_file(s.dir) == result
    assert read_status(s.dir).stages["corner_tags"].state == "done"


@pytest.mark.unit
def test_run_corner_tags_asks_with_the_clip_window_4_and_12_and_the_strict_question(
    tmp_path: Path,
) -> None:
    s = Setup(tmp_path)

    s.run()

    assert {c[2:] for c in s.clipper.calls} == {(4.0, 12.0)}
    for system, prompt, video, seconds in s.model.requests:
        assert "JSON" in system
        assert "4-й секунды" in prompt
        assert video == b"fake mp4"
        assert seconds == 16.0


@pytest.mark.unit
def test_run_corner_tags_second_run_asks_nothing_and_force_asks_again(tmp_path: Path) -> None:
    s = Setup(tmp_path, [corner_reply()] * 6)
    s.run()
    assert len(s.model.requests) == 3

    s.run()
    assert len(s.model.requests) == 3

    s.run(force=True)
    assert len(s.model.requests) == 6


@pytest.mark.unit
def test_run_corner_tags_goes_on_from_the_answers_kept_after_a_failure(tmp_path: Path) -> None:
    s = Setup(tmp_path, [corner_reply(), ExternalServiceError("502")])

    with pytest.raises(ExternalServiceError):
        s.run()

    kept = read_tags_file(s.dir)
    assert kept is not None
    assert [t.event_id for t in kept.tags] == ["p:3"]
    assert read_status(s.dir).stages["corner_tags"].state == "failed"

    s.model = FakeVideoModel([corner_reply(), corner_reply()])
    result = s.run()

    assert result is not None
    assert len(result.tags) == 3
    assert len(s.model.requests) == 2  # the first corner is not asked again


@pytest.mark.unit
def test_run_corner_tags_retries_an_unusable_answer_once(tmp_path: Path) -> None:
    s = Setup(tmp_path, [reply("???", 0.001), corner_reply(0.004), corner_reply(), corner_reply()])

    result = s.run()

    assert result is not None
    first = result.tags[0]
    assert first.status == "retried"
    assert first.cost_usd == pytest.approx(0.005)
    assert first.fields.zone == "far_post"


@pytest.mark.unit
def test_run_corner_tags_two_unusable_answers_become_unclear_everywhere(tmp_path: Path) -> None:
    s = Setup(tmp_path, [reply("a"), reply("{}"), corner_reply(), corner_reply()])

    result = s.run()

    assert result is not None
    first = result.tags[0]
    assert first.status == "invalid"
    assert first.fields == CornerFields.all_unclear()
    assert len(s.model.requests) == 4  # two for the first corner, one each for the others


@pytest.mark.unit
def test_run_corner_tags_a_new_model_asks_again(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.run()
    s.model = FakeVideoModel([corner_reply()] * 3, model="test/other")

    s.run()

    assert len(s.model.requests) == 3


@pytest.mark.unit
def test_run_corner_tags_a_moved_corner_is_asked_again(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.run()
    s.db.unlink()
    moved = [(t, 1003.0 if t_v == 1000.0 else t_v, h, team) for t, t_v, h, team in DEFAULT_SPECS]
    s.db = write_db(s.dir, moved)  # the same ids, the first corner 3 s later
    s.model = FakeVideoModel([corner_reply()] * 3)

    result = s.run()

    assert result is not None
    assert len(s.model.requests) == 1  # only the moved corner is asked again
    assert [t.event_id for t in result.tags] == ["p:3", "p:5", "p:7"]
    assert result.tags[0].t_v == 1003.0


@pytest.mark.unit
def test_run_corner_tags_a_clip_that_cannot_be_cut_is_noted_and_skipped(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.clipper = FakeClipper(raises=ExportError("ffmpeg failed"))

    result = s.run()

    assert result is not None
    assert result.tags == []
    assert len(result.notes) == 3


@pytest.mark.unit
def test_run_corner_tags_without_a_database_or_a_key_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Setup(tmp_path)
    s.db.unlink()

    assert s.run() is None
    assert read_status(s.dir).stages["corner_tags"].state == "skipped"

    s.db = write_db(s.dir)

    def no_key(*_args: object, **_kwargs: object) -> object:
        msg = "OPENROUTER_API_KEY is not set"
        raise ConfigError(msg)

    monkeypatch.setattr("aicoach.corner_tags.stage.make_video_model", no_key)
    skipped = run_corner_tags(s.dir, MATCH_ID, db_path=s.db, video=s.video, clipper=s.clipper)

    assert skipped is None
    assert not (s.dir / RESULT_FILE).exists()


@pytest.mark.unit
def test_run_corner_tags_without_the_video_is_skipped(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.video.unlink()

    assert s.run() is None
    assert s.model.requests == []


@pytest.mark.unit
def test_run_corner_tags_a_match_without_corners_writes_an_empty_file(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.db.unlink()
    s.db = write_db(s.dir, [(EventType.GOAL, 100.0, 1, "home")])

    result = s.run()

    assert result is not None
    assert result.tags == []
    assert s.model.requests == []


@pytest.mark.unit
def test_corner_tags_config_rejects_a_negative_or_huge_window() -> None:
    with pytest.raises(ValueError, match="before_s"):
        CornerTagsConfig(before_s=-1)
    with pytest.raises(ValueError, match="after_s"):
        CornerTagsConfig(after_s=999)


@pytest.mark.unit
def test_read_tags_file_of_a_missing_or_broken_file_is_none(tmp_path: Path) -> None:
    assert read_tags_file(tmp_path) is None
    (tmp_path / RESULT_FILE).write_text("{oops", encoding="utf-8")
    assert read_tags_file(tmp_path) is None


def tagged(tmp_path: Path) -> tuple[CornerTagsFile, list]:  # type: ignore[type-arg]
    s = Setup(
        tmp_path,
        [
            corner_reply(attacking_kit="white", zone="near_post"),
            corner_reply(attacking_kit="red", outcome="shot"),
            corner_reply(attacking_kit="unclear", delivery="unclear"),
        ],
    )
    result = s.run()
    assert result is not None
    corners = [make_event(t.event_id, EventType.CORNER, t.t_v, team=None) for t in result.tags]
    return result, corners


@pytest.mark.unit
def test_build_views_maps_the_kit_colour_to_the_attacking_side(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)

    views = build_views(result, corners, TEAMS)

    assert views["p:3"].side == "home"
    assert views["p:5"].side == "away"
    assert views["p:7"].side is None  # unclear kit: the team is not known


@pytest.mark.unit
def test_build_views_ignores_tags_of_a_corner_that_was_moved(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)
    moved = corners[0].model_copy(update={"t_v": corners[0].t_v + 5})

    views = build_views(result, [moved, *corners[1:]], TEAMS)

    assert "p:3" not in views
    assert set(views) == {"p:5", "p:7"}


@pytest.mark.unit
def test_build_views_without_tags_is_empty() -> None:
    assert build_views(None, [make_event()], TEAMS) == {}


@pytest.mark.unit
def test_build_views_a_colour_nobody_wears_gives_no_side(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)
    teams = {**TEAMS, "away": EffectiveTeam(name="K", name_source="lineup")}

    views = build_views(result, corners, teams)

    assert views["p:5"].side is None


@pytest.mark.unit
def test_load_views_reads_the_file_of_the_match_directory(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)

    assert set(load_views(tmp_path / "match", corners, TEAMS)) == {"p:3", "p:5", "p:7"}
    assert load_views(tmp_path, corners, TEAMS) == {}


@pytest.mark.unit
def test_matches_by_team_and_by_fields(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)
    views = build_views(result, corners, TEAMS)

    assert matches(views["p:3"], CornerFilter(team="home"))
    assert not matches(views["p:3"], CornerFilter(team="away"))
    assert matches(views["p:7"], CornerFilter(team="unknown"))
    assert matches(views["p:5"], CornerFilter(fields={"outcome": "shot"}))
    assert not matches(views["p:3"], CornerFilter(fields={"outcome": "shot"}))
    assert matches(views["p:7"], CornerFilter(fields={"delivery": "unclear"}))
    assert matches(views["p:3"], CornerFilter(team="home", fields={"zone": "near_post"}))
    assert not matches(views["p:3"], CornerFilter(team="home", fields={"zone": "center"}))


@pytest.mark.unit
def test_matches_an_empty_filter_passes_everything_and_a_filter_needs_tags() -> None:
    assert matches(None, CornerFilter())
    assert not matches(None, CornerFilter(team="home"))
    assert not matches(None, CornerFilter(fields={"zone": "center"}))
    assert not CornerFilter().active


@pytest.mark.unit
def test_apply_filter_keeps_every_other_event_and_filters_only_corners(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)
    views = build_views(result, corners, TEAMS)
    goal = make_event("p:1", EventType.GOAL, 10.0)
    untagged = make_event("p:9", EventType.CORNER, 9999.0)

    kept = apply_filter([goal, *corners, untagged], views, CornerFilter(team="home"))

    assert [e.event_id for e in kept] == ["p:1", "p:3"]
    assert [e.event_id for e in apply_filter([goal, untagged], views, CornerFilter())] == [
        "p:1",
        "p:9",
    ]


@pytest.mark.unit
def test_describe_says_the_team_the_known_fields_and_that_it_is_not_verified(
    tmp_path: Path,
) -> None:
    result, corners = tagged(tmp_path)
    views = build_views(result, corners, TEAMS)

    text = describe(views["p:3"], TEAMS)

    assert text.startswith("подаёт: Актобе")
    assert "подача: навес" in text
    assert "зона: ближняя штанга" in text
    assert text.endswith("(по клипу, не проверено)")
    assert "команда не определена" in describe(views["p:7"], TEAMS)
    assert "подача:" not in describe(views["p:7"], TEAMS)  # unclear fields are left out


@pytest.mark.unit
def test_describe_a_clip_without_a_corner_says_so_with_the_mark() -> None:
    tag = CornerFields.model_validate({**CORNER_JSON, "is_corner": "no"})
    view = CornerView(_tag(tag), None)

    assert describe(view, TEAMS) == "по клипу это не угловой (по клипу, не проверено)"


def _tag(fields: CornerFields):  # type: ignore[no-untyped-def]  # a plain tag for the view
    from aicoach.corner_tags.schemas import CornerTag

    return CornerTag(
        event_id="p:1",
        t_v=1.0,
        fields=fields,
        status="ok",
        model="m",
        prompt_version=1,
        before_s=4.0,
        after_s=12.0,
    )


@pytest.mark.unit
def test_texts_gives_a_line_per_tagged_corner(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)
    views = build_views(result, corners, TEAMS)

    assert set(texts(views, TEAMS)) == {"p:3", "p:5", "p:7"}


@pytest.mark.unit
def test_labels_in_russian() -> None:
    assert value_label("zone", "far_post") == "дальняя штанга"
    assert value_label("zone", "odd") == "odd"
    assert team_label(None, TEAMS) == "команда не определена"
    assert team_label("away", TEAMS) == "Кайрат"


@pytest.mark.unit
def test_run_corner_tags_a_failure_keeps_the_kept_tags_of_corners_not_reached_yet(
    tmp_path: Path,
) -> None:
    s = Setup(tmp_path)
    s.run()
    s.db.unlink()
    moved = [(t, t_v + 3 if t is EventType.CORNER and t_v < 4500 else t_v, h, team)
             for t, t_v, h, team in DEFAULT_SPECS]  # fmt: skip
    s.db = write_db(s.dir, moved)  # the first two corners moved, the third is as it was
    s.model = FakeVideoModel([corner_reply(), ExternalServiceError("502")])

    with pytest.raises(ExternalServiceError):
        s.run()

    kept = read_tags_file(s.dir)
    assert kept is not None
    assert [t.event_id for t in kept.tags] == ["p:3", "p:7"]  # the new one and the paid one


@pytest.mark.unit
@pytest.mark.parametrize(
    ("missing", "word"), [("video", "src.mp4"), ("db", "aicoach.db"), ("key", "OPENROUTER_API_KEY")]
)
def test_run_corner_tags_skipped_says_the_real_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str, word: str
) -> None:
    s = Setup(tmp_path)
    model: FakeVideoModel | None = s.model
    if missing == "video":
        s.video.unlink()
    elif missing == "db":
        s.db.unlink()
    else:
        model = None

        def no_key(*_args: object, **_kwargs: object) -> object:
            msg = "OPENROUTER_API_KEY is not set"
            raise ConfigError(msg)

        monkeypatch.setattr("aicoach.corner_tags.stage.make_video_model", no_key)

    assert (
        run_corner_tags(
            s.dir, MATCH_ID, db_path=s.db, video=s.video, model=model, clipper=s.clipper
        )
        is None
    )

    entry = read_status(s.dir).stages["corner_tags"]
    assert entry.state == "skipped"
    assert word in (entry.error or "")


@pytest.mark.unit
def test_describe_marks_the_zone_as_low_reliability_and_only_the_zone(tmp_path: Path) -> None:
    result, corners = tagged(tmp_path)
    views = build_views(result, corners, TEAMS)

    text = describe(views["p:3"], TEAMS)

    assert "зона: ближняя штанга (низкая надёжность)" in text
    assert text.count("низкая надёжность") == 1
    assert "(низкая надёжность)" not in text.split("зона")[0]


@pytest.mark.unit
def test_the_zone_filter_label_says_low_reliability() -> None:
    from aicoach.i18n import ru

    assert "низкая надёжность" in ru.CORNER_FIELD_LABELS["zone"]
    assert all("надёжность" not in v for k, v in ru.CORNER_FIELD_LABELS.items() if k != "zone")
