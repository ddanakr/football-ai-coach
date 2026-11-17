"""The `teams.json` store, the override of the user and the `teams` stage (D-078)."""

import json
from pathlib import Path

import pytest

from aicoach.errors import ConfigError, ExternalServiceError, StorageError
from aicoach.status import read_status
from aicoach.teams.stage import TeamsConfig, cache_key, resolve_names, run_teams
from aicoach.teams.store import (
    OVERRIDE_FILE,
    RESULT_FILE,
    EffectiveTeam,
    Side,
    SideOverride,
    TeamEntry,
    TeamsOverride,
    TeamsResult,
    load_effective,
    read_override,
    read_teams,
    save_override,
    side_of_kit,
)
from aicoach.teams.title import INFO_FILE
from tests.fixtures.corner_synth import MATCH_ID, kit_reply, write_db, write_lineup, write_video
from tests.fixtures.episode_synth import FakeClipper, FakeVideoModel, reply


def result_file(match_dir: Path, home_kit: str | None = None) -> None:
    home = TeamEntry(
        name="Актобе",
        name_source="lineup",
        kit=home_kit,  # type: ignore[arg-type]
        kit_source="vlm" if home_kit else "undetermined",
    )
    away = TeamEntry(name="Кайрат", name_source="lineup")
    teams = TeamsResult(match_id="m", cache_key="k", home=home, away=away)
    (match_dir / RESULT_FILE).write_text(teams.model_dump_json(), encoding="utf-8")


@pytest.mark.unit
def test_load_effective_without_any_file_gives_the_default_words(tmp_path: Path) -> None:
    teams = load_effective(tmp_path)

    assert teams["home"] == EffectiveTeam(name="хозяева", name_source="default")
    assert teams["away"].name == "гости"
    assert teams["away"].kit is None


@pytest.mark.unit
def test_load_effective_takes_the_names_of_the_lineup_when_there_is_no_teams_file(
    tmp_path: Path,
) -> None:
    write_lineup(tmp_path)

    teams = load_effective(tmp_path)

    assert (teams["home"].name, teams["home"].name_source) == ("Актобе", "lineup")
    assert teams["away"].name == "Кайрат"


@pytest.mark.unit
def test_load_effective_override_wins_over_the_stage_and_keeps_the_rest(tmp_path: Path) -> None:
    result_file(tmp_path, home_kit="white")
    save_override(
        tmp_path,
        TeamsOverride(home=SideOverride(kit="red"), away=SideOverride(name="  FC  Kairat ")),
    )

    teams = load_effective(tmp_path)

    assert (teams["home"].kit, teams["home"].kit_source) == ("red", "manual")
    assert (teams["home"].name, teams["home"].name_source) == ("Актобе", "lineup")
    assert (teams["away"].name, teams["away"].name_source) == ("FC Kairat", "manual")
    assert teams["away"].kit_source == "undetermined"


@pytest.mark.unit
def test_load_effective_stage_kit_is_used_without_an_override(tmp_path: Path) -> None:
    result_file(tmp_path, home_kit="white")

    teams = load_effective(tmp_path)

    assert (teams["home"].kit, teams["home"].kit_source) == ("white", "vlm")


@pytest.mark.unit
@pytest.mark.parametrize("name", [RESULT_FILE, OVERRIDE_FILE])
def test_broken_files_are_ignored_not_fatal(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("{not json", encoding="utf-8")

    assert load_effective(tmp_path)["home"].name == "хозяева"
    assert read_teams(tmp_path) is None
    assert read_override(tmp_path) == TeamsOverride()


@pytest.mark.unit
def test_save_override_round_trips_and_an_empty_override_removes_the_file(tmp_path: Path) -> None:
    save_override(tmp_path, TeamsOverride(home=SideOverride(name="Актобе", kit="white")))

    assert read_override(tmp_path).home.kit == "white"
    assert (tmp_path / OVERRIDE_FILE).is_file()

    save_override(tmp_path, TeamsOverride())

    assert not (tmp_path / OVERRIDE_FILE).exists()
    save_override(tmp_path, TeamsOverride())  # nothing to remove: not an error


@pytest.mark.unit
def test_save_override_to_a_missing_directory_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        save_override(tmp_path / "nope", TeamsOverride(home=SideOverride(name="x")))


@pytest.mark.unit
def test_side_override_rejects_an_unknown_colour_and_a_long_name() -> None:
    with pytest.raises(ValueError, match="kit"):
        SideOverride.model_validate({"kit": "pink"})
    with pytest.raises(ValueError, match="name"):
        SideOverride(name="x" * 61)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kit", "expected"),
    [("white", "home"), ("red", "away"), ("blue", None), (None, None)],
)
def test_side_of_kit_maps_a_colour_to_the_side_that_wears_it(
    kit: str | None, expected: str | None
) -> None:
    teams: dict[Side, EffectiveTeam] = {
        "home": EffectiveTeam(name="a", name_source="default", kit="white"),
        "away": EffectiveTeam(name="b", name_source="default", kit="red"),
    }

    assert side_of_kit(kit, teams) == expected


@pytest.mark.unit
def test_side_of_kit_is_none_when_both_teams_wear_the_colour() -> None:
    teams: dict[Side, EffectiveTeam] = {
        "home": EffectiveTeam(name="a", name_source="default", kit="white"),
        "away": EffectiveTeam(name="b", name_source="default", kit="white"),
    }

    assert side_of_kit("white", teams) is None


@pytest.mark.unit
def test_resolve_names_lineup_then_title_then_default_per_side(tmp_path: Path) -> None:
    (tmp_path / INFO_FILE).write_text(json.dumps({"title": "Тобол - Астана"}), encoding="utf-8")
    (tmp_path / "lineups.csv").write_text(
        "team,side,number,name\nАктобе,home,1,Иванов Иван\n", encoding="utf-8"
    )

    names = resolve_names(tmp_path)

    assert names["home"] == ("Актобе", "lineup")
    assert names["away"] == ("Астана", "title")


@pytest.mark.unit
def test_resolve_names_without_sources_gives_the_default_words(tmp_path: Path) -> None:
    assert resolve_names(tmp_path) == {
        "home": ("хозяева", "default"),
        "away": ("гости", "default"),
    }


class Setup:
    """A match directory with a database, a video and the fakes the stage needs."""

    def __init__(self, tmp_path: Path, script: list[object] | None = None) -> None:
        self.dir = tmp_path / "match"
        self.db = write_db(self.dir)
        self.video = write_video(self.dir)
        self.clipper = FakeClipper()
        default = [kit_reply("white"), kit_reply("white"), kit_reply("red"), kit_reply("red")]
        self.model = FakeVideoModel(default if script is None else script)  # type: ignore[arg-type]

    def run(self, **kwargs: object) -> TeamsResult:
        return run_teams(
            self.dir,
            MATCH_ID,
            db_path=self.db,
            video=self.video,
            model=self.model,
            clipper=self.clipper,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_run_teams_votes_the_kit_of_each_side_from_goal_clips(tmp_path: Path) -> None:
    s = Setup(
        tmp_path, [kit_reply("white"), kit_reply("white"), kit_reply("red"), kit_reply("red")]
    )
    write_lineup(s.dir)

    result = s.run()

    assert (result.home.name, result.home.kit, result.home.kit_source) == ("Актобе", "white", "vlm")
    assert (result.away.name, result.away.kit, result.away.kit_votes) == (
        "Кайрат",
        "red",
        {"red": 2},
    )
    assert len(result.votes) == 4
    assert result.kits_skipped is None
    assert read_status(s.dir).stages["teams"].state == "done"
    assert (s.dir / RESULT_FILE).is_file()


@pytest.mark.unit
def test_run_teams_asks_a_strict_question_with_the_role_and_no_names(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    write_lineup(s.dir)

    s.run()

    assert len(s.model.requests) == 4
    for system, prompt, video, seconds in s.model.requests:
        assert "JSON" in system
        assert "забила гол" in prompt
        assert "Актобе" not in system + prompt
        assert "Кайрат" not in system + prompt
        assert video == b"fake mp4"
        assert seconds == 16.0
    assert {call[2:] for call in s.clipper.calls} == {(4.0, 12.0)}


@pytest.mark.unit
def test_run_teams_second_run_is_cached_and_asks_nothing(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    first = s.run()
    asked = len(s.model.requests)

    second = s.run()

    assert len(s.model.requests) == asked
    assert second.cache_key == first.cache_key


@pytest.mark.unit
def test_run_teams_force_reuses_the_votes_already_made(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.run()
    asked = len(s.model.requests)

    s.run(force=True)

    assert len(s.model.requests) == asked


@pytest.mark.unit
def test_run_teams_a_new_lineup_changes_the_names_and_keeps_the_votes(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.run()
    asked = len(s.model.requests)
    write_lineup(s.dir, "Тобол", "Астана")

    result = s.run()

    assert result.home.name == "Тобол"
    assert len(s.model.requests) == asked


@pytest.mark.unit
def test_run_teams_unusable_answer_is_asked_again_then_counts_as_no_vote(tmp_path: Path) -> None:
    s = Setup(tmp_path, [reply("не знаю"), reply("{}")])

    result = s.run(config=TeamsConfig(max_votes_per_side=1))

    first = result.votes[0]
    assert (first.status, first.kit) == ("invalid", "unclear")
    assert first.tokens_in == 4200  # both requests are counted
    assert result.home.kit is None


@pytest.mark.unit
def test_run_teams_second_answer_valid_is_a_retried_vote(tmp_path: Path) -> None:
    s = Setup(tmp_path, [reply("cannot say"), kit_reply("white")])

    result = s.run(config=TeamsConfig(max_votes_per_side=1))

    assert (result.votes[0].status, result.votes[0].kit) == ("retried", "white")


@pytest.mark.unit
def test_run_teams_conflicting_votes_leave_the_kit_undetermined_with_a_note(
    tmp_path: Path,
) -> None:
    s = Setup(
        tmp_path, [kit_reply("white"), kit_reply("red"), kit_reply("blue"), kit_reply("blue")]
    )

    result = s.run()

    assert result.home.kit is None
    assert result.away.kit == "blue"
    assert any(note.startswith("home: kit not determined") for note in result.notes)


@pytest.mark.unit
def test_run_teams_without_a_key_writes_the_names_and_records_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_key(*_args: object, **_kwargs: object) -> object:
        msg = "OPENROUTER_API_KEY is not set"
        raise ConfigError(msg)

    monkeypatch.setattr("aicoach.teams.stage.make_video_model", no_key)
    s = Setup(tmp_path)
    write_lineup(s.dir)

    result = run_teams(s.dir, MATCH_ID, db_path=s.db, video=s.video, clipper=s.clipper)

    assert (result.home.name, result.home.kit) == ("Актобе", None)
    assert result.kits_skipped is not None
    entry = read_status(s.dir).stages["teams"]
    assert entry.state == "skipped"
    assert "OPENROUTER_API_KEY" in (entry.error or "")
    assert s.clipper.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["db", "video"])
def test_run_teams_without_the_database_or_the_video_skips_the_colours(
    tmp_path: Path, missing: str
) -> None:
    s = Setup(tmp_path)
    if missing == "db":
        s.db.unlink()
    else:
        s.video.unlink()

    result = s.run()

    assert result.kits_skipped is not None
    assert s.model.requests == []
    assert read_status(s.dir).stages["teams"].state == "skipped"


@pytest.mark.unit
def test_run_teams_a_database_without_goals_or_cards_skips_the_colours(tmp_path: Path) -> None:
    from aicoach.domain.events import EventType

    s = Setup(tmp_path)
    s.db.unlink()
    s.db = write_db(s.dir, [(EventType.CORNER, 100.0, 1, None)])

    result = s.run()

    assert result.kits_skipped is not None
    assert result.home.name == "хозяева"


@pytest.mark.unit
def test_run_teams_a_clip_that_cannot_be_cut_is_skipped_with_a_note(tmp_path: Path) -> None:
    from aicoach.errors import ExportError

    s = Setup(tmp_path)
    s.clipper = FakeClipper(raises=ExportError("ffmpeg failed"))

    result = s.run()

    assert result.votes == []
    assert any(note.startswith("no clip for") for note in result.notes)


@pytest.mark.unit
def test_run_teams_a_service_failure_is_raised_and_recorded_as_failed(tmp_path: Path) -> None:
    s = Setup(tmp_path, [ExternalServiceError("502")])

    with pytest.raises(ExternalServiceError):
        s.run()

    assert read_status(s.dir).stages["teams"].state == "failed"
    assert not (s.dir / RESULT_FILE).exists()


@pytest.mark.unit
def test_cache_key_changes_with_the_model_and_the_config() -> None:
    names = {"home": ("a", "default"), "away": ("b", "default")}
    base = cache_key(names, [], "m1", TeamsConfig())  # type: ignore[arg-type]

    assert cache_key(names, [], "m2", TeamsConfig()) != base  # type: ignore[arg-type]
    assert cache_key(names, [], "m1", TeamsConfig(min_votes=3)) != base  # type: ignore[arg-type]
    assert cache_key(names, [], "m1", TeamsConfig()) == base  # type: ignore[arg-type]


@pytest.mark.unit
def test_run_teams_a_later_run_without_a_key_keeps_the_votes_and_the_kits_already_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Setup(tmp_path)
    first = s.run()
    assert first.home.kit == "white"

    def no_key(*_args: object, **_kwargs: object) -> object:
        msg = "OPENROUTER_API_KEY is not set"
        raise ConfigError(msg)

    monkeypatch.setattr("aicoach.teams.stage.make_video_model", no_key)
    again = run_teams(s.dir, MATCH_ID, db_path=s.db, video=s.video, clipper=s.clipper)

    assert (again.home.kit, again.away.kit) == ("white", "red")
    assert len(again.votes) == len(first.votes)
    assert again.kits_skipped is not None
    assert read_teams(s.dir) == again


@pytest.mark.unit
def test_run_teams_a_stopped_run_keeps_its_votes_and_goes_on_from_them(tmp_path: Path) -> None:
    s = Setup(tmp_path, [kit_reply("white"), kit_reply("white"), ExternalServiceError("502")])

    with pytest.raises(ExternalServiceError):
        s.run()

    partial = read_teams(s.dir)
    assert partial is not None
    assert partial.cache_key == "incomplete"  # never taken for a finished result
    assert len(partial.votes) == 2

    s.model = FakeVideoModel([kit_reply("red"), kit_reply("red")])
    result = s.run()

    assert len(s.model.requests) == 2  # the two paid votes are not asked again
    assert (result.home.kit, result.away.kit) == ("white", "red")
    assert result.cache_key != "incomplete"


@pytest.mark.unit
def test_run_teams_a_clip_that_failed_is_tried_again_on_the_next_run(tmp_path: Path) -> None:
    from aicoach.errors import ExportError

    s = Setup(tmp_path)
    s.clipper = FakeClipper(raises=ExportError("ffmpeg failed"))
    assert s.run().votes == []

    s.clipper = FakeClipper()
    result = s.run()

    assert len(result.votes) == 4
    assert result.home.kit == "white"


@pytest.mark.unit
def test_run_teams_a_new_model_does_not_claim_the_votes_of_the_old_one_in_a_partial_file(
    tmp_path: Path,
) -> None:
    s = Setup(tmp_path)
    s.run()  # four votes paid for
    first_model = s.model
    # A new model asks again for every source; the first answer arrives, the second fails.
    s.model = FakeVideoModel([kit_reply("white"), ExternalServiceError("502")], model="test/other")
    with pytest.raises(ExternalServiceError):
        s.run()
    assert first_model.requests  # the first run did ask

    partial = read_teams(s.dir)
    assert partial is not None
    assert partial.cache_key == "incomplete"
    # One new vote from the new model; nothing of the old model's votes is claimed for it.
    assert [v.model for v in partial.votes] == ["test/other"]


@pytest.mark.unit
def test_run_teams_a_partial_file_keeps_the_votes_of_later_sources_from_the_same_model(
    tmp_path: Path,
) -> None:
    s = Setup(tmp_path)
    first = s.run()
    # Two home goals moved: the first is asked again and answers, the second fails; the two
    # votes of the away team, paid for earlier, stay in the saved file.
    s.db.unlink()
    from aicoach.domain.events import EventType
    from tests.fixtures.corner_synth import DEFAULT_SPECS

    moved = [
        (t, t_v + 3 if t is EventType.GOAL and t_v in (600.0, 4000.0) else t_v, h, team)
        for t, t_v, h, team in DEFAULT_SPECS
    ]
    s.db = write_db(s.dir, moved)
    s.model = FakeVideoModel([kit_reply("white"), ExternalServiceError("502")])

    with pytest.raises(ExternalServiceError):
        s.run()

    partial = read_teams(s.dir)
    assert partial is not None
    assert len(first.votes) == 4
    assert sorted(v.side for v in partial.votes) == ["away", "away", "home"]
    assert partial.cache_key == "incomplete"
