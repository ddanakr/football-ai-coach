"""The fusion stage: files, database, cache, edit guard, failures."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from aicoach.domain.events import EventType
from aicoach.errors import FusionError, SpottingError, StorageError
from aicoach.fusion import stage as stage_module
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.rules import NO_MODEL_WARNING
from aicoach.fusion.schemas import CandidatesResult, Source
from aicoach.fusion.stage import (
    DB_FILE,
    RESULT_FILE,
    cache_key,
    db_source,
    run_fuse,
    to_new_prediction,
)
from aicoach.spotting.schemas import TDeedFile
from aicoach.status import read_status
from aicoach.storage import add_edit, connect, count_effective_edits, get_current_events
from aicoach.storage import latest_run_id as stored_run
from tests.fixtures.fusion_synth import plate, write_inputs, write_tdeed
from tests.fixtures.plates_synth import change

MATCH = "m1"
GOAL_UP = change((0, 0), (1, 0), 199.0, 200.0)


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.fixture
def match_dir(tmp_path: Path) -> Path:
    """A match directory with a goal, a goal plate, a substitution and a card."""
    write_inputs(
        tmp_path,
        [GOAL_UP],
        [
            plate("p1", "goal", 230.0),
            plate("p2", "subst", 600.0, team="away"),
            plate("p3", "card", 700.0),
        ],
    )
    return tmp_path


def events(match_dir: Path) -> list[tuple[str, float]]:
    """`(type, t_v)` of the current events in the database."""
    with connect(match_dir / DB_FILE) as conn:
        return [(e.type.value, e.t_v) for e in get_current_events(conn, MATCH)]


@pytest.mark.unit
def test_run_fuse_without_model_writes_events_candidates_and_status(match_dir: Path) -> None:
    result = run_fuse(match_dir, MATCH, run_id="r1")

    foul_t = 700.0 - FuseConfig().card_foul_lead_s  # the card plate is at 700 s
    assert events(match_dir) == [("goal", 200.0), ("subst", 600.0), ("foul", foul_t)]
    assert result.mode == "no_model"
    assert NO_MODEL_WARNING in result.warnings
    assert (result.run_id, result.counts) == (
        "r1",
        {"goal": {"1": 1}, "subst": {"1": 1}, "foul": {"1": 1}},
    )
    on_disk = CandidatesResult.model_validate_json((match_dir / RESULT_FILE).read_text("utf-8"))
    assert on_disk == result
    assert read_status(match_dir).stages["fuse"].state == "done"


@pytest.mark.unit
def test_run_fuse_with_model_file_adds_model_events_and_marks_the_mode(match_dir: Path) -> None:
    tdeed = match_dir / "tdeed_x.json"
    # frame / 25 = 300 s corner, 350 s goal (ignored), 690 s foul before the card plate at 700 s
    write_tdeed(tdeed, [(7500, "Corner", 0.9), (8750, "Goal", 0.9), (17250, "Foul", 0.3)])

    result = run_fuse(match_dir, MATCH, tdeed, run_id="r1")

    assert result.mode == "with_model"
    assert result.model_file == "tdeed_x.json"
    assert NO_MODEL_WARNING not in result.warnings
    assert events(match_dir) == [
        ("goal", 200.0),
        ("corner", 300.0),
        ("subst", 600.0),
        ("foul", 690.0),
    ]
    foul = next(c for c in result.candidates if c.type is EventType.FOUL)
    assert (foul.score, foul.source) == (0.9, "model+plates")


@pytest.mark.unit
def test_run_fuse_stores_db_sources_and_halves(match_dir: Path) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")

    with connect(match_dir / DB_FILE) as conn:
        current = get_current_events(conn, MATCH)
    assert {(e.type.value, e.source, e.half, e.status) for e in current} == {
        ("goal", "plates", 1, "auto"),
        ("subst", "plates", 1, "auto"),
        ("foul", "plates", 1, "auto"),
    }


@pytest.mark.unit
def test_run_fuse_second_run_with_same_inputs_comes_from_the_cache(match_dir: Path) -> None:
    first = run_fuse(match_dir, MATCH, run_id="r1")

    with capture_logs() as logs:
        second = run_fuse(match_dir, MATCH, run_id="r2")

    assert second == first
    assert any(log["event"] == "stage_skipped" for log in logs)
    with connect(match_dir / DB_FILE) as conn:
        assert stored_run(conn, MATCH) == "r1"


@pytest.mark.unit
def test_run_fuse_force_makes_a_new_run_with_the_same_events(match_dir: Path) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    before = events(match_dir)

    result = run_fuse(match_dir, MATCH, force=True, run_id="r2")

    assert result.run_id == "r2"
    assert events(match_dir) == before


@pytest.mark.unit
@pytest.mark.parametrize("change_what", ["config", "model", "plates"])
def test_run_fuse_recomputes_when_an_input_or_a_parameter_changes(
    match_dir: Path, change_what: str
) -> None:
    tdeed = match_dir / "tdeed_x.json"
    write_tdeed(tdeed, [(7500, "Corner", 0.9)])
    run_fuse(match_dir, MATCH, tdeed, run_id="r1")
    config = FuseConfig()
    if change_what == "config":
        config = FuseConfig(goal_unconfirmed_score=0.7)
    elif change_what == "model":
        write_tdeed(tdeed, [(7500, "Corner", 0.8)])
    else:
        write_inputs(match_dir, [GOAL_UP], [plate("p9", "subst", 610.0)])

    result = run_fuse(match_dir, MATCH, tdeed, config, run_id="r2")

    assert result.run_id == "r2"


@pytest.mark.unit
def test_run_fuse_refills_a_deleted_database_instead_of_trusting_the_cache(
    match_dir: Path,
) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    for suffix in ("", "-wal", "-shm"):
        (match_dir / (DB_FILE + suffix)).unlink(missing_ok=True)

    result = run_fuse(match_dir, MATCH, run_id="r2")

    assert result.run_id == "r2"
    assert len(events(match_dir)) == 3


@pytest.mark.unit
def test_run_fuse_refuses_when_the_match_has_user_edits(match_dir: Path) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    with connect(match_dir / DB_FILE) as conn:
        target = get_current_events(conn, MATCH)[0].event_id
        add_edit(conn, MATCH, "verify", author="analyst", target=target)

    with pytest.raises(FusionError, match="--discard-edits"):
        run_fuse(match_dir, MATCH, force=True, run_id="r2")

    with connect(match_dir / DB_FILE) as conn:
        assert stored_run(conn, MATCH) == "r1"
    status = read_status(match_dir).stages["fuse"]
    assert status.state == "failed"
    assert status.error is not None
    assert "user edit" in status.error


@pytest.mark.unit
def test_run_fuse_with_discard_edits_makes_a_new_run_and_keeps_the_journal(
    match_dir: Path,
) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    with connect(match_dir / DB_FILE) as conn:
        target = get_current_events(conn, MATCH)[0].event_id
        add_edit(conn, MATCH, "verify", author="analyst", target=target)

    result = run_fuse(match_dir, MATCH, force=True, discard_edits=True, run_id="r2")

    assert result.run_id == "r2"
    with connect(match_dir / DB_FILE) as conn:
        assert stored_run(conn, MATCH) == "r2"
        assert count_effective_edits(conn, MATCH) == 0
        assert conn.execute("SELECT COUNT(*) FROM edits").fetchone()[0] == 1


@pytest.mark.unit
def test_run_fuse_cached_run_does_not_refuse_even_with_edits(match_dir: Path) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    with connect(match_dir / DB_FILE) as conn:
        add_edit(conn, MATCH, "add", author="analyst", type=EventType.CORNER, t_v=300.0)

    result = run_fuse(match_dir, MATCH, run_id="r2")

    assert result.run_id == "r1"


@pytest.mark.unit
def test_run_fuse_empty_inputs_store_nothing_and_create_no_database(tmp_path: Path) -> None:
    write_inputs(tmp_path)

    result = run_fuse(tmp_path, MATCH, run_id="r1")

    assert (result.candidates, result.run_id) == ([], None)
    assert not (tmp_path / DB_FILE).exists()
    assert run_fuse(tmp_path, MATCH, run_id="r2") == result


@pytest.mark.unit
def test_run_fuse_empty_run_leaves_the_previous_run_visible_and_warns(match_dir: Path) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    write_inputs(match_dir)

    with capture_logs() as logs:
        result = run_fuse(match_dir, MATCH, run_id="r2")

    assert result.run_id is None
    assert len(events(match_dir)) == 3
    assert any(log["event"] == "empty_run_not_stored" for log in logs)


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["scoreboard.json", "plates.json"])
def test_run_fuse_missing_input_raises_fusion_error_with_the_command_to_run(
    match_dir: Path, missing: str
) -> None:
    (match_dir / missing).unlink()

    with pytest.raises(FusionError, match="run first: python -m aicoach.cli"):
        run_fuse(match_dir, MATCH)

    assert read_status(match_dir).stages["fuse"].state == "failed"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["scoreboard.json", "plates.json"])
def test_run_fuse_invalid_input_raises_fusion_error(match_dir: Path, name: str) -> None:
    (match_dir / name).write_text(json.dumps({"video": "x"}), encoding="utf-8")

    with pytest.raises(FusionError, match="not a valid"):
        run_fuse(match_dir, MATCH)


@pytest.mark.unit
def test_run_fuse_plates_made_from_another_scoreboard_raises_fusion_error(match_dir: Path) -> None:
    plates = json.loads((match_dir / "plates.json").read_text("utf-8"))
    plates["scoreboard_version"] = "stale"
    (match_dir / "plates.json").write_text(json.dumps(plates), encoding="utf-8")

    with pytest.raises(FusionError, match="another scoreboard.json"):
        run_fuse(match_dir, MATCH)


@pytest.mark.unit
def test_run_fuse_scoreboard_with_one_clock_segment_raises_fusion_error(tmp_path: Path) -> None:
    write_inputs(tmp_path, segments=[(100.0, 1000.0)])

    with pytest.raises(FusionError, match="two are needed"):
        run_fuse(tmp_path, MATCH)


@pytest.mark.unit
def test_run_fuse_broken_model_file_raises_and_keeps_the_earlier_result(match_dir: Path) -> None:
    first = run_fuse(match_dir, MATCH, run_id="r1")
    tdeed = match_dir / "tdeed_x.json"
    tdeed.write_text("not json", encoding="utf-8")

    with pytest.raises(SpottingError):
        run_fuse(match_dir, MATCH, tdeed, run_id="r2")

    on_disk = CandidatesResult.model_validate_json((match_dir / RESULT_FILE).read_text("utf-8"))
    assert on_disk == first


@pytest.mark.unit
def test_run_fuse_missing_model_file_raises_spotting_error(match_dir: Path) -> None:
    with pytest.raises(SpottingError, match="cannot read"):
        run_fuse(match_dir, MATCH, match_dir / "nope.json")


@pytest.mark.unit
def test_run_fuse_events_outside_the_halves_are_dropped_and_listed(tmp_path: Path) -> None:
    write_inputs(tmp_path, plates=[plate("p1", "subst", 1200.0)])

    result = run_fuse(tmp_path, MATCH, run_id="r1")

    assert result.candidates == []
    assert [(d.type, d.reason) for d in result.dropped] == [(EventType.SUBST, "outside_halves")]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("plates", "plates"),
        ("scoreboard", "plates"),
        ("scoreboard+plates", "plates"),
        ("plates_only", "plates"),
        ("model", "model"),
        ("model+plates", "merge"),
    ],
)
def test_db_source_maps_candidate_sources_to_the_columns_the_schema_allows(
    source: Source, expected: str
) -> None:
    assert db_source(source) == expected


@pytest.mark.unit
def test_to_new_prediction_keeps_type_time_half_score_status_and_team(match_dir: Path) -> None:
    result = run_fuse(match_dir, MATCH, run_id="r1")
    goal = next(c for c in result.candidates if c.type is EventType.GOAL)

    row = to_new_prediction(goal)

    assert (row.type, row.t_v, row.half, row.score, row.status, row.team, row.source) == (
        EventType.GOAL,
        200.0,
        1,
        1.0,
        "auto",
        "home",
        "plates",
    )


@pytest.mark.unit
def test_run_fuse_unreadable_candidates_file_is_recomputed_with_a_warning(match_dir: Path) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    (match_dir / RESULT_FILE).write_text("{broken", encoding="utf-8")

    with capture_logs() as logs:
        result = run_fuse(match_dir, MATCH, force=False, run_id="r2")

    assert result.run_id == "r2"
    assert any(log["event"] == "cache_unreadable" for log in logs)


@pytest.mark.unit
def test_run_fuse_input_that_cannot_be_read_raises_fusion_error(match_dir: Path) -> None:
    (match_dir / "scoreboard.json").unlink()
    (match_dir / "scoreboard.json").mkdir()

    with pytest.raises(FusionError, match="cannot read"):
        run_fuse(match_dir, MATCH)


@pytest.mark.unit
def test_run_fuse_model_file_that_vanishes_after_loading_raises_fusion_error(
    match_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stage_module, "load_tdeed_file", lambda _path: TDeedFile(predictions=()))

    with pytest.raises(FusionError, match="cannot read"):
        run_fuse(match_dir, MATCH, match_dir / "gone.json")


@pytest.mark.unit
def test_run_fuse_broken_database_raises_storage_error_and_keeps_candidates(
    match_dir: Path,
) -> None:
    first = run_fuse(match_dir, MATCH, run_id="r1")
    for suffix in ("-wal", "-shm"):
        (match_dir / (DB_FILE + suffix)).unlink(missing_ok=True)
    (match_dir / DB_FILE).write_bytes(b"this is not a database" * 100)

    with pytest.raises(StorageError):
        run_fuse(match_dir, MATCH, run_id="r2")

    on_disk = CandidatesResult.model_validate_json((match_dir / RESULT_FILE).read_text("utf-8"))
    assert on_disk == first


@pytest.mark.unit
def test_run_fuse_empty_run_with_edits_is_not_refused_because_it_hides_nothing(
    match_dir: Path,
) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    with connect(match_dir / DB_FILE) as conn:
        target = get_current_events(conn, MATCH)[0].event_id
        add_edit(conn, MATCH, "verify", author="analyst", target=target)
    write_inputs(match_dir)

    result = run_fuse(match_dir, MATCH, run_id="r2")

    assert result.run_id is None
    with connect(match_dir / DB_FILE) as conn:
        assert stored_run(conn, MATCH) == "r1"
        assert count_effective_edits(conn, MATCH) == 1


@pytest.mark.unit
def test_cache_key_changes_with_every_input_and_with_the_rules_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = cache_key("s", "p", "m", FuseConfig())

    assert cache_key("s", "p", "m", FuseConfig()) == base
    assert cache_key("s2", "p", "m", FuseConfig()) != base
    assert cache_key("s", "p2", "m", FuseConfig()) != base
    assert cache_key("s", "p", None, FuseConfig()) != base
    assert cache_key("s", "p", "m", FuseConfig(merge_radius_s=2.0)) != base
    monkeypatch.setattr(stage_module, "RULES_VERSION", 2)
    assert cache_key("s", "p", "m", FuseConfig()) != base


@pytest.mark.unit
def test_run_fuse_unusable_database_is_logged_and_recomputation_fails_loudly(
    match_dir: Path,
) -> None:
    run_fuse(match_dir, MATCH, run_id="r1")
    for suffix in ("-wal", "-shm"):
        (match_dir / (DB_FILE + suffix)).unlink(missing_ok=True)
    (match_dir / DB_FILE).write_bytes(b"this is not a database" * 100)

    with capture_logs() as logs, pytest.raises(StorageError):
        run_fuse(match_dir, MATCH, run_id="r2")

    assert any(log["event"] == "database_unreadable" for log in logs)
