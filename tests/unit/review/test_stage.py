"""The review-candidates stage: inputs, cache, failures, status."""

import hashlib
import json
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from aicoach.domain.events import EventType
from aicoach.errors import FusionError, SpottingError
from aicoach.fusion.schemas import CandidatesResult
from aicoach.fusion.stage import RESULT_FILE as CANDIDATES_FILE
from aicoach.fusion.stage import run_fuse
from aicoach.review.schemas import ReviewCandidatesResult
from aicoach.review.stage import (
    LAYER_VERSION,
    RESULT_FILE,
    STAGE,
    cache_key,
    load_review_candidates,
    run_review_candidates,
)
from aicoach.status import read_status
from tests.fixtures.fusion_synth import write_tdeed
from tests.fixtures.review_synth import write_review_match

MATCH = "m1"


@pytest.fixture
def match(tmp_path: Path) -> tuple[Path, Path]:
    """The match directory and its T-DEED file, after the fusion stage."""
    _, tdeed = write_review_match(tmp_path, MATCH)
    return tmp_path, tdeed


def layer(result: ReviewCandidatesResult) -> list[tuple[str, float, int]]:
    return [(c.type.value, c.t_v, c.half) for c in result.candidates]


@pytest.mark.unit
def test_stage_writes_the_layer_without_events_of_the_catalog(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match

    result = run_review_candidates(match_dir, MATCH, tdeed)

    assert layer(result) == [
        ("shot", 210.0, 1),
        ("shot", 600.0, 1),
        ("shot", 700.0, 1),
        ("shot", 2000.0, 2),
    ]
    assert [c.id for c in result.candidates] == ["r:1", "r:2", "r:3", "r:4"]
    # 8 shot detections: 2 near the shot event (it is one of them), 1 merged into a stronger
    # neighbour, 1 in the break. Goals, substitutions, corners and fouls are not counted.
    assert (result.n_model, result.n_existing, result.n_merged, result.n_outside_halves) == (
        8,
        2,
        1,
        1,
    )
    assert result.counts == {"shot": {"1": 3, "2": 1}}
    assert load_review_candidates(match_dir) == result
    assert result.model_file == tdeed.name
    assert result.match_id == MATCH


@pytest.mark.unit
def test_every_candidate_is_away_from_every_event_of_its_type(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match
    result = run_review_candidates(match_dir, MATCH, tdeed)
    events = CandidatesResult.model_validate_json(
        (match_dir / CANDIDATES_FILE).read_text(encoding="utf-8")
    )

    for candidate in result.candidates:
        same_type = [e.t_v for e in events.candidates if e.type is candidate.type]
        assert all(abs(candidate.t_v - t) > result.merge_radius_s for t in same_type)


@pytest.mark.unit
def test_second_run_reads_the_cache_and_force_builds_again(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match
    first = run_review_candidates(match_dir, MATCH, tdeed)
    path = match_dir / RESULT_FILE
    stamp = path.stat().st_mtime_ns

    second = run_review_candidates(match_dir, MATCH, tdeed)
    assert second == first
    assert path.stat().st_mtime_ns == stamp  # nothing was written

    third = run_review_candidates(match_dir, MATCH, tdeed, force=True)
    assert third == first
    assert path.stat().st_mtime_ns != stamp


@pytest.mark.unit
def test_a_result_of_another_match_id_is_not_taken_from_the_cache(
    match: tuple[Path, Path],
) -> None:
    match_dir, tdeed = match
    first = run_review_candidates(match_dir, MATCH, tdeed)

    other = run_review_candidates(match_dir, "other", tdeed)

    assert other.match_id == "other"
    assert other.candidates == first.candidates


@pytest.mark.unit
def test_status_goes_to_done(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match

    run_review_candidates(match_dir, MATCH, tdeed)

    entry = read_status(match_dir).stages[STAGE]
    assert (entry.state, entry.progress) == ("done", 1.0)


@pytest.mark.unit
def test_cache_key_depends_on_each_input() -> None:
    base = cache_key("a", "b", "c")

    assert cache_key("a", "b", "c") == base
    changed = {cache_key("x", "b", "c"), cache_key("a", "x", "c"), cache_key("a", "b", "x")}
    assert base not in changed
    assert len(changed) == 3
    assert LAYER_VERSION >= 1


@pytest.mark.unit
@pytest.mark.parametrize("missing", [CANDIDATES_FILE, "scoreboard.json"])
def test_missing_input_is_a_fusion_error_that_says_what_to_run(
    match: tuple[Path, Path], missing: str
) -> None:
    match_dir, tdeed = match
    (match_dir / missing).unlink()

    with pytest.raises(FusionError, match="not found"):
        run_review_candidates(match_dir, MATCH, tdeed)

    assert read_status(match_dir).stages[STAGE].state == "failed"
    assert not (match_dir / RESULT_FILE).exists()


@pytest.mark.unit
def test_missing_model_file_is_a_fusion_error(match: tuple[Path, Path]) -> None:
    match_dir, _ = match

    with pytest.raises(FusionError, match="not found"):
        run_review_candidates(match_dir, MATCH, match_dir / "nope.json")


@pytest.mark.unit
def test_candidates_made_from_another_model_file_are_refused(match: tuple[Path, Path]) -> None:
    match_dir, _ = match
    other = match_dir / "tdeed_other.json"
    write_tdeed(other, [(1000, "Foul", 0.5)])

    with pytest.raises(FusionError, match="another T-DEED file"):
        run_review_candidates(match_dir, MATCH, other)


@pytest.mark.unit
def test_candidates_json_of_a_no_model_run_is_refused(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match
    run_fuse(match_dir, MATCH, None, force=True, discard_edits=True, run_id="run2")

    with pytest.raises(FusionError, match="another T-DEED file"):
        run_review_candidates(match_dir, MATCH, tdeed)


@pytest.mark.unit
def test_garbage_input_file_is_a_fusion_error_not_a_traceback(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match
    (match_dir / CANDIDATES_FILE).write_text("{not json", encoding="utf-8")

    with pytest.raises(FusionError, match="not valid"):
        run_review_candidates(match_dir, MATCH, tdeed)


@pytest.mark.unit
def test_a_broken_model_file_is_a_spotting_error(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match
    # The fusion output must name the broken file by its hash, or the hash check fires first.
    tdeed.write_text(json.dumps({"predictions": [{"frame": "x"}]}), encoding="utf-8")
    path = match_dir / CANDIDATES_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model_version"] = hashlib.sha256(tdeed.read_bytes()).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SpottingError):
        run_review_candidates(match_dir, MATCH, tdeed)


@pytest.mark.unit
def test_load_review_candidates_is_none_without_a_valid_file(tmp_path: Path) -> None:
    assert load_review_candidates(tmp_path) is None
    (tmp_path / RESULT_FILE).write_text("[]", encoding="utf-8")
    assert load_review_candidates(tmp_path) is None


@pytest.mark.unit
def test_a_broken_layer_file_is_logged_but_a_missing_one_is_not(tmp_path: Path) -> None:
    with capture_logs() as quiet:
        assert load_review_candidates(tmp_path) is None
    (tmp_path / RESULT_FILE).write_text("{broken", encoding="utf-8")
    with capture_logs() as loud:
        assert load_review_candidates(tmp_path) is None

    assert quiet == []
    assert [e["event"] for e in loud] == ["review_candidates_unreadable"]


@pytest.mark.unit
def test_event_types_of_the_layer_are_known_categories(match: tuple[Path, Path]) -> None:
    match_dir, tdeed = match

    result = run_review_candidates(match_dir, MATCH, tdeed)

    assert {c.type for c in result.candidates} <= set(EventType)
