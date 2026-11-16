"""The `final-eval` body on synthetic files: guard of half 2, config check, document."""

import json
from pathlib import Path

import pytest

from aicoach.errors import EvalError
from aicoach.eval.final_run import FinalPaths, run_final_eval
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.stage import run_fuse
from aicoach.storage import add_edit, connect, get_current_events
from tests.fixtures.fusion_synth import plate, write_inputs, write_tdeed
from tests.fixtures.plates_synth import change

PENDING = "будет заполнено после заморозки"
FPS = 25


@pytest.fixture
def paths(tmp_path: Path) -> FinalPaths:
    """A match directory with inputs, both databases, truth, baseline, journal and status."""
    match_dir = tmp_path / "m1"
    match_dir.mkdir()
    write_inputs(
        match_dir,
        [change((0, 0), (1, 0), 199.0, 200.0)],
        [
            plate("p1", "goal", 230.0),
            plate("p2", "subst", 600.0, team="away"),
            plate("p3", "card", 700.0),
        ],
    )
    tdeed = match_dir / "tdeed.json"
    write_tdeed(
        tdeed,
        [(300 * FPS, "Shots on target", 0.9), (2000 * FPS, "Shots on target", 0.9)],
    )
    run_fuse(match_dir, "m1", None, db_path=match_dir / "nomodel.db")
    run_fuse(match_dir, "m1", tdeed, db_path=match_dir / "model.db", force=True)
    truth = {
        "video": "x.mp4",
        "halves": {"1": {"start": 100, "end": 1000}, "2": {"start": 1500, "end": 2500}},
        "events": [
            {"type": "goal", "t_v": 200, "half": 1},
            {"type": "shot", "t_v": 301, "half": 1},
            {"type": "shot", "t_v": 2001, "half": 2},
            {"type": "foul", "t_v": 2100, "half": 2},
        ],
    }
    (match_dir / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    baseline = [{"type": "shot", "t": "00:33:20", "note": "SECRETNAME"}]
    (match_dir / "gemini.txt").write_text("source\n" + json.dumps(baseline), encoding="utf-8")
    record = {"tokens_in": 10, "tokens_out": 2, "duration_ms": 1000, "validator": "first_try"}
    (match_dir / "agent_log.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    stages = {
        "download": {
            "state": "done",
            "progress": 1.0,
            "started_at": "2026-11-16T10:00:00.000+00:00",
            "finished_at": "2026-11-16T10:03:00.000+00:00",
            "duration_s": 180.0,
        },
        "plate_names": {"state": "skipped", "progress": 1.0},
    }
    (match_dir / "status.json").write_text(json.dumps({"stages": stages}), encoding="utf-8")
    return FinalPaths(
        match_dir=match_dir,
        ground_truth=match_dir / "truth.json",
        model_db=match_dir / "model.db",
        nomodel_db=match_dir / "nomodel.db",
        tdeed=tdeed,
        gemini=match_dir / "gemini.txt",
        ingest_dir=match_dir,
    )


def test_run_without_final_leaves_half_two_and_the_baseline_unevaluated(
    paths: FinalPaths, tmp_path: Path
) -> None:
    out = tmp_path / "metrics.md"

    text = run_final_eval(paths, out)

    assert out.read_text(encoding="utf-8") == text
    assert "## 1-й тайм" in text
    assert "подбирались на этом тайме" in text
    assert f"## 2-й тайм\n\nНе посчитано: {PENDING}" in text
    assert "(d) Gemini-база | нет данных" in text
    assert "Сдвиг времён: " + PENDING in text
    assert "Событий в файле (разобрано) | 1" in text
    assert "внутри границ 2-го тайма" not in text
    assert "SECRETNAME" not in text
    assert "| download | done | 2026-11-16T10:00:00.000+00:00 |" in text
    assert "| 180.0 |" in text
    assert "| plate_names | skipped | — | — | — |" in text


def test_run_with_final_evaluates_half_two_and_checks_the_baseline_shift(
    paths: FinalPaths, tmp_path: Path
) -> None:
    text = run_final_eval(paths, tmp_path / "metrics.md", final=True)

    assert "ничего не подбиралось" in text
    assert "Не посчитано" not in text
    assert "Сдвиг времён: пар 1" in text
    assert "SECRETNAME" not in text
    assert "| сумма по матчу | ±5 с | " + PENDING not in text


def test_run_refuses_when_a_database_does_not_match_the_frozen_config(
    paths: FinalPaths, tmp_path: Path
) -> None:
    other = paths.match_dir / "other.db"
    run_fuse(
        paths.match_dir,
        "m1",
        None,
        FuseConfig(card_foul_lead_s=5.0),
        db_path=other,
        force=True,
    )
    stale = FinalPaths(**{**paths.__dict__, "nomodel_db": other})

    with pytest.raises(EvalError, match="frozen configuration"):
        run_final_eval(stale, tmp_path / "metrics.md")
    assert not (tmp_path / "metrics.md").exists()


@pytest.mark.parametrize("field", ["model_db", "nomodel_db", "gemini", "ground_truth", "tdeed"])
def test_run_with_a_missing_input_raises_eval_or_spotting_error(
    paths: FinalPaths, tmp_path: Path, field: str
) -> None:
    broken = FinalPaths(**{**paths.__dict__, field: paths.match_dir / "missing.file"})

    with pytest.raises(Exception, match=r"missing\.file|not found|cannot read"):
        run_final_eval(broken, tmp_path / "metrics.md")


def test_run_with_a_bad_status_file_raises(paths: FinalPaths, tmp_path: Path) -> None:
    (paths.match_dir / "status.json").write_text("[]", encoding="utf-8")

    with pytest.raises(EvalError, match="stage states"):
        run_final_eval(paths, tmp_path / "metrics.md")


def test_run_ignores_user_edits(paths: FinalPaths, tmp_path: Path) -> None:
    before = run_final_eval(paths, tmp_path / "a.md")
    with connect(paths.model_db) as conn:
        target = get_current_events(conn, "m1")[0].event_id
        add_edit(conn, "m1", "delete", author="u", target=target)

    after = run_final_eval(paths, tmp_path / "b.md")

    assert after == before


def test_run_without_an_agent_journal_shows_no_data(paths: FinalPaths, tmp_path: Path) -> None:
    (paths.match_dir / "agent_log.jsonl").unlink()

    text = run_final_eval(paths, tmp_path / "metrics.md")

    assert "Записей в журнале | 0" in text


def test_mismatch_message_has_no_counts(paths: FinalPaths, tmp_path: Path) -> None:
    other = paths.match_dir / "other.db"
    run_fuse(paths.match_dir, "m1", None, FuseConfig(card_foul_lead_s=5.0), db_path=other)
    stale = FinalPaths(**{**paths.__dict__, "nomodel_db": other})

    with pytest.raises(EvalError) as err:
        run_final_eval(stale, tmp_path / "metrics.md")

    assert not any(ch.isdigit() for ch in str(err.value).split(":", 1)[1].split("run `fuse`")[0])
