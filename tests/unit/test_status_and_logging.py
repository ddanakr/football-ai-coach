"""status.json handling and the structlog setup."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from aicoach.errors import ConfigError, StorageError
from aicoach.logging_setup import configure_logging, stage_logger
from aicoach.status import STATUS_FILE, read_status, write_status


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """structlog configuration is global: restore the defaults even if a test fails."""
    yield
    structlog.reset_defaults()


@pytest.mark.unit
def test_read_status_missing_file_gives_empty_status_without_a_warning(tmp_path: Path) -> None:
    with capture_logs() as logs:
        assert read_status(tmp_path).stages == {}
    assert logs == []


@pytest.mark.unit
def test_read_status_missing_directory_gives_empty_status(tmp_path: Path) -> None:
    assert read_status(tmp_path / "nope").stages == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "",
        '{"stages": {"s": {"state": "bogus", "progress": 5}}}',
        '{"stages": []}',
    ],
)
def test_read_status_broken_or_invalid_file_gives_empty_status_and_warns(
    tmp_path: Path, content: str
) -> None:
    (tmp_path / STATUS_FILE).write_text(content, encoding="utf-8")
    with capture_logs() as logs:
        assert read_status(tmp_path).stages == {}
    assert len(logs) == 1
    assert logs[0]["event"] == "status_file_unreadable"
    assert logs[0]["log_level"] == "warning"
    assert logs[0]["match_id"] == tmp_path.name
    assert "content" not in logs[0]


@pytest.mark.unit
def test_read_status_empty_object_is_an_empty_status(tmp_path: Path) -> None:
    (tmp_path / STATUS_FILE).write_text("{}", encoding="utf-8")
    assert read_status(tmp_path).stages == {}


@pytest.mark.unit
def test_write_status_keeps_other_stages(tmp_path: Path) -> None:
    write_status(tmp_path, "ingest", "done", 1.0)
    write_status(tmp_path, "scoreboard", "running", 0.25)
    status = read_status(tmp_path)
    assert status.stages["ingest"].state == "done"
    assert status.stages["scoreboard"].progress == 0.25


@pytest.mark.unit
def test_write_status_overwrites_the_same_stage_and_clears_the_error(tmp_path: Path) -> None:
    write_status(tmp_path, "scoreboard", "failed", 0.4, error="boom")
    write_status(tmp_path, "scoreboard", "done", 1.0)
    stage = read_status(tmp_path).stages["scoreboard"]
    assert (stage.state, stage.progress, stage.error) == ("done", 1.0, None)


@pytest.mark.unit
def test_write_status_records_the_error_text_including_non_ascii(tmp_path: Path) -> None:
    write_status(tmp_path, "scoreboard", "failed", 0.4, error="нет образцов Қ")
    data = json.loads((tmp_path / STATUS_FILE).read_text(encoding="utf-8"))
    assert data["stages"]["scoreboard"] == {
        "state": "failed",
        "progress": 0.4,
        "error": "нет образцов Қ",
    }


@pytest.mark.unit
@pytest.mark.parametrize(("given", "stored"), [(-0.5, 0.0), (1.7, 1.0), (0.0, 0.0), (1.0, 1.0)])
def test_write_status_clamps_progress(tmp_path: Path, given: float, stored: float) -> None:
    write_status(tmp_path, "scoreboard", "running", given)
    assert read_status(tmp_path).stages["scoreboard"].progress == stored


@pytest.mark.unit
def test_write_status_rejects_nan_progress(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NaN"):
        write_status(tmp_path, "scoreboard", "running", float("nan"))


@pytest.mark.unit
def test_write_status_replaces_a_broken_file(tmp_path: Path) -> None:
    (tmp_path / STATUS_FILE).write_text("{broken", encoding="utf-8")
    write_status(tmp_path, "scoreboard", "done", 1.0)
    assert read_status(tmp_path).stages["scoreboard"].state == "done"


@pytest.mark.unit
def test_write_status_leaves_no_temporary_file(tmp_path: Path) -> None:
    write_status(tmp_path, "scoreboard", "done", 1.0)
    assert [p.name for p in tmp_path.iterdir()] == [STATUS_FILE]


@pytest.mark.unit
def test_write_status_into_a_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="cannot write"):
        write_status(tmp_path / "missing", "scoreboard", "done", 1.0)


@pytest.mark.unit
def test_stage_logger_carries_stage_and_match_id() -> None:
    with capture_logs() as logs:
        stage_logger("scoreboard", "match01").info("stage_started")
    assert logs[0]["stage"] == "scoreboard"
    assert logs[0]["match_id"] == "match01"
    assert logs[0]["event"] == "stage_started"


@pytest.mark.unit
def test_configure_logging_writes_json_lines_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    stage_logger("scoreboard", "match01").warning("clock_missing", t_v=12.0)
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "clock_missing"
    assert record["level"] == "warning"
    assert record["stage"] == "scoreboard"


@pytest.mark.unit
def test_configure_logging_filters_below_the_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING")
    stage_logger("scoreboard", "m").info("hidden")
    assert capsys.readouterr().err == ""


@pytest.mark.unit
def test_configure_logging_can_be_called_twice_and_accepts_lower_case(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("error")
    configure_logging("info")
    stage_logger("scoreboard", "m").info("shown")
    assert "shown" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize("level", ["FOO", "", "  "])
def test_configure_logging_unknown_level_raises(level: str) -> None:
    with pytest.raises(ConfigError, match="unknown log level"):
        configure_logging(level)
