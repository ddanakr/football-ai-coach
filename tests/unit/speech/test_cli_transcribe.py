"""The `transcribe` command: numbers only, never the text of the commentary."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from aicoach import cli
from aicoach.speech.transcript import TranscriptResult
from aicoach.status import write_status
from tests.fixtures.transcript_synth import make_result


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.mark.unit
def test_transcribe_command_prints_the_counts_the_gaps_the_cost_and_no_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def fake(match_dir: Path, match_id: str, **kwargs: Any) -> TranscriptResult:
        seen.update(match_dir=match_dir, match_id=match_id, **kwargs)
        return make_result()

    monkeypatch.setattr(cli, "run_transcribe", fake)

    code = cli.main(["transcribe", "--match-dir", str(tmp_path), "--model", "x/y", "--force"])

    out = capsys.readouterr().out
    assert code == 0
    assert (
        "1 segments, 2 words, 0 player mentions, 1 chunks; model fake/stt; cost 0.0123 USD" in out
    )
    assert "half 1: 1 segments, 1 gaps, 20 s (20.0%) without speech" in out
    assert "warning: a warning" in out
    assert "Тест" not in out  # the commentary text is never printed
    assert seen["match_id"] == tmp_path.resolve().name
    assert seen["model"] == "x/y"
    assert seen["force"] is True
    assert seen["lineup_path"] is None


@pytest.mark.unit
def test_transcribe_command_unknown_cost_is_said_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_transcribe", lambda *_a, **_k: make_result(cost=None))

    assert cli.main(["transcribe", "--match-dir", str(tmp_path), "--match-id", "m"]) == 0

    assert "cost unknown" in capsys.readouterr().out


@pytest.mark.unit
def test_transcribe_command_skipped_step_exits_zero_and_shows_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_status(tmp_path, "transcribe", "skipped", 1.0, error="no key")
    monkeypatch.setattr(cli, "run_transcribe", lambda *_a, **_k: None)

    code = cli.main(["transcribe", "--match-dir", str(tmp_path)])

    assert code == 0
    assert "transcribe skipped: no key" in capsys.readouterr().out


@pytest.mark.unit
def test_transcribe_command_without_a_sound_is_a_handled_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["transcribe", "--match-dir", str(tmp_path)])

    assert code == 2
    assert "run ingest first" in capsys.readouterr().err
