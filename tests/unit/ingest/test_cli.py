"""`python -m aicoach.cli ingest`: arguments, messages and exit codes (0, 2, 3)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach import cli
from aicoach.errors import IngestError
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.schemas import CandidatesResult
from aicoach.ingest.stage import EXIT_WAITING_TDEED, IngestResult, IngestTools

VID = "abcDEF_-123"  # invented id


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """`cli.main` binds the logger to the captured stderr; later tests must not inherit it."""
    yield
    structlog.reset_defaults()


def empty_fusion() -> CandidatesResult:
    return CandidatesResult(
        match_id=VID,
        cache_key="k",
        run_id=None,
        mode="with_model",
        scoreboard_version="a",
        plates_version="b",
        model_version="c",
        model_file="tdeed_x.json",
        config=FuseConfig(),
        half_bounds={},
        candidates=[],
        dropped=[],
        warnings=[],
        counts={},
    )


@pytest.mark.unit
def test_cli_ingest_waiting_for_tdeed_exits_3_and_explains_what_to_bring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, object] = {}

    def fake(url: str, data_root: Path, **kwargs: object) -> IngestResult:
        seen.update(url=url, data_root=data_root, **kwargs)
        return IngestResult(VID, VID, data_root / VID, None)

    monkeypatch.setattr(cli, "run_ingest", fake)

    code = cli.main(
        [
            "ingest",
            f"https://youtu.be/{VID}",
            "--data-root",
            str(tmp_path),
            "--templates",
            str(tmp_path / "t"),
            "--cpu",
        ]
    )

    out = capsys.readouterr().out
    assert code == EXIT_WAITING_TDEED == 3
    assert "tdeed_<имя>.json" in out
    assert str(tmp_path / VID) in out
    assert "ещё раз" in out
    assert seen == {
        "url": f"https://youtu.be/{VID}",
        "data_root": tmp_path,
        "match_id": None,
        "templates": tmp_path / "t",
        "hw_accel": False,
        "tools": None,
    }


@pytest.mark.unit
def test_cli_ingest_done_exits_0_and_reports_the_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake(_url: str, data_root: Path, **_kwargs: object) -> IngestResult:
        return IngestResult(VID, "m", data_root / "m", empty_fusion())

    monkeypatch.setattr(cli, "run_ingest", fake)

    code = cli.main(["ingest", "x", "--data-root", str(tmp_path), "--match-id", "m"])

    out = capsys.readouterr().out
    assert code == 0
    assert "candidates.json" in out
    assert str(tmp_path / "m") in out


@pytest.mark.unit
def test_cli_ingest_error_exits_2_with_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake(_url: str, _data_root: Path, **_kwargs: object) -> IngestResult:
        msg = "YouTube download failed: Video unavailable"
        raise IngestError(msg)

    monkeypatch.setattr(cli, "run_ingest", fake)

    code = cli.main(["ingest", "x", "--data-root", str(tmp_path)])

    assert code == 2
    assert "Video unavailable" in capsys.readouterr().err


@pytest.mark.unit
def test_cli_ingest_bad_link_exits_2_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["ingest", "https://example.com/v", "--data-root", str(tmp_path / "d")])

    assert code == 2
    assert "not a YouTube link" in capsys.readouterr().err
    assert not (tmp_path / "d").exists()


@pytest.mark.unit
def test_cli_ingest_data_root_defaults_to_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake(_url: str, data_root: Path, **_kwargs: object) -> IngestResult:
        seen["data_root"] = data_root
        return IngestResult(VID, VID, data_root / VID, empty_fusion())

    monkeypatch.setattr(cli, "run_ingest", fake)
    monkeypatch.setenv("AICOACH_DATA_ROOT", str(tmp_path / "from_env"))

    assert cli.main(["ingest", f"https://youtu.be/{VID}"]) == 0
    assert seen["data_root"] == tmp_path / "from_env"

    assert (
        cli.main(["ingest", f"https://youtu.be/{VID}", "--data-root", str(tmp_path / "cli")]) == 0
    )
    assert seen["data_root"] == tmp_path / "cli"


@pytest.mark.unit
def test_cli_ingest_without_any_data_root_exits_2_and_says_where_to_set_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["ingest", f"https://youtu.be/{VID}"])

    assert code == 2
    assert "AICOACH_DATA_ROOT" in capsys.readouterr().err


@pytest.mark.unit
def test_cli_ingest_passes_tesseract_cmd_to_the_names_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake(_url: str, data_root: Path, **kwargs: object) -> IngestResult:
        seen.update(kwargs)
        return IngestResult(VID, VID, data_root / VID, empty_fusion())

    monkeypatch.setattr(cli, "run_ingest", fake)
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "tess.exe"))
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "run_plate_names", lambda *_a, **k: calls.append(k))

    assert cli.main(["ingest", f"https://youtu.be/{VID}", "--data-root", str(tmp_path)]) == 0

    tools = seen["tools"]
    assert isinstance(tools, IngestTools)
    tools.stages.plate_names(tmp_path, "m", force=False)
    assert calls == [{"force": False, "tesseract_cmd": tmp_path / "tess.exe"}]
