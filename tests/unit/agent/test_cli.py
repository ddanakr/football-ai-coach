"""`python -m aicoach.cli agent-eval` with a scripted model (no network)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach import cli
from aicoach.agent.evaluation import ANSWERS_FILE
from aicoach.agent.journal import JOURNAL_FILE
from tests.fixtures.agent_synth import FakeLLM, turn
from tests.fixtures.ui_synth import write_match_dir

FAKE_KEY = "AIzaFakeKeyForTests0123456789abcdefXYZ"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()  # main() points the logger at the captured stderr


def questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "q.csv"
    path.write_text(
        "id,type,question,expected,what_we_check\n1,count,Сколько было событий?,3,число\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_agent_eval_writes_answers_and_the_journal_and_prints_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")
    llm = FakeLLM([turn("", ("match_info", {})), turn("Система нашла 3 события.")])
    monkeypatch.setattr(cli, "gemini_client", lambda _settings: llm)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    code = cli.main(
        [
            "agent-eval",
            "--questions",
            str(questions_file(tmp_path)),
            "--video",
            str(tmp_path / "src.mp4"),
            "--match-id",
            "m1",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "validator passed on the first try: 1" in out
    assert "refused (could not confirm with data): 0" in out
    assert FAKE_KEY not in out
    answers = (tmp_path / ANSWERS_FILE).read_text(encoding="utf-8-sig")
    assert "Система нашла 3 события." in answers
    assert (tmp_path / JOURNAL_FILE).is_file()


@pytest.mark.unit
def test_agent_eval_without_a_key_exits_2_with_the_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")

    code = cli.main(
        [
            "agent-eval",
            "--questions",
            str(questions_file(tmp_path)),
            "--video",
            str(tmp_path / "src.mp4"),
            "--match-id",
            "m1",
        ]
    )

    assert code == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / ANSWERS_FILE).exists()


@pytest.mark.unit
def test_agent_eval_model_flag_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")
    seen: list[str] = []

    def fake_client(settings: object) -> FakeLLM:
        seen.append(getattr(settings, "gemini_model", ""))
        return FakeLLM([turn("Не знаю.")])

    monkeypatch.setattr(cli, "gemini_client", fake_client)
    monkeypatch.setenv("GEMINI_MODEL", "model-from-env")

    code = cli.main(
        [
            "agent-eval",
            "--questions",
            str(questions_file(tmp_path)),
            "--video",
            str(tmp_path / "src.mp4"),
            "--match-id",
            "m1",
            "--model",
            "model-from-cli",
        ]
    )

    assert code == 0
    assert seen == ["model-from-cli"]
    assert "model: model-from-cli" in capsys.readouterr().out


@pytest.mark.unit
def test_agent_eval_bad_questions_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "q.csv"
    bad.write_text("id,type\n1,a\n", encoding="utf-8")

    code = cli.main(["agent-eval", "--questions", str(bad), "--video", str(tmp_path / "src.mp4")])

    assert code == 2
    assert "no column question" in capsys.readouterr().err
