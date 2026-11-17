"""`speech-test` picks the provider, the model and the vocabulary run (no request is sent)."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import structlog

from aicoach import cli
from aicoach.speech.check_file import CheckFragment
from aicoach.speech.run import FragmentRun
from aicoach.speech.transcriber import Transcriber

LINEUP = "team,side,number,name\nА,home,1,Ветров Павел\n"


class Capture:
    """Stands in for `run_speech_test` and keeps what the command passed to it."""

    def __init__(self) -> None:
        self.transcriber: Transcriber | None = None
        self.vocabulary: Sequence[str] | None = None

    def __call__(
        self,
        video: Path,  # noqa: ARG002
        fragments: Sequence[CheckFragment],  # noqa: ARG002
        out_dir: Path,  # noqa: ARG002
        transcriber: Transcriber,
        *,
        vocabulary: Sequence[str] | None = None,
    ) -> list[FragmentRun]:
        self.transcriber = transcriber
        self.vocabulary = vocabulary
        return []


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


def match_dir(tmp_path: Path, *, lineup: bool = True) -> Path:
    (tmp_path / "speech_check.txt").write_text("00:00:10 - 00:01:00 ru\nслово\n", encoding="utf-8")
    if lineup:
        (tmp_path / "lineups.csv").write_text(LINEUP, encoding="utf-8")
    return tmp_path


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[int, Capture]:
    capture = Capture()
    monkeypatch.setattr(cli, "run_speech_test", capture)
    code = cli.main(["speech-test", "--video", str(tmp_path / "src.mp4"), *extra])
    return code, capture


@pytest.mark.unit
def test_default_provider_is_openrouter_with_the_gemini_transcribe_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")

    code, capture = run(match_dir(tmp_path), monkeypatch)

    out = capsys.readouterr().out
    assert code == 0
    assert capture.transcriber is not None
    assert capture.transcriber.model == "google/gemini-3.5-transcribe"
    assert "provider: openrouter" in out
    assert capture.vocabulary == ["Ветров", "Павел"]


@pytest.mark.unit
def test_model_flag_chooses_the_scribe_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")

    _, capture = run(match_dir(tmp_path), monkeypatch, "--model", "elevenlabs/scribe-v2")

    assert capture.transcriber is not None
    assert capture.transcriber.model == "elevenlabs/scribe-v2"


@pytest.mark.unit
def test_no_vocabulary_flag_skips_the_extra_run_even_with_a_lineup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-for-the-fake-client")

    _, capture = run(match_dir(tmp_path), monkeypatch, "--no-vocabulary")

    assert capture.vocabulary is None


@pytest.mark.unit
def test_gemini_provider_flag_uses_the_gemini_key_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "key-for-the-fake-client")
    monkeypatch.setenv("AICOACH_LLM_PROVIDER", "openrouter")

    code, capture = run(match_dir(tmp_path), monkeypatch, "--provider", "gemini")

    assert code == 0
    assert capture.transcriber is not None
    assert capture.transcriber.model == "gemini-3.5-transcribe"
    assert "provider: gemini" in capsys.readouterr().out


@pytest.mark.unit
def test_missing_key_exits_2_with_the_hint_of_the_chosen_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, capture = run(match_dir(tmp_path), monkeypatch)

    assert code == 2
    assert capture.transcriber is None
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
