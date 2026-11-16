"""Sound cutting with a replaced FFmpeg, and the whole test flow with a fake model."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from aicoach.errors import TranscriptionError
from aicoach.speech.audio import build_audio_command, cut_audio
from aicoach.speech.check_file import CheckFragment
from aicoach.speech.run import (
    format_summary,
    format_transcript,
    lineup_terms,
    run_speech_test,
)
from aicoach.speech.segments import TimedWord
from aicoach.speech.transcriber import RawTranscript


def _done(code: int = 0, stderr: bytes = b"") -> "subprocess.CompletedProcess[bytes]":
    return subprocess.CompletedProcess([], code, b"", stderr)


def _snd(video: Path, output: Path, start_s: float, length_s: float) -> bytes:  # noqa: ARG001 - signature of the cutter
    return b"snd"


@pytest.mark.unit
def test_command_cuts_mono_16k_mp3_in_the_given_window(tmp_path: Path) -> None:
    command = build_audio_command(tmp_path / "v.mp4", tmp_path / "o.mp3", 935.0, 99.0)

    assert command[command.index("-ss") + 1] == "935.000"
    assert command[command.index("-t") + 1] == "99.000"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert "-vn" in command


@pytest.mark.unit
def test_cut_audio_returns_bytes_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    def runner(command: list[str], timeout_s: float) -> "subprocess.CompletedProcess[bytes]":  # noqa: ARG001
        Path(command[-1]).write_bytes(b"mp3 bytes")
        return _done()

    output = tmp_path / "out" / "f1.mp3"

    assert cut_audio(tmp_path / "v.mp4", output, 1.0, 5.0, runner) == b"mp3 bytes"
    assert [p.name for p in output.parent.iterdir()] == ["f1.mp3"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start", "length", "code", "writes"),
    [(-1.0, 5.0, 0, True), (1.0, 0.0, 0, True), (1.0, 5.0, 1, True), (1.0, 5.0, 0, False)],
)
def test_cut_audio_fails_loudly(
    tmp_path: Path, start: float, length: float, code: int, writes: bool
) -> None:
    def runner(command: list[str], timeout_s: float) -> "subprocess.CompletedProcess[bytes]":  # noqa: ARG001
        if writes:
            Path(command[-1]).write_bytes(b"x")
        return _done(code, b"boom")

    with pytest.raises(TranscriptionError):
        cut_audio(tmp_path / "v.mp4", tmp_path / "o.mp3", start, length, runner)
    assert not list(tmp_path.glob("*.part"))


class _FakeTranscriber:
    model = "fake-model"

    def __init__(self) -> None:
        self.vocabularies: list[Sequence[str] | None] = []

    def transcribe(
        self,
        audio: bytes,  # noqa: ARG002 - Protocol signature
        mime_type: str,  # noqa: ARG002
        *,
        vocabulary: Sequence[str] | None = None,
    ) -> RawTranscript:
        self.vocabularies.append(vocabulary)
        if vocabulary is not None:
            return RawTranscript("Ветров тащил мяч", (), 7, 3)
        words = (TimedWord("Ветров", 1.0, 1.5, "spk:0"), TimedWord("тащил", 1.6, 2.0, "spk:0"))
        return RawTranscript("Ветров тащил", words, 10, 2)


def _fragment() -> CheckFragment:
    return CheckFragment(1, 100.0, 160.0, "ru", ("Ветров тащил мяч",))


@pytest.mark.unit
def test_run_writes_json_per_mode_and_a_summary(tmp_path: Path) -> None:
    fake = _FakeTranscriber()
    out = tmp_path / "speech_test"

    runs = run_speech_test(
        tmp_path / "v.mp4", [_fragment()], out, fake, vocabulary=["Ветров"], cutter=_snd
    )

    assert [r.mode for r in runs] == ["timed", "vocabulary"]
    assert fake.vocabularies == [None, ["Ветров"]]
    timed = json.loads((out / "fragment1_timed.json").read_text(encoding="utf-8"))
    assert timed["segments"][0]["t_v_start"] == 101.0
    assert "words" not in timed and "heard" not in timed
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [(r["found"], r["total"]) for r in summary["runs"]] == [(2, 3), (3, 3)]
    assert summary["runs"][0]["missing"] == ["мяч"]


@pytest.mark.unit
def test_run_without_vocabulary_makes_only_the_timed_run(tmp_path: Path) -> None:
    fake = _FakeTranscriber()

    runs = run_speech_test(tmp_path / "v.mp4", [_fragment()], tmp_path / "o", fake, cutter=_snd)

    assert [r.mode for r in runs] == ["timed"]


@pytest.mark.unit
def test_format_summary_and_transcript(tmp_path: Path) -> None:
    fake = _FakeTranscriber()
    runs = run_speech_test(
        tmp_path / "v.mp4", [_fragment()], tmp_path / "o", fake, vocabulary=["x"], cutter=_snd
    )

    text = format_summary(runs)
    assert "found 2/3 (67%)" in text
    assert "missing: мяч" in text
    assert "[101.0-102.0 s] spk:0 ru: Ветров тащил" in format_transcript(runs[0])
    assert "Ветров тащил мяч" in format_transcript(runs[1])


@pytest.mark.unit
def test_lineup_terms_from_a_file_and_none_without_it(tmp_path: Path) -> None:
    csv = tmp_path / "lineups.csv"
    csv.write_text(
        "team,side,number,name\nА,home,1,Ветров Павел\nА,home,2,Сомов Игорь\n"
        "Б,away,3,Ветров Олег\n",
        encoding="utf-8",
    )

    assert lineup_terms(csv) == ["Ветров", "Павел", "Сомов", "Игорь", "Олег"]
    assert lineup_terms(tmp_path / "none.csv") is None
    with pytest.raises(TranscriptionError):
        lineup_terms(tmp_path / "none.csv", required=True)
    assert lineup_terms(None) is None
