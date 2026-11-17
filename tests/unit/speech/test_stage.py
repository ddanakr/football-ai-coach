"""The transcription stage with a fake model and a fake FFmpeg (no network, no real sound)."""

import json
import subprocess
import wave
from pathlib import Path

import pytest

from aicoach.config import DEFAULT_SPEECH_MODEL, Settings, load_settings
from aicoach.errors import ExternalServiceError, TranscriptionError
from aicoach.scoreboard.video import VideoInfo
from aicoach.speech.chunks import Chunk
from aicoach.speech.segments import TimedWord
from aicoach.speech.stage import (
    CHUNKS_DIR,
    AudioSource,
    chunk_key,
    default_model_name,
    find_audio_source,
    run_transcribe,
)
from aicoach.speech.transcriber import RawTranscript
from aicoach.speech.transcript import RESULT_FILE, TranscriptResult
from aicoach.status import read_status
from tests.fixtures.plates_synth import scoreboard_result

VIDEO_FPS = 25.0
DURATION_S = 400.0  # three chunks of 180 s with 10 s overlap: [0,180] [170,350] [340,400]


def probe(_path: Path) -> VideoInfo:
    return VideoInfo(VIDEO_FPS, int(DURATION_S * VIDEO_FPS), 1280, 720)


def runner(command: list[str], _timeout_s: float) -> "subprocess.CompletedProcess[bytes]":
    """A fake FFmpeg: the 'sound' is the start of the cut, so the fake model knows where it is."""
    start = command[command.index("-ss") + 1]
    Path(command[-1]).write_bytes(start.encode())
    return subprocess.CompletedProcess(command, 0, b"", b"")


def failing_runner(command: list[str], _timeout_s: float) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.CompletedProcess(command, 1, b"", b"no such stream")


def speech_at(start_s: float) -> list[TimedWord]:
    """Commentary known by the video time: a word of Тестов every 20 s from 100 s on, times
    counted from the start of the chunk (what a model returns)."""
    return [
        TimedWord("Тестова", float(t - start_s), t - start_s + 0.5, "s0")
        for t in range(100, 400, 20)
        if start_s <= t <= start_s + 180.0
    ]


class FakeSTT:
    """Answers by the start second of the chunk; may fail at a given request number."""

    def __init__(self, *, fail_at: int | None = None, cost: float | None = 0.01) -> None:
        self.model = "fake/stt"
        self.starts: list[float] = []
        self.fail_at = fail_at
        self.cost = cost
        self.untimed = False

    def transcribe(
        self,
        audio: bytes,
        mime_type: str,  # noqa: ARG002
        *,
        vocabulary: object = None,  # noqa: ARG002
        duration_s: float | None = None,
    ) -> RawTranscript:
        if self.fail_at is not None and len(self.starts) == self.fail_at:
            msg = "service down"
            raise ExternalServiceError(msg)
        start = float(audio.decode())
        self.starts.append(start)
        if self.untimed:
            return RawTranscript("some text", (), 0, 0, duration_s, self.cost)
        words = tuple(speech_at(start))
        return RawTranscript(" ".join(w.text for w in words), words, 0, 0, duration_s, self.cost)


def make_match(tmp_path: Path, *, scoreboard: bool = True, lineup: str | None = None) -> Path:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    (match_dir / "src.mp4").write_bytes(b"video")
    if scoreboard:
        board = scoreboard_result(segments=[(100.0, 380.0), (390.0, 400.0)], n_samples=400)
        (match_dir / "scoreboard.json").write_text(board.model_dump_json(), encoding="utf-8")
    if lineup is not None:
        (match_dir / "lineups.csv").write_text(lineup, encoding="utf-8")
    return match_dir


def go(match_dir: Path, stt: FakeSTT | None, *, force: bool = False) -> TranscriptResult | None:
    return run_transcribe(
        match_dir,
        "m",
        transcriber=stt,
        ffmpeg=runner,
        probe=probe,
        settings=load_settings(env_file=match_dir / "none.env"),
        force=force,
    )


LINEUP = "side,name,number,aliases\nhome,Тестов Иван,9,Тесто\naway,Образцов Марат,7,\n"


@pytest.mark.unit
def test_stage_sends_every_chunk_once_and_writes_a_transcript_without_duplicates(
    tmp_path: Path,
) -> None:
    match_dir = make_match(tmp_path)
    stt = FakeSTT()

    result = go(match_dir, stt)

    assert result is not None
    assert stt.starts == [0.0, 170.0, 340.0]
    # The words at 180 and 340 s are heard by two chunks each: one copy is kept.
    assert [s.t_start for s in result.segments] == [float(t) for t in range(100, 400, 20)]
    assert result.stats.n_chunks == 3
    assert result.stats.n_words == len(result.segments)
    assert result.stats.cost_usd == pytest.approx(0.03)
    assert [c.cost_usd for c in result.chunks] == [0.01, 0.01, 0.01]
    on_disk = TranscriptResult.model_validate_json((match_dir / RESULT_FILE).read_text("utf-8"))
    assert on_disk == result
    assert read_status(match_dir).stages["transcribe"].state == "done"
    assert not list((match_dir / CHUNKS_DIR).glob("*.mp3"))  # the cut sound is not kept


@pytest.mark.unit
def test_stage_marks_the_half_of_each_segment_from_the_scoreboard(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)

    result = go(match_dir, FakeSTT())

    assert result is not None
    by_start = {s.t_start: s.half for s in result.segments}
    assert by_start[100.0] == 1
    assert by_start[380.0] == 1
    assert {h.half for h in result.halves} == {1, 2}


@pytest.mark.unit
def test_stage_second_run_asks_nothing(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)
    first = go(match_dir, FakeSTT())
    stt = FakeSTT()

    second = go(match_dir, stt)

    assert stt.starts == []
    assert second == first


@pytest.mark.unit
def test_stage_stopped_run_resumes_from_the_first_chunk_without_an_answer(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)
    broken = FakeSTT(fail_at=1)

    with pytest.raises(ExternalServiceError, match="service down"):
        go(match_dir, broken)

    assert broken.starts == [0.0]
    assert not (match_dir / RESULT_FILE).exists()
    assert (match_dir / CHUNKS_DIR / "chunk_0000.json").is_file()  # the paid answer is kept
    status = read_status(match_dir).stages["transcribe"]
    assert status.state == "failed"
    assert "service down" in (status.error or "")

    healthy = FakeSTT()
    result = go(match_dir, healthy)

    assert healthy.starts == [170.0, 340.0]
    assert result is not None
    assert read_status(match_dir).stages["transcribe"].state == "done"


@pytest.mark.unit
def test_stage_new_lineup_rebuilds_the_mentions_without_a_new_request(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)
    plain = go(match_dir, FakeSTT())
    assert plain is not None
    assert plain.stats.n_mentions == 0
    (match_dir / "lineups.csv").write_text(LINEUP, encoding="utf-8")
    stt = FakeSTT()

    result = go(match_dir, stt)

    assert stt.starts == []
    assert result is not None
    assert result.lineup == "lineups.csv"
    assert result.stats.n_mentions == result.stats.n_segments
    assert result.segments[0].players[0].name == "Тестов Иван"
    assert result.segments[0].players[0].number == 9


@pytest.mark.unit
def test_stage_force_rebuilds_from_kept_answers_and_does_not_ask_again(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)
    go(match_dir, FakeSTT())
    (match_dir / RESULT_FILE).write_text("garbage", encoding="utf-8")
    stt = FakeSTT()

    result = go(match_dir, stt, force=True)

    assert stt.starts == []
    assert result is not None
    assert json.loads((match_dir / RESULT_FILE).read_text("utf-8"))["stats"]["n_chunks"] == 3


@pytest.mark.unit
def test_stage_unreadable_chunk_file_is_asked_again(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)
    go(match_dir, FakeSTT())
    (match_dir / CHUNKS_DIR / "chunk_0001.json").write_text("{broken", encoding="utf-8")
    (match_dir / RESULT_FILE).unlink()
    stt = FakeSTT()

    go(match_dir, stt)

    assert stt.starts == [170.0]


@pytest.mark.unit
def test_stage_without_a_key_is_skipped_with_a_reason_and_writes_nothing(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)
    settings = load_settings(env_file=tmp_path / "none.env")  # no OPENROUTER_API_KEY anywhere

    result = run_transcribe(match_dir, "m", settings=settings, ffmpeg=runner, probe=probe)

    assert result is None
    status = read_status(match_dir).stages["transcribe"]
    assert status.state == "skipped"
    assert status.error
    assert not (match_dir / RESULT_FILE).exists()


@pytest.mark.unit
def test_stage_no_sound_at_all_fails_loudly(tmp_path: Path) -> None:
    match_dir = tmp_path / "empty"
    match_dir.mkdir()

    with pytest.raises(TranscriptionError, match="run ingest first"):
        go(match_dir, FakeSTT())

    assert read_status(match_dir).stages["transcribe"].state == "failed"


@pytest.mark.unit
def test_stage_ffmpeg_failure_is_a_transcription_error(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)

    with pytest.raises(TranscriptionError):
        run_transcribe(
            match_dir,
            "m",
            transcriber=FakeSTT(),
            ffmpeg=failing_runner,
            probe=probe,
            settings=load_settings(env_file=tmp_path / "none.env"),
        )

    assert read_status(match_dir).stages["transcribe"].state == "failed"


@pytest.mark.unit
def test_stage_text_without_word_times_is_a_warning_not_silence(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)
    stt = FakeSTT()
    stt.untimed = True

    result = go(match_dir, stt)

    assert result is not None
    assert result.stats.n_segments == 0
    assert any("without word times" in w for w in result.warnings)
    assert result.halves[0].gap_share == 1.0  # the metric shows the loss


@pytest.mark.unit
def test_stage_without_scoreboard_counts_gaps_over_the_whole_sound_and_says_so(
    tmp_path: Path,
) -> None:
    match_dir = make_match(tmp_path, scoreboard=False)

    result = go(match_dir, FakeSTT())

    assert result is not None
    assert [h.half for h in result.halves] == [None]
    assert any("halves unknown" in w for w in result.warnings)
    assert all(s.half is None for s in result.segments)


@pytest.mark.unit
def test_stage_costs_not_reported_leave_the_total_unknown(tmp_path: Path) -> None:
    match_dir = make_match(tmp_path)

    result = go(match_dir, FakeSTT(cost=None))

    assert result is not None
    assert result.stats.cost_usd is None


@pytest.mark.unit
def test_find_audio_source_prefers_audio_wav_and_reads_its_length(tmp_path: Path) -> None:
    (tmp_path / "src.mp4").write_bytes(b"video")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(100)
        handle.writeframes(b"\x00\x00" * 750)

    def never(_path: Path) -> VideoInfo:
        msg = "the video must not be probed when audio.wav is there"
        raise AssertionError(msg)

    source = find_audio_source(tmp_path, never)

    assert source.path.name == "audio.wav"
    assert source.duration_s == pytest.approx(7.5)


@pytest.mark.unit
def test_find_audio_source_broken_wav_is_a_transcription_error(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").write_bytes(b"not a wave file")

    with pytest.raises(TranscriptionError, match="audio.wav"):
        find_audio_source(tmp_path, probe)


@pytest.mark.unit
def test_chunk_key_depends_on_model_sound_and_stretch() -> None:
    source = AudioSource(Path("a.wav"), 100.0, "a.wav:1:1")
    other = AudioSource(Path("a.wav"), 100.0, "a.wav:2:1")
    chunk = Chunk(0, 0.0, 50.0)

    keys = {
        chunk_key("m1", source, chunk),
        chunk_key("m2", source, chunk),
        chunk_key("m1", other, chunk),
        chunk_key("m1", source, Chunk(0, 0.0, 60.0)),
    }

    assert len(keys) == 4
    assert chunk_key("m1", source, chunk) == chunk_key("m1", source, Chunk(7, 0.0, 50.0))


@pytest.mark.unit
def test_default_model_is_scribe_through_openrouter_and_the_gemini_model_for_gemini() -> None:
    openrouter = Settings(_env_file=None)  # type: ignore[call-arg]
    gemini = Settings(_env_file=None, aicoach_llm_provider="gemini")  # type: ignore[call-arg]

    assert default_model_name(openrouter, None) == DEFAULT_SPEECH_MODEL == "elevenlabs/scribe-v2"
    assert default_model_name(gemini, None) == gemini.gemini_transcribe_model
    assert default_model_name(openrouter, "x/y") == "x/y"


@pytest.mark.unit
def test_stage_bad_setting_is_a_failure_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aicoach.errors import ConfigError

    match_dir = make_match(tmp_path)
    monkeypatch.setenv("AICOACH_BUDGET_USD", "abc")

    with pytest.raises(ConfigError, match="AICOACH_BUDGET_USD"):
        run_transcribe(match_dir, "m", ffmpeg=runner, probe=probe)

    assert read_status(match_dir).stages["transcribe"].state == "failed"
