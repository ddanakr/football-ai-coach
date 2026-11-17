"""The transcription stage: the whole match sound to `transcript.json` (FR-2, D-070).

Input: `audio.wav` (else the sound of `src.mp4`), `scoreboard.json` (the halves) and an optional
`lineups.csv` (player mentions) in the match directory. The sound goes to the model in overlapping
chunks (`chunks.py`); the answer of every chunk is kept in `speech_chunks/`, so a stopped run goes
on from the chunk where it stopped and a new lineup or new parameters rebuild the transcript
without a new request. Output: `transcript.json` and the state in `status.json`. Without a key of
the provider the stage is "skipped", not failed. The transcript does not go to `predictions`.
"""

import hashlib
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from aicoach.clips import Runner, subprocess_runner
from aicoach.config import DEFAULT_SPEECH_MODEL, Settings, load_settings
from aicoach.errors import AicoachError, ConfigError, TranscriptionError
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.halves import half_bounds
from aicoach.i18n import ru
from aicoach.ingest.download import VIDEO_FILE
from aicoach.ingest.media import AUDIO_FILE
from aicoach.logging_setup import stage_logger
from aicoach.plate_names.lineup import LINEUP_FILE, Player
from aicoach.plate_names.stage import load_lineup_file
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.scoreboard.stage import RESULT_FILE as SCOREBOARD_FILE
from aicoach.scoreboard.video import VideoInfo, probe_video
from aicoach.speech.audio import AUDIO_MIME, cut_audio
from aicoach.speech.chunks import Chunk, ChunkWords, merge_chunks, plan_chunks
from aicoach.speech.segments import TimedWord
from aicoach.speech.service import make_transcriber
from aicoach.speech.transcriber import Transcriber
from aicoach.speech.transcript import (
    RESULT_FILE,
    RULES_VERSION,
    SCHEMA_VERSION,
    ChunkInfo,
    TranscribeConfig,
    TranscriptResult,
    TranscriptStats,
    build_transcript_segments,
    find_gaps,
)
from aicoach.stage_common import (
    Logger,
    Progress,
    elapsed_ms,
    record_failure,
    set_status,
    write_atomic,
)

STAGE = "transcribe"
CHUNKS_DIR = "speech_chunks"
CHUNK_FORMAT = 1  # raise when the cached chunk files change their meaning
_MS = 1000
_COST_DIGITS = 6

Probe = Callable[[Path], VideoInfo]


class MissingKeyError(ConfigError):
    """The key of the provider is not set: the stage is skipped, not failed."""


def _require_key(settings: Settings) -> None:
    """Only a missing key is "skipped"; any other ConfigError is a real failure.

    Raises:
        MissingKeyError: No key for the provider of the settings.
    """
    provider = settings.aicoach_llm_provider
    key = settings.gemini_api_key if provider == "gemini" else settings.openrouter_api_key
    if key is None:
        raise MissingKeyError(ru.MISSING_SPEECH_KEY.format(provider=provider))


class _CachedWord(BaseModel):
    text: str
    start_s: float
    end_s: float
    speaker: str | None = None


class _CachedChunk(BaseModel):
    """What one request returned, as kept in `speech_chunks/`. `untimed_chars` is the length of a
    text the model returned without word times (nothing usable came of it)."""

    key: str
    index: int
    start_s: float
    end_s: float
    model: str
    words: list[_CachedWord]
    untimed_chars: int = 0
    audio_seconds: float | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class AudioSource:
    """The file the chunks are cut from, its length and a cheap fingerprint for the cache."""

    path: Path
    duration_s: float
    fingerprint: str


def _fingerprint(path: Path) -> str:
    info = path.stat()
    return f"{path.name}:{info.st_size}:{info.st_mtime_ns}"


def find_audio_source(match_dir: Path, probe: Probe = probe_video) -> AudioSource:
    """`audio.wav` of the match, else the video `src.mp4`.

    Raises:
        TranscriptionError: Neither file is there, or its length cannot be read.
    """
    audio, video = match_dir / AUDIO_FILE, match_dir / VIDEO_FILE
    if audio.is_file():
        try:
            with wave.open(str(audio), "rb") as handle:
                duration = handle.getnframes() / handle.getframerate()
        except (wave.Error, EOFError, OSError, ZeroDivisionError) as err:
            msg = f"cannot read the length of {audio.name}: {type(err).__name__}"
            raise TranscriptionError(msg) from err
        return AudioSource(audio, duration, _fingerprint(audio))
    if video.is_file():
        try:
            duration = probe(video).duration_s
        except AicoachError as err:
            msg = f"cannot read the length of {video.name}: {err}"
            raise TranscriptionError(msg) from err
        return AudioSource(video, duration, _fingerprint(video))
    msg = f"no {AUDIO_FILE} and no {VIDEO_FILE} in {match_dir}: run ingest first"
    raise TranscriptionError(msg)


def half_spans(match_dir: Path, log: Logger) -> dict[int, tuple[float, float]]:
    """The video seconds of both halves, as the fusion stage finds them from `scoreboard.json`
    (the margins of `FuseConfig`); empty when the file is missing or has no two clock segments."""
    path = match_dir / SCOREBOARD_FILE
    try:
        board = ScoreboardResult.model_validate_json(path.read_text(encoding="utf-8"))
        config = FuseConfig()
        bounds = half_bounds(
            board.offset_segments, config.half_start_margin_s, config.half_end_margin_s
        )
    except (OSError, ValueError, AicoachError) as err:
        log.warning("no_half_bounds", reason=type(err).__name__)
        return {}
    return {int(half): (span.start, span.end) for half, span in bounds.items()}


def chunk_key(model: str, source: AudioSource, chunk: Chunk) -> str:
    """What makes a chunk answer the same: the model, the sound, the stretch, the file format."""
    text = (
        f"{CHUNK_FORMAT}|{model}|{source.fingerprint}|"
        f"{round(chunk.start_s * _MS)}|{round(chunk.end_s * _MS)}"
    )
    return hashlib.sha256(text.encode()).hexdigest()


def cache_key(
    model: str,
    source: AudioSource,
    config: TranscribeConfig,
    lineup_version: str | None,
    spans: dict[int, tuple[float, float]],
) -> str:
    """A hash of everything that changes `transcript.json`: the model, the sound, the parameters,
    the lineup file, the halves and the version of the rules."""
    digest = hashlib.sha256()
    digest.update(
        f"{RULES_VERSION}|{SCHEMA_VERSION}|{CHUNK_FORMAT}|{model}|{source.fingerprint}|"
        f"{lineup_version}|{sorted(spans.items())}".encode()
    )
    digest.update(config.model_dump_json().encode())
    return digest.hexdigest()


def _chunk_path(match_dir: Path, chunk: Chunk) -> Path:
    return match_dir / CHUNKS_DIR / f"chunk_{chunk.index:04d}.json"


def _load_chunk(match_dir: Path, chunk: Chunk, key: str, log: Logger) -> _CachedChunk | None:
    path = _chunk_path(match_dir, chunk)
    try:
        cached = _CachedChunk.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError) as err:
        log.warning("chunk_cache_unreadable", path=path.name, reason=type(err).__name__)
        return None
    return cached if cached.key == key else None


def _load_cached_result(match_dir: Path, key: str, log: Logger) -> TranscriptResult | None:
    path = match_dir / RESULT_FILE
    try:
        result = TranscriptResult.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError) as err:
        log.warning("cache_unreadable", path=str(path), reason=type(err).__name__)
        return None
    return result if result.cache_key == key else None


def _request_chunk(
    match_dir: Path,
    source: AudioSource,
    chunk: Chunk,
    key: str,
    model: Transcriber,
    runner: Runner,
) -> _CachedChunk:
    """Cut the chunk, send it, keep the answer. The cut MP3 is not kept."""
    mp3 = match_dir / CHUNKS_DIR / f"chunk_{chunk.index:04d}.mp3"
    try:
        audio = cut_audio(source.path, mp3, chunk.start_s, chunk.length_s, runner)
    finally:
        mp3.unlink(missing_ok=True)
    raw = model.transcribe(audio, AUDIO_MIME, duration_s=chunk.length_s)
    cached = _CachedChunk(
        key=key,
        index=chunk.index,
        start_s=chunk.start_s,
        end_s=chunk.end_s,
        model=model.model,
        words=[
            _CachedWord(text=w.text, start_s=w.start_s, end_s=w.end_s, speaker=w.speaker)
            for w in raw.words
        ],
        untimed_chars=0 if raw.words else len(raw.text),
        audio_seconds=raw.audio_seconds,
        cost_usd=raw.cost_usd,
    )
    write_atomic(_chunk_path(match_dir, chunk), cached.model_dump_json())
    return cached


def _total_cost(chunks: list[_CachedChunk]) -> float | None:
    costs = [c.cost_usd for c in chunks if c.cost_usd is not None]
    return round(sum(costs), _COST_DIGITS) if costs else None


def default_model_name(settings: Settings, model: str | None) -> str:
    """The model the stage uses: the given one, else scribe-v2 through OpenRouter, else the
    transcription model of the Gemini setting."""
    if model:
        return model
    if settings.aicoach_llm_provider == "gemini":
        return settings.gemini_transcribe_model
    return DEFAULT_SPEECH_MODEL


@dataclass(frozen=True)
class _Request:
    """Everything one run needs besides the match directory."""

    config: TranscribeConfig
    lineup_path: Path | None
    force: bool
    model_name: str
    make_model: Callable[[], Transcriber]
    runner: Runner
    probe: Probe


def _run(
    match_dir: Path, request: _Request, progress: Progress, log: Logger
) -> tuple[TranscriptResult, bool]:
    """Returns the result and whether it came from the cache."""
    config = request.config
    source = find_audio_source(match_dir, request.probe)
    chunks = plan_chunks(source.duration_s, config.chunk_s, config.overlap_s)
    loaded = load_lineup_file(match_dir, request.lineup_path)
    lineup: list[Player] | None = None if loaded is None else loaded[0]
    lineup_version = None if loaded is None else loaded[1]
    spans = half_spans(match_dir, log)
    key = cache_key(request.model_name, source, config, lineup_version, spans)
    if not request.force:
        cached_result = _load_cached_result(match_dir, key, log)
        if cached_result is not None:
            return cached_result, True
    set_status(match_dir, STAGE, "running", 0.0, log)
    model: Transcriber | None = None
    done: list[_CachedChunk] = []
    requested = 0
    for chunk in chunks:
        ckey = chunk_key(request.model_name, source, chunk)
        cached = _load_chunk(match_dir, chunk, ckey, log)
        if cached is None:
            model = model or request.make_model()
            cached = _request_chunk(match_dir, source, chunk, ckey, model, request.runner)
            requested += 1
            log.info("chunk_done", index=chunk.index, of=len(chunks), words=len(cached.words))
        done.append(cached)
        progress.update(len(done) / len(chunks))
    log.info("chunks_ready", n_chunks=len(chunks), requested=requested)
    return _assemble(
        match_dir, request, source, chunks, done, lineup, lineup_version, spans, key, log
    ), False


def _assemble(
    match_dir: Path,
    request: _Request,
    source: AudioSource,
    chunks: list[Chunk],
    done: list[_CachedChunk],
    lineup: list[Player] | None,
    lineup_version: str | None,
    spans: dict[int, tuple[float, float]],
    key: str,
    log: Logger,
) -> TranscriptResult:
    config = request.config
    results = [
        ChunkWords(chunk, tuple(TimedWord(w.text, w.start_s, w.end_s, w.speaker) for w in c.words))
        for chunk, c in zip(chunks, done, strict=True)
    ]
    words, unresolved = merge_chunks(results)
    segments = build_transcript_segments(words, spans, lineup, config.min_similarity)
    gaps, halves = find_gaps(segments, spans, source.duration_s, config.min_gap_s)
    warnings = [
        f"chunk {c.index}: {c.untimed_chars} characters of text came without word times"
        for c in done
        if c.untimed_chars
    ]
    if not spans:
        warnings.append(
            "halves unknown (no usable scoreboard.json): gaps counted over the whole sound"
        )
    for text in warnings:
        log.warning("transcribe_warning", detail=text)
    result = TranscriptResult(
        cache_key=key,
        model=request.model_name,
        source=source.path.name,
        duration_s=source.duration_s,
        config=config,
        lineup=None if lineup is None else (request.lineup_path or Path(LINEUP_FILE)).name,
        lineup_version=lineup_version,
        chunks=[
            ChunkInfo(
                index=c.index,
                start_s=c.start_s,
                end_s=c.end_s,
                n_words=len(c.words),
                cost_usd=c.cost_usd,
            )
            for c in done
        ],
        segments=segments,
        gaps=gaps,
        halves=halves,
        stats=TranscriptStats(
            n_segments=len(segments),
            n_words=len(words),
            n_mentions=sum(len(s.players) for s in segments),
            n_chunks=len(chunks),
            cost_usd=_total_cost(done),
            speakers_unresolved=unresolved,
        ),
        warnings=warnings,
    )
    write_atomic(match_dir / RESULT_FILE, result.model_dump_json(indent=2))
    return result


def run_transcribe(
    match_dir: Path,
    match_id: str,
    *,
    lineup_path: Path | None = None,
    config: TranscribeConfig | None = None,
    force: bool = False,
    model: str | None = None,
    settings: Settings | None = None,
    transcriber: Transcriber | None = None,
    ffmpeg: Runner | None = None,
    probe: Probe = probe_video,
) -> TranscriptResult | None:
    """Transcribe the whole match of `match_dir`.

    A transcript made from the same sound, model, parameters, lineup file and halves is returned
    without a request unless `force` is set; `force` rebuilds the transcript from the answers kept
    in `speech_chunks/` and does not ask the model again (delete that directory to pay again).
    A run that stopped (service error, budget) goes on from the first chunk without an answer.
    Without a key of the provider the step is recorded as `skipped` and None is returned.

    Args:
        match_dir: Directory with `audio.wav` or `src.mp4`, and optionally `scoreboard.json` and
            `lineups.csv`.
        match_id: Match id for the logs.
        lineup_path: Lineup file; default `lineups.csv` in `match_dir` when it exists.
        config: Parameters; default `TranscribeConfig()`.
        force: Build the transcript again even if an up-to-date one exists.
        model: The model; default scribe-v2 through OpenRouter (the Gemini model for the
            `gemini` provider).
        settings: Provider, keys and budget; loaded from the environment when absent.
        transcriber: The speech model (a fake in tests); default the provider of the settings,
            wrapped in the spending ledger and the budget.
        ffmpeg: The FFmpeg runner (a fake in tests).
        probe: Reads the length of a video (a fake in tests).

    Raises:
        TranscriptionError: No sound, FFmpeg failed, or an answer cannot be read.
        ExternalServiceError: The service failed for good (the chunks done stay kept).
        BudgetExceededError: The monthly budget does not allow the next chunk (same).
        PlatesError: The lineup file is invalid.
    """
    log = stage_logger(STAGE, match_id)
    progress = Progress(match_dir, STAGE, log)
    started = time.perf_counter()
    try:
        if transcriber is not None:
            name = transcriber.model
        else:
            settings = settings or load_settings()
            name = default_model_name(settings, model)
        chosen = settings

        def make_model() -> Transcriber:
            if transcriber is not None:
                return transcriber
            active = chosen or load_settings()
            _require_key(active)
            return make_transcriber(active, match_dir, name)

        request = _Request(
            config or TranscribeConfig(),
            lineup_path,
            force,
            name,
            make_model,
            ffmpeg or subprocess_runner,
            probe,
        )
        result, from_cache = _run(match_dir, request, progress, log)
    except MissingKeyError as err:
        reason = ru.TRANSCRIBE_SKIPPED.format(reason=err)
        log.warning("stage_skipped_no_key", reason=str(err))
        set_status(match_dir, STAGE, "skipped", 1.0, log, error=reason)
        return None
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        record_failure(match_dir, STAGE, progress.value, err, log, started)
        raise
    set_status(match_dir, STAGE, "done", 1.0, log)
    log.info(
        "stage_skipped" if from_cache else "stage_done",
        n_segments=result.stats.n_segments,
        n_chunks=result.stats.n_chunks,
        duration_ms=elapsed_ms(started),
    )
    return result
