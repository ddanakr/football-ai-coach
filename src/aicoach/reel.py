"""Reel export (FR-10): all events of one category cut into one video.

The events come from `events_current` (user edits applied) through `load_context`, so the game
minute, the half and the card of a foul are the ones the viewer and the agent show. Every piece
is a clip made by `aicoach.clips` (the same encoding), with a short caption such as "Гол · 5'" over
its first seconds. The pieces are joined with the FFmpeg `concat` demuxer without encoding again
when their streams have the same parameters, and re-encoded with the same flags when they do not.
The reel lies in `<match directory>/clips/reels/`; its name carries a hash of the chosen events and
the parameters, so an unchanged request returns the finished file at once (D-076).
"""

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from aicoach.agent.context import EventFacts, load_context
from aicoach.clips import (
    CLIPS_DIR,
    DEFAULT_AFTER_S,
    DEFAULT_BEFORE_S,
    ENCODE_ARGS,
    FFMPEG_TIMEOUT_S,
    ClipCaption,
    Probe,
    Runner,
    export_clip,
    run_ffmpeg,
    subprocess_runner,
)
from aicoach.domain.events import EventType
from aicoach.errors import ExportError
from aicoach.i18n import ru
from aicoach.scoreboard.video import probe_video
from aicoach.stage_common import elapsed_ms

REELS_DIR = "reels"
DEFAULT_MAX_EVENTS = 30
CONCAT_TIMEOUT_S = 600.0
FFPROBE_TIMEOUT_S = 30.0
CARD = "card"
# Every kind a reel can be made of: the five event categories and the "card" view of the fouls.
KINDS: tuple[str, ...] = (*(t.value for t in EventType), CARD)
Progress = Callable[[int, int], None]
StreamSignature = tuple[tuple[tuple[str, str], ...], ...]
# Where a font with the Cyrillic letters usually lies. A system font is looked up, not shipped
# with the repository; `AICOACH_FONT_FILE` points to any other one.
FONT_CANDIDATES: tuple[Path, ...] = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
)

logger = structlog.get_logger()


@dataclass(frozen=True)
class ReelFilter:
    """Which events go into the reel: a kind (an event category or `card`), and optionally one
    half and one team side. `card` is the fouls that a card plate made."""

    kind: str
    half: int | None = None
    side: str | None = None


@dataclass(frozen=True)
class ReelOptions:
    """Margins around each event, the most events a reel may hold, and the caption font file
    (None: look in the usual places)."""

    before_s: float = DEFAULT_BEFORE_S
    after_s: float = DEFAULT_AFTER_S
    max_events: int = DEFAULT_MAX_EVENTS
    font_file: Path | None = None


@dataclass(frozen=True)
class _Tools:
    """What `build_reel` hands to the cutting of the pieces: progress, FFmpeg runner, probe."""

    progress: Progress | None
    runner: Runner
    probe: Probe
    timeout_s: float


@dataclass(frozen=True)
class Selection:
    """The chosen events in time order, and how many events of the kind were left out because
    their team is unknown while a side was asked for (the reel must not guess a team)."""

    events: tuple[EventFacts, ...]
    team_unknown: int = 0


@dataclass(frozen=True)
class ReelResult:
    """A finished reel: the file, the number of pieces, whether it came from the cache, whether
    the pieces carry captions, whether the join re-encoded them, the length in seconds (None when
    FFprobe could not tell) and the warnings."""

    path: Path
    n_events: int
    cached: bool
    captioned: bool
    reencoded: bool = False
    duration_s: float | None = None
    warnings: tuple[str, ...] = ()
    team_unknown: int = 0


def _matches_kind(facts: EventFacts, kind: str) -> bool:
    if kind == CARD:
        return facts.event.type == EventType.FOUL and facts.card is not None
    return facts.event.type.value == kind


def select_events(facts: Sequence[EventFacts], flt: ReelFilter) -> Selection:
    """The events that fit the filter, ordered by video time.

    Args:
        facts: Every current event of the match.
        flt: The kind, and the optional half and side. An event with an unknown team is not
            taken when a side is asked for; it is counted in `Selection.team_unknown`.

    Raises:
        ValueError: The kind is not one of `KINDS`, the half is not 1 or 2, or the side is not
            `home` or `away`.
    """
    if flt.kind not in KINDS:
        msg = f"unknown reel kind {flt.kind!r}; expected one of {', '.join(KINDS)}"
        raise ValueError(msg)
    if flt.half not in (None, 1, 2) or flt.side not in (None, "home", "away"):
        msg = f"bad reel filter: half {flt.half!r}, side {flt.side!r}"
        raise ValueError(msg)
    chosen: list[EventFacts] = []
    team_unknown = 0
    for item in facts:
        event = item.event
        if not _matches_kind(item, flt.kind) or (flt.half is not None and event.half != flt.half):
            continue
        if flt.side is not None and event.team is None:
            team_unknown += 1
            continue
        if flt.side is None or event.team == flt.side:
            chosen.append(item)
    chosen.sort(key=lambda f: (f.event.t_v, f.event.event_id))
    return Selection(tuple(chosen), team_unknown)


def caption_text(facts: EventFacts, kind: str) -> str:
    """The caption of a piece: the category and the game minute, `Гол · 5'`.

    In a `card` reel the category is "Карточка". Without a known minute the category alone.
    """
    label = ru.REEL_CARD_LABEL if kind == CARD else ru.TYPE_LABELS[facts.event.type]
    if facts.minute_text is None:
        return label
    return ru.REEL_CAPTION.format(type=label, minute=facts.minute_text)


def find_font(configured: Path | None, candidates: Sequence[Path] = FONT_CANDIDATES) -> Path | None:
    """The font file for the captions, or None when there is none.

    A configured file that does not exist is not replaced by a system font: the user named it,
    so the caller reports that it is missing.
    """
    if configured is not None:
        return configured if configured.is_file() else None
    return next((path for path in candidates if path.is_file()), None)


def reel_filename(
    flt: ReelFilter, selection: Selection, options: ReelOptions, *, captioned: bool
) -> str:
    """`reel_<kind>[_h<half>][_<side>]_<hash>.mp4`.

    The hash covers everything that changes the picture: the id, the time and the caption of every
    piece, the margins and whether there are captions. A moved event or another filter gives
    another name, an unchanged request the same one.
    """
    pieces = [
        [f.event.event_id, round(f.event.t_v * 1000), caption_text(f, flt.kind)]
        for f in selection.events
    ]
    payload = json.dumps(
        [flt.kind, options.before_s, options.after_s, captioned, pieces], ensure_ascii=False
    )
    digest = hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:10]
    parts = ["reel", flt.kind]
    if flt.half is not None:
        parts.append(f"h{flt.half}")
    if flt.side is not None:
        parts.append(flt.side)
    return "_".join([*parts, digest]) + ".mp4"


def concat_list_text(pieces: Sequence[Path]) -> str:
    """The list file of the concat demuxer: `file '<absolute path>'` per line, quotes escaped."""
    lines = []
    for piece in pieces:
        escaped = piece.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


def build_concat_command(
    list_file: Path, output: Path, *, reencode: bool, ffmpeg: str = "ffmpeg"
) -> list[str]:
    """FFmpeg arguments that join the listed files: a stream copy, or (`reencode`) the clip
    encoding of `aicoach.clips`. The container is set by `-f mp4` (the output has a temporary
    name until the join has finished)."""
    command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    command += ["-f", "concat", "-safe", "0", "-i", str(list_file)]
    if reencode:
        command += ["-map", "0:v:0", "-map", "0:a:0?", *ENCODE_ARGS]
    else:
        command += ["-c", "copy", "-movflags", "+faststart"]
    return [*command, "-f", "mp4", str(output)]


_SIGNATURE_FIELDS = (
    "codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,time_base,sample_rate,channels"
)


def stream_signature(path: Path, runner: Runner = subprocess_runner) -> StreamSignature | None:
    """The stream parameters that must match for a join without encoding, or None when FFprobe is
    missing or cannot read the file (the pieces are then re-encoded, which always works)."""
    command = ["ffprobe", "-v", "error", "-show_entries", f"stream={_SIGNATURE_FIELDS}"]
    command += ["-of", "json", str(path)]
    try:
        done = runner(command, FFPROBE_TIMEOUT_S)
        streams = json.loads(done.stdout)["streams"] if done.returncode == 0 else None
    except (OSError, ValueError, KeyError, TypeError) as err:
        logger.warning("stream_signature_unavailable", stage="reel", reason=type(err).__name__)
        return None
    if not isinstance(streams, list):
        logger.warning("stream_signature_unavailable", stage="reel", reason="ffprobe_failed")
        return None
    return tuple(sorted(tuple(sorted((str(k), str(v)) for k, v in s.items())) for s in streams))


def media_duration(path: Path, runner: Runner = subprocess_runner) -> float | None:
    """The length of a media file in seconds from FFprobe, or None when it cannot be told."""
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json"]
    try:
        done = runner([*command, str(path)], FFPROBE_TIMEOUT_S)
        value = float(json.loads(done.stdout)["format"]["duration"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        logger.warning("media_duration_unavailable", stage="reel", reason=type(err).__name__)
        return None
    if not math.isfinite(value) or value < 0:
        logger.warning("media_duration_unavailable", stage="reel", reason="bad_value")
        return None
    return value


def _check_selection(selection: Selection, flt: ReelFilter, max_events: int) -> None:
    if not selection.events:
        msg = f"no events of kind {flt.kind!r} match the filters"
        raise ExportError(msg)
    if len(selection.events) > max_events:
        msg = (
            f"{len(selection.events)} events of kind {flt.kind!r} match, more than --max "
            f"{max_events}: narrow the reel with --half or --side, or raise --max"
        )
        raise ExportError(msg)


def _font_warning(configured: Path | None) -> str:
    if configured is not None:
        return f"font file {configured} does not exist: the reel has no captions"
    return (
        "no font with the Cyrillic letters found (set AICOACH_FONT_FILE): the reel has no captions"
    )


def _join(pieces: Sequence[Path], output: Path, runner: Runner, timeout_s: float) -> bool:
    """Join the pieces into `output`; return True if they had to be re-encoded.

    The list file and the half-written output are removed on every outcome except success.
    """
    signatures = [stream_signature(p, runner) for p in pieces]
    reencode = None in signatures or len(set(signatures)) > 1
    partial = output.with_name(output.name + ".part")
    list_file = output.with_name(output.name + ".list.txt")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        list_file.write_text(concat_list_text(pieces), encoding="utf-8")
        run_ffmpeg(build_concat_command(list_file, partial, reencode=reencode), timeout_s, runner)
        partial.replace(output)
    except OSError as err:
        msg = f"cannot write the reel {output}: {err}"
        raise ExportError(msg) from err
    finally:
        list_file.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return reencode


def _cut_pieces(
    video: Path,
    selection: Selection,
    flt: ReelFilter,
    options: ReelOptions,
    font: Path | None,
    tools: _Tools,
) -> list[Path]:
    total = len(selection.events)
    pieces: list[Path] = []
    for index, item in enumerate(selection.events):
        if tools.progress is not None:
            tools.progress(index, total)
        caption = None if font is None else ClipCaption(caption_text(item, flt.kind), font)
        pieces.append(
            export_clip(
                video,
                item.event,
                options.before_s,
                options.after_s,
                timeout_s=tools.timeout_s,
                runner=tools.runner,
                probe=tools.probe,
                caption=caption,
            )
        )
    if tools.progress is not None:
        tools.progress(total, total)
    return pieces


def build_reel(
    video: Path,
    match_dir: Path,
    facts: Sequence[EventFacts],
    flt: ReelFilter,
    options: ReelOptions = ReelOptions(),  # noqa: B008 - a frozen value, safe as a default
    *,
    progress: Progress | None = None,
    runner: Runner = subprocess_runner,
    probe: Probe = probe_video,
    timeout_s: float = FFMPEG_TIMEOUT_S,
) -> ReelResult:
    """Cut the chosen events and join them into `<match_dir>/clips/reels/`.

    Each piece is `export_clip` with a caption (so it is cached like any clip); an existing reel
    of the same events and parameters is returned without running FFmpeg. Without a usable font
    the pieces have no captions and the result carries a warning.

    Args:
        video: The match video.
        match_dir: The match directory; the reel goes to its `clips/reels/`.
        facts: Every current event of the match (`MatchContext.events`).
        flt: Which events.
        options: Margins, the limit and the font file.
        progress: Called with `(done, total)` before the first piece and after every piece.
        runner: Runs FFmpeg and FFprobe (a fake in tests).
        probe: Reads the video length for the clip windows.
        timeout_s: Time limit of one FFmpeg run that cuts a piece.

    Raises:
        ExportError: No event fits, more than `options.max_events` fit, a piece or the join
            failed.
        ValueError: The kind of the filter is unknown.
    """
    selection = select_events(facts, flt)
    _check_selection(selection, flt, options.max_events)
    font = find_font(options.font_file)
    warnings = () if font is not None else (_font_warning(options.font_file),)
    name = reel_filename(flt, selection, options, captioned=font is not None)
    output = match_dir / CLIPS_DIR / REELS_DIR / name
    log = logger.bind(stage="reel", kind=flt.kind, n_events=len(selection.events))
    for text in warnings:
        log.warning("reel_without_captions", reason=text)
    cached = output.is_file()
    reencoded = False
    if cached:
        log.info("reel_cached", path=output.name)
    else:
        started = time.perf_counter()
        tools = _Tools(progress, runner, probe, timeout_s)
        pieces = _cut_pieces(video, selection, flt, options, font, tools)
        reencoded = _join(pieces, output, runner, CONCAT_TIMEOUT_S)
        log.info(
            "reel_built", path=output.name, reencoded=reencoded, duration_ms=elapsed_ms(started)
        )
    return ReelResult(
        path=output,
        n_events=len(selection.events),
        cached=cached,
        captioned=font is not None,
        reencoded=reencoded,
        duration_s=media_duration(output, runner),
        warnings=warnings,
        team_unknown=selection.team_unknown,
    )


def export_event_reel(
    db_path: Path,
    match_id: str,
    match_dir: Path,
    video: Path,
    flt: ReelFilter,
    options: ReelOptions = ReelOptions(),  # noqa: B008 - a frozen value, safe as a default
    *,
    progress: Progress | None = None,
    runner: Runner = subprocess_runner,
    probe: Probe = probe_video,
) -> ReelResult:
    """Read the current events of the match and build the reel (see `build_reel`).

    Raises:
        StorageError: The database is missing or SQLite failed.
        ExportError: See `build_reel`.
    """
    context = load_context(db_path, match_id, match_dir)
    return build_reel(
        video,
        match_dir,
        context.events,
        flt,
        options,
        progress=progress,
        runner=runner,
        probe=probe,
    )
