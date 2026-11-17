"""The `corner_tags` stage: structured tags of every corner from its clip (D-078).

For each corner of `events_current` (user edits applied) the clip `[t_v - 4 s, t_v + 12 s]` goes
to the video model with the strict question of `tagger.py`. Answers are kept in
`corner_tags.json` and written after every new answer, so a stopped run (service failure, budget)
goes on from the first corner without one; an answer is reused when the event id, its time, the
model, the question version and the window are the same. The database, `predictions` and the
fusion are only read. The spending goes through the ledger (D-067) with the model of the video
questions.
"""

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aicoach.agent.context import EventFacts, load_context
from aicoach.agent.service import make_video_model
from aicoach.clips import export_clip
from aicoach.config import Settings, load_settings
from aicoach.corner_tags import tagger
from aicoach.corner_tags.schemas import (
    RESULT_FILE,
    CornerFields,
    CornerTag,
    CornerTagsFile,
    read_tags_file,
)
from aicoach.domain.events import EventType
from aicoach.errors import ConfigError, ExportError
from aicoach.i18n import ru
from aicoach.ingest.download import VIDEO_FILE
from aicoach.logging_setup import stage_logger
from aicoach.stage_common import (
    Logger,
    Progress,
    elapsed_ms,
    record_failure,
    set_status,
    write_atomic,
)
from aicoach.storage import CurrentEvent
from aicoach.vlm import VideoModel, ask_json

STAGE = "corner_tags"
DB_FILE = "aicoach.db"

Clipper = Callable[[Path, CurrentEvent, float, float], Path]
TagKey = tuple[str, int, str, int, float, float]


class CornerTagsConfig(BaseModel):
    """The clip window of a corner: 4 s before the kick and 12 s after (the flight of the ball
    and what follows it). Starting values, not tuned on data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    before_s: float = Field(default=4.0, ge=0, le=30)
    after_s: float = Field(default=12.0, ge=0, le=30)


def tag_key(tag: CornerTag) -> TagKey:
    """What identifies an answer: the same clip asked the same way."""
    return (
        tag.event_id,
        round(tag.t_v * 1000),
        tag.model,
        tag.prompt_version,
        tag.before_s,
        tag.after_s,
    )


def _event_key(event: CurrentEvent, model: str, config: CornerTagsConfig) -> TagKey:
    return (
        event.event_id,
        round(event.t_v * 1000),
        model,
        tagger.PROMPT_VERSION,
        config.before_s,
        config.after_s,
    )


def _corners(facts: Sequence[EventFacts]) -> list[CurrentEvent]:
    return sorted(
        (f.event for f in facts if f.event.type == EventType.CORNER and not f.event.deleted),
        key=lambda e: e.t_v,
    )


def _ask_tag(
    event: CurrentEvent,
    half: int | None,
    video: Path,
    model: VideoModel,
    config: CornerTagsConfig,
    clipper: Clipper,
) -> CornerTag:
    """Cut the clip and ask the model once (twice when the answer is unusable)."""
    clip = clipper(video, event, config.before_s, config.after_s)
    try:
        data = clip.read_bytes()
    except OSError as err:
        msg = f"cannot read the clip {clip.name}: {err}"
        raise ExportError(msg) from err
    answer = ask_json(
        model,
        tagger.SYSTEM_PROMPT,
        tagger.build_prompt(min(config.before_s, event.t_v)),
        data,
        tagger.parse_corner_answer,
        video_seconds=config.before_s + config.after_s,
    )
    status = "invalid" if answer.value is None else "ok" if answer.attempts == 1 else "retried"
    return CornerTag(
        event_id=event.event_id,
        t_v=event.t_v,
        half=half,
        fields=answer.value or CornerFields.all_unclear(),
        status=status,
        model=model.model,
        prompt_version=tagger.PROMPT_VERSION,
        before_s=config.before_s,
        after_s=config.after_s,
        tokens_in=answer.usage.input_tokens,
        tokens_out=answer.usage.output_tokens,
        cost_usd=answer.usage.cost_usd,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _write(
    match_dir: Path, match_id: str, tags: list[CornerTag], notes: list[str]
) -> CornerTagsFile:
    result = CornerTagsFile(match_id=match_id, tags=tags, notes=notes)
    write_atomic(match_dir / RESULT_FILE, result.model_dump_json(indent=2))
    return result


def _run(
    match_dir: Path,
    match_id: str,
    db_path: Path,
    video: Path,
    config: CornerTagsConfig,
    force: bool,
    tools: tuple[VideoModel | None, Settings | None, Clipper],
    progress: Progress,
    log: Logger,
) -> CornerTagsFile | str:
    """The tags, or the reason (a text) why the step cannot run."""
    if not db_path.is_file():
        log.warning("corner_tags_no_db", path=str(db_path))
        return ru.CORNER_TAGS_NO_DB
    facts = load_context(db_path, match_id, match_dir).events
    corners = _corners(facts)
    halves = {f.event.event_id: f.event.half for f in facts}
    model, settings, clipper = tools
    if model is None:
        try:
            model = make_video_model(settings or load_settings(), match_dir)
        except ConfigError as err:
            log.warning("corner_tags_skipped_no_key", reason=str(err)[:200])
            return ru.CORNER_TAGS_SKIPPED
    if corners and not video.is_file():
        log.warning("corner_tags_no_video", path=str(video))
        return ru.CORNER_TAGS_NO_VIDEO
    set_status(match_dir, STAGE, "running", 0.0, log)
    previous = read_tags_file(match_dir)
    old = {} if previous is None or force else {tag_key(t): t for t in previous.tags}
    tags: list[CornerTag] = []
    notes: list[str] = []
    for i, event in enumerate(corners):
        half = halves.get(event.event_id)
        kept = old.get(_event_key(event, model.model, config))
        if kept is not None:
            tags.append(kept.model_copy(update={"half": half}))
        else:
            try:
                tags.append(_ask_tag(event, half, video, model, config, clipper))
            except ExportError as err:
                log.warning("corner_clip_failed", event_id=event.event_id, reason=str(err)[:200])
                notes.append(f"no clip for {event.event_id}: not tagged")
            else:
                # Written with the answers of the later corners that are kept from an earlier run,
                # so a failure at the next corner loses nothing already paid for.
                later = [
                    old[k]
                    for e in corners[i + 1 :]
                    if (k := _event_key(e, model.model, config)) in old
                ]
                _write(match_dir, match_id, [*tags, *later], notes)
        progress.update((i + 1) / len(corners))
    return _write(match_dir, match_id, tags, notes)


def run_corner_tags(
    match_dir: Path,
    match_id: str,
    *,
    db_path: Path | None = None,
    video: Path | None = None,
    config: CornerTagsConfig | None = None,
    force: bool = False,
    model: VideoModel | None = None,
    settings: Settings | None = None,
    clipper: Clipper | None = None,
) -> CornerTagsFile | None:
    """Tag every corner of the match in `match_dir`; `None` when the step cannot run (no
    database, or no key of the video model: recorded as `skipped`).

    A corner already tagged the same way (same event, time, model, question and window) is not
    asked again unless `force` is set. The clips are cut next to the video (`clips/`).

    Args:
        match_dir: The match directory.
        match_id: Match id in the database and the logs.
        db_path: The events database; default `aicoach.db` in `match_dir`.
        video: The match video; default `src.mp4` in `match_dir`.
        config: The clip window; default `CornerTagsConfig()`.
        force: Ask again for every corner (the old answers are not reused).
        model: The video model (a fake in tests); default the one of the settings, wrapped in
            the spending ledger and the budget.
        settings: Keys, budget and model name; loaded from the environment when absent.
        clipper: Cuts the clip of an event (a fake in tests); default `clips.export_clip`.

    Raises:
        ExternalServiceError: The video model failed for good (the answers so far are kept).
        BudgetExceededError: The monthly budget does not allow a request (same).
        StorageError: The result or the database cannot be written or read.
    """
    config = config or CornerTagsConfig()
    log = stage_logger(STAGE, match_id)
    progress = Progress(match_dir, STAGE, log)
    started = time.perf_counter()
    try:
        result = _run(
            match_dir,
            match_id,
            db_path or match_dir / DB_FILE,
            video or match_dir / VIDEO_FILE,
            config,
            force,
            (model, settings, clipper or export_clip),
            progress,
            log,
        )
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        record_failure(match_dir, STAGE, progress.value, err, log, started)
        raise
    if isinstance(result, str):
        set_status(match_dir, STAGE, "skipped", 1.0, log, error=result)
        return None
    set_status(match_dir, STAGE, "done", 1.0, log)
    log.info(
        "stage_done",
        n_tags=len(result.tags),
        n_invalid=sum(t.status == "invalid" for t in result.tags),
        duration_ms=elapsed_ms(started),
    )
    return result
