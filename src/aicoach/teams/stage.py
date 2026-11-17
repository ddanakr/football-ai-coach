"""The `teams` stage: team names and kit colours of a match (D-078).

Names: `lineups.csv` (`team` column), else the video title (`src.info.json`), else "хозяева" and
"гости". Kit colours: a video model looks at clips of events whose side is known (goals: the
scoring team; card fouls: the offender) and the colours are voted, see `kits.py`. The result is
`teams.json` in the match directory plus the state in `status.json`. The event database is only
read. Without the key of the video model (or the video) the names are still written and the
colours stay undetermined, with a note; the viewer can set them by hand (`teams_override.json`).
"""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aicoach.agent.context import load_context
from aicoach.agent.service import make_video_model
from aicoach.clips import export_clip
from aicoach.config import Settings, load_settings
from aicoach.errors import ConfigError, ExportError
from aicoach.i18n import ru
from aicoach.ingest.download import VIDEO_FILE
from aicoach.logging_setup import stage_logger
from aicoach.plate_names.lineup import LINEUP_FILE, parse_team_names
from aicoach.stage_common import (
    Logger,
    Progress,
    elapsed_ms,
    record_failure,
    set_status,
    write_atomic,
)
from aicoach.storage import CurrentEvent
from aicoach.teams import kits
from aicoach.teams.kits import KitSource, decide_kits, select_sources
from aicoach.teams.store import (
    KIT_COLORS,
    KIT_UNCLEAR,
    RESULT_FILE,
    SCHEMA_VERSION,
    SIDES,
    Kit,
    KitVote,
    NameSource,
    Side,
    TeamEntry,
    TeamsResult,
    VoteStatus,
    read_teams,
)
from aicoach.teams.title import parse_title_teams, read_title
from aicoach.vlm import VideoModel, ask_json

STAGE = "teams"
DB_FILE = "aicoach.db"
PARTIAL_KEY = "incomplete"

Clipper = Callable[[Path, CurrentEvent, float, float], Path]
VoteKey = tuple[str, int, str, str, str, int, float, float]
Names = dict[Side, tuple[str, NameSource]]


class TeamsConfig(BaseModel):
    """Parameters of the stage; every number is a starting value, not tuned on data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    before_s: float = Field(default=4.0, ge=0, le=30)
    after_s: float = Field(default=12.0, ge=0, le=30)
    # A side has few goals and cards in a match: five clips cost a few cents and are enough.
    max_votes_per_side: int = Field(default=5, ge=1, le=20)
    # One clip can be wrong, so a colour needs two agreeing clips.
    min_votes: int = Field(default=2, ge=1, le=20)


def resolve_names(match_dir: Path) -> Names:
    """Name and source of each side: the lineup, else the video title, else the default words.

    The sources are chosen per side, so a side missing from the lineup falls back to the title
    or the default for that side only.
    """
    lineup: dict[Side, str] = {}
    path = match_dir / LINEUP_FILE
    if path.is_file():
        try:
            lineup = dict(parse_team_names(path.read_bytes()))
        except OSError:
            lineup = {}
    title = read_title(match_dir)
    from_title = parse_title_teams(title) if title else {}
    names: Names = {}
    for side in SIDES:
        if lineup.get(side):
            names[side] = (lineup[side], "lineup")
        elif from_title.get(side):
            names[side] = (from_title[side], "title")
        else:
            names[side] = (ru.TEAM_LABELS[side], "default")
    return names


def vote_key(v: KitVote) -> VoteKey:
    """What identifies a vote: the same clip asked the same way (event, time, role, side, model,
    question version, window)."""
    return (
        v.event_id,
        round(v.t_v * 1000),
        v.role,
        v.side,
        v.model,
        v.prompt_version,
        v.before_s,
        v.after_s,
    )


def _source_key(source: KitSource, model: str, config: TeamsConfig) -> VoteKey:
    return (
        source.event.event_id,
        round(source.event.t_v * 1000),
        source.role,
        source.side,
        model,
        kits.PROMPT_VERSION,
        config.before_s,
        config.after_s,
    )


def cache_key(
    names: Names, sources: Sequence[KitSource], model: str | None, config: TeamsConfig
) -> str:
    """A hash of everything the result depends on: the names, the clips asked about, the model,
    the parameters and the versions of the question and the file."""
    payload = {
        "names": {s: list(v) for s, v in names.items()},
        "sources": [(s.event.event_id, round(s.event.t_v * 1000), s.role, s.side) for s in sources],
        "model": model,
        "config": config.model_dump(),
        "prompt": kits.PROMPT_VERSION,
        "schema": SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sources(db_path: Path, match_id: str, match_dir: Path, config: TeamsConfig) -> list[KitSource]:
    """The events that tell a kit colour; none when there is no database yet."""
    if not db_path.is_file():
        return []
    context = load_context(db_path, match_id, match_dir)
    pairs = [(facts.event, facts.card) for facts in context.events]
    return select_sources(pairs, config.max_votes_per_side)


def _ask_vote(
    source: KitSource, video: Path, model: VideoModel, config: TeamsConfig, clipper: Clipper
) -> KitVote:
    """Cut the clip of the event and ask the model once (twice when the answer is unusable)."""
    clip = clipper(video, source.event, config.before_s, config.after_s)
    try:
        data = clip.read_bytes()
    except OSError as err:
        msg = f"cannot read the clip {clip.name}: {err}"
        raise ExportError(msg) from err
    offset = min(config.before_s, source.event.t_v)
    answer = ask_json(
        model,
        kits.SYSTEM_PROMPT,
        kits.build_prompt(source.role, offset),
        data,
        kits.parse_kit_answer,
        video_seconds=config.before_s + config.after_s,
    )
    status: VoteStatus = (
        "invalid" if answer.value is None else "ok" if answer.attempts == 1 else "retried"
    )
    return KitVote(
        event_id=source.event.event_id,
        t_v=source.event.t_v,
        role=source.role,
        side=source.side,
        kit=answer.value or KIT_UNCLEAR,
        status=status,
        model=model.model,
        prompt_version=kits.PROMPT_VERSION,
        before_s=config.before_s,
        after_s=config.after_s,
        tokens_in=answer.usage.input_tokens,
        tokens_out=answer.usage.output_tokens,
        cost_usd=answer.usage.cost_usd,
    )


def _collect_votes(
    sources: Sequence[KitSource],
    old: dict[VoteKey, KitVote],
    video: Path,
    model: VideoModel,
    config: TeamsConfig,
    clipper: Clipper,
    progress: Progress,
    log: Logger,
    save: Callable[[list[KitVote]], None],
) -> tuple[list[KitVote], list[str]]:
    """One vote per source: from an earlier `teams.json` when it was asked the same way, else a
    new request (`save` keeps the votes so far, so a stopped run does not pay for them again).
    A clip that cannot be cut is skipped with a note."""
    votes: list[KitVote] = []
    notes: list[str] = []
    for i, source in enumerate(sources):
        kept = old.get(_source_key(source, model.model, config))
        if kept is not None:
            votes.append(kept)
        else:
            try:
                votes.append(_ask_vote(source, video, model, config, clipper))
                # The votes of the sources not reached yet that an earlier run paid for stay in
                # the saved file, so a failure at the next source loses nothing.
                later = [
                    kept_later
                    for s in sources[i + 1 :]
                    if (kept_later := old.get(_source_key(s, model.model, config)))
                ]
                save([*votes, *later])
            except ExportError as err:
                log.warning(
                    "kit_clip_failed", event_id=source.event.event_id, reason=str(err)[:200]
                )
                notes.append(f"no clip for {source.event.event_id}: vote skipped")
        progress.update((i + 1) / max(1, len(sources)))
    return votes, notes


def _entries(
    names: Names, votes: Sequence[KitVote], config: TeamsConfig
) -> tuple[dict[Side, TeamEntry], list[str]]:
    decisions, notes = decide_kits(votes, config.min_votes)
    entries: dict[Side, TeamEntry] = {}
    for side in SIDES:
        decision = decisions[side]
        name, source = names[side]
        kit: Kit | None = None
        if decision.kit in KIT_COLORS:
            kit = decision.kit  # type: ignore[assignment]  # a member of KIT_COLORS, checked
        entries[side] = TeamEntry(
            name=name,
            name_source=source,
            kit=kit,
            kit_source="vlm" if kit is not None else "undetermined",
            kit_votes=decision.counts,
        )
        if kit is None and votes:
            notes.append(f"{side}: kit not determined ({decision.reason})")
    return entries, notes


def _pick_model(
    model: VideoModel | None, settings: Settings | None, match_dir: Path, log: Logger
) -> tuple[VideoModel | None, str | None]:
    """The video model to use and, when there is none, why (no key)."""
    if model is not None:
        return model, None
    try:
        return make_video_model(settings or load_settings(), match_dir), None
    except ConfigError as err:
        log.warning("kits_skipped_no_key", reason=str(err)[:200])
        return None, ru.TEAMS_KITS_SKIPPED.format(reason=str(err)[:200])


def _run(
    match_dir: Path,
    match_id: str,
    db_path: Path,
    video: Path,
    config: TeamsConfig,
    force: bool,
    tools: tuple[VideoModel | None, Settings | None, Clipper],
    progress: Progress,
    log: Logger,
) -> TeamsResult:
    names = resolve_names(match_dir)
    sources = _sources(db_path, match_id, match_dir, config)
    notes: list[str] = []
    chosen: VideoModel | None = None
    skipped: str | None = None
    if not db_path.is_file():
        skipped = ru.TEAMS_NO_DB
    elif not sources:
        skipped = ru.TEAMS_NO_SOURCES
    elif not video.is_file():
        skipped = ru.TEAMS_NO_VIDEO
    else:
        chosen, skipped = _pick_model(tools[0], tools[1], match_dir, log)
    if skipped:
        notes.append(skipped)
    previous = read_teams(match_dir)
    key = cache_key(names, sources, None if chosen is None else chosen.model, config)
    if previous is not None and previous.cache_key == key and not force:
        return previous
    set_status(match_dir, STAGE, "running", 0.0, log)
    old = {} if previous is None else {vote_key(v): v for v in previous.votes}
    votes: list[KitVote] = []
    model_name = None if chosen is None else chosen.model
    complete = True
    if chosen is not None:

        def save(so_far: list[KitVote]) -> None:
            # A half-done run is written under a key no run matches, so it is never taken for
            # a finished one; its votes are reused by the next run.
            partial, _ = _entries(names, so_far, config)
            write_atomic(
                match_dir / RESULT_FILE,
                TeamsResult(
                    match_id=match_id,
                    cache_key=PARTIAL_KEY,
                    model=chosen.model,
                    home=partial["home"],
                    away=partial["away"],
                    votes=so_far,
                ).model_dump_json(indent=2),
            )

        votes, vote_notes = _collect_votes(
            sources, old, video, chosen, config, tools[2], progress, log, save
        )
        notes += vote_notes
        complete = not vote_notes  # a clip that could not be cut is tried again next time
    elif previous is not None:
        # No key now, but an earlier run paid for votes: keep them and their decision. The
        # colour belongs to the team, so votes of events that are gone stay (there may be no
        # database at all); the next run with a key recounts them against the sources.
        votes, model_name = list(previous.votes), previous.model
    entries, decision_notes = _entries(names, votes, config)
    result = TeamsResult(
        match_id=match_id,
        cache_key=key if complete else PARTIAL_KEY,
        model=model_name,
        home=entries["home"],
        away=entries["away"],
        votes=votes,
        notes=[*notes, *decision_notes],
        kits_skipped=skipped,
    )
    write_atomic(match_dir / RESULT_FILE, result.model_dump_json(indent=2))
    return result


def run_teams(
    match_dir: Path,
    match_id: str,
    *,
    db_path: Path | None = None,
    video: Path | None = None,
    config: TeamsConfig | None = None,
    force: bool = False,
    model: VideoModel | None = None,
    settings: Settings | None = None,
    clipper: Clipper | None = None,
) -> TeamsResult:
    """Names and kit colours of the match in `match_dir`.

    A result made from the same names, events, model and parameters is returned without a
    request unless `force` is set; votes already in `teams.json` are reused even when the
    rest changed. Clips are cut next to the video (`clips/`, cached by name).

    Args:
        match_dir: The match directory (`lineups.csv`, `src.info.json`, `src.mp4`).
        match_id: Match id in the database and the logs.
        db_path: The events database; default `aicoach.db` in `match_dir`.
        video: The match video; default `src.mp4` in `match_dir`.
        config: Parameters; default `TeamsConfig()`.
        force: Ask again even if an up-to-date result exists (the votes are reused).
        model: The video model (a fake in tests); default the one of the settings, wrapped in
            the spending ledger and the budget. Without a key the colours stay undetermined.
        settings: Keys, budget and model name; loaded from the environment when absent.
        clipper: Cuts the clip of an event (a fake in tests); default `clips.export_clip`.

    Raises:
        ExternalServiceError: The video model failed for good.
        BudgetExceededError: The monthly budget does not allow a request.
        StorageError: The result or the database cannot be written or read.
    """
    config = config or TeamsConfig()
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
    if result.kits_skipped:  # the names are written, the colours were not asked for
        set_status(match_dir, STAGE, "skipped", 1.0, log, error=result.kits_skipped)
    else:
        set_status(match_dir, STAGE, "done", 1.0, log)
    log.info(
        "stage_done",
        n_votes=len(result.votes),
        home_kit=result.home.kit,
        away_kit=result.away.kit,
        duration_ms=elapsed_ms(started),
    )
    return result
