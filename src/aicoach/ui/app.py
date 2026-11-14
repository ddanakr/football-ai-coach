"""Streamlit viewer and editor of the event catalog (D12a view, D12b manual edits, D15 new
matches from a YouTube link).

Thin: the logic is in `logic.py` and `edits.py`. Started by `python -m aicoach.cli ui`; the
arguments after `--` are read by `parse_app_args`. The page only draws; every write to the
database happens in a button callback (`_on_*`), which runs once per click, so a rerun of the
page can never write an edit twice.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import streamlit as st
import structlog

from aicoach import clips
from aicoach.domain.events import EventType
from aicoach.errors import ConfigError, ExportError, IngestError, PlatesError, StorageError
from aicoach.i18n import ru
from aicoach.ingest.download import VIDEO_FILE
from aicoach.ingest.stage import STAGE as INGEST_STAGE
from aicoach.status import read_status
from aicoach.storage import CurrentEvent
from aicoach.ui.agent_panel import render_agent_panel
from aicoach.ui.data import (
    apply_edit,
    count_edits,
    load_events,
    load_player_texts,
    load_time_bounds,
    undo_last,
)
from aicoach.ui.edits import (
    EditInputError,
    EditPlan,
    TimeBounds,
    plan_add,
    plan_delete,
    plan_move,
    plan_retype,
    with_derived_half,
)
from aicoach.ui.ingest_jobs import (
    IngestJob,
    busy_pid,
    job_phase,
    launch,
    plan_job,
    save_lineup,
    step_rows,
)
from aicoach.ui.launch import AppArgs, StartArgs, parse_start_args
from aicoach.ui.logic import (
    EventFilter,
    EventRow,
    HalfFilter,
    available_types,
    filter_events,
    format_mmss,
    player_start,
    selected_row,
    table_records,
    to_row,
    to_rows,
    youtube_link,
)
from aicoach.youtube import watch_url

PLAYER_COLUMNS = [1, 4, 1]  # the player sits in the middle one, it is not stretched over the page
TABLE_HEIGHT_PX = 380
TABLE_KEY = "events_table"
_SELECTED = "selected_event_id"  # id of the chosen event, not its row: rows move after an edit
_ROWS_SEEN = "rows_seen"  # ids shown in the table on the previous run
_PICK_SEEN = "pick_seen"  # table selection reported on the previous run
_FLASH = "flash"  # (level, text) of the last edit, shown once
_CLIP = "clip_result"  # (path, caption) of the last exported clip; kept across reruns
_CLIP_ERROR = "clip_error"  # why the last export failed; cleared by the next export
_CLIP_BEFORE = "clip_before"
_CLIP_AFTER = "clip_after"
_OPENED = "opened_match"  # AppArgs of a match opened from the "new match" block
_JOB = "ingest_job"  # the IngestJob the sidebar follows
_PROC = "ingest_process"  # its process, while this session started it
_INGEST_FLASH = "ingest_flash"  # (level, text) of the last start attempt
_URL_KEY = "ingest_url"
_LINEUP_KEY = "ingest_lineup"  # the uploaded lineups.csv, if any (a number is added to it)
_LINEUP_ROUND = "ingest_lineup_round"  # a new uploader key after each start: it clears the file
STATUS_REFRESH_S = 2  # how often the step list rereads status.json
_HALF_OPTIONS: list[HalfFilter] = [None, 1, 2]
_TYPES = list(EventType)

logger = structlog.get_logger()


def _half_text(half: HalfFilter) -> str:
    return ru.HALF_ALL if half is None else ru.HALF_TEXT.format(half=half)


def _type_text(event_type: EventType) -> str:
    return ru.TYPE_LABELS[event_type]


def _sidebar(events: list[CurrentEvent]) -> EventFilter:
    """Draw the filters in the sidebar and return what the user chose."""
    with st.sidebar:
        st.header(ru.SIDEBAR_FILTERS)
        present = available_types(events)
        types = st.multiselect(ru.FILTER_TYPES, present, default=present, format_func=_type_text)
        half = st.radio(ru.FILTER_HALF, _HALF_OPTIONS, format_func=_half_text)
        min_score = st.slider(
            ru.FILTER_MIN_SCORE, 0.0, 1.0, 0.0, 0.05, help=ru.FILTER_MIN_SCORE_HELP
        )
    return EventFilter(types=frozenset(types), half=half, min_score=min_score)


def _flash(level: str, text: str) -> None:
    st.session_state[_FLASH] = (level, text)


def _show_flash() -> None:
    """Show the result of the last edit, once."""
    flash: tuple[str, str] | None = st.session_state.pop(_FLASH, None)
    if flash is not None:
        level, text = flash
        (st.success if level == "ok" else st.error)(text)


def _require_bounds(args: AppArgs) -> TimeBounds:
    bounds = load_time_bounds(args.results_dir)
    if bounds is None:
        raise EditInputError(ru.ERR_NO_BOUNDS)
    return bounds


def _event_by_id(args: AppArgs, event_id: str) -> CurrentEvent:
    for event in load_events(args.db, args.match_id):
        if event.event_id == event_id:
            return event
    raise EditInputError(ru.EDIT_EVENT_GONE)


def _write(args: AppArgs, make_plan: Callable[[], EditPlan]) -> None:
    """Plan an edit, write it, and leave a message for the next draw. Runs in a callback only."""
    try:
        plan = make_plan()
        record = apply_edit(args.db, args.match_id, plan)
    except EditInputError as err:
        _flash("error", str(err))
        return
    except StorageError as err:
        logger.warning("edit_failed", stage="ui", match_id=args.match_id, reason=str(err))
        _flash("error", ru.EDIT_FAILED.format(error=err))
        return
    logger.info(
        "edit_written", stage="ui", match_id=args.match_id, action=plan.action, edit=record.id
    )
    if plan.action == "add":
        st.session_state[_SELECTED] = record.event_id  # the new event, so the player goes to it
    elif plan.action == "delete":
        st.session_state[_SELECTED] = None
    _flash("ok", plan.message)


def _on_move(args: AppArgs, event_id: str, time_key: str) -> None:
    _write(
        args,
        lambda: plan_move(
            _event_by_id(args, event_id), st.session_state.get(time_key, ""), _require_bounds(args)
        ),
    )


def _on_retype(args: AppArgs, event_id: str, type_key: str) -> None:
    _write(args, lambda: plan_retype(_event_by_id(args, event_id), st.session_state[type_key]))


def _on_delete(args: AppArgs, event_id: str) -> None:
    _write(args, lambda: plan_delete(_event_by_id(args, event_id)))


def _on_add(args: AppArgs, type_key: str, time_key: str) -> None:
    _write(
        args,
        lambda: plan_add(
            st.session_state[type_key], st.session_state.get(time_key, ""), _require_bounds(args)
        ),
    )


def _on_undo(args: AppArgs) -> None:
    try:
        undo_last(args.db, args.match_id)
    except StorageError as err:
        logger.warning("undo_failed", stage="ui", match_id=args.match_id, reason=str(err))
        _flash("error", ru.EDIT_FAILED.format(error=err))
        return
    logger.info("edit_undone", stage="ui", match_id=args.match_id)
    _flash("ok", ru.EDIT_UNDONE)


def _on_export(args: AppArgs, video: Path, event_id: str) -> None:
    """Cut the clip of the chosen event. Runs in a callback only, so once per click."""
    st.session_state.pop(_CLIP_ERROR, None)
    before = float(st.session_state[_CLIP_BEFORE])
    after = float(st.session_state[_CLIP_AFTER])
    try:
        event = _event_by_id(args, event_id)
        path = clips.export_clip(video, event, before, after)
    except EditInputError as err:
        st.session_state[_CLIP_ERROR] = str(err)
        return
    except (ExportError, StorageError) as err:
        logger.warning(
            "clip_export_failed", stage="ui", match_id=args.match_id, reason=str(err)[:200]
        )
        st.session_state[_CLIP_ERROR] = ru.CLIP_FAILED.format(error=err)
        return
    row = to_row(event)
    caption = ru.CLIP_READY.format(
        type=row.type_label, time=row.time_text, before=f"{before:g}", after=f"{after:g}"
    )
    st.session_state[_CLIP] = (str(path), caption)


def _show_clip() -> None:
    """The error of the last export, or the last clip with a download button."""
    error: str | None = st.session_state.get(_CLIP_ERROR)
    if error is not None:
        st.error(error)
    result: tuple[str, str] | None = st.session_state.get(_CLIP)
    if result is None:
        return
    path = Path(result[0])
    try:
        data = path.read_bytes()
    except OSError:
        st.session_state.pop(_CLIP, None)
        st.error(ru.CLIP_FILE_GONE.format(name=path.name))
        return
    st.caption(result[1])
    st.video(data, format="video/mp4")
    st.download_button(
        ru.CLIP_DOWNLOAD, data=data, file_name=path.name, mime="video/mp4", key="btn_clip_download"
    )


def _render_clip_panel(args: AppArgs, selected: CurrentEvent | None) -> None:
    """Margins, the export button for the chosen event, and the last clip (in the sidebar)."""
    with st.sidebar:
        st.header(ru.CLIP_HEADER)
        before_col, after_col = st.columns(2)
        before_col.number_input(
            ru.CLIP_BEFORE, min_value=0.0, value=clips.DEFAULT_BEFORE_S, step=1.0, key=_CLIP_BEFORE
        )
        after_col.number_input(
            ru.CLIP_AFTER, min_value=0.0, value=clips.DEFAULT_AFTER_S, step=1.0, key=_CLIP_AFTER
        )
        if args.video is None:
            st.caption(ru.CLIP_NO_VIDEO)
        elif selected is None:
            st.caption(ru.CLIP_PICK_FIRST)
        st.button(
            ru.CLIP_EXPORT,
            key="btn_clip",
            on_click=_on_export,
            args=(args, args.video, None if selected is None else selected.event_id),
            disabled=args.video is None or selected is None,
            help=ru.CLIP_EXPORT_HELP,
        )
        _show_clip()


def _render_table(rows: list[EventRow]) -> tuple[int, ...]:
    """Draw the events table over the full width; return the indices of the picked rows."""
    state = st.dataframe(
        table_records(rows),
        key=TABLE_KEY,
        hide_index=True,
        height=TABLE_HEIGHT_PX,
        on_select="rerun",
        selection_mode="single-row",
        column_config={ru.COLUMN_SCORE: st.column_config.NumberColumn(format="%.2f")},
    )
    return tuple(state.selection.rows)


def _take_pick(rows: list[EventRow], picked: tuple[int, ...]) -> None:
    """Make a click on the table the chosen event.

    Only a changed selection on an unchanged list counts as a click. When the list itself
    changed (a filter, or an edit that moved or removed rows) the indices the table reports
    may point at other events than before, so they are noted and not acted on: the chosen
    event is kept by its id.
    """
    ids = tuple(r.event_id for r in rows)
    same_list = st.session_state.get(_ROWS_SEEN) == ids
    changed = st.session_state.get(_PICK_SEEN, ()) != picked
    st.session_state[_ROWS_SEEN] = ids
    st.session_state[_PICK_SEEN] = picked
    chosen = selected_row(rows, picked)
    if same_list and changed and chosen is not None:
        st.session_state[_SELECTED] = chosen.event_id


def _render_player(args: AppArgs, selected: CurrentEvent | None) -> None:
    t_v = 0.0 if selected is None else selected.t_v
    # Nothing chosen yet: show the start of the video and do not play by itself.
    start = 0 if selected is None else player_start(t_v)
    if selected is not None:
        row = to_row(selected)
        st.markdown(f"**{ru.PLAYING.format(type=row.type_label, time=row.time_text)}**")
    else:
        st.caption(ru.TABLE_HINT)
    st.video(watch_url(args.youtube_id), start_time=start, autoplay=selected is not None)
    link = (
        youtube_link(args.youtube_id, t_v) if selected is not None else watch_url(args.youtube_id)
    )
    st.markdown(f"{ru.LINK_TEXT}: [{link}]({link})")
    st.caption(ru.TIME_NOTE)


def _render_selected_controls(args: AppArgs, event: CurrentEvent) -> None:
    """Move, change the type, or delete the chosen event.

    Widget keys carry the event id and its time, so the fields show the current values again
    after every edit.
    """
    row = to_row(event)
    st.markdown(f"**{ru.EDIT_SELECTED.format(type=row.type_label, time=row.time_text)}**")
    time_key = f"move_time_{event.event_id}_{int(event.t_v)}"
    type_key = f"retype_{event.event_id}_{event.type}"
    time_col, type_col = st.columns(2)
    time_col.text_input(ru.EDIT_TIME_LABEL, value=format_mmss(event.t_v), key=time_key)
    type_col.selectbox(
        ru.EDIT_TYPE_LABEL,
        _TYPES,
        index=_TYPES.index(event.type),
        format_func=_type_text,
        key=type_key,
    )
    move_col, retype_col, delete_col = st.columns(3)
    move_col.button(
        ru.EDIT_MOVE, key="btn_move", on_click=_on_move, args=(args, event.event_id, time_key)
    )
    retype_col.button(
        ru.EDIT_RETYPE, key="btn_retype", on_click=_on_retype, args=(args, event.event_id, type_key)
    )
    delete_col.button(
        ru.EDIT_DELETE, key="btn_delete", on_click=_on_delete, args=(args, event.event_id)
    )


def _render_add_form(args: AppArgs, selected: CurrentEvent | None) -> None:
    """The form for a new event; its time starts as the chosen event's time."""
    suffix = "none" if selected is None else f"{selected.event_id}_{int(selected.t_v)}"
    type_key, time_key = f"add_type_{suffix}", f"add_time_{suffix}"
    default_type = EventType.GOAL if selected is None else selected.type
    default_time = "" if selected is None else format_mmss(selected.t_v)
    with st.form("add_form"):
        st.markdown(f"**{ru.EDIT_ADD_HEADER}**")
        st.selectbox(
            ru.EDIT_ADD_TYPE_LABEL,
            _TYPES,
            index=_TYPES.index(default_type),
            format_func=_type_text,
            key=type_key,
        )
        st.text_input(ru.EDIT_ADD_TIME_LABEL, value=default_time, key=time_key)
        st.form_submit_button(
            ru.EDIT_ADD_SUBMIT, key="btn_add", on_click=_on_add, args=(args, type_key, time_key)
        )


def _render_edit_panel(args: AppArgs, selected: CurrentEvent | None, n_edits: int) -> None:
    st.subheader(ru.EDIT_HEADER)
    if selected is None:
        st.caption(ru.EDIT_PICK_FIRST)
    else:
        _render_selected_controls(args, selected)
    _render_add_form(args, selected)
    st.button(
        ru.EDIT_UNDO,
        key="btn_undo",
        on_click=_on_undo,
        args=(args,),
        disabled=n_edits == 0,
        help=ru.EDIT_UNDO_HELP,
    )
    st.caption(ru.EDIT_COUNT.format(count=n_edits) if n_edits else ru.EDIT_NO_EDITS)


def _process_alive(job: IngestJob) -> bool:
    """The ingest of `job` still runs: our own process, or any process holding its lock."""
    proc = st.session_state.get(_PROC)
    if proc is not None and proc.poll() is None:
        return True
    return busy_pid(job) is not None


def _lineup_key() -> str:
    """Key of the lineup uploader; it changes after every start, so the next match does not
    silently get the file of the previous one."""
    return f"{_LINEUP_KEY}_{st.session_state.get(_LINEUP_ROUND, 0)}"


def _on_start_ingest(start: StartArgs) -> None:
    """Start `cli ingest` in the background for the typed link. Runs in a callback only."""
    st.session_state.pop(_INGEST_FLASH, None)
    if start.data_root is None:
        return
    try:
        job = plan_job(st.session_state.get(_URL_KEY, ""), start.data_root, start.templates)
    except IngestError as err:
        st.session_state[_INGEST_FLASH] = ("error", ru.INGEST_BAD_URL.format(error=err))
        return
    tracked: IngestJob | None = st.session_state.get(_JOB)
    proc = st.session_state.get(_PROC)
    pid = busy_pid(job)
    same = tracked is not None and tracked.match_dir == job.match_dir
    if same and proc is not None and proc.poll() is None:
        pid = proc.pid  # started a moment ago: the process may not hold its lock yet
    if pid is not None:
        st.session_state[_JOB] = job
        st.session_state[_INGEST_FLASH] = ("error", ru.INGEST_BUSY.format(pid=pid))
        return
    upload = st.session_state.get(_lineup_key())
    if upload is not None:
        try:
            save_lineup(job.match_dir, upload.getvalue())
        except (PlatesError, StorageError) as err:
            st.session_state[_INGEST_FLASH] = ("error", ru.INGEST_LINEUP_BAD.format(error=err))
            return
    try:
        started = launch(job)
    except OSError as err:
        logger.warning("ingest_start_failed", stage="ui", match_id=job.video_id, reason=str(err))
        st.session_state[_INGEST_FLASH] = ("error", ru.INGEST_START_FAILED.format(error=err))
        return
    logger.info("ingest_started", stage="ui", match_id=job.video_id, pid=started.pid)
    st.session_state[_JOB] = job
    st.session_state[_PROC] = started
    st.session_state[_LINEUP_ROUND] = st.session_state.get(_LINEUP_ROUND, 0) + 1
    st.session_state[_INGEST_FLASH] = ("ok", ru.INGEST_STARTED.format(log=job.log_path))


def _show_phase(job: IngestJob, phase: str, error: str | None) -> None:
    if phase == "waiting":
        video = job.match_dir / VIDEO_FILE
        st.info(f"{ru.TDEED_WAITING.format(video=video, dir=job.match_dir)} {ru.TDEED_RESUME_UI}")
    elif phase == "failed":
        st.error(ru.INGEST_FAILED.format(error=error))
    elif phase == "gone":
        st.warning(ru.INGEST_PROCESS_GONE.format(log=job.log_path))
    elif phase == "done":
        st.success(ru.INGEST_DONE)
        if st.button(ru.INGEST_OPEN, key="btn_open_match"):
            st.session_state[_OPENED] = job.view_args()
            st.rerun()  # the whole page, not only this fragment: it shows the new match


@st.fragment(run_every=STATUS_REFRESH_S)
def _job_panel() -> None:
    """Steps of the followed ingest; reread every few seconds without blocking the page."""
    job: IngestJob | None = st.session_state.get(_JOB)
    if job is None:
        return
    status = read_status(job.match_dir)
    phase = job_phase(status, _process_alive(job))
    st.markdown(f"**{ru.INGEST_MATCH.format(match_id=job.match_dir.name)}**")
    for row in step_rows(status):
        line = f"{row.label}: {row.state}"
        if row.state == ru.INGEST_STATE_LABELS["running"] and row.progress > 0:
            line += f" ({row.progress:.0%})"
        st.text(line)
        if row.error and row.state == ru.INGEST_STATE_LABELS["skipped"]:
            st.caption(row.error)  # why the optional step did not run
    entry = status.stages.get(INGEST_STAGE)
    _show_phase(job, phase, None if entry is None else entry.error)
    st.caption(ru.INGEST_LOG.format(log=job.log_path))


def _render_ingest_panel(start: StartArgs) -> None:
    """The "new match" block at the top of the sidebar."""
    with st.sidebar:
        st.header(ru.INGEST_HEADER)
        if start.data_root is None:
            st.caption(ru.INGEST_NO_ROOT)
            return
        st.text_input(ru.INGEST_URL, key=_URL_KEY, help=ru.INGEST_URL_HELP)
        st.file_uploader(
            ru.INGEST_LINEUPS, type=["csv"], key=_lineup_key(), help=ru.INGEST_LINEUPS_HELP
        )
        st.button(ru.INGEST_START, key="btn_ingest", on_click=_on_start_ingest, args=(start,))
        flash: tuple[str, str] | None = st.session_state.get(_INGEST_FLASH)
        if flash is not None:
            (st.success if flash[0] == "ok" else st.error)(flash[1])
        if st.session_state.get(_JOB) is not None:
            _job_panel()


def _select_event(event_id: str) -> None:
    """Make an event the chosen one (a link of the agent's answer): the player shows it."""
    st.session_state[_SELECTED] = event_id


def _load(args: AppArgs) -> tuple[list[CurrentEvent], int]:
    """Events (the half of edited ones derived from their time) and the edits in force."""
    bounds = load_time_bounds(args.results_dir)
    events = [with_derived_half(e, bounds) for e in load_events(args.db, args.match_id)]
    return events, count_edits(args.db, args.match_id)


def main() -> None:
    """Draw the page; every user action reruns it."""
    st.set_page_config(page_title=ru.TITLE, layout="wide")
    st.title(ru.TITLE)
    try:
        start = parse_start_args(sys.argv[1:])
    except ConfigError as err:
        st.error(ru.BAD_ARGS.format(error=err))
        st.stop()
    _render_ingest_panel(start)
    args: AppArgs | None = st.session_state.get(_OPENED) or start.match
    if args is None:
        st.info(ru.NO_MATCH_OPEN)
        st.stop()
    if not args.db.is_file():
        st.error(ru.NO_DB.format(path=args.db))
        st.stop()
    try:
        events, n_edits = _load(args)
    except StorageError as err:
        st.error(ru.DB_ERROR.format(path=args.db, error=err))
        st.stop()
    _show_flash()
    if not events:
        st.warning(ru.NO_EVENTS.format(match_id=args.match_id))
        if n_edits == 0:
            st.stop()  # nothing to show, edit or undo
    by_id = {e.event_id: e for e in events}
    if st.session_state.get(_SELECTED) not in by_id:
        st.session_state[_SELECTED] = None  # deleted, undone, or replaced by a new run

    shown = filter_events(events, _sidebar(events))
    player_slot = st.container()  # filled below the table code: the pick is known only then
    agent_slot = st.container()  # the agent panel sits between the player and the table
    st.caption(ru.SHOWN.format(shown=len(shown), total=len(events)))
    if shown:
        rows = to_rows(shown, load_player_texts(args.results_dir))
        _take_pick(rows, _render_table(rows))
    else:
        st.info(ru.NOTHING_MATCHES)
    selected = by_id.get(st.session_state[_SELECTED])
    with player_slot:
        _, middle, _ = st.columns(PLAYER_COLUMNS)
        with middle:
            _render_player(args, selected)
    with agent_slot:
        render_agent_panel(args, _select_event)
    _render_edit_panel(args, selected, n_edits)
    _render_clip_panel(args, selected)


main()
