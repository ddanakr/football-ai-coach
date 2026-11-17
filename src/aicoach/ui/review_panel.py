"""The review mode of the viewer (D-075): events and candidates in one list, ✓ and ✗ on each row.

Drawn by `app.py`. The decisions are written in button callbacks only (`_on_decide`), which run
once per click, so a rerun of the page cannot write one twice. A decision that changes the
catalog (adding a candidate, deleting an event) goes through the `edits` journal of the store
like any manual edit; every decision is also a line of `review_log.jsonl`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import streamlit as st
import structlog

from aicoach.domain.events import EventType
from aicoach.errors import StorageError
from aicoach.i18n import ru
from aicoach.review.log import (
    Action,
    ReviewLogEntry,
    append_entry,
    deleted_ids,
    entries_for_layer,
    make_entry,
    read_log,
    review_state,
)
from aicoach.review.stage import load_review_candidates
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.storage import CurrentEvent
from aicoach.ui.data import apply_edit, load_clock_segments, load_events, load_time_bounds
from aicoach.ui.edits import EditInputError, plan_add_at, plan_delete
from aicoach.ui.launch import AppArgs
from aicoach.ui.logic import format_mmss, type_label
from aicoach.ui.review_logic import (
    ReviewRow,
    build_review_rows,
    next_row_id,
    review_progress,
)

MODE_KEY = "review_mode"
SHOW_ALL_KEY = "review_show_all"
LIST_HEIGHT_PX = 380
_ORDER_KEY = "review_order"  # ids of the rows drawn on the previous run, in screen order
_COLUMNS = [1.3, 1.5, 1.4, 1, 1.4, 0.7, 0.7]

Notify = Callable[[str, str], None]  # (level, text) of the message shown after the click
Select = Callable[[str | None], None]  # makes a row the chosen one (the player shows it)

logger = structlog.get_logger()


@dataclass(frozen=True)
class ReviewView:
    """What the page needs of the review: the rows to show, every row by id, the progress."""

    rows: list[ReviewRow]
    lookup: dict[str, ReviewRow]
    checked: int
    total: int
    has_layer: bool

    @property
    def types(self) -> set[EventType]:
        """The categories present among all rows."""
        return {r.type for r in self.lookup.values()}


def render_toggle() -> tuple[bool, bool]:
    """The switch of the mode and of "show all" in the sidebar; returns `(on, show_all)`."""
    with st.sidebar:
        on = st.toggle(ru.REVIEW_MODE, key=MODE_KEY, help=ru.REVIEW_MODE_HELP)
        show_all = (
            st.checkbox(ru.REVIEW_SHOW_ALL, key=SHOW_ALL_KEY, help=ru.REVIEW_SHOW_ALL_HELP)
            if on
            else False
        )
    return on, show_all


def load_review_view(
    args: AppArgs, events: Sequence[CurrentEvent], *, show_all: bool
) -> ReviewView:
    """Read the layer and the journal of the match and build the rows.

    Raises:
        StorageError: The journal cannot be read.
    """
    layer = load_review_candidates(args.results_dir)
    candidates = [] if layer is None else layer.candidates
    entries = entries_for_layer(
        read_log(args.results_dir), [(c.id, c.type, c.t_v) for c in candidates]
    )
    state = review_state(entries, (e.event_id for e in events))
    segments = load_clock_segments(args.results_dir)
    max_extrapolation_s = ScoreboardConfig().timeline.max_extrapolation_s
    shown = build_review_rows(
        events, candidates, state, segments, max_extrapolation_s, show_all=show_all
    )
    everything = build_review_rows(
        events, candidates, state, segments, max_extrapolation_s, show_all=True
    )
    checked, total = review_progress(events, candidates, state, deleted_ids(entries))
    return ReviewView(shown, {r.row_id: r for r in everything}, checked, total, layer is not None)


def _event_by_id(args: AppArgs, event_id: str) -> CurrentEvent:
    for event in load_events(args.db, args.match_id):
        if event.event_id == event_id:
            return event
    raise EditInputError(ru.REVIEW_GONE)


def _log_after_edit(args: AppArgs, entry: ReviewLogEntry) -> None:
    """Write the journal line of a decision whose edit is already in the store. A failure is
    reported with that fact, so the reviewer knows the row is not counted as decided."""
    try:
        append_entry(args.results_dir, entry)
    except StorageError as err:
        logger.warning("review_log_failed", stage="ui", match_id=args.match_id, reason=str(err))
        raise EditInputError(ru.REVIEW_LOG_FAILED.format(error=err)) from err


def _decide_candidate(args: AppArgs, row_id: str, action: Action) -> str:
    layer = load_review_candidates(args.results_dir)
    candidate = next((c for c in (layer.candidates if layer else []) if c.id == row_id), None)
    if candidate is None:
        raise EditInputError(ru.REVIEW_GONE)
    label, time = type_label(candidate.type), format_mmss(candidate.t_v)
    if action == "reject":
        entry = make_entry(row_id, "reject", candidate.t_v, candidate.type)
        append_entry(args.results_dir, entry)
        return ru.REVIEW_REJECTED.format(type=label, time=time)
    bounds = load_time_bounds(args.results_dir)
    if bounds is None:
        raise EditInputError(ru.ERR_NO_BOUNDS)
    record = apply_edit(args.db, args.match_id, plan_add_at(candidate.type, candidate.t_v, bounds))
    entry = make_entry(row_id, "accept", candidate.t_v, candidate.type, event_id=record.event_id)
    _log_after_edit(args, entry)
    return ru.REVIEW_ACCEPTED.format(type=label, time=time)


def _decide_event(args: AppArgs, row_id: str, action: Action) -> str:
    event = _event_by_id(args, row_id)
    label, time = type_label(event.type), format_mmss(event.t_v)
    entry = make_entry(row_id, action, event.t_v, event.type)
    if action == "confirm":
        append_entry(args.results_dir, entry)
        return ru.REVIEW_CONFIRMED.format(type=label, time=time)
    plan = plan_delete(event)
    apply_edit(args.db, args.match_id, plan)
    _log_after_edit(args, entry)
    return plan.message


def _on_decide(args: AppArgs, row_id: str, action: Action, notify: Notify, select: Select) -> None:
    """Write one decision, then move on to the next row. Runs in a callback only."""
    try:
        message = (
            _decide_candidate(args, row_id, action)
            if row_id.startswith("r:")
            else _decide_event(args, row_id, action)
        )
    except EditInputError as err:
        notify("error", str(err))
        return
    except StorageError as err:
        logger.warning("review_failed", stage="ui", match_id=args.match_id, reason=str(err))
        notify("error", ru.EDIT_FAILED.format(error=err))
        return
    logger.info("review_decided", stage="ui", match_id=args.match_id, action=action)
    order: tuple[str, ...] = st.session_state.get(_ORDER_KEY, ())
    select(next_row_id(order, row_id))
    notify("ok", message)


def _on_next(select: Select, current: str | None) -> None:
    order: tuple[str, ...] = st.session_state.get(_ORDER_KEY, ())
    select(next_row_id(order, current))


def _draw_row(args: AppArgs, row: ReviewRow, chosen: bool, notify: Notify, select: Select) -> None:
    """One row: the time (a button that puts the player there), the texts, ✓ and ✗."""
    candidate = row.kind == "candidate"
    cells = st.columns(_COLUMNS, vertical_alignment="center")
    cells[0].button(
        f"▶ {format_mmss(row.t_v)}",
        key=f"rv_play_{row.row_id}",
        on_click=select,
        args=(row.row_id,),
        help=ru.REVIEW_PLAY_HELP,
        type="primary" if chosen else "secondary",
    )
    label = type_label(row.type)
    cells[1].markdown(f"**{label}**" if chosen else label)
    half = "" if row.half is None else ru.HALF_TEXT.format(half=row.half)
    cells[2].text(" · ".join(part for part in (half, row.minute_text or "") if part))
    cells[3].text("" if row.score is None else f"{row.score:.2f}")
    mark = f"{row.source_text} ({ru.REVIEW_DECIDED_MARK})" if row.decided else row.source_text
    cells[4].text(mark)
    ok_action: Action = "accept" if candidate else "confirm"
    no_action: Action = "reject" if candidate else "delete"
    cells[5].button(
        "✓",
        key=f"rv_ok_{row.row_id}",
        on_click=_on_decide,
        args=(args, row.row_id, ok_action, notify, select),
        help=ru.REVIEW_OK_HELP_CANDIDATE if candidate else ru.REVIEW_OK_HELP_EVENT,
    )
    cells[6].button(
        "✗",
        key=f"rv_no_{row.row_id}",
        on_click=_on_decide,
        args=(args, row.row_id, no_action, notify, select),
        help=ru.REVIEW_NO_HELP_CANDIDATE if candidate else ru.REVIEW_NO_HELP_EVENT,
    )


def render_review_list(
    args: AppArgs,
    view: ReviewView,
    rows: Sequence[ReviewRow],
    chosen: str | None,
    notify: Notify,
    select: Select,
) -> None:
    """Draw the progress, the "next" button and the list (the full width, a fixed height).

    Args:
        args: The open match.
        view: The review as read from the files.
        rows: The rows to draw (`view.rows` after the sidebar filters).
        chosen: Id of the row the player shows, if any.
        notify: Shows a message after a click.
        select: Makes a row the chosen one.
    """
    st.subheader(ru.REVIEW_HEADER)
    progress_col, next_col = st.columns([4, 1], vertical_alignment="center")
    progress_col.caption(ru.REVIEW_PROGRESS.format(checked=view.checked, total=view.total))
    next_col.button(ru.REVIEW_NEXT, key="review_next", on_click=_on_next, args=(select, chosen))
    if not view.has_layer:
        st.caption(ru.REVIEW_NO_LAYER)
    st.session_state[_ORDER_KEY] = tuple(r.row_id for r in rows)
    with st.container(height=LIST_HEIGHT_PX, border=True):
        if not rows:
            st.info(ru.REVIEW_NOTHING_LEFT)
        for row in rows:
            _draw_row(args, row, row.row_id == chosen, notify, select)
