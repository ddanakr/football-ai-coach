"""The review mode through the Streamlit script (`AppTest`): the switch, ✓ and ✗ on candidates and
events, the journal, rows that are not shown twice, the counter and the "next" button."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.review_synth import write_review_layer

from aicoach.domain.events import EventType
from aicoach.errors import StorageError
from aicoach.i18n import ru
from aicoach.review.log import LOG_FILE, ReviewLogEntry, append_entry, make_entry, read_log
from aicoach.review.stage import RESULT_FILE
from aicoach.storage import connect
from aicoach.ui.data import load_events
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command

VID = "abcDEF_-123"
MATCH = "m1"
SELECTED = "selected_event_id"

# Same loopback exception as `test_app.py`: AppTest needs an asyncio socket pair on Windows.
pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])


def start(monkeypatch: pytest.MonkeyPatch, match_dir: Path) -> AppTest:
    args = AppArgs(db=match_dir / "aicoach.db", match_id=MATCH, youtube_id=VID, match_dir=match_dir)
    monkeypatch.setattr("sys.argv", [str(app_script()), *build_streamlit_command(args)[6:]])
    return AppTest.from_file(str(app_script()), default_timeout=30).run()


def review_on(app: AppTest, *, show_all: bool = False) -> AppTest:
    app.toggle(key="review_mode").set_value(True).run()
    if show_all:
        app.checkbox(key="review_show_all").set_value(True).run()
    return app


@pytest.fixture
def match_dir(tmp_path: Path) -> Path:
    """Four events (goal, shot, substitution, foul) and four shot candidates (210 s, 600 s, 700 s,
    2000 s)."""
    write_review_layer(tmp_path, MATCH)
    return tmp_path


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, match_dir: Path) -> AppTest:
    return start(monkeypatch, match_dir)


def row_ids(app: AppTest, prefix: str = "rv_ok_") -> list[str]:
    """Ids of the rows in the list, in screen order."""
    return [b.key[len(prefix) :] for b in app.button if b.key and b.key.startswith(prefix)]


def events_of(match_dir: Path, kind: EventType) -> list[str]:
    return [e.event_id for e in load_events(match_dir / "aicoach.db", MATCH) if e.type is kind]


def edits(match_dir: Path) -> list[tuple[str, str | None, str | None, float | None]]:
    with connect(match_dir / "aicoach.db") as conn:
        rows = conn.execute("SELECT action, target, type, t_v FROM edits ORDER BY id").fetchall()
    return [(r["action"], r["target"], r["type"], r["t_v"]) for r in rows]


def log(match_dir: Path) -> list[ReviewLogEntry]:
    return read_log(match_dir)


def messages(app: AppTest) -> list[str]:
    return [str(m.value) for m in (*app.success, *app.error)]


def progress(app: AppTest) -> str:
    return next(str(c.value) for c in app.caption if str(c.value).startswith("Проверено"))


@pytest.mark.unit
def test_review_list_is_off_until_the_switch_is_turned_on(app: AppTest) -> None:
    assert not app.exception
    assert row_ids(app) == []
    assert len(app.dataframe) == 1  # the usual table

    review_on(app)

    assert not app.exception
    assert len(app.dataframe) == 0
    assert len(row_ids(app)) == 8  # 4 events + 4 candidates
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=0, total=8)


@pytest.mark.unit
def test_rows_are_in_time_order_and_candidates_are_marked(app: AppTest) -> None:
    review_on(app)

    play = [b.label for b in app.button if b.key and b.key.startswith("rv_play_")]
    times = [label.removeprefix("▶ ") for label in play]
    assert times == sorted(times)  # mm:ss sorts like the time
    ids = row_ids(app)
    assert [i[0] for i in ids].count("r") == 4
    marked = [t.value for t in app.text if t.value == ru.REVIEW_CANDIDATE_MARK]
    assert len(marked) == 4


@pytest.mark.unit
def test_accepting_a_candidate_adds_an_event_through_edits_and_logs_it(
    app: AppTest, match_dir: Path
) -> None:
    review_on(app)
    shot_candidate = next(i for i in row_ids(app) if i.startswith("r:"))  # the earliest: 210 s
    shots_before = len(events_of(match_dir, EventType.SHOT))

    app.button(key=f"rv_ok_{shot_candidate}").click().run()

    assert not app.exception
    added = [e for e in edits(match_dir) if e[0] == "add"]
    assert added == [("add", None, "shot", 210.0)]  # the exact time, not cut to the second
    assert len(events_of(match_dir, EventType.SHOT)) == shots_before + 1
    (entry,) = log(match_dir)
    assert (entry.id, entry.action, entry.t_v, entry.type) == (
        shot_candidate,
        "accept",
        210.0,
        EventType.SHOT,
    )
    assert entry.event_id is not None and entry.event_id.startswith("e:")
    assert shot_candidate not in row_ids(app)
    assert entry.event_id not in row_ids(app)  # the added event is decided by being added
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=1, total=8)
    assert ru.REVIEW_ACCEPTED.format(type=ru.TYPE_LABELS[EventType.SHOT], time="03:30") in messages(
        app
    )


@pytest.mark.unit
def test_rejecting_a_candidate_only_writes_the_journal(app: AppTest, match_dir: Path) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))

    app.button(key=f"rv_no_{candidate}").click().run()

    assert edits(match_dir) == []
    (entry,) = log(match_dir)
    assert (entry.id, entry.action, entry.event_id) == (candidate, "reject", None)
    assert candidate not in row_ids(app)
    assert len(row_ids(app)) == 7
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=1, total=8)


@pytest.mark.unit
def test_confirming_an_event_only_writes_the_journal(app: AppTest, match_dir: Path) -> None:
    review_on(app)
    event = next(i for i in row_ids(app) if i.startswith("p:"))

    app.button(key=f"rv_ok_{event}").click().run()

    assert edits(match_dir) == []  # the predictions and the edits stay as they were
    (entry,) = log(match_dir)
    assert (entry.id, entry.action) == (event, "confirm")
    assert event not in row_ids(app)
    assert len(events_of(match_dir, EventType.GOAL) + events_of(match_dir, EventType.SHOT)) == 2


@pytest.mark.unit
def test_deleting_an_event_goes_through_edits_and_is_logged(app: AppTest, match_dir: Path) -> None:
    review_on(app)
    event = next(i for i in row_ids(app) if i.startswith("p:"))
    total = len(load_events(match_dir / "aicoach.db", MATCH))

    app.button(key=f"rv_no_{event}").click().run()

    assert edits(match_dir) == [("delete", event, None, None)]
    assert len(load_events(match_dir / "aicoach.db", MATCH)) == total - 1
    (entry,) = log(match_dir)
    assert (entry.id, entry.action) == (event, "delete")
    assert event not in row_ids(app)
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=1, total=8)  # deleted rows count


@pytest.mark.unit
def test_decided_rows_are_not_shown_again_after_a_reload_but_show_all_brings_them_back(
    monkeypatch: pytest.MonkeyPatch, app: AppTest, match_dir: Path
) -> None:
    review_on(app)
    ids = row_ids(app)
    candidate = next(i for i in ids if i.startswith("r:"))
    event = next(i for i in ids if i.startswith("p:"))
    app.button(key=f"rv_no_{candidate}").click().run()
    app.button(key=f"rv_ok_{event}").click().run()

    fresh = review_on(start(monkeypatch, match_dir))
    assert candidate not in row_ids(fresh) and event not in row_ids(fresh)
    assert len(row_ids(fresh)) == 6

    shown = review_on(start(monkeypatch, match_dir), show_all=True)
    assert candidate in row_ids(shown) and event in row_ids(shown)
    marks = [t.value for t in shown.text if ru.REVIEW_DECIDED_MARK in str(t.value)]
    assert len(marks) == 2


@pytest.mark.unit
def test_a_decision_on_a_candidate_of_an_older_layer_does_not_hide_the_new_one(
    app: AppTest, match_dir: Path
) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))
    # The journal names r:N, but with the time of another detection: the layer was rebuilt.
    entry = make_entry(candidate, "reject", 12345.0, EventType.SHOT)
    append_entry(match_dir, entry)

    app.run()

    assert candidate in row_ids(app)
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=0, total=8)


@pytest.mark.unit
def test_a_decision_in_show_all_wins_over_the_earlier_one(app: AppTest, match_dir: Path) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))
    app.button(key=f"rv_no_{candidate}").click().run()
    review_on(app, show_all=True)

    app.button(key=f"rv_ok_{candidate}").click().run()

    assert [e.action for e in log(match_dir)] == ["reject", "accept"]
    assert len([e for e in edits(match_dir) if e[0] == "add"]) == 1


@pytest.mark.unit
def test_undoing_the_added_event_brings_the_candidate_back(app: AppTest) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))
    app.button(key=f"rv_ok_{candidate}").click().run()
    assert candidate not in row_ids(app)

    app.button(key="btn_undo").click().run()

    assert candidate in row_ids(app)
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=0, total=8)


@pytest.mark.unit
def test_undoing_a_deletion_brings_the_event_back_for_review(app: AppTest) -> None:
    review_on(app)
    event = next(i for i in row_ids(app) if i.startswith("p:"))
    app.button(key=f"rv_no_{event}").click().run()
    assert event not in row_ids(app)

    app.button(key="btn_undo").click().run()

    assert event in row_ids(app)
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=0, total=8)


@pytest.mark.unit
def test_a_click_writes_once_even_if_the_page_reruns(app: AppTest, match_dir: Path) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))

    app.button(key=f"rv_ok_{candidate}").click().run()
    app.run()
    app.run()

    assert len(log(match_dir)) == 1
    assert len([e for e in edits(match_dir) if e[0] == "add"]) == 1


@pytest.mark.unit
def test_next_button_walks_the_rows_and_stops_after_the_last(app: AppTest) -> None:
    review_on(app)
    ids = row_ids(app)

    app.button(key="review_next").click().run()
    assert app.session_state[SELECTED] == ids[0]
    app.button(key="review_next").click().run()
    assert app.session_state[SELECTED] == ids[1]
    for _ in ids[2:]:
        app.button(key="review_next").click().run()
    assert app.session_state[SELECTED] == ids[-1]
    app.button(key="review_next").click().run()
    assert app.session_state[SELECTED] is None  # nothing after the last row


@pytest.mark.unit
def test_play_button_chooses_a_row_and_the_player_shows_it(app: AppTest) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))

    app.button(key=f"rv_play_{candidate}").click().run()

    assert not app.exception
    assert app.session_state[SELECTED] == candidate
    playing = [m.value for m in app.markdown if ru.REVIEW_CANDIDATE_MARK in str(m.value)]
    assert any("03:30" in str(v) for v in playing)  # the player line names the candidate and time


@pytest.mark.unit
def test_a_decision_moves_the_player_to_the_next_row(app: AppTest) -> None:
    review_on(app)
    ids = row_ids(app)
    app.button(key=f"rv_play_{ids[0]}").click().run()

    app.button(
        key=f"rv_no_{ids[0]}" if ids[0].startswith("r:") else f"rv_ok_{ids[0]}"
    ).click().run()

    assert app.session_state[SELECTED] == ids[1]


@pytest.mark.unit
def test_a_candidate_that_disappeared_gives_a_message_and_no_journal_line(
    app: AppTest, match_dir: Path
) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))
    (match_dir / RESULT_FILE).unlink()

    app.button(key=f"rv_ok_{candidate}").click().run()

    assert ru.REVIEW_GONE in messages(app)
    assert edits(match_dir) == []
    assert not (match_dir / LOG_FILE).exists()


@pytest.mark.unit
def test_a_failing_journal_after_the_edit_says_so_and_the_row_stays_open(
    monkeypatch: pytest.MonkeyPatch, app: AppTest, match_dir: Path
) -> None:
    review_on(app)
    candidate = next(i for i in row_ids(app) if i.startswith("r:"))

    def broken(*_: object) -> None:
        message = "disk full"
        raise StorageError(message)

    monkeypatch.setattr("aicoach.ui.review_panel.append_entry", broken)
    app.button(key=f"rv_ok_{candidate}").click().run()

    assert any("disk full" in m and "Событие уже изменено" in m for m in messages(app))
    assert len([e for e in edits(match_dir) if e[0] == "add"]) == 1  # the edit is in the store
    assert candidate in row_ids(app)  # but the row is not counted as decided


@pytest.mark.unit
def test_without_the_layer_file_only_events_are_listed_with_a_hint(
    monkeypatch: pytest.MonkeyPatch, match_dir: Path
) -> None:
    (match_dir / RESULT_FILE).unlink()

    app = review_on(start(monkeypatch, match_dir))

    assert not app.exception
    assert len(row_ids(app)) == 4
    assert any(ru.REVIEW_NO_LAYER in str(c.value) for c in app.caption)


@pytest.mark.unit
def test_a_broken_journal_is_an_error_and_the_usual_table_stays(
    monkeypatch: pytest.MonkeyPatch, match_dir: Path
) -> None:
    (match_dir / LOG_FILE).write_text("not json\n", encoding="utf-8")

    app = review_on(start(monkeypatch, match_dir))

    assert not app.exception
    assert any(str(e.value).startswith("Журнал проверки не читается") for e in app.error)
    assert row_ids(app) == []
    assert len(app.dataframe) == 1


@pytest.mark.unit
def test_filters_apply_to_candidates_too(app: AppTest) -> None:
    review_on(app)

    app.multiselect[0].unselect(EventType.FOUL).unselect(EventType.GOAL).run()

    # The shot event, the substitution and the four shot candidates: no goal, no foul.
    assert len(set(row_ids(app))) == 6


@pytest.mark.unit
def test_all_rows_decided_shows_the_finished_message(app: AppTest) -> None:
    review_on(app)
    for row in row_ids(app):
        key = f"rv_no_{row}" if row.startswith("r:") else f"rv_ok_{row}"
        app.button(key=key).click().run()

    assert row_ids(app) == []
    assert any(ru.REVIEW_NOTHING_LEFT in str(i.value) for i in app.info)
    assert progress(app) == ru.REVIEW_PROGRESS.format(checked=8, total=8)


@pytest.mark.unit
def test_the_agent_does_not_see_the_candidates(match_dir: Path) -> None:
    from aicoach.agent.context import load_context

    context = load_context(match_dir / "aicoach.db", MATCH, match_dir)

    assert len(context.events) == 4  # the events of the catalog, not the 4 candidates
    assert all(not f.event.event_id.startswith("r:") for f in context.events)
