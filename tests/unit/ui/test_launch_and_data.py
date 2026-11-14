"""Start-up arguments, the Streamlit command and reading events through the store."""

import sys
from pathlib import Path

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import ConfigError, StorageError
from aicoach.storage import NewPrediction, add_edit, add_predictions, connect, init_db
from aicoach.ui.data import load_events
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command, parse_app_args

VID = "abcDEF_-123"


@pytest.mark.unit
def test_app_script_exists() -> None:
    assert app_script().is_file()
    assert app_script().name == "app.py"


@pytest.mark.unit
def test_command_runs_streamlit_with_arguments_after_double_dash(tmp_path: Path) -> None:
    args = AppArgs(db=tmp_path / "a.db", match_id="m1", youtube_id=VID)

    command = build_streamlit_command(args, python="py")

    assert command[:4] == ["py", "-m", "streamlit", "run"]
    assert command[4] == str(app_script())
    assert command[5] == "--"
    assert command[6:] == [
        "--db",
        str(tmp_path / "a.db"),
        "--match-id",
        "m1",
        f"--youtube-id={VID}",
    ]


@pytest.mark.unit
def test_match_dir_is_passed_on_and_read_back(tmp_path: Path) -> None:
    args = AppArgs(db=tmp_path / "a.db", match_id="m1", youtube_id=VID, match_dir=tmp_path / "v")

    command = build_streamlit_command(args, python="py")

    assert command[-2:] == ["--match-dir", str(tmp_path / "v")]
    assert parse_app_args(command[6:]) == args
    assert args.results_dir == tmp_path / "v"


@pytest.mark.unit
def test_results_dir_defaults_to_the_directory_of_the_database(tmp_path: Path) -> None:
    args = AppArgs(db=tmp_path / "a.db", match_id="m1", youtube_id=VID)

    assert args.results_dir == tmp_path


@pytest.mark.unit
def test_command_defaults_to_the_current_interpreter(tmp_path: Path) -> None:
    command = build_streamlit_command(AppArgs(db=tmp_path / "a.db", match_id="m", youtube_id=VID))

    assert command[0] == sys.executable


@pytest.mark.unit
def test_parse_app_args_roundtrips_the_command_arguments(tmp_path: Path) -> None:
    args = AppArgs(db=tmp_path / "a.db", match_id="m1", youtube_id="-abcDEF_123")
    command = build_streamlit_command(args, python="py")

    assert parse_app_args(command[6:]) == args


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--db", "a.db", "--match-id", "m"],
        ["--db", "a.db", "--youtube-id", VID],
        ["--match-id", "m", "--youtube-id", VID],
        ["--db", "a.db", "--match-id", "", "--youtube-id", VID],
    ],
)
def test_parse_app_args_missing_argument_raises(argv: list[str]) -> None:
    with pytest.raises(ConfigError, match="needs"):
        parse_app_args(argv)


@pytest.mark.unit
def test_parse_app_args_bad_video_id_raises() -> None:
    with pytest.raises(ConfigError, match="11 characters"):
        parse_app_args(["--db", "a.db", "--match-id", "m", "--youtube-id", "bad"])


@pytest.mark.unit
def test_parse_app_args_ignores_foreign_arguments() -> None:
    argv = ["--db", "a.db", "--match-id", "m", "--youtube-id", VID, "--other", "x"]

    assert parse_app_args(argv).match_id == "m"


@pytest.mark.unit
def test_load_events_missing_file_raises_and_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "none.db"

    with pytest.raises(StorageError, match="does not exist"):
        load_events(path, "m")

    assert not path.exists()


@pytest.mark.unit
def test_load_events_file_that_is_not_a_database_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.db"
    path.write_text("this is not a database at all, just text" * 10, encoding="utf-8")

    with pytest.raises(StorageError):
        load_events(path, "m")


@pytest.mark.unit
def test_load_events_returns_current_events_in_time_order_without_deleted(tmp_path: Path) -> None:
    path = tmp_path / "e.db"
    init_db(path)
    with connect(path) as conn:
        ids = add_predictions(
            conn,
            "m",
            "run1",
            [
                NewPrediction(type=EventType.FOUL, t_v=50.0, source="model", half=1, score=0.4),
                NewPrediction(type=EventType.GOAL, t_v=10.0, source="plates", half=1),
                NewPrediction(type=EventType.SHOT, t_v=30.0, source="model"),
            ],
        )
        add_edit(conn, "m", "delete", author="a", target=f"p:{ids[2]}")
        add_edit(conn, "m", "add", author="a", type=EventType.CORNER, t_v=20.0)

    events = load_events(path, "m")

    assert [(e.type, e.t_v) for e in events] == [
        (EventType.GOAL, 10.0),
        (EventType.CORNER, 20.0),
        (EventType.FOUL, 50.0),
    ]
    assert load_events(path, "other") == []


@pytest.mark.unit
def test_parse_app_args_option_without_value_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="invalid viewer argument"):
        parse_app_args(["--db"])


@pytest.mark.unit
def test_video_is_passed_on_and_read_back(tmp_path: Path) -> None:
    args = AppArgs(
        db=tmp_path / "a.db",
        match_id="m1",
        youtube_id=VID,
        match_dir=tmp_path,
        video=tmp_path / "src.mp4",
    )

    command = build_streamlit_command(args, python="py")

    assert command[-2:] == ["--video", str(tmp_path / "src.mp4")]
    assert parse_app_args(command[6:]) == args
