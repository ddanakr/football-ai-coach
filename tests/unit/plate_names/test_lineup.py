"""Tests of the lineup file and the search of a name among the players of one side."""

from pathlib import Path

import pytest

from aicoach.errors import PlatesError
from aicoach.plate_names.lineup import Player, find_player, load_lineup, parse_lineup, similarity
from tests.fixtures.plate_names_synth import LINEUP_CSV

HOME = Player("home", "Лебедев Роман", 9)
HOME2 = Player("home", "Стрельцов Антон Павлович", 14)
AWAY = Player("away", "Нурланов Айдар", 10)
AWAY2 = Player("away", "Лебедев Артур", 7)
PLAYERS = [HOME, HOME2, AWAY, AWAY2]


@pytest.mark.unit
def test_parse_lineup_reads_the_columns_of_the_spec() -> None:
    players = parse_lineup(LINEUP_CSV.encode("utf-8"))

    assert len(players) == 6
    assert players[0] == Player("home", "Лебедев Роман", 9)
    assert players[-1] == Player("away", "Қосанов Ерлан", 21)


@pytest.mark.unit
def test_parse_lineup_accepts_bom_extra_spaces_and_semicolons() -> None:
    text = (
        "team ; Side ;number;name\r\n"
        " Север ; HOME ; 9 ; Лебедев   Роман \r\n"
        "\r\n"
        "Юг;away;x;Нурланов Айдар\r\n"
    )

    players = parse_lineup(("﻿" + text).encode("utf-8"))

    assert players == [Player("home", "Лебедев Роман", 9), Player("away", "Нурланов Айдар", None)]


@pytest.mark.unit
@pytest.mark.parametrize("cell", ["9" * 5, "9" * 5000, "-3", "1.5"])
def test_parse_lineup_garbage_shirt_number_is_none_not_an_error(cell: str) -> None:
    players = parse_lineup(f"side,number,name\nhome,{cell},Лебедев Роман\n".encode())

    assert players == [Player("home", "Лебедев Роман", None)]


@pytest.mark.unit
def test_parse_lineup_tolerates_short_rows_and_other_columns_missing() -> None:
    players = parse_lineup("side,name\nhome,Лебедев Роман\naway,Нурланов Айдар,extra".encode())

    assert [p.name for p in players] == ["Лебедев Роман", "Нурланов Айдар"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "empty"),
        (b"side,name\n", "no players"),
        (b"name\n\xd0\x9b\n", "no column side"),
        (b"side\nhome\n", "no column name"),
        ("side,name\nleft,Лебедев Роман\n".encode(), "line 2: side must be home or away"),
        ("side,name\nhome,Лебедев Роман\naway,  \n".encode(), "line 3: the name is empty"),
        ("side,name\nhome,Лебедев Роман\n,Нурланов\n".encode(), "line 3: side"),
        ("side,name\nhome,Лебедев\n".encode("cp1251"), "not UTF-8"),
        (b"\xff\xfe\x00\x01", "not UTF-8"),
    ],
)
def test_parse_lineup_broken_file_raises_a_plates_error(data: bytes, message: str) -> None:
    with pytest.raises(PlatesError, match=message):
        parse_lineup(data)


@pytest.mark.unit
def test_parse_lineup_a_field_over_the_csv_limit_is_a_plates_error_not_a_csv_error() -> None:
    with pytest.raises(PlatesError, match="not a CSV"):
        parse_lineup(b"side,name\nhome," + b"x" * 200_000 + b"\n")


@pytest.mark.unit
def test_parse_lineup_error_does_not_quote_the_content() -> None:
    with pytest.raises(PlatesError) as caught:
        parse_lineup("side,name\nleft,СекретнаяФамилия\n".encode())

    assert "СекретнаяФамилия" not in str(caught.value)


@pytest.mark.unit
def test_load_lineup_reads_a_file_and_names_a_missing_one(tmp_path: Path) -> None:
    path = tmp_path / "lineups.csv"
    path.write_text(LINEUP_CSV, encoding="utf-8-sig")

    assert len(load_lineup(path)) == 6
    with pytest.raises(PlatesError, match="cannot read"):
        load_lineup(tmp_path / "none.csv")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Лебедев Роман", "ЛЕБЕДЕВ РОМАН", 1.0),
        ("Роман Лебедев", "Лебедев Роман", 1.0),  # the plate puts the given name first
        ("", "Лебедев Роман", 0.0),
        ("Лебедев Роман", "", 0.0),
        ("123", "Лебедев Роман", 0.0),
    ],
)
def test_similarity_ignores_case_and_word_order(a: str, b: str, expected: float) -> None:
    assert similarity(a, b) == pytest.approx(expected)


@pytest.mark.unit
def test_find_player_exact_name_in_the_plate_order() -> None:
    match, best = find_player("Роман Лебедев", PLAYERS, "home", 0.88)

    assert match is not None
    assert match.player == HOME
    assert match.similarity == pytest.approx(1.0)
    assert best == pytest.approx(1.0)


@pytest.mark.unit
def test_find_player_one_wrong_letter_still_matches() -> None:
    match, best = find_player("Айдар Нурлонов", PLAYERS, "away", 0.88)

    assert match is not None
    assert match.player == AWAY
    assert 0.88 <= match.similarity < 1.0
    assert best == match.similarity


@pytest.mark.unit
def test_find_player_below_the_threshold_is_not_recognized_and_reports_the_best_score() -> None:
    match, best = find_player("Ковальчук Игорь", PLAYERS, "home", 0.88)

    assert match is None
    assert best is not None
    assert best < 0.88


@pytest.mark.unit
def test_find_player_never_picks_a_player_of_the_other_side() -> None:
    match, best = find_player("Айдар Нурланов", PLAYERS, "home", 0.88)

    assert match is None  # the exact name belongs to the away side
    assert best is not None
    assert best < 0.88


@pytest.mark.unit
def test_find_player_same_surname_on_both_sides_takes_the_given_side() -> None:
    home, _ = find_player("Роман Лебедев", PLAYERS, "home", 0.88)
    away, _ = find_player("Артур Лебедев", PLAYERS, "away", 0.88)

    assert home is not None
    assert away is not None
    assert (home.player, away.player) == (HOME, AWAY2)


@pytest.mark.unit
def test_find_player_two_players_of_one_side_with_the_same_name_is_ambiguous() -> None:
    twins = [Player("home", "Лебедев Роман", 9), Player("home", "Роман Лебедев", 19)]

    match, best = find_player("Лебедев Роман", twins, "home", 0.88)

    assert match is None
    assert best == pytest.approx(1.0)


@pytest.mark.unit
def test_find_player_listed_twice_with_one_name_is_not_ambiguous() -> None:
    doubled = [HOME, Player("home", HOME.name, 9)]

    match, _ = find_player("Лебедев Роман", doubled, "home", 0.88)

    assert match is not None


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "12", "Ұ"])
def test_find_player_text_without_letters_is_not_recognized(text: str) -> None:
    match, best = find_player(text, PLAYERS, "home", 0.88)

    assert match is None
    assert best is not None
    assert best < 0.88


@pytest.mark.unit
def test_find_player_side_without_players_compares_nothing() -> None:
    assert find_player("Лебедев Роман", [HOME], "away", 0.88) == (None, None)
    assert find_player("Лебедев Роман", [], "home", 0.88) == (None, None)
