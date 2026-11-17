"""Tests of the lineup file."""

from pathlib import Path

import pytest

from aicoach.errors import PlatesError
from aicoach.plate_names.lineup import Player, load_lineup, parse_lineup
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
def test_parse_lineup_reads_aliases_separated_by_semicolons() -> None:
    text = (
        "side,name,number,aliases\n"
        'home,Мельников Артём,17,"Мельня; Большой  Артём;;Мельня"\n'
        "home,Тестов Иван,5,\n"
    )

    players = parse_lineup(text.encode("utf-8"))

    assert players[0].aliases == ("Мельня", "Большой Артём")  # trimmed, no repeats
    assert players[1].aliases == ()


@pytest.mark.unit
def test_parse_lineup_without_an_aliases_column_has_no_aliases() -> None:
    players = parse_lineup(LINEUP_CSV.encode("utf-8"))

    assert all(p.aliases == () for p in players)
