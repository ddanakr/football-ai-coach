"""Tests of the pure assembly of the names of a plate."""

import pytest

from aicoach.plate_names.assemble import build_plate_names, count_names, read_player
from aicoach.plate_names.lineup import Player
from tests.fixtures.plate_names_synth import make_plate

LINEUP = [
    Player("home", "Лебедев Роман", 9),
    Player("home", "Мирзоев Тимур", 5),
    Player("away", "Нурланов Айдар", 10),
]


@pytest.mark.unit
def test_read_player_with_a_lineup_uses_the_lineup_spelling() -> None:
    player = read_player("player", "9 РОМАН ЛЕБЕДЕВ", "home", LINEUP, 0.88)

    assert (player.status, player.source) == ("matched", "ocr+lineup")
    assert player.name == "Лебедев Роман"
    assert player.raw_text == "9 РОМАН ЛЕБЕДЕВ"
    assert player.similarity == 1.0


@pytest.mark.unit
def test_read_player_one_misread_letter_is_matched_with_a_score_below_one() -> None:
    player = read_player("player", "5 ТИМУР МИРЗОЕВ".replace("О", "Q"), "home", LINEUP, 0.88)

    assert player.status == "matched"
    assert player.name == "Мирзоев Тимур"
    assert player.similarity is not None
    assert player.similarity < 1.0


@pytest.mark.unit
def test_read_player_without_a_lineup_gives_the_cleaned_text_unchecked() -> None:
    player = read_player("player", "9 РОМАН ЛЕБЕДЕВ", "home", None, 0.88)

    assert (player.status, player.source) == ("unchecked", "ocr")
    assert player.name == "Роман Лебедев"
    assert player.similarity is None


@pytest.mark.unit
def test_read_player_with_a_lineup_but_no_side_is_unchecked() -> None:
    player = read_player("player", "9 РОМАН ЛЕБЕДЕВ", None, LINEUP, 0.88)

    assert (player.status, player.name) == ("unchecked", "Роман Лебедев")


@pytest.mark.unit
def test_read_player_below_the_threshold_has_no_name_and_keeps_the_best_score() -> None:
    player = read_player("player", "7 ИГОРЬ КОВАЛЬЧУК", "home", LINEUP, 0.88)

    assert (player.status, player.name, player.source) == ("unrecognized", None, "ocr")
    assert player.similarity is not None
    assert player.similarity < 0.88
    assert player.raw_text == "7 ИГОРЬ КОВАЛЬЧУК"


@pytest.mark.unit
def test_read_player_the_other_side_is_not_chosen() -> None:
    player = read_player("player", "10 АЙДАР НУРЛАНОВ", "home", LINEUP, 0.88)

    assert (player.status, player.name) == ("unrecognized", None)


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["", "  ", "9", "|| ~", "Ұ"])
@pytest.mark.parametrize("lineup", [None, LINEUP])
def test_read_player_text_without_letters_is_unrecognized(
    raw: str, lineup: list[Player] | None
) -> None:
    player = read_player("player", raw, "home", lineup, 0.88)

    assert (player.status, player.name, player.similarity) == ("unrecognized", None, None)


@pytest.mark.unit
def test_build_plate_names_substitution_keeps_both_names_with_their_roles() -> None:
    plate = make_plate("plate_006", "subst", 2844.5, team="away", half=1)

    names = build_plate_names(
        plate,
        {"in": "21 ЕРЛАН ҚОСАНОВ", "out": "10 АЙДАР НУРЛАНОВ"},
        [*LINEUP, Player("away", "Қосанов Ерлан", 21)],
        0.88,
    )

    assert (names.plate_id, names.type, names.t_v, names.half, names.side) == (
        "plate_006",
        "subst",
        2844.5,
        1,
        "away",
    )
    assert [(p.role, p.name) for p in names.players] == [
        ("in", "Қосанов Ерлан"),
        ("out", "Нурланов Айдар"),
    ]


@pytest.mark.unit
def test_build_plate_names_goal_has_one_player() -> None:
    plate = make_plate("plate_001", "goal", 10.0)

    names = build_plate_names(plate, {"player": "9 РОМАН ЛЕБЕДЕВ"}, LINEUP, 0.88)

    assert [(p.role, p.name) for p in names.players] == [("player", "Лебедев Роман")]


@pytest.mark.unit
def test_count_names_counts_every_status() -> None:
    goal = build_plate_names(
        make_plate("a", "goal", 1.0), {"player": "9 РОМАН ЛЕБЕДЕВ"}, LINEUP, 0.88
    )
    subst = build_plate_names(
        make_plate("b", "subst", 2.0),
        {"in": "5 ТИМУР МИРЗОЕВ", "out": "7 ИГОРЬ КОВАЛЬЧУК"},
        LINEUP,
        0.88,
    )
    loose = build_plate_names(
        make_plate("c", "card", 3.0, team=None), {"player": "7 ИВАН"}, LINEUP, 0.88
    )

    stats = count_names([goal, subst, loose])

    assert (stats.n_plates, stats.n_names) == (3, 4)
    assert (stats.n_matched, stats.n_unchecked, stats.n_unrecognized) == (2, 1, 1)


@pytest.mark.unit
def test_count_names_without_plates_is_all_zero() -> None:
    stats = count_names([])

    assert (stats.n_plates, stats.n_names, stats.n_matched) == (0, 0, 0)
