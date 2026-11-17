"""Matching of names: folding, words, whole names against the lineup, mentions in free text."""

import pytest

from aicoach.namematch import (
    NameMatch,
    find_mentions,
    find_player,
    name_key,
    name_matches,
    same_word,
    similarity,
    sorted_key,
    words,
)
from aicoach.plate_names.lineup import Player


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("заруцкого", "заруцкий", True),
        ("тестова", "тестов", True),
        ("тестов", "тестов", True),
        ("петров", "петренко", False),
        ("петров", "петровский", False),
        ("иванов", "иванчук", False),
        ("ким", "кима", False),  # short words must be equal
        ("аскар", "аскара", True),
        ("", "", True),
    ],
)
def test_same_word_ignores_case_endings_only(a: str, b: str, expected: bool) -> None:
    assert same_word(a, b) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "name", "expected"),
    [
        ("Тестова", "Тестов Иван", True),
        ("иван тестов", "Тестов Иван", True),
        ("ӘНУАРБЕКОВ", "Ануарбеков Нурлан", True),
        ("Иван", "Тестов Иван", True),
        ("Пробный Иван", "Тестов Иван", False),
        ("", "Тестов Иван", False),
        ("   ", "Тестов Иван", False),
        ("12", "Тестов Иван", False),
    ],
)
def test_name_matches_needs_every_query_word(query: str, name: str, expected: bool) -> None:
    assert name_matches(query, name) is expected


@pytest.mark.unit
def test_words_of_punctuation_only_is_empty() -> None:
    assert words("!!! --- 123") == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("ҚҰДАЙБЕРГЕНОВ", "кудаибергенов"),  # Kazakh letters fold to the Russian ones
        ("Ёлкин Ғалым", "елкин галым"),
        ("Mиронов", "миронов"),  # a Latin M inside Cyrillic
        ("Иванов-Петров", "иванов петров"),
        ("  Лебедев   Роман  ", "лебедев роман"),
        ("9 Лебедев", "лебедев"),
        ("Д'Артаньян", "д артаньян"),
        ("", ""),
        ("123", ""),
        ("--  ..", ""),
    ],
)
def test_name_key_folds_case_letters_and_punctuation(text: str, key: str) -> None:
    assert name_key(text) == key


@pytest.mark.unit
def test_sorted_key_ignores_the_order_of_the_words() -> None:
    assert sorted_key("АЙДАР НУРЛАНОВ") == sorted_key("Нурланов Айдар")


HOME = Player("home", "Лебедев Роман", 9)
HOME2 = Player("home", "Стрельцов Антон Павлович", 14)
AWAY = Player("away", "Нурланов Айдар", 10)
AWAY2 = Player("away", "Лебедев Артур", 7)
PLAYERS = [HOME, HOME2, AWAY, AWAY2]


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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("letter", "folded"),
    [
        ("ё", "е"),
        ("ә", "а"),
        ("ғ", "г"),
        ("қ", "к"),
        ("ң", "н"),
        ("ө", "о"),
        ("ұ", "у"),
        ("ү", "у"),
        ("һ", "х"),
        ("і", "и"),
    ],
)
def test_name_key_folds_every_kazakh_letter_in_both_cases(letter: str, folded: str) -> None:
    assert name_key(letter) == folded
    assert name_key(letter.upper()) == folded


@pytest.mark.unit
def test_name_key_makes_plate_protocol_and_asr_spellings_meet() -> None:
    assert name_key("ҚҰДАЙБЕРГЕНОВ") == name_key("Кудайбергенов") == "кудаибергенов"


ALIASED = Player("home", "Мельников Артём Иванович", 17, aliases=("Мельня", "Большой Артём"))
OTHER = Player("away", "Тестов Иван", 5)
SQUAD = [ALIASED, OTHER, HOME, AWAY]


@pytest.mark.unit
def test_find_player_matches_an_alias_with_the_lineup_spelling_and_says_so() -> None:
    match, best = find_player("МЕЛЬНЯ", SQUAD, "home", 0.88)

    assert match is not None
    assert match.player is ALIASED
    assert match.via == "alias"
    assert match.source == "МЕЛЬНЯ"
    assert match.ref.name == "Мельников Артём Иванович"
    assert (match.ref.side, match.ref.number) == ("home", 17)
    assert best == pytest.approx(1.0)


@pytest.mark.unit
def test_find_player_alias_of_the_other_side_is_not_used() -> None:
    match, _ = find_player("Мельня", SQUAD, "away", 0.88)

    assert match is None


@pytest.mark.unit
def test_find_player_without_a_side_looks_at_everybody() -> None:
    match, _ = find_player("Айдар Нурланов", SQUAD, None, 0.88)

    assert match is not None
    assert match.player is AWAY


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "surname"),
    [
        ("Тестова пробил мимо", "Тестов Иван"),  # case ending
        ("удар ТЕСТОВ!", "Тестов Иван"),
        ("мяч у Тестову", "Тестов Иван"),
    ],
)
def test_find_mentions_finds_the_surname_in_free_text(text: str, surname: str) -> None:
    found = find_mentions(text, SQUAD, 0.88)

    assert [m.player.name for m in found] == [surname]
    assert found[0].via == "name"


@pytest.mark.unit
def test_find_mentions_keeps_the_word_as_said_and_gives_the_lineup_spelling() -> None:
    (found,) = find_mentions("Передача на Мельникова.", SQUAD, 0.88)

    assert found.source == "Мельникова"
    assert found.ref.name == "Мельников Артём Иванович"
    assert found.ref.number == 17


@pytest.mark.unit
def test_find_mentions_alias_nickname_finds_the_player() -> None:
    (found,) = find_mentions("и снова Мельня на краю штрафной", SQUAD, 0.88)

    assert found.player is ALIASED
    assert found.via == "alias"
    assert found.source == "Мельня"


@pytest.mark.unit
def test_find_mentions_multiword_alias_must_be_said_in_full_and_wins_over_a_word() -> None:
    (found,) = find_mentions("Большой Артём сыграл", SQUAD, 0.88)
    assert found.via == "alias"
    assert found.source == "Большой Артём"
    # "Артём" alone is a given name, not a surname or an alias: nobody.
    assert find_mentions("Артём сыграл", SQUAD, 0.88) == []


@pytest.mark.unit
def test_find_mentions_typo_of_a_long_surname_is_found_a_short_one_must_be_equal() -> None:
    assert [m.player for m in find_mentions("Нурланоу", SQUAD, 0.88)] == [AWAY]
    short = [Player("home", "Ли Вэй", 3)]
    assert [m.player for m in find_mentions("Ли", short, 0.88)] == short
    assert find_mentions("Лис", short, 0.88) == []


@pytest.mark.unit
def test_find_mentions_one_surname_on_both_sides_is_not_guessed() -> None:
    assert find_mentions("Лебедев на подборе", PLAYERS, 0.88) == []


@pytest.mark.unit
def test_find_mentions_side_filter_resolves_the_shared_surname() -> None:
    (found,) = find_mentions("Лебедев на подборе", PLAYERS, 0.88, side="away")

    assert found.player is AWAY2


@pytest.mark.unit
def test_find_mentions_two_players_in_the_order_of_the_text() -> None:
    found = find_mentions("Нурланов отдал Мельникову, тот сам", SQUAD, 0.88)

    assert [m.player for m in found] == [AWAY, ALIASED]


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "   ", "12 34", "!!! ---"])
def test_find_mentions_text_without_words_gives_nothing(text: str) -> None:
    assert find_mentions(text, SQUAD, 0.88) == []


@pytest.mark.unit
def test_find_mentions_empty_lineup_gives_nothing() -> None:
    assert find_mentions("Тестов", [], 0.88) == []


@pytest.mark.unit
def test_name_match_ref_is_a_plain_reference() -> None:
    match = NameMatch(HOME, 1.0, "Лебедев")

    assert (match.ref.side, match.ref.number, match.ref.name) == ("home", 9, "Лебедев Роман")
