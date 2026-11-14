"""Fuzzy name comparison."""

import pytest

from aicoach.agent.names import name_matches, same_word, words


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
