"""Tests of the name text helpers."""

import pytest

from aicoach.plate_names.text import clean_text, name_key, sorted_key


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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("9 ИВАН ПЕТР СЕРГЕЕВИЧ ЛЕБЕДЕВ", "Иван Петр Сергеевич Лебедев"),
        ("14  РОМАН   КОВАЛЬЧУК\n", "Роман Ковальчук"),
        ("|| 7 ~ ИВАНОВ .", "Иванов"),
        ("9 ИВАНОВ-ПЕТРОВ", "Иванов-Петров"),
        ("22 ЕРЛАН ҚОСАНОВ", "Ерлан Қосанов"),
        ("5", ""),
        ("", ""),
        ("   ", ""),
        ("12 Ұ", ""),  # one letter is noise
        ("@@ ## 99", ""),
    ],
)
def test_clean_text_cuts_the_number_and_noise(raw: str, cleaned: str) -> None:
    assert clean_text(raw) == cleaned
