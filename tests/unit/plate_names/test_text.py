"""Tests of the name text helpers."""

import pytest

from aicoach.plate_names.text import clean_text


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
