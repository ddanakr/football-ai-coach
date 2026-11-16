"""The listener's check file: parsing, including junk."""

from pathlib import Path

import pytest

from aicoach.errors import TranscriptionError
from aicoach.speech.check_file import load_check_file, parse_check_text

SAMPLE = """﻿noise before the first header

00:15:35 - 00:17:14 kz

Қала
соңғы кезең


00:17:19-00:18:24 mixed
быстрая атака
ещё один навес
"""


@pytest.mark.unit
def test_parse_reads_headers_phrases_and_languages() -> None:
    first, second = parse_check_text(SAMPLE)

    assert (first.index, first.start_s, first.end_s, first.language) == (1, 935.0, 1034.0, "kk")
    assert first.phrases == ("Қала", "соңғы кезең")
    assert (second.start_s, second.end_s, second.language) == (1039.0, 1104.0, "mixed")
    assert second.length_s == 65.0
    assert len(second.phrases) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "",
        "only words\nno header\n",
        "00:10:00 - 00:09:00 ru\nслово\n",
        "00:10:00 - 00:10:00 ru\n",
        "00:10:00 - 00:11:00 en\n",
        "00:10:00 - 00:11:00\n",
        "00:75:00 - 00:76:00 ru\n",
    ],
)
def test_parse_rejects_missing_or_bad_headers(text: str) -> None:
    with pytest.raises(TranscriptionError):
        parse_check_text(text)


@pytest.mark.unit
def test_fragment_without_words_is_kept_with_no_phrases() -> None:
    (fragment,) = parse_check_text("00:00:10 - 00:00:20 ru\n")

    assert fragment.phrases == ()


@pytest.mark.unit
def test_load_reads_a_file_and_reports_unreadable_ones(tmp_path: Path) -> None:
    good = tmp_path / "check.txt"
    good.write_text("00:00:10 - 00:00:20 ru\nслово\n", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00\x80\x81")

    assert load_check_file(good)[0].phrases == ("слово",)
    with pytest.raises(TranscriptionError):
        load_check_file(bad)
    with pytest.raises(TranscriptionError):
        load_check_file(tmp_path / "missing.txt")
