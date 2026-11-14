"""The answer validator: links, time-like text, names."""

import pytest

from aicoach.agent.validator import Violation, describe, find_violations

SEEN = {"p:2", "p:6"}
EXISTING = {"p:2", "p:6", "p:7"}
NAMES = {"Тестов Иван", "Альфа", "Бета"}


def check(text: str) -> list[Violation]:
    return find_violations(text, seen_ids=SEEN, existing_ids=EXISTING, allowed_names=NAMES)


def codes(text: str) -> list[str]:
    return [v.code for v in check(text)]


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "Система нашла два гола: [E:p:2] и [E:p:6].",
        "Не знаю: система не нашла такого события.",
        "Гол забил Тестов Иван [E:p:2].",
        "Гол забил Тестова [E:p:2].",  # a declined surname of a returned name
        "Счёт на табло 2:1, система нашла 3 удара.",  # a score is not a time
        "Команда не определена. Игрок Иван не назван.",
        "Матч Альфа против Бета: на 23-й минуте гол [E:p:2].",
    ],
)
def test_correct_answers_pass(text: str) -> None:
    assert check(text) == []


@pytest.mark.unit
def test_link_to_an_id_the_tools_did_not_return_is_rejected() -> None:
    # p:7 exists in the database, but the tools did not hand it out in this dialog.
    assert codes("Удар [E:p:7].") == ["id_not_from_tools"]


@pytest.mark.unit
def test_link_to_an_unknown_event_is_rejected() -> None:
    assert codes("Гол [E:p:99].") == ["id_not_from_tools"]


@pytest.mark.unit
def test_link_to_an_event_that_disappeared_is_rejected() -> None:
    result = find_violations(
        "Гол [E:p:2].", seen_ids=SEEN, existing_ids={"p:6"}, allowed_names=NAMES
    )

    assert [v.code for v in result] == ["id_not_in_events"]


@pytest.mark.unit
@pytest.mark.parametrize("link", ["[E:12]", "[E:]", "[E:p:x]", "[E:p:2 ]", "[E:p: 2]"])
def test_malformed_link_is_rejected(link: str) -> None:
    assert "bad_link" in codes(f"Гол {link}.")


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "Гол на 12:34 видео.",
        "Гол в 1:02:03.",
        "Гол на 1234 секунде.",
        "Гол на 1234-й секунде видео.",
        "Через 5 секунд после удара.",
        "Это 38 минута видео.",
        "Видео: 2300.",
    ],
)
def test_text_that_looks_like_a_video_time_is_rejected(text: str) -> None:
    assert "timecode" in codes(text)


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    ["Смотри https://youtu.be/abc", "www.youtube.com", "ссылка ?t=1234s"],
)
def test_addresses_are_rejected(text: str) -> None:
    assert "url" in codes(text)


@pytest.mark.unit
def test_a_game_minute_and_a_half_are_allowed() -> None:
    assert check("Во 2-м тайме, на 81-й минуте, команда Бета забила [E:p:6].") == []


@pytest.mark.unit
def test_surname_that_the_tools_did_not_return_is_rejected() -> None:
    result = check("Гол забил Выдумов [E:p:2].")

    assert [(v.code, "Выдумов" in v.detail) for v in result] == [("unknown_name", True)]


@pytest.mark.unit
def test_surname_of_a_player_known_to_the_system_but_not_returned_is_rejected() -> None:
    assert "unknown_name" in codes("Угловой подавал Образцов [E:p:6].")


@pytest.mark.unit
def test_an_invented_surname_at_the_start_of_a_sentence_is_a_known_gap() -> None:
    # Capitals at the start of a sentence are normal, so the heuristic cannot see a name there.
    assert check("Выдумов забил гол [E:p:2].") == []


@pytest.mark.unit
def test_a_name_in_the_middle_of_a_list_line_is_checked() -> None:
    assert "unknown_name" in codes("Список:\n- Гол — Выдумов [E:p:2]")


@pytest.mark.unit
def test_each_unknown_name_is_reported_once() -> None:
    assert codes("Гол забил Выдумов, потом снова Выдумов.") == ["unknown_name"]


@pytest.mark.unit
def test_kazakh_letters_in_a_returned_name_are_matched_loosely() -> None:
    result = find_violations(
        "Гол забил Нурланов [E:p:6].",
        seen_ids=SEEN,
        existing_ids=EXISTING,
        allowed_names={"Нұрланов Ұлан"},
    )

    assert result == []


@pytest.mark.unit
def test_all_capitals_words_are_not_names() -> None:
    assert check("Гол на 5-й минуте, но потом VAR отменил его.") == []


@pytest.mark.unit
def test_describe_lists_the_reasons_and_tells_how_to_fix() -> None:
    text = describe(check("Гол на 12:34 [E:p:99] забил Выдумов."))

    assert "12:34" in text
    assert "p:99" in text
    assert "Выдумов" in text
    assert "[E:<id>]" in text


@pytest.mark.unit
def test_a_name_after_a_colon_or_a_semicolon_is_checked() -> None:
    assert "unknown_name" in codes("Автор гола: Выдумов [E:p:2]")
    assert "unknown_name" in codes("Гол; Выдумов [E:p:2]")


@pytest.mark.unit
def test_a_surname_written_in_capitals_is_checked() -> None:
    assert "unknown_name" in codes("Гол забил ВЫДУМОВ [E:p:2].")
    assert check("Гол забил ТЕСТОВ [E:p:2].") == []


@pytest.mark.unit
@pytest.mark.parametrize("invented", ["Тестеров", "Тестилин"])
def test_a_different_surname_with_the_same_beginning_is_not_the_returned_one(
    invented: str,
) -> None:
    assert "unknown_name" in codes(f"Гол забил {invented} [E:p:2].")


@pytest.mark.unit
def test_the_preposition_s_after_a_number_is_not_a_time() -> None:
    assert check("Система нашла 3 гола, один из них с пенальти.") == []
    assert check("Забито 1 с игры [E:p:2].") == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    ["Вот ![картинка](//evil.example/x.png)", "Смотри [тут](x) подробнее", "см. evil.example.com"],
)
def test_markdown_links_images_and_bare_domains_are_rejected(text: str) -> None:
    assert "url" in codes(text)
