"""Check of an answer before the user sees it (D-061).

An answer is accepted only if
- every `[E:<id>]` link names an event the tools returned in this very conversation and which
  still exists in `events_current`;
- every number written with digits (`38-й`, `45+2`, `3 гола`) is a number the tools returned
  (or took as an argument) in this question, or one from the question text;
- it contains nothing that looks like a video time (`1:02:03`, `12:34`, "на 1234 секунде"),
  a web address, or "minute of the video";
- every capitalized word in the middle of a sentence (a surname, a team) is a name the tools
  returned or a team name.

The last rule is a heuristic: it cannot tell an invented surname from a rare common noun in
capitals. It errs on the strict side; a wrongly rejected answer is retried once.
"""

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from aicoach.namematch import name_key, same_word, words

LINK = re.compile(r"\[E:([^\]\s]*)\]")
# `p:` a prediction, `e:` an added event, `s:` a stretch of the commentary (D-070).
EVENT_ID = re.compile(r"[pes]:\d+")
_ANY_LINK_LIKE = re.compile(r"\[E:[^\]]*\]")
_TIMECODES = (
    re.compile(r"(?<![\d:])\d{1,2}:\d{2}:\d{2}(?![\d:])"),  # 1:02:03
    re.compile(r"(?<![\d:])\d{1,3}:[0-5]\d(?![\d:])"),  # 12:34 (a score such as 2:1 has 1 digit)
    re.compile(r"\d+\s*(?:-\s*[а-яa-z]{1,2}\s+)?(?:сек\w*|sec\w*)", re.IGNORECASE),
    re.compile(r"(?:мин\w*|minute\w*)\s+видео|видео\s*[:\-]?\s*\d", re.IGNORECASE),
)
# Addresses, and the Markdown forms that make a link or an image out of text.
_URL = re.compile(
    r"https?://|www\.|youtu\.?be|\bt=\d|\]\(|!\[|//[a-z0-9.-]+\.[a-z]{2,}|\b[a-z0-9-]+\.(?:com|net|org|ru|kz|io|info|me|ly|xyz)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W\d_][\w'’-]*", re.UNICODE)
_SENTENCE_START = frozenset(".!?\n")
_NOT_A_WORD_START = ' \t*_>#-•–—"«“(['
_MIN_NAME_LETTERS = 3
# Capitalized words that are not names of players or teams and appear in answers.
ALLOWED_WORDS = frozenset({"кпл", "казахстан", "премьер", "лига", "var", "вар", "xg"})
MAX_LISTED = 5
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class Violation:
    """One reason to reject an answer. `code` is stable (tests, journal), `detail` is for the
    model's retry message."""

    code: str
    detail: str


def _strip_links(text: str) -> str:
    return _ANY_LINK_LIKE.sub(" ", text)


def _link_violations(
    text: str, seen_ids: Collection[str], existing_ids: Collection[str]
) -> list[Violation]:
    found: list[Violation] = []
    for raw in _ANY_LINK_LIKE.findall(text):
        match = LINK.fullmatch(raw)
        event_id = match.group(1) if match else ""
        if not EVENT_ID.fullmatch(event_id):
            found.append(Violation("bad_link", f"ссылка {raw!r} имеет неверный вид"))
        elif event_id not in seen_ids:
            found.append(
                Violation("id_not_from_tools", f"{event_id}: этот id инструменты не возвращали")
            )
        elif event_id not in existing_ids:
            found.append(Violation("id_not_in_events", f"{event_id}: такого события больше нет"))
    return found


def _time_violations(text: str) -> list[Violation]:
    found: list[Violation] = []
    for pattern in _TIMECODES:
        found.extend(
            Violation("timecode", f"похоже на время: {m.group(0)!r}")
            for m in pattern.finditer(text)
        )
    found.extend(Violation("url", f"адрес или ссылка: {m.group(0)!r}") for m in _URL.finditer(text))
    return found


def _starts_sentence(text: str, start: int) -> bool:
    """The word at `start` opens a sentence, a line or a list item (capitals are normal there)."""
    before = text[:start].rstrip(_NOT_A_WORD_START)  # a line break is not stripped: it stays
    return not before or before[-1] in _SENTENCE_START


def _is_allowed(token: str, allowed: Collection[str]) -> bool:
    """A token is allowed when every word of it is: a hyphenated token ("Премьер-лиге") is
    checked word by word, like the names it may repeat (the key reads a hyphen as a space)."""
    keys = name_key(token).split()
    if not keys or token.casefold() in ALLOWED_WORDS:
        return True
    known = [word for name in allowed for word in words(name)]
    return all(k in ALLOWED_WORDS or any(same_word(k, w) for w in known) for k in keys)


def _name_violations(text: str, allowed_names: Collection[str]) -> list[Violation]:
    found: list[Violation] = []
    reported: set[str] = set()
    for match in _WORD.finditer(text):
        token = match.group(0)
        letters = sum(ch.isalpha() for ch in token)
        capitalized = token[0].isupper() and letters >= _MIN_NAME_LETTERS
        if not capitalized or _starts_sentence(text, match.start()):
            continue
        if _is_allowed(token, allowed_names) or token.casefold() in reported:
            continue
        reported.add(token.casefold())
        found.append(Violation("unknown_name", f"имя {token!r} не приходило из инструментов"))
    return found


def _without_times(text: str) -> str:
    for pattern in _TIMECODES:
        text = pattern.sub(" ", text)
    return text


def numbers_in(text: str) -> set[float]:
    """The numbers written with digits in a text (`45+2` gives 45 and 2, `0,95` gives 0.95)."""
    return {float(m.group(0).replace(",", ".")) for m in _NUMBER.finditer(text)}


def _number_violations(text: str, allowed: Iterable[float]) -> list[Violation]:
    allowed_set = set(allowed)
    found: list[Violation] = []
    reported: set[float] = set()
    for match in _NUMBER.finditer(text):
        shown = match.group(0)
        value = float(shown.replace(",", "."))
        if value in allowed_set or value in reported:
            continue
        reported.add(value)
        found.append(Violation("unknown_number", f"число {shown} не приходило из инструментов"))
    return found


def find_violations(
    text: str,
    *,
    seen_ids: Collection[str],
    existing_ids: Collection[str],
    allowed_names: Collection[str],
    allowed_numbers: Iterable[float],
) -> list[Violation]:
    """All reasons the answer cannot be shown; empty means it passes.

    Args:
        text: The model's answer.
        seen_ids: Event ids the tools returned in this conversation.
        existing_ids: Ids in `events_current` now and of the commentary segments.
        allowed_names: Names the tools returned plus the team names (full names, any case).
        allowed_numbers: Numbers of the tool results and arguments of this question, and the
            numbers of the question text. Numbers spelled as words are not checked.
    """
    plain = _strip_links(text)
    return [
        *_link_violations(text, seen_ids, existing_ids),
        *_time_violations(plain),
        # A time already counts as a time violation: its digits are not reported a second time.
        *_number_violations(_without_times(plain), allowed_numbers),
        *_name_violations(plain, allowed_names),
    ]


def describe(violations: list[Violation]) -> str:
    """The retry message for the model: what was wrong and how to fix it."""
    listed = "; ".join(v.detail for v in violations[:MAX_LISTED])
    hint = ""
    if any(v.code == "unknown_number" for v in violations):
        # The usual cause is a count the model made itself (per half, per team): the tool can
        # return exactly that number if it is asked with the right filter.
        hint = (
            " Число, которого нет в результатах, получи у инструмента: вызови query_events с "
            "нужными фильтрами (half, side, type) и назови его поле total; или не называй число."
        )
    return (
        f"Ответ отклонён проверкой: {listed}. Перепиши ответ: ссылки [E:<id>] только на id из "
        "результатов инструментов этого разговора, без времени, секунд и адресов, фамилии "
        "и числа (цифрами) только из результатов инструментов. Если данных нет, так и скажи." + hint
    )
