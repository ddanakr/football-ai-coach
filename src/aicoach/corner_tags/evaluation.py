"""Accuracy of the corner tags against the truth file `corner_truth.csv` (D-078).

The person who watched the corners fills one row per corner (the template is made from
`events_current`); values are codes or Russian names, `unclear` or an empty cell where nothing can
be said. The score is per field, never one total: for every field of the six, over the corners
whose truth value is known, the model's answer is a hit, a miss, or `unclear` ("не видно"), and
"unclear" is counted on its own: it is not an error and not a hit. The other five fields are
scored only on corners the truth says are corners (`is_corner` is not `no`).

The protocol is that of the project (D-008): the question is tuned on the first half only; the
second half is a check made once with an unchanged question (`--final`).
"""

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aicoach.corner_tags.schemas import FIELDS, IS_CORNER, UNCLEAR, CornerTag, normalize_value
from aicoach.errors import EvalError

BEGIN_MARKER: Final = "<!-- corner-eval:begin -->"
END_MARKER: Final = "<!-- corner-eval:end -->"
REQUIRED_COLUMNS: Final = ("id", *FIELDS)


@dataclass(frozen=True)
class TruthRow:
    """One corner of the truth file; a field is a code, or `None` when the cell is empty."""

    event_id: str
    half: int | None
    values: Mapping[str, str | None]


@dataclass(frozen=True)
class FieldScore:
    """The score of one field over one set of corners.

    `n_truth`: corners with a known truth value (not empty, not `unclear`). Of them the model
    answered right (`hits`), wrong (`misses`) or said `unclear` (`model_unclear`); a corner the
    model has no tags for is `untagged`. `skipped` counts corners left out because the truth
    value is empty or `unclear`, or the truth says it is no corner.
    """

    field: str
    n_truth: int
    hits: int
    misses: int
    model_unclear: int
    untagged: int
    skipped: int

    @property
    def answered(self) -> int:
        """Corners where the model gave a definite value."""
        return self.hits + self.misses

    @property
    def accuracy_of_answers(self) -> float | None:
        """Hits among the definite answers; `None` when there were none."""
        return None if self.answered == 0 else self.hits / self.answered

    @property
    def accuracy_of_all(self) -> float | None:
        """Hits among all corners with a known truth value (an `unclear` answer counts against
        it); `None` when there are none."""
        return None if self.n_truth == 0 else self.hits / self.n_truth


def _delimiter(text: str) -> str:
    first = next((line for line in text.splitlines() if not line.startswith("#")), "")
    return ";" if first.count(";") > first.count(",") else ","


def parse_truth(data: bytes) -> list[TruthRow]:
    """The rows of a truth file (UTF-8 with or without BOM, `,` or `;`, lines starting with `#`
    are comments).

    Raises:
        EvalError: Not UTF-8, a required column is missing, an id is empty or repeated, a value
            is not allowed for its field (the message names the row and the column).
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as err:
        msg = "the truth file is not UTF-8"
        raise EvalError(msg) from err
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        msg = "the truth file has no header row"
        raise EvalError(msg)
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=_delimiter(text))
    header = [h.strip().lower() for h in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        msg = f"the truth file lacks the columns: {', '.join(missing)}"
        raise EvalError(msg)
    rows: list[TruthRow] = []
    seen: set[str] = set()
    for number, raw in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        rows.append(_row(row, number, seen))
    return rows


def _row(row: Mapping[str, str], number: int, seen: set[str]) -> TruthRow:
    event_id = row["id"]
    if not event_id or event_id in seen:
        msg = f"truth file row {number}: empty or repeated id {event_id!r}"
        raise EvalError(msg)
    seen.add(event_id)
    values: dict[str, str | None] = {}
    for name in FIELDS:
        cell = row[name]
        code = normalize_value(name, cell)
        if cell and code is None:
            msg = f"truth file row {number} ({event_id}), column {name}: {cell!r} is not allowed"
            raise EvalError(msg)
        values[name] = code
    half_text = row.get("half", "")
    half = int(half_text) if half_text in ("1", "2") else None
    return TruthRow(event_id, half, values)


def load_truth(path: Path) -> list[TruthRow]:
    """`parse_truth` of a file.

    Raises:
        EvalError: The file cannot be read or is invalid.
    """
    try:
        data = path.read_bytes()
    except OSError as err:
        msg = f"cannot read the truth file {path}: {err}"
        raise EvalError(msg) from err
    return parse_truth(data)


def filled_rows(rows: Sequence[TruthRow]) -> int:
    """How many rows have any of the six fields filled in."""
    return sum(any(v is not None for v in r.values.values()) for r in rows)


def score_field(name: str, rows: Sequence[TruthRow], tags: Mapping[str, CornerTag]) -> FieldScore:
    """The score of one field over `rows` (see `FieldScore` for what is counted)."""
    hits = misses = model_unclear = untagged = skipped = 0
    for row in rows:
        truth = row.values[name]
        not_a_corner = name != IS_CORNER and row.values[IS_CORNER] == "no"
        if truth is None or truth == UNCLEAR or not_a_corner:
            skipped += 1
            continue
        tag = tags.get(row.event_id)
        if tag is None:
            untagged += 1
            continue
        answer = str(getattr(tag.fields, name))
        if answer == UNCLEAR:
            model_unclear += 1
        elif answer == truth:
            hits += 1
        else:
            misses += 1
    return FieldScore(
        name,
        hits + misses + model_unclear + untagged,
        hits,
        misses,
        model_unclear,
        untagged,
        skipped,
    )


def score_rows(rows: Sequence[TruthRow], tags: Mapping[str, CornerTag]) -> list[FieldScore]:
    """The score of every field over `rows`."""
    return [score_field(name, rows, tags) for name in FIELDS]


def rows_of_half(
    rows: Sequence[TruthRow], tags: Mapping[str, CornerTag], half: int
) -> list[TruthRow]:
    """The corners of one half: by the truth file's `half` column, else by the tag's half."""
    return [r for r in rows if _half_of(r, tags) == half]


def _half_of(row: TruthRow, tags: Mapping[str, CornerTag]) -> int | None:
    if row.half is not None:
        return row.half
    tag = tags.get(row.event_id)
    return None if tag is None else tag.half


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def score_table(scores: Sequence[FieldScore]) -> str:
    """A Markdown table of the scores, one row per field."""
    lines = [
        "| Поле | Эталон (n) | Совпало | Не совпало | Модель «не видно» | Нет тегов "
        "| Точность среди ответов | Точность от всех |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {s.field} | {s.n_truth} | {s.hits} | {s.misses} | {s.model_unclear} "
        f"| {s.untagged} | {_percent(s.accuracy_of_answers)} | {_percent(s.accuracy_of_all)} |"
        for s in scores
    ]
    return "\n".join(lines)


def render_section(
    halves: Mapping[int, Sequence[FieldScore]],
    *,
    n_corners: Mapping[int, int],
    model: str,
    prompt_version: int,
    cost_usd: float | None,
) -> str:
    """The section of `docs/metrics.md` between the markers.

    Args:
        halves: Scores per half that was evaluated (1, and 2 only with `--final`).
        n_corners: The corners of the truth file per half.
        model: The video model of the tags.
        prompt_version: Version of the question.
        cost_usd: What the tagging cost (sum of the stored answers), `None` when unknown.
    """
    cost = "неизвестна" if cost_usd is None else f"{cost_usd:.4f} USD"
    parts = [
        BEGIN_MARKER,
        "## Теги угловых (расширение сверх ТЗ, D-078)",
        "",
        f"Модель: `{model}`, версия запроса {prompt_version}; стоимость разметки всех угловых "
        f"матча {cost}. Эталон заполнен человеком по видео. Это пилот на одном матче: выборка "
        "в единицы угловых, доверительные интервалы очень широки, цифры показывают "
        "направление, а не точность. «Не видно» у модели считается отдельно: это не ошибка и "
        "не попадание. «Точность среди ответов» = совпало / (совпало + не совпало); «от всех» "
        "= совпало / n. Поля кроме `is_corner` считаются только на угловых, которые эталон "
        "признаёт угловыми.",
        "",
    ]
    for half, scores in sorted(halves.items()):
        role = (
            "запрос подбирался на этих угловых"
            if half == 1
            else "проверка, запрос не менялся после подбора"
        )
        parts += [f"### {half}-й тайм (угловых в эталоне: {n_corners.get(half, 0)}; {role})", ""]
        parts += [score_table(scores), ""]
    parts.append(END_MARKER)
    return "\n".join(parts)


def replace_section(document: str, section: str) -> str:
    """The document with the corner section replaced between its markers, or appended after a
    blank line when the markers are not there yet; the text outside is not touched.

    Raises:
        EvalError: Only one of the two markers is present, so the extent is unknown.
    """
    has_begin, has_end = BEGIN_MARKER in document, END_MARKER in document
    if has_begin != has_end:
        msg = f"the document has only one of the markers {BEGIN_MARKER} and {END_MARKER}"
        raise EvalError(msg)
    if not has_begin:
        return document.rstrip("\n") + "\n\n" + section + "\n" if document else section + "\n"
    head, _, rest = document.partition(BEGIN_MARKER)
    _, _, tail = rest.partition(END_MARKER)
    return head + section + tail
