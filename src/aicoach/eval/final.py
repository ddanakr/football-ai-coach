"""Final evaluation tables: systems, edit counts, Gemini shift, agent cost, Markdown.

Pure functions over predictions and reports; reading files is `aicoach.eval.final_run`.
The document holds numbers and shares only: no event times, names, teams or scores.
"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.metrics import EvaluationReport, Metrics, ToleranceResult
from aicoach.eval.schemas import GroundTruthEvent, Prediction

Half = Literal[1, 2]
_EMPTY: Final = "—"
SHIFT_LIMIT_S: Final = 10.0
SHIFT_WINDOW_S: Final = 60.0
_FINAL_HALF: Final = 2


@dataclass(frozen=True)
class SystemRow:
    """One system on one half: the report, or `None` with the reason there is no data."""

    title: str
    report: EvaluationReport | None
    note: str = ""


@dataclass(frozen=True)
class HalfSection:
    """Everything computed for one half; `caveat` is printed under the heading."""

    half: Half
    n_truth: dict[str, int]
    rows: tuple[SystemRow, ...]
    caveat: str


def check_final_allowed(half: int, *, final: bool) -> None:
    """Half 2 is evaluated only with the explicit `final` switch.

    Raises:
        EvalError: `half` is 2 without `final`, or not a half.
    """
    if half not in (1, _FINAL_HALF):
        msg = f"half must be 1 or 2, got {half!r}"
        raise EvalError(msg)
    if half == _FINAL_HALF and not final:
        msg = "half 2 is reserved for the final evaluation: run once with --final"
        raise EvalError(msg)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    """Harmonic mean; `None` if either is undefined, 0 when both are 0."""
    if precision is None or recall is None:
        return None
    total = precision + recall
    return 0.0 if total == 0 else 2 * precision * recall / total


def count_edits(result: ToleranceResult) -> dict[str, tuple[int, int]]:
    """Edits an analyst needs to reach the ground truth: (deletes = FP, adds = FN) per class.

    The key `all` is the micro total.
    """
    counts = {kind.value: (m.fp, m.fn) for kind, m in result.per_class.items()}
    counts["all"] = (result.micro.fp, result.micro.fn)
    return counts


def total_edits(report: EvaluationReport, tolerance_s: float) -> int:
    """Deletes plus adds over all classes at one tolerance."""
    fp, fn = count_edits(report.result_for(tolerance_s))["all"]
    return fp + fn


@dataclass(frozen=True)
class ShiftCheck:
    """Signed time difference `t_pred - t_gt` to the nearest same-class truth event."""

    n_pairs: int
    median_signed_s: float | None
    median_abs_s: float | None
    systematic: bool


def gemini_shift(
    predictions: Sequence[Prediction],
    truth: Sequence[GroundTruthEvent],
    *,
    window_s: float = SHIFT_WINDOW_S,
    limit_s: float = SHIFT_LIMIT_S,
) -> ShiftCheck:
    """Report a systematic offset of the baseline times; nothing is corrected.

    Each prediction is paired with the nearest truth event of its class within `window_s`.
    The offset is systematic when the median signed difference exceeds `limit_s` in absolute
    value.
    """
    diffs: list[float] = []
    for pred in predictions:
        near = [pred.t_v - g.t_v for g in truth if g.type is pred.type]
        near = [d for d in near if abs(d) <= window_s]
        if near:
            diffs.append(min(near, key=abs))
    if not diffs:
        return ShiftCheck(0, None, None, systematic=False)
    signed = statistics.median(diffs)
    return ShiftCheck(
        len(diffs),
        signed,
        statistics.median(abs(d) for d in diffs),
        systematic=abs(signed) > limit_s,
    )


@dataclass(frozen=True)
class AgentCost:
    """Averages over the records of the agent journal."""

    n_records: int
    n_bad_lines: int
    mean_tokens_in: float | None
    mean_tokens_out: float | None
    mean_cost_usd: float | None
    n_with_cost: int
    mean_duration_s: float | None
    first_try_share: float | None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_agent_log(lines: Sequence[str]) -> AgentCost:
    """Average tokens, cost and time per question from `agent_log.jsonl` lines.

    A line that is not a JSON object is counted in `n_bad_lines`; a record without a field
    simply does not enter that average.
    """
    records: list[Mapping[str, object]] = []
    bad = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(item, dict):
            records.append(item)
        else:
            bad += 1

    def numbers(key: str) -> list[float]:
        return [
            float(v)
            for r in records
            if isinstance(v := r.get(key), int | float) and not isinstance(v, bool)
        ]

    verdicts = [r["validator"] for r in records if "validator" in r]
    costs = numbers("cost_usd")
    return AgentCost(
        n_records=len(records),
        n_bad_lines=bad,
        mean_tokens_in=_mean(numbers("tokens_in")),
        mean_tokens_out=_mean(numbers("tokens_out")),
        mean_cost_usd=_mean(costs),
        n_with_cost=len(costs),
        mean_duration_s=_mean([v / 1000 for v in numbers("duration_ms")]),
        first_try_share=(
            sum(v == "first_try" for v in verdicts) / len(verdicts) if verdicts else None
        ),
    )


def truth_counts(events: Sequence[GroundTruthEvent], half: Half) -> dict[str, int]:
    """Ground-truth events of one half by class."""
    counts = {k.value: 0 for k in EventType}
    for e in events:
        if e.half == half:
            counts[e.type.value] += 1
    return counts


def _num(value: float | None, digits: int = 3) -> str:
    return _EMPTY if value is None else f"{value:.{digits}f}"


def _ci(interval: tuple[float, float] | None) -> str:
    return _EMPTY if interval is None else f"[{interval[0]:.3f}; {interval[1]:.3f}]"


def _class_row(name: str, m: Metrics) -> str:
    cells = [
        name,
        str(m.tp),
        str(m.fp),
        str(m.fn),
        _num(m.precision),
        _ci(m.precision_ci),
        _num(m.recall),
        _ci(m.recall_ci),
        _num(f1_score(m.precision, m.recall)),
        _num(m.median_error_s, 2),
        _num(m.p90_error_s, 2),
    ]
    return "| " + " | ".join(cells) + " |"


_CLASS_HEADER = (
    "| Класс | TP | FP | FN | Precision | 95 % CI | Recall | 95 % CI | F1 "
    "| Медиана ошибки, с | P90 ошибки, с |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def class_table(report: EvaluationReport) -> str:
    """Per-class tables, one per tolerance."""
    blocks = []
    for result in report.results:
        rows = [_class_row(k.value, m) for k, m in result.per_class.items()]
        rows.append(_class_row("все (micro)", result.micro))
        blocks.append(f"Допуск ±{result.tolerance_s:g} с\n\n{_CLASS_HEADER}\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def comparison_table(rows: Sequence[SystemRow], tolerances_s: Sequence[float]) -> str:
    """Micro Precision, Recall and F1 of every system at every tolerance, plus timing error.

    The timing error columns use the first tolerance.
    """
    header = ["Система"]
    for t in tolerances_s:
        header += [f"P ±{t:g}", f"R ±{t:g}", f"F1 ±{t:g}"]
    header += ["Медиана ошибки, с", "P90 ошибки, с"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        if row.report is None:
            lines.append(f"| {row.title} | {row.note or _EMPTY} |")
            continue
        cells = [row.title]
        for t in tolerances_s:
            m = row.report.result_for(t).micro
            cells += [_num(m.precision), _num(m.recall), _num(f1_score(m.precision, m.recall))]
        first = row.report.result_for(tolerances_s[0]).micro
        cells += [_num(first.median_error_s, 2), _num(first.p90_error_s, 2)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def edits_table(report: EvaluationReport) -> str:
    """Edits per class: deletes (FP) plus adds (FN), at every tolerance."""
    blocks = []
    for result in report.results:
        lines = [
            f"Допуск ±{result.tolerance_s:g} с",
            "",
            "| Класс | Удалить (FP) | Добавить (FN) | Всего правок |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {k} | {fp} | {fn} | {fp + fn} |" for k, (fp, fn) in count_edits(result).items()
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def stage_table(states: Mapping[str, str]) -> str:
    """Stage states from `status.json`; time is not stored there, so it is "нет данных"."""
    lines = ["| Стадия | Состояние | Время |", "| --- | --- | --- |"]
    lines += [f"| {name} | {state} | нет данных |" for name, state in states.items()]
    return "\n".join(lines)


def cost_table(cost: AgentCost) -> str:
    """Agent cost and time per question."""
    first = _EMPTY if cost.first_try_share is None else f"{cost.first_try_share:.3f}"
    rows = [
        ("Записей в журнале", str(cost.n_records)),
        ("Нечитаемых строк журнала", str(cost.n_bad_lines)),
        ("Токенов на вход на вопрос (среднее)", _num(cost.mean_tokens_in, 0)),
        ("Токенов на выход на вопрос (среднее)", _num(cost.mean_tokens_out, 0)),
        (
            f"Оценка стоимости на вопрос, USD (среднее по {cost.n_with_cost})",
            _num(cost.mean_cost_usd, 4),
        ),
        ("Время ответа на вопрос, с (среднее)", _num(cost.mean_duration_s, 1)),
        ("Доля ответов, прошедших проверку с первой попытки", first),
    ]
    body = "\n".join(f"| {a} | {b} |" for a, b in rows)
    return "| Показатель | Значение |\n| --- | ---: |\n" + body
