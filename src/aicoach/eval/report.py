"""Markdown rendering of an evaluation report."""

from aicoach.eval.metrics import EvaluationReport, Metrics

_EMPTY = "—"
_HEADER = (
    "| Class | TP | FP | FN | Precision | Precision 95% CI | Recall | Recall 95% CI "
    "| Median err, s | P90 err, s |"
)
_SEPARATOR = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"


def _number(value: float | None, digits: int) -> str:
    return _EMPTY if value is None else f"{value:.{digits}f}"


def _interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return _EMPTY
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _row(name: str, m: Metrics) -> str:
    cells = [
        name,
        str(m.tp),
        str(m.fp),
        str(m.fn),
        _number(m.precision, 3),
        _interval(m.precision_ci),
        _number(m.recall, 3),
        _interval(m.recall_ci),
        _number(m.median_error_s, 2),
        _number(m.p90_error_s, 2),
    ]
    return "| " + " | ".join(cells) + " |"


def format_markdown(report: EvaluationReport) -> str:
    """Render the report as markdown: one table per tolerance, classes as rows.

    Undefined values (no predictions, no ground truth, no matched pairs) are shown as "—".
    """
    scope = "all halves" if report.half is None else f"half {report.half}"
    blocks: list[str] = []
    for result in report.results:
        width = result.tolerance_s * 2
        lines = [
            f"### Tolerance ±{result.tolerance_s:g} s (window {width:g} s), {scope}",
            "",
            _HEADER,
            _SEPARATOR,
        ]
        lines.extend(_row(event_type.value, m) for event_type, m in result.per_class.items())
        lines.append(_row("all (micro)", result.micro))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
