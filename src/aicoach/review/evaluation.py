"""`review-eval`: Recall of the union "events + candidates" and the number of rows to review.

Per half and per class, at the two tolerances of the project. The layer is built here in memory
from the frozen predictions of the model database and the T-DEED file by the same function as the
stage (`build_review_candidates`), so the numbers do not depend on a file that may be stale. The
threshold 0.05 was fixed by the file itself (it is how the T-DEED output was exported) and is not
tuned on any half (D-075). The section is written into `docs/metrics.md` between two markers;
nothing else in the file changes. Numbers and shares only, no event times, names or scores.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aicoach.domain.events import EventType
from aicoach.errors import EvalError, StorageError
from aicoach.eval.final import Half, truth_counts
from aicoach.eval.final_run import verify_run
from aicoach.eval.metrics import DEFAULT_TOLERANCES_S, EvaluationReport, Metrics, evaluate
from aicoach.eval.schemas import GroundTruth, Prediction, load_ground_truth
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.halves import half_bounds
from aicoach.plates.schemas import PlatesResult
from aicoach.review.candidates import build_review_candidates
from aicoach.review.schemas import ReviewCandidate
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.spotting.convert import to_predictions
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.storage import connect, get_latest_predictions
from aicoach.storage.models import NewPrediction

BEGIN_MARKER: Final = "<!-- review-eval:begin -->"
END_MARKER: Final = "<!-- review-eval:end -->"
_EMPTY: Final = "—"
_HALVES: Final[tuple[Half, ...]] = (1, 2)


@dataclass(frozen=True)
class ReviewEvalPaths:
    """Input files of one match (names inside `match_dir` for the database and the T-DEED)."""

    match_dir: Path
    ground_truth: Path
    model_db: Path
    tdeed: Path


@dataclass(frozen=True)
class HalfReview:
    """The numbers of one half: the events alone and the union with the candidates."""

    half: Half
    n_truth: dict[str, int]
    events: EvaluationReport
    union: EvaluationReport


def candidate_predictions(candidates: Sequence[ReviewCandidate]) -> list[Prediction]:
    """The candidates as predictions (their half is known)."""
    return [
        Prediction(type=c.type, t_v=c.t_v, score=c.score, status="auto", half=c.half)
        for c in candidates
    ]


def build_half_review(
    half: Half,
    truth: GroundTruth,
    events: Sequence[Prediction],
    candidates: Sequence[ReviewCandidate],
) -> HalfReview:
    """Evaluate the events alone and the union of events and candidates on one half."""

    def report(preds: Sequence[Prediction]) -> EvaluationReport:
        return evaluate(
            preds,
            truth.events,
            tolerances_s=DEFAULT_TOLERANCES_S,
            half=half,
            halves=truth.halves,
        )

    return HalfReview(
        half=half,
        n_truth=truth_counts(truth.events, half),
        events=report(events),
        union=report([*events, *candidate_predictions(candidates)]),
    )


def _recall(metrics: Metrics) -> str:
    return _EMPTY if metrics.recall is None else f"{metrics.recall:.3f}"


def _rows(metrics: Metrics) -> int:
    return metrics.tp + metrics.fp  # every prediction is a row to look at


def _row(name: str, n_truth: int, events: Metrics, union: Metrics) -> str:
    cells = [
        name,
        str(n_truth),
        str(_rows(events)),
        f"{events.tp}/{n_truth}",
        _recall(events),
        str(_rows(union) - _rows(events)),
        str(_rows(union)),
        f"{union.tp}/{n_truth}",
        _recall(union),
    ]
    return "| " + " | ".join(cells) + " |"


_HEADER = (
    "| Класс | Отметок эталона | События: строк | Найдено | Recall | Кандидатов "
    "| Объединение: строк к проверке | Найдено | Recall |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def half_tables(review: HalfReview) -> str:
    """One table per tolerance: the events alone, the candidates and the union per class."""
    blocks = []
    for events_result, union_result in zip(
        review.events.results, review.union.results, strict=True
    ):
        lines = [
            _row(
                kind.value,
                review.n_truth[kind.value],
                events_result.per_class[kind],
                union_result.per_class[kind],
            )
            for kind in EventType
        ]
        lines.append(
            _row(
                "все (micro)",
                sum(review.n_truth.values()),
                events_result.micro,
                union_result.micro,
            )
        )
        blocks.append(f"Допуск ±{events_result.tolerance_s:g} с\n\n{_HEADER}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def render_section(reviews: Sequence[HalfReview], *, n_model: int, merge_radius_s: float) -> str:
    """The section of `docs/metrics.md`, markers included."""
    parts = [
        BEGIN_MARKER,
        "## Режим проверки: Recall объединения «события + кандидаты» (`review-eval`)\n",
        "Сгенерировано командой `review-eval` (см. `SKILLS.md`). Не запуск MLflow. Только числа: "
        "времён событий, имён, команд и счёта здесь нет. Раздел дописан, цифры выше не менялись.\n",
        "Слой кандидатов (D-075): только удары. Все срабатывания файла T-DEED в классах ударов, "
        f"кроме попавших в окно склейки ({merge_radius_s:g} с) к удару среди событий; дубли внутри "
        "слоя схлопнуты тем же правилом. Кандидатов в других классах нет: голы и замены дают "
        "табло и плашки, у фолов кандидаты дали слишком мало. Это решение принято после того, "
        "как был виден расчёт 2-го тайма. "
        f"Срабатываний модели в классах ударов: {n_model}.\n",
        "**Порог.** Порог 0.05 задан заранее: так выгружен файл T-DEED (`inference.py "
        "--inference_threshold 0.05`). Он не подбирался и под 2-й тайм не подбирается. Параметры "
        "слияния, из которых состоят «События», подбирались на 1-м тайме (D-053).\n",
        "Оговорки: один матч, один размечающий, эталон 2-го тайма размечен вторым проходом, "
        "результат это пилот. «Строк» это число строк, которые просмотрит человек: события или "
        "события и кандидаты. «Найдено» это отметки эталона, у которых есть строка в пределах "
        "допуска (жадное сопоставление по классу, одна строка на одну отметку). Допуск ±T это "
        "половина окна сопоставления.\n",
    ]
    for review in reviews:
        counts = ", ".join(f"{k} {n}" for k, n in review.n_truth.items())
        parts.append(
            f"### {review.half}-й тайм\n\nОтметок эталона: {counts}.\n\n{half_tables(review)}\n"
        )
    parts.append(END_MARKER)
    return "\n".join(parts)


def replace_section(document: str, section: str) -> str:
    """The document with the review section replaced between its markers, or appended (after a
    blank line) when the markers are not there yet. Text outside the markers is not touched.

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


def _read_model_rows(db: Path, match_id: str) -> list[NewPrediction]:
    if not db.is_file():
        msg = f"database not found: {db}"
        raise EvalError(msg)
    try:
        with connect(db) as conn:
            return get_latest_predictions(conn, match_id)
    except StorageError as err:
        msg = f"{db}: {err}"
        raise EvalError(msg) from err


def _write_lf(path: Path, text: str) -> None:
    """Write with LF line ends on every platform (the repository keeps its Markdown in LF), to
    a temporary file renamed over the target, so a failure keeps the old file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        msg = f"cannot write {path}"
        raise StorageError(msg) from err


def run_review_eval(paths: ReviewEvalPaths, output: Path) -> str:
    """Evaluate both halves and write the section into `output` (created if missing).

    The stored run must be what the frozen fusion configuration gives from the same inputs
    (the check of `final-eval`), so the events are the frozen ones.

    Returns:
        The section text.

    Raises:
        EvalError: An input is missing or broken, or the stored run does not match the frozen
            configuration.
        StorageError: The output cannot be written.
    """
    match_id = paths.match_dir.resolve().name
    truth = load_ground_truth(paths.ground_truth)
    config = FuseConfig()
    try:
        board = ScoreboardResult.model_validate_json(
            (paths.match_dir / "scoreboard.json").read_text(encoding="utf-8")
        )
        plates = PlatesResult.model_validate_json(
            (paths.match_dir / "plates.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as err:
        msg = f"cannot read scoreboard.json or plates.json: {err}"
        raise EvalError(msg) from err
    model = to_predictions(load_tdeed_file(paths.tdeed), config.model_fps)
    rows = _read_model_rows(paths.model_db, match_id)
    verify_run(rows, board, plates, model, config, "model database")
    events = [
        Prediction(type=r.type, t_v=r.t_v, score=r.score, status=r.status, half=r.half)
        for r in rows
    ]
    bounds = half_bounds(
        board.offset_segments, config.half_start_margin_s, config.half_end_margin_s
    )
    build = build_review_candidates(
        model,
        [(r.type, r.t_v) for r in rows],
        bounds,
        board.offset_segments,
        merge_radius_s=config.merge_radius_s,
        max_extrapolation_s=ScoreboardConfig().timeline.max_extrapolation_s,
    )
    reviews = [build_half_review(h, truth, events, build.candidates) for h in _HALVES]
    section = render_section(reviews, n_model=build.n_model, merge_radius_s=config.merge_radius_s)
    try:
        document = output.read_text(encoding="utf-8") if output.is_file() else ""
    except OSError as err:
        msg = f"cannot read {output}: {err}"
        raise EvalError(msg) from err
    _write_lf(output, replace_section(document, section))
    return section
