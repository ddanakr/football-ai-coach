"""The `final-eval` command body: read files, check the frozen configuration, evaluate, write.

Half 1 is always evaluated (the development half). Half 2 only with `final=True`, once, after
the freeze. Predictions are read from `predictions`, never from `edits` (D-055).
"""

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import structlog

from aicoach.errors import EvalError, StorageError
from aicoach.eval.bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED
from aicoach.eval.final import (
    AgentCost,
    Half,
    HalfSection,
    ShiftCheck,
    SystemRow,
    check_final_allowed,
    class_table,
    comparison_table,
    cost_table,
    edits_table,
    gemini_shift,
    stage_table,
    summarize_agent_log,
    total_edits,
    truth_counts,
)
from aicoach.eval.gemini_baseline import GeminiBaseline, parse_gemini_baseline
from aicoach.eval.matching import filter_predictions_by_half
from aicoach.eval.metrics import DEFAULT_TOLERANCES_S, EvaluationReport, evaluate
from aicoach.eval.schemas import GroundTruth, Prediction, load_ground_truth
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.rules import RULES_VERSION, fuse_signals
from aicoach.plates.schemas import PlatesResult
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.spotting.convert import (
    filter_by_confidence,
    merge_duplicates,
    predictions_in_half,
    to_predictions,
)
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.stage_common import elapsed_ms, write_atomic
from aicoach.status import StageStatus, StatusFile
from aicoach.storage import connect, get_latest_predictions
from aicoach.storage.models import NewPrediction

STAGE: Final = "final_eval"
DEFAULT_TDEED_THRESHOLD: Final = 0.2
_PENDING = "будет заполнено после заморозки"
_DEV_CAVEAT = "пороги и смещения подбирались на этом тайме, цифры оптимистичны"
_FINAL_CAVEAT = "тайм, на котором ничего не подбиралось; эталон размечен вторым проходом"


@dataclass(frozen=True)
class FinalPaths:
    """Input files of one match; `ingest_dir` holds the `status.json` of the full ingest."""

    match_dir: Path
    ground_truth: Path
    model_db: Path
    nomodel_db: Path
    tdeed: Path
    gemini: Path
    ingest_dir: Path


def _to_predictions(rows: Sequence[NewPrediction]) -> list[Prediction]:
    return [
        Prediction(type=r.type, t_v=r.t_v, score=r.score, status=r.status, half=r.half)
        for r in rows
    ]


def _key(
    items: Sequence[tuple[str, float, int | None, str | None, float | None, str]],
) -> list[tuple[str, float, int | None, str | None, float | None, str]]:
    return sorted(
        (t, round(v, 6), h, tm, None if s is None else round(s, 6), st)
        for t, v, h, tm, s, st in items
    )


def verify_run(
    stored: Sequence[NewPrediction],
    board: ScoreboardResult,
    plates: PlatesResult,
    model: Sequence[Prediction] | None,
    config: FuseConfig,
    name: str,
) -> None:
    """Check that a stored run is what the frozen configuration produces from the same inputs.

    Raises:
        EvalError: The rules give another set of events; the message holds counts only.
    """
    out = fuse_signals(board, plates, model, config)
    expected = _key(
        [(str(c.type), c.t_v, c.half, c.team, c.score, c.status) for c in out.candidates]
    )
    got = _key([(str(p.type), p.t_v, p.half, p.team, p.score, p.status) for p in stored])
    if expected != got:
        # No counts: they would include half 2 before the final run.
        msg = (
            f"{name}: the stored predictions do not match the frozen configuration: "
            "run `fuse` again into a new database"
        )
        raise EvalError(msg)


def _report(predictions: Sequence[Prediction], truth: GroundTruth, half: Half) -> EvaluationReport:
    return evaluate(
        predictions,
        truth.events,
        tolerances_s=DEFAULT_TOLERANCES_S,
        half=half,
        halves=truth.halves,
    )


def build_section(
    half: Half,
    *,
    truth: GroundTruth,
    nomodel: Sequence[Prediction],
    tdeed: Sequence[Prediction],
    model_run: Sequence[Prediction],
    gemini: Sequence[Prediction] | None,
    tdeed_threshold: float,
) -> HalfSection:
    """Evaluate the four systems on one half.

    `gemini` is `None` when the baseline must not be compared (half 1 has none).
    """
    in_half = predictions_in_half(tdeed, half, truth.halves)
    bare = filter_by_confidence(merge_duplicates(in_half), tdeed_threshold)
    rows = [
        SystemRow("(a) без модели: табло и плашки", _report(nomodel, truth, half)),
        SystemRow(f"(b) T-DEED без слияния, порог {tdeed_threshold:g}", _report(bare, truth, half)),
        SystemRow("(c) слияние с моделью (основная)", _report(model_run, truth, half)),
    ]
    if gemini is None:
        rows.append(
            SystemRow("(d) Gemini-база", None, "нет данных: база снята только на 2-м тайме")
        )
    else:
        rows.append(SystemRow("(d) Gemini-база", _report(gemini, truth, half)))
    caveat = _DEV_CAVEAT if half == 1 else _FINAL_CAVEAT
    return HalfSection(half, truth_counts(truth.events, half), tuple(rows), caveat)


def _gemini_summary(baseline: GeminiBaseline, truth: GroundTruth, *, final: bool) -> str:
    """Parse statistics; the share inside the half-2 bounds is shown only in the final run."""
    counts = Counter(str(e.type) for e in baseline.events)
    lines = [
        "| Показатель | Значение |",
        "| --- | ---: |",
        f"| Событий в файле (разобрано) | {len(baseline.events)} |",
    ]
    if final:
        inside = len(filter_predictions_by_half(baseline.events, 2, truth.halves))
        lines.append(f"| из них внутри границ 2-го тайма | {inside} |")
    lines += [f"| {kind} | {n} |" for kind, n in sorted(counts.items())]
    lines += [f"| пропущено записей ({why}) | {n} |" for why, n in sorted(baseline.skipped.items())]
    return "\n".join(lines)


def _shift_text(shift: ShiftCheck | None) -> str:
    if shift is None:
        return _PENDING
    if shift.median_signed_s is None:
        return "пар с эталоном нет"
    verdict = (
        "систематический сдвиг больше 10 с (не исправлялся)"
        if shift.systematic
        else "сдвиг не больше 10 с"
    )
    return (
        f"пар {shift.n_pairs}; медиана разности {shift.median_signed_s:+.1f} с; "
        f"медиана модуля {shift.median_abs_s:.1f} с; {verdict}"
    )


_SYSTEM_TITLES: Final = (
    "(a) без модели: табло и плашки",
    "(b) T-DEED без слияния, порог {threshold:g}",
    "(c) слияние с моделью (основная)",
    "(d) Gemini-база",
)
_CLASS_NAMES: Final = ("goal", "shot", "corner", "foul", "subst", "все (micro)")


def _pending_block(half: Half, threshold: float) -> str:
    """The half that is not evaluated yet: the same tables with empty rows."""
    systems = [SystemRow(t.format(threshold=threshold), None, _PENDING) for t in _SYSTEM_TITLES]
    classes = "\n".join(f"| {kind} | {_PENDING} |" for kind in _CLASS_NAMES)
    return (
        f"## {half}-й тайм\n\n"
        f"Не посчитано: {_PENDING}.\n\n"
        f"### Основная система (c) по классам\n\n"
        f"| Класс | Значение |\n| --- | --- |\n{classes}\n\n"
        f"### Сравнение систем\n\n{comparison_table(systems, DEFAULT_TOLERANCES_S)}\n\n"
        f"### Число правок для системы (c)\n\n{_PENDING}.\n"
    )


def _half_block(section: HalfSection | None, half: Half, threshold: float) -> str:
    if section is None:
        return _pending_block(half, threshold)
    counts = ", ".join(f"{k} {n}" for k, n in section.n_truth.items())
    main = section.rows[2].report
    assert main is not None  # noqa: S101 - row (c) always has a report
    return (
        f"## {half}-й тайм\n\n"
        f"Оговорка: {section.caveat}. Отметок эталона: {counts}.\n\n"
        f"### Основная система (c) по классам\n\n{class_table(main)}\n\n"
        f"### Сравнение систем\n\n{comparison_table(section.rows, DEFAULT_TOLERANCES_S)}\n\n"
        f"### Число правок для системы (c)\n\n{edits_table(main)}\n"
    )


def render_document(
    sections: dict[Half, HalfSection],
    *,
    gemini: str,
    shift: ShiftCheck | None,
    cost: AgentCost,
    stages: dict[str, StageStatus],
    threshold: float,
) -> str:
    """The whole `docs/metrics.md`: numbers and shares only."""
    edit_lines = ["| Тайм | Допуск | Правок (FP + FN) |", "| --- | ---: | ---: |"]
    totals = dict.fromkeys(DEFAULT_TOLERANCES_S, 0)
    complete = len(sections) == 2  # noqa: PLR2004 - both halves of a match
    for half, section in sorted(sections.items()):
        main = section.rows[2].report
        assert main is not None  # noqa: S101
        for t in DEFAULT_TOLERANCES_S:
            n = total_edits(main, t)
            totals[t] += n
            edit_lines.append(f"| {half} | ±{t:g} с | {n} |")
    edit_lines.extend(
        f"| сумма по матчу | ±{t:g} с | {totals[t] if complete else _PENDING} |"
        for t in DEFAULT_TOLERANCES_S
    )
    parts = [
        "# Метрики (match01)\n",
        "Сгенерировано командой `final-eval` (см. `SKILLS.md`). Не запуск MLflow. Только числа "
        "и доли: времён событий, имён, команд и счёта здесь нет.\n",
        f"Версия правил слияния: {RULES_VERSION}.\n",
        "Оговорки: один матч, один размечающий, эталон 2-го тайма размечен вторым проходом; "
        f"интервалы 95 % bootstrap по событиям ({DEFAULT_N_BOOT} ресэмплов, seed {DEFAULT_SEED}), "
        "результат это "
        "пилот. Допуск ±T это половина окна сопоставления.\n",
        _half_block(sections.get(1), 1, threshold),
        _half_block(sections.get(2), 2, threshold),
        "## Правки на матч для системы (c)\n\n"
        + "\n".join(edit_lines)
        + "\n\nПравка это удаление ложного события (FP) или добавление пропущенного (FN). "
        "Сумма включает 1-й тайм с оговоркой выше.\n",
        f"## Gemini-база: разбор файла\n\n{gemini}\n\nСдвиг времён: {_shift_text(shift)}.\n",
        "## Время стадий и стоимость\n",
        "Время стадий (`status.json` полной обработки; у пропущенных стадий нет длительности):"
        "\n\n" + stage_table(stages) + "\n",
        "Агент (журнал `agent_log.jsonl`, все записи журнала):\n\n" + cost_table(cost) + "\n",
        "Время T-DEED в Colab, стоимость Gemini-базы: нет данных.\n",
    ]
    return "\n".join(parts)


def _read_stage_states(path: Path) -> dict[str, StageStatus]:
    try:
        return StatusFile.model_validate_json(path.read_text(encoding="utf-8")).stages
    except (OSError, ValueError) as err:
        msg = f"{path}: cannot read the stage states: {err}"
        raise EvalError(msg) from err


def _stored(db: Path, match_id: str) -> list[NewPrediction]:
    if not db.is_file():
        msg = f"database not found: {db}"
        raise EvalError(msg)
    try:
        with connect(db) as conn:
            return get_latest_predictions(conn, match_id)
    except StorageError as err:
        msg = f"{db}: {err}"
        raise EvalError(msg) from err


def run_final_eval(
    paths: FinalPaths,
    output: Path,
    *,
    final: bool = False,
    tdeed_threshold: float = DEFAULT_TDEED_THRESHOLD,
) -> str:
    """Evaluate half 1 (and half 2 with `final`), write the Markdown document.

    Args:
        paths: Input files.
        output: Markdown file to write.
        final: Evaluate half 2 too and compare the Gemini baseline; once, after the freeze.
        tdeed_threshold: Single confidence threshold for system (b).

    Returns:
        The document text.

    Raises:
        EvalError: An input is missing or broken, or a stored run does not match the frozen
            configuration.
    """
    started = time.perf_counter()
    halves: tuple[Half, ...] = (1, 2) if final else (1,)
    for half in halves:
        check_final_allowed(half, final=final)
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
    tdeed = to_predictions(load_tdeed_file(paths.tdeed), config.model_fps)
    nomodel_rows = _stored(paths.nomodel_db, match_id)
    model_rows = _stored(paths.model_db, match_id)
    verify_run(nomodel_rows, board, plates, None, config, "no-model database")
    verify_run(model_rows, board, plates, tdeed, config, "model database")
    try:
        baseline = parse_gemini_baseline(paths.gemini.read_text(encoding="utf-8"))
    except OSError as err:
        msg = f"{paths.gemini}: cannot read the Gemini baseline: {err}"
        raise EvalError(msg) from err
    nomodel, model_run = _to_predictions(nomodel_rows), _to_predictions(model_rows)
    sections: dict[Half, HalfSection] = {}
    for half in halves:
        sections[half] = build_section(
            half,
            truth=truth,
            nomodel=nomodel,
            tdeed=tdeed,
            model_run=model_run,
            gemini=baseline.events if half == 2 and final else None,  # noqa: PLR2004
            tdeed_threshold=tdeed_threshold,
        )
    shift = (
        gemini_shift(
            filter_predictions_by_half(baseline.events, 2, truth.halves),
            [e for e in truth.events if e.half == 2],  # noqa: PLR2004
        )
        if final
        else None
    )
    try:
        log_lines = (paths.match_dir / "agent_log.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        log_lines = []  # no journal: the cost table shows "no data"
    text = render_document(
        sections,
        gemini=_gemini_summary(baseline, truth, final=final),
        shift=shift,
        cost=summarize_agent_log(log_lines),
        stages=_read_stage_states(paths.ingest_dir / "status.json"),
        threshold=tdeed_threshold,
    )
    write_atomic(output, text)
    structlog.get_logger().info(
        "final_eval_done",
        stage=STAGE,
        match_id=match_id,
        final=final,
        duration_ms=elapsed_ms(started),
    )
    return text
