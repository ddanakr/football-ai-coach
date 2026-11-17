"""The command `corner-eval`: read the truth file and the tags, write the section of
`docs/metrics.md` (D-078).

Half 1 only, unless `final` is set: the question of the model is tuned on the corners of the first
half, and the second half is a check that is made once with the question unchanged (D-008). A
truth file with nothing filled in stops the command with a clear message instead of reporting
empty tables.
"""

from pathlib import Path

from aicoach.corner_tags.evaluation import (
    FieldScore,
    filled_rows,
    load_truth,
    render_section,
    replace_section,
    rows_of_half,
    score_rows,
)
from aicoach.corner_tags.schemas import read_tags_file
from aicoach.errors import EvalError, StorageError

TRUTH_FILE = "corner_truth.csv"


def _write_lf(path: Path, text: str) -> None:
    """Write with LF line ends (the repository keeps its Markdown in LF) through a temporary
    file renamed over the target."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        msg = f"cannot write {path}"
        raise StorageError(msg) from err


def run_corner_eval(
    match_dir: Path, truth_path: Path, output: Path, *, final: bool
) -> dict[int, list[FieldScore]]:
    """Score the tags of the match against the truth file and write the report section.

    Args:
        match_dir: The match directory (`corner_tags.json`).
        truth_path: The filled truth file.
        output: The Markdown file whose corner section is written (created if missing).
        final: Also evaluate the second half. Without it the second half is not looked at.

    Returns:
        The scores per evaluated half.

    Raises:
        EvalError: The truth file is missing, invalid or empty, there are no tags, or the
            output file has only one of the two markers.
        StorageError: The output cannot be written.
    """
    rows = load_truth(truth_path)
    if filled_rows(rows) == 0:
        msg = f"{truth_path} is not filled in yet: no corner has any value (see its header)"
        raise EvalError(msg)
    tagged = read_tags_file(match_dir)
    if tagged is None:
        msg = f"no corner_tags.json in {match_dir}: run corner-tags first"
        raise EvalError(msg)
    tags = {t.event_id: t for t in tagged.tags}
    halves = (1, 2) if final else (1,)
    scores: dict[int, list[FieldScore]] = {}
    counts: dict[int, int] = {}
    for half in halves:
        in_half = rows_of_half(rows, tags, half)
        scores[half] = score_rows(in_half, tags)
        counts[half] = len(in_half)
    costs = [t.cost_usd for t in tagged.tags]
    cost = None if not costs or any(c is None for c in costs) else sum(c or 0.0 for c in costs)
    first = tagged.tags[0] if tagged.tags else None
    section = render_section(
        scores,
        n_corners=counts,
        model=first.model if first else "—",
        prompt_version=first.prompt_version if first else 0,
        cost_usd=cost,
    )
    try:
        document = output.read_text(encoding="utf-8") if output.is_file() else ""
    except OSError as err:
        msg = f"cannot read {output}: {err}"
        raise EvalError(msg) from err
    _write_lf(output, replace_section(document, section))
    return scores
