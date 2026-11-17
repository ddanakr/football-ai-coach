"""The speech test: cut each fragment, transcribe it, score it, write the files (D-066).

Per fragment two runs are possible: `timed` (words with times and speakers; the segments) and, when
a lineup gives terms, `vocabulary` (plain text with the surnames as hints; the model cannot do
both at once). Results go to a directory outside git; the printed summary carries counts and
shares, the transcripts are shown to the owner, not stored in the repository or in MEMORY.md.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from aicoach.errors import TranscriptionError
from aicoach.plate_names.lineup import load_lineup
from aicoach.speech.audio import AUDIO_MIME, cut_audio
from aicoach.speech.check_file import CheckFragment
from aicoach.speech.metrics import Coverage, Gap, coverage, time_gaps
from aicoach.speech.segments import Segment, build_segments, language_shares
from aicoach.speech.transcriber import Transcriber
from aicoach.stage_common import write_atomic

logger = structlog.get_logger()

OUTPUT_DIR = "speech_test"
SUMMARY_FILE = "summary.json"
Cutter = Callable[[Path, Path, float, float], bytes]


@dataclass(frozen=True)
class FragmentRun:
    """The result of one request for one fragment."""

    fragment: CheckFragment
    mode: str  # "timed" or "vocabulary"
    text: str
    segments: tuple[Segment, ...]
    coverage: Coverage
    seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None  # what the service charged, when it says
    gaps: tuple[Gap, ...] = ()  # stretches of 15 s and more without a segment (timed mode)

    @property
    def shares(self) -> dict[str, float]:
        """Share of segments per language label (empty in vocabulary mode)."""
        return language_shares(list(self.segments))


def lineup_terms(lineup: Path | None, *, required: bool = False) -> list[str] | None:
    """Words of the player names of `lineups.csv` as vocabulary terms; `None` without the file.

    Args:
        lineup: The lineup file, or `None`.
        required: The path was given by the user: a missing file is an error, not a skip.

    Raises:
        TranscriptionError: `required` and the file is missing.
        PlatesError: The lineup file is broken (raised by the loader, content not quoted).
    """
    if lineup is not None and required and not lineup.is_file():
        msg = f"lineup file not found: {lineup.name}"
        raise TranscriptionError(msg)
    if lineup is None or not lineup.is_file():
        return None
    words = (w for p in load_lineup(lineup) for w in p.name.split())
    return list(dict.fromkeys(words))


def _file_name(run: FragmentRun) -> str:
    return f"fragment{run.fragment.index}_{run.mode}.json"


def run_to_json(run: FragmentRun, model: str) -> str:
    """The JSON file content of a run (utterance level; no word labels, no listener's words)."""
    frag = run.fragment
    doc = {
        "fragment": frag.index,
        "mode": run.mode,
        "model": model,
        "video_start_s": frag.start_s,
        "video_end_s": frag.end_s,
        "noted_language": frag.language,
        "text": run.text,
        "segments": [s.model_dump() for s in run.segments],
        "seconds": round(run.seconds, 2),
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cost_usd": run.cost_usd,
        "gaps": [_gap_json(g) for g in run.gaps],
    }
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def _gap_json(gap: Gap) -> dict[str, float]:
    return {"start_s": round(gap.start_s, 1), "end_s": round(gap.end_s, 1)}


def _one_run(
    transcriber: Transcriber,
    audio: bytes,
    fragment: CheckFragment,
    vocabulary: Sequence[str] | None,
) -> FragmentRun:
    started = time.perf_counter()
    raw = transcriber.transcribe(
        audio, AUDIO_MIME, vocabulary=vocabulary, duration_s=fragment.length_s
    )
    seconds = time.perf_counter() - started
    segments = build_segments(list(raw.words), fragment.start_s)
    # Gaps only mean something where the model gave times (the vocabulary mode gives none).
    gaps = time_gaps(segments, fragment.length_s) if vocabulary is None else []
    return FragmentRun(
        fragment,
        "timed" if vocabulary is None else "vocabulary",
        raw.text,
        tuple(segments),
        coverage(list(fragment.phrases), raw.text),
        seconds,
        raw.input_tokens,
        raw.output_tokens,
        raw.cost_usd,
        tuple(gaps),
    )


def run_speech_test(
    video: Path,
    fragments: Sequence[CheckFragment],
    out_dir: Path,
    transcriber: Transcriber,
    *,
    vocabulary: Sequence[str] | None = None,
    cutter: Cutter = cut_audio,
) -> list[FragmentRun]:
    """Transcribe every fragment and write `fragmentN_<mode>.json` and `summary.json`.

    Args:
        video: The match video.
        fragments: Fragments of the check file.
        out_dir: Where the sound cuts and the JSON files go (outside git).
        transcriber: The speech-to-text model.
        vocabulary: Terms for the extra `vocabulary` run of each fragment; `None` skips it.
        cutter: Cuts the sound (replaceable in tests).

    Raises:
        ExternalServiceError: The model failed for good on some fragment.
        TranscriptionError: Bad fragment bounds, no sound, or an unreadable answer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[FragmentRun] = []
    for fragment in fragments:
        audio = cutter(video, out_dir / f"fragment{fragment.index}.mp3", fragment.start_s,
                       fragment.length_s)  # fmt: skip
        modes: list[Sequence[str] | None] = [None] if vocabulary is None else [None, vocabulary]
        for terms in modes:
            run = _one_run(transcriber, audio, fragment, terms)
            write_atomic(out_dir / _file_name(run), run_to_json(run, transcriber.model))
            logger.info(
                "speech_fragment",
                stage="speech",
                fragment=fragment.index,
                mode=run.mode,
                duration_ms=round(run.seconds * 1000),
            )
            runs.append(run)
    write_atomic(out_dir / SUMMARY_FILE, summary_to_json(runs, transcriber.model))
    return runs


def _share_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def summary_to_json(runs: Sequence[FragmentRun], model: str) -> str:
    """The summary file: counts, shares, missing words, language shares, time and tokens."""
    rows = [
        {
            "fragment": r.fragment.index,
            "mode": r.mode,
            "found": r.coverage.found,
            "total": r.coverage.total,
            "share": r.coverage.share,
            "missing": list(r.coverage.missing),
            "segment_language_shares": r.shares,
            "segments": len(r.segments),
            "seconds": round(r.seconds, 2),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
            "gaps": [_gap_json(g) for g in r.gaps],
        }
        for r in runs
    ]
    return json.dumps({"model": model, "runs": rows}, ensure_ascii=False, indent=2) + "\n"


def format_summary(runs: Sequence[FragmentRun]) -> str:
    """The summary as text, one block per run."""
    lines: list[str] = []
    for r in runs:
        frag = r.fragment
        shares = ", ".join(f"{k} {v:.0%}" for k, v in r.shares.items()) or "-"
        cost = "n/a" if r.cost_usd is None else f"{r.cost_usd:.5f}"
        lines.append(
            f"fragment {frag.index} ({frag.language}, {frag.length_s:.0f} s) {r.mode}: "
            f"found {r.coverage.found}/{r.coverage.total} ({_share_text(r.coverage.share)}); "
            f"segment languages: {shares}; {r.seconds:.1f} s; "
            f"tokens in {r.input_tokens}, out {r.output_tokens}; cost USD {cost}"
        )
        lines.append(f"  missing: {', '.join(r.coverage.missing) or '-'}")
        if r.gaps:
            shown = ", ".join(f"{g.start_s:.0f}-{g.end_s:.0f} s" for g in r.gaps)
            lines.append(f"  no segments for 15 s and more: {shown}")
    return "\n".join(lines) + "\n"


def format_transcript(run: FragmentRun) -> str:
    """The full text of a run: segments with time, speaker and language, or the plain text."""
    head = f"--- fragment {run.fragment.index} ({run.mode}) ---"
    if not run.segments:
        return f"{head}\n{run.text}\n"
    rows = [
        f"[{s.t_v_start:.1f}-{s.t_v_end:.1f} s] {s.speaker or '?'} {s.language}: {s.text}"
        for s in run.segments
    ]
    return head + "\n" + "\n".join(rows) + "\n"
