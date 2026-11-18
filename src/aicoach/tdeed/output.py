"""From the raw output of `inference.py` to the file that `fuse` reads.

A whole-match run keeps the output as it is. A segment run moves the frame numbers to the time
of the whole match (the segment starts at its own frame 0) and keeps only what lies inside the
piece the user asked for.
"""

import json
from pathlib import Path

from aicoach.errors import SpottingError
from aicoach.spotting.schemas import TDeedFile, TDeedPrediction, load_tdeed_file
from aicoach.stage_common import write_atomic
from aicoach.tdeed.config import (
    BATCH_SIZE,
    FPS,
    MODEL,
    OUTPUT_PREFIX,
    SEGMENT_PREFIX,
    THRESHOLD,
)
from aicoach.tdeed.timecode import Segment


def output_name(segment: Segment | None) -> str:
    """The file name: `tdeed_<model>_thr<threshold>_bs<batch>.json` for a whole match (the name of
    the frozen file), `tdeedseg_..._<from>-<to>s.json` for a segment (which `fuse` does not see)."""
    base = f"{MODEL}_thr{THRESHOLD:g}_bs{BATCH_SIZE}"
    if segment is None:
        return f"{OUTPUT_PREFIX}{base}.json"
    return f"{SEGMENT_PREFIX}{base}_{round(segment.start_s)}-{round(segment.end_s)}s.json"


def shift_to_match_time(raw: TDeedFile, segment: Segment) -> TDeedFile:
    """Predictions of a segment run in the frames of the whole match, those inside the piece only.

    Raises:
        SpottingError: The segment does not start on a whole frame of the 25 fps video.
    """
    offset = segment.start_s * FPS
    if offset != int(offset):
        msg = f"the segment start {segment.start_s} s is not a whole frame at {FPS} fps"
        raise SpottingError(msg)
    first, last = segment.start_s * FPS, segment.end_s * FPS
    shifted = [
        TDeedPrediction(frame=p.frame + int(offset), label=p.label, confidence=p.confidence)
        for p in raw.predictions
    ]
    kept = tuple(p for p in shifted if first <= p.frame <= last)
    return TDeedFile(predictions=kept)


def write_output(path: Path, data: TDeedFile) -> None:
    """Write the file the way T-DEED does (`indent=4`), atomically.

    Raises:
        StorageError: The file cannot be written.
    """
    payload = {
        "predictions": [
            {"frame": p.frame, "label": p.label.value, "confidence": p.confidence}
            for p in data.predictions
        ]
    }
    write_atomic(path, json.dumps(payload, indent=4))


def convert_output(raw_path: Path, target: Path, segment: Segment | None) -> int:
    """Validate the raw output and write `target`; returns the number of predictions written.

    Raises:
        SpottingError: The raw file is missing or not a valid T-DEED output.
    """
    raw = load_tdeed_file(raw_path)
    if segment is None:
        write_atomic(target, raw_path.read_text(encoding="utf-8"))
        return len(raw.predictions)
    data = shift_to_match_time(raw, segment)
    write_output(target, data)
    return len(data.predictions)
