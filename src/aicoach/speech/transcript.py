"""The transcript of a whole match: the models of `transcript.json` and the pure assembly of
segments, player mentions, halves and gaps from the joined words (D-070). No files, no network."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from aicoach.namematch import DEFAULT_MIN_SIMILARITY, find_mentions
from aicoach.plate_names.lineup import Player
from aicoach.speech.metrics import MIN_GAP_S, span_gaps
from aicoach.speech.segments import TimedWord, build_segments

SCHEMA_VERSION = 1
# Rules changed in a way the cache key does not see (segmenting, mentions, gaps): raise this.
RULES_VERSION = 1
RESULT_FILE = "transcript.json"
SEGMENT_ID_PREFIX = "s:"
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Half = Literal[1, 2]
_DIGITS = 3


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TranscribeConfig(_Frozen):
    """Parameters of the stage; they are part of the cache key.

    `chunk_s` 180 s: far below the 25 MB request cap (an MP3 of 32 kbit/s is 0.7 MB) and under
    the 60 s the provider spends on a request (scribe took 5-23 s for 65-99 s of sound, D-069).
    `min_similarity` is the threshold of a fuzzy word, the same as for the plates (D-059).
    """

    chunk_s: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    overlap_s: float = Field(default=10.0, ge=0, allow_inf_nan=False)
    min_similarity: float = Field(default=DEFAULT_MIN_SIMILARITY, gt=0, le=1)
    min_gap_s: float = Field(default=MIN_GAP_S, gt=0, allow_inf_nan=False)


class Mention(_Frozen):
    """A player the commentary speaks of: the lineup spelling and number, the word as said."""

    name: str
    word: str
    side: Literal["home", "away"]
    number: int | None = None
    via: Literal["name", "alias"]
    similarity: float = Field(ge=0, le=1)


class TranscriptSegment(_Frozen):
    """One utterance of one speaker. `id` is `s:<n>` in the order of time. `language` is a
    guess from the letters (D-066)."""

    id: str
    t_start: VideoTime
    t_end: VideoTime
    half: Half | None = None
    speaker: str | None = None
    language: str
    text: str
    players: list[Mention] = Field(default_factory=list)


class ChunkInfo(_Frozen):
    """One request: its stretch, the words it returned and what it cost (None: not reported)."""

    index: int
    start_s: VideoTime
    end_s: VideoTime
    n_words: int
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TimeGap(_Frozen):
    """A stretch of a half with no speech for at least `min_gap_s` seconds."""

    t_start: VideoTime
    t_end: VideoTime
    half: Half | None = None


class HalfStats(_Frozen):
    """Speech and gaps in a half. `gap_share` is the gaps over the length of the half."""

    half: Half | None
    start: VideoTime
    end: VideoTime
    n_segments: int
    n_gaps: int
    gap_s: float
    gap_share: float


class TranscriptStats(_Frozen):
    """Counters of the whole transcript. `cost_usd` is the sum of the costs the service
    reported for the chunks (None when it reported none)."""

    n_segments: int
    n_words: int
    n_mentions: int
    n_chunks: int
    cost_usd: float | None
    speakers_unresolved: int


class TranscriptResult(_Frozen):
    """The whole `transcript.json`. `lineup` is the file used for mentions (None: none)."""

    schema_version: int = SCHEMA_VERSION
    cache_key: str
    model: str
    source: str
    duration_s: VideoTime
    config: TranscribeConfig
    lineup: str | None
    lineup_version: str | None
    chunks: list[ChunkInfo]
    segments: list[TranscriptSegment]
    gaps: list[TimeGap]
    halves: list[HalfStats]
    stats: TranscriptStats
    warnings: list[str] = Field(default_factory=list)


def half_of(t_v: float, spans: Mapping[int, tuple[float, float]]) -> Half | None:
    """The half whose span holds `t_v` (ends included); None in the break or the studio."""
    for half in (1, 2):
        span = spans.get(half)
        if span is not None and span[0] <= t_v <= span[1]:
            return half
    return None


def build_transcript_segments(
    words: Sequence[TimedWord],
    spans: Mapping[int, tuple[float, float]],
    lineup: list[Player] | None,
    min_similarity: float,
) -> list[TranscriptSegment]:
    """Utterance segments of the words (video seconds) with half, language and the players the
    text speaks of. No words give no segments; without a lineup no player is named."""
    found: list[TranscriptSegment] = []
    for number, seg in enumerate(build_segments(list(words), 0.0)):
        mentions = [
            Mention(
                name=m.player.name,
                word=m.source,
                side=m.player.side,
                number=m.player.number,
                via=m.via,
                similarity=min(1.0, m.similarity),
            )
            for m in (find_mentions(seg.text, lineup, min_similarity) if lineup else [])
        ]
        found.append(
            TranscriptSegment(
                id=f"{SEGMENT_ID_PREFIX}{number}",
                t_start=seg.t_v_start,
                t_end=seg.t_v_end,
                half=half_of((seg.t_v_start + seg.t_v_end) / 2, spans),
                speaker=seg.speaker,
                language=seg.language,
                text=seg.text,
                players=mentions,
            )
        )
    return found


def find_gaps(
    segments: Sequence[TranscriptSegment],
    spans: Mapping[int, tuple[float, float]],
    duration_s: float,
    min_gap_s: float,
) -> tuple[list[TimeGap], list[HalfStats]]:
    """Gaps of at least `min_gap_s` seconds inside each half, and the per-half numbers.

    Only the halves are looked at (the break and the studio are silent or not commentary). Without
    known halves the whole sound is one span with `half = None`.
    """
    targets: list[tuple[Half | None, float, float]] = [
        (half, span[0], min(span[1], duration_s))
        for half in (1, 2)
        if (span := spans.get(half)) is not None
    ] or [(None, 0.0, duration_s)]
    gaps: list[TimeGap] = []
    stats: list[HalfStats] = []
    for half, start, end in targets:
        inside = [s for s in segments if s.t_end > start and s.t_start < end]
        found = span_gaps(((s.t_start, s.t_end) for s in inside), start, end, min_gap_s)
        gaps.extend(TimeGap(t_start=g.start_s, t_end=g.end_s, half=half) for g in found)
        gap_s = sum(g.length_s for g in found)
        length = max(end - start, 0.0)
        stats.append(
            HalfStats(
                half=half,
                start=start,
                end=end,
                n_segments=len(inside),
                n_gaps=len(found),
                gap_s=round(gap_s, _DIGITS),
                gap_share=round(gap_s / length, _DIGITS) if length else 0.0,
            )
        )
    return gaps, stats
