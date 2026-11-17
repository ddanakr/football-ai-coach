"""A small `TranscriptResult` for tests of the command line (invented text)."""

from aicoach.speech.transcript import (
    HalfStats,
    TranscribeConfig,
    TranscriptResult,
    TranscriptSegment,
    TranscriptStats,
)


def make_result(*, cost: float | None = 0.0123) -> TranscriptResult:
    """One segment, one half with one gap."""
    segment = TranscriptSegment(
        id="s:0", t_start=10.0, t_end=14.0, half=1, language="ru", text="Тест тест"
    )
    return TranscriptResult(
        cache_key="k",
        model="fake/stt",
        source="audio.wav",
        duration_s=100.0,
        config=TranscribeConfig(),
        lineup=None,
        lineup_version=None,
        chunks=[],
        segments=[segment],
        gaps=[],
        halves=[
            HalfStats(half=1, start=0, end=100, n_segments=1, n_gaps=1, gap_s=20, gap_share=0.2)
        ],
        stats=TranscriptStats(
            n_segments=1,
            n_words=2,
            n_mentions=0,
            n_chunks=1,
            cost_usd=cost,
            speakers_unresolved=0,
        ),
        warnings=["a warning"],
    )
