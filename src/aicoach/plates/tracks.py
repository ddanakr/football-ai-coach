"""Join per-frame detections into plates (pure functions, no video).

Hits of one type that follow each other with a small gap are one plate, unless the picture of the
panel changes (another person's name): then it is the next plate. Plates of different types never
merge. Refused candidates are joined the same way, by their reason.
"""

from dataclasses import dataclass, field

from aicoach.plates.detect import FrameReading, signature_distance
from aicoach.plates.schemas import CardColor, PlateType, RejectReason

_EPS = 1e-9


@dataclass(frozen=True, eq=False)
class Hit:
    """A candidate frame: its sample index, its video time and what was read."""

    k: int
    t_v: float
    reading: FrameReading


@dataclass(eq=False)
class Track:
    """Consecutive hits of one label: `kind` for a plate, `reject` for a refused candidate."""

    kind: PlateType | None
    reject: RejectReason | None
    hits: list[Hit] = field(default_factory=list)

    @property
    def t_first(self) -> float:
        """Video time of the first hit."""
        return self.hits[0].t_v

    @property
    def t_last(self) -> float:
        """Video time of the last hit."""
        return self.hits[-1].t_v

    @property
    def mid_hit(self) -> Hit:
        """The hit closest to the middle of the plate's life (the earlier one on a tie)."""
        middle = (self.t_first + self.t_last) / 2
        return min(self.hits, key=lambda h: (abs(h.t_v - middle), h.t_v))


def _continues(last: Hit, hit: Hit, merge_gap_s: float, split_diff: float) -> bool:
    """Does `hit` continue the plate whose last hit is `last`: small gap, same panel picture."""
    if hit.t_v - last.t_v > merge_gap_s + _EPS:
        return False
    sig_a, sig_b = last.reading.signature, hit.reading.signature
    if sig_a is None or sig_b is None:
        return True
    return signature_distance(sig_a, sig_b) <= split_diff


def _join(hits: list[Hit], merge_gap_s: float, split_diff: float) -> list[list[Hit]]:
    """Split time-ordered hits of one label into runs."""
    runs: list[list[Hit]] = []
    for hit in hits:
        if runs and _continues(runs[-1][-1], hit, merge_gap_s, split_diff):
            runs[-1].append(hit)
        else:
            runs.append([hit])
    return runs


def build_tracks(
    hits: list[Hit], merge_gap_s: float, split_diff: float, min_hits: int
) -> tuple[list[Track], list[Track]]:
    """Plates and refused candidates, both ordered by time.

    Hits are grouped by label (the plate type, or the reject reason), so a flicker of another
    label inside a plate does not cut it. A plate run with fewer than `min_hits` hits becomes a
    refused candidate with the reason `too_short`.
    """
    ordered = sorted(hits, key=lambda h: h.t_v)
    plates: list[Track] = []
    rejected: list[Track] = []
    labels = {(h.reading.kind, h.reading.reject) for h in ordered}
    for kind, reject in sorted(labels, key=lambda x: (str(x[0]), str(x[1]))):
        group = [h for h in ordered if (h.reading.kind, h.reading.reject) == (kind, reject)]
        for run in _join(group, merge_gap_s, split_diff):
            if kind is not None and len(run) < min_hits:
                rejected.append(Track(None, "too_short", run))
            elif kind is not None:
                plates.append(Track(kind, None, run))
            else:
                rejected.append(Track(None, reject, run))
    plates.sort(key=lambda t: t.t_first)
    rejected.sort(key=lambda t: t.t_first)
    return plates, rejected


def card_color_of(track: Track) -> CardColor | None:
    """The card colour of a card plate (the most frequent over its hits), None for other types."""
    if track.kind != "card":
        return None
    colours = [h.reading.card_color for h in track.hits if h.reading.card_color is not None]
    if not colours:
        return None
    return max(sorted(set(colours)), key=colours.count)


def count_overlaps(plates: list[Track]) -> int:
    """Number of pairs of plates of different types whose time spans intersect."""
    count = 0
    for i, first in enumerate(plates):
        for second in plates[i + 1 :]:
            if first.kind != second.kind and second.t_first <= first.t_last:
                count += 1
    return count
